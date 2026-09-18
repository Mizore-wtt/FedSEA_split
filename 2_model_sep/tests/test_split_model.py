import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_model import Partition, SplitQwen2ForCausalLM


class SplitModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, tie_word_embeddings=True,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        config._attn_implementation = "eager"
        cls.reference = Qwen2ForCausalLM(config).eval()
        cls.reference.requires_grad_(False)

    def setUp(self):
        self.split = SplitQwen2ForCausalLM(self.reference, Partition(2, 2, 2))
        self.ids = torch.tensor([[5, 11, 7, 20]])

    def assert_forward_equal(self, split=None, **inputs):
        split = split or self.split
        with torch.inference_mode():
            expected = self.reference(**inputs, use_cache=False).logits
            actual = split(**inputs, use_cache=False).logits
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_full_forward_exact(self):
        self.assert_forward_equal(input_ids=self.ids)

    def test_alternative_partitions(self):
        for partition in (Partition(1, 4, 1), Partition(1, 1, 4), Partition(4, 1, 1)):
            with self.subTest(partition=partition):
                split = SplitQwen2ForCausalLM(self.reference, partition)
                self.assert_forward_equal(split, input_ids=self.ids)

    def test_no_parameter_copies(self):
        self.assertEqual(
            {id(p) for p in self.reference.parameters()},
            {id(p) for p in self.split.parameters()},
        )

    def test_tied_embedding_preserved(self):
        self.assertIs(self.split.model.stage_a.embed_tokens.weight, self.split.lm_head.weight)
        self.assertIs(self.split.lm_head, self.reference.lm_head)

    def test_layer_order_and_indices(self):
        layers = (
            list(self.split.model.stage_a.layers) + list(self.split.model.stage_b.layers)
            + list(self.split.model.stage_c.layers)
        )
        for index, layer in enumerate(layers):
            self.assertIs(layer, self.reference.model.layers[index])
            self.assertEqual(layer.self_attn.layer_idx, index)

    def test_reference_cache_settings_unchanged(self):
        self.assertTrue(self.reference.config.use_cache)
        self.assertFalse(self.split.config.use_cache)
        self.assertTrue(self.reference.generation_config.use_cache)

    def test_no_reference_forward_shortcut(self):
        with patch.object(self.reference, "forward", side_effect=AssertionError("shortcut")):
            with torch.inference_mode():
                output = self.split(input_ids=self.ids, use_cache=False)
            self.assertEqual(output.logits.shape, (1, 4, 97))

    def test_variable_lengths(self):
        for length in (1, 16, 128):
            ids = torch.arange(length).unsqueeze(0) % 90 + 3
            self.assert_forward_equal(input_ids=ids, attention_mask=torch.ones_like(ids))

    def test_left_padding_and_positions(self):
        ids = torch.tensor([[0, 0, 5, 11], [5, 11, 7, 20]])
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        self.assert_forward_equal(input_ids=ids, attention_mask=mask, position_ids=positions)

    def test_future_tokens_do_not_change_prefix(self):
        changed = self.ids.clone()
        changed[0, -1] = 30
        with torch.inference_mode():
            left = self.split(input_ids=self.ids).logits[:, :-1]
            right = self.split(input_ids=changed).logits[:, :-1]
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_inputs_embeds(self):
        embedding = self.reference.get_input_embeddings()(self.ids)
        self.assert_forward_equal(inputs_embeds=embedding)

    def test_last_logits_slice(self):
        with torch.inference_mode():
            expected = self.reference(input_ids=self.ids, use_cache=False, logits_to_keep=1).logits
            actual = self.split(input_ids=self.ids, use_cache=False, logits_to_keep=1).logits
        self.assertEqual(actual.shape, (1, 1, 97))
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_generation_multiple_steps(self):
        config = GenerationConfig(
            max_new_tokens=5, min_new_tokens=5, use_cache=False,
            do_sample=False, pad_token_id=0, eos_token_id=2,
        )
        with torch.inference_mode():
            expected = self.reference.generate(self.ids, generation_config=config)
            actual = self.split.generate(self.ids, generation_config=config)
        self.assertEqual(actual.shape[-1], self.ids.shape[-1] + 5)
        self.assertTrue(torch.equal(expected, actual))

    def test_cache_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support KV"):
            self.split(input_ids=self.ids, use_cache=True)
        with self.assertRaisesRegex(ValueError, "does not support KV"):
            self.split(input_ids=self.ids, past_key_values=object())

    def test_invalid_partitions(self):
        for partition in (Partition(0, 4, 2), Partition(-1, 5, 2),
                          Partition(2, 3, 2), Partition(True, 3, 2), Partition(1.0, 3, 2)):
            with self.subTest(partition=partition):
                with self.assertRaises(ValueError):
                    SplitQwen2ForCausalLM(self.reference, partition)

    def test_invalid_inputs(self):
        bad_inputs = [
            {},
            {"input_ids": self.ids, "inputs_embeds": torch.zeros(1, 4, 32)},
            {"input_ids": torch.empty((1, 0), dtype=torch.long)},
            {"input_ids": self.ids, "attention_mask": torch.ones(1, 3)},
            {"input_ids": self.ids, "position_ids": torch.zeros(1, 3, dtype=torch.long)},
            {"input_ids": self.ids, "cache_position": torch.arange(4) + 1},
        ]
        for inputs in bad_inputs:
            with self.subTest(keys=list(inputs)):
                with self.assertRaises(ValueError):
                    self.split(**inputs)


if __name__ == "__main__":
    unittest.main()
