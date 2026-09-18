"""Exercise the MoE adapter with small random weights, not the 30B checkpoint."""

import sys
import unittest
from pathlib import Path

import torch
from transformers import GenerationConfig, Qwen3MoeConfig, Qwen3MoeForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_model import Partition, build_split_model


class Qwen3MoeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen3MoeConfig(
            vocab_size=97, hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
            num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        cls.reference = Qwen3MoeForCausalLM(config).eval()
        cls.reference.requires_grad_(False)
        cls.ids = torch.tensor([[5, 7, 9, 11]])

    def test_forward_for_multiple_partitions(self):
        for partition in (Partition(2, 2, 2), Partition(1, 2, 3), Partition(3, 2, 1)):
            with self.subTest(partition=partition), torch.inference_mode():
                split = build_split_model(self.reference, partition)
                expected = self.reference(input_ids=self.ids, use_cache=False).logits
                actual = split(input_ids=self.ids, use_cache=False).logits
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_experts_and_untied_head_are_reused(self):
        split = build_split_model(self.reference, Partition(2, 2, 2))
        self.assertIs(split.model.stage_b.layers[0].mlp, self.reference.model.layers[2].mlp)
        self.assertIs(split.lm_head, self.reference.lm_head)
        self.assertIsNot(split.lm_head.weight, split.model.stage_a.embed_tokens.weight)
        self.assertEqual({id(x) for x in split.parameters()}, {id(x) for x in self.reference.parameters()})

    def test_generation(self):
        split = build_split_model(self.reference, Partition(1, 2, 3))
        config = GenerationConfig(
            max_new_tokens=5, min_new_tokens=5, use_cache=False, do_sample=False,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        with torch.inference_mode():
            left = self.reference.generate(self.ids, generation_config=config)
            right = split.generate(self.ids, generation_config=config)
        self.assertTrue(torch.equal(left, right))

    def test_padding(self):
        split = build_split_model(self.reference, Partition(2, 2, 2))
        ids = torch.tensor([[0, 0, 5, 7], [5, 7, 9, 11]])
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        with torch.inference_mode():
            left = self.reference(
                input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
            ).logits
            right = split(input_ids=ids, attention_mask=mask, position_ids=positions).logits
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_wrong_adapter_rejected(self):
        with self.assertRaises(ValueError):
            build_split_model(self.reference, Partition(2, 2, 2), "qwen2")

    def test_router_traces_rejected(self):
        split = build_split_model(self.reference, Partition(2, 2, 2))
        with self.assertRaises(ValueError):
            split(input_ids=self.ids, output_router_logits=True)


if __name__ == "__main__":
    unittest.main()
