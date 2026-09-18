"""Small random Qwen2 and MoE models; no real checkpoints or downloads."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cached_model import Partition, build_cached_model
from session import CachedSession


class CachedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        common = dict(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        configs = [
            Qwen2Config(**common, tie_word_embeddings=True),
            Qwen3MoeConfig(**common, head_dim=8, num_experts=4, num_experts_per_tok=2,
                           moe_intermediate_size=32, tie_word_embeddings=False),
        ]
        cls.models = []
        for config, model_class in zip(configs, (Qwen2ForCausalLM, Qwen3MoeForCausalLM)):
            config._attn_implementation = "eager"
            model = model_class(config).eval()
            model.requires_grad_(False)
            cls.models.append(model)
        cls.ids = torch.tensor([[5, 7, 9, 11, 13, 15, 17]])
        cls.generation = GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=4, min_new_tokens=4,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )

    def inputs(self, ids=None):
        ids = self.ids[:, :4] if ids is None else ids
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def test_prefill_and_incremental_exact_all_partitions(self):
        for ref in self.models:
            for part in (Partition(2, 2, 2), Partition(1, 2, 3), Partition(3, 2, 1)):
                with self.subTest(model=ref.config.model_type, partition=part), torch.inference_mode():
                    split = build_cached_model(ref, part)
                    expected_cache, actual_cache = DynamicCache(config=ref.config), split.new_cache()
                    start = 0
                    for stop in (4, 5, 6, 7):
                        inputs = {
                            "input_ids": self.ids[:, start:stop],
                            "attention_mask": torch.ones_like(self.ids[:, :stop]),
                        }
                        left = ref(**inputs, past_key_values=expected_cache, use_cache=True)
                        right = split(**inputs, past_key_values=actual_cache, use_cache=True)
                        torch.testing.assert_close(left.logits, right.logits, atol=0, rtol=0)
                        self.assertEqual(actual_cache.assert_consistent(), stop)
                        for a, b in zip(expected_cache.layers, actual_cache.layers):
                            torch.testing.assert_close(a.keys, b.keys, atol=0, rtol=0)
                            torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
                        start = stop

    def test_generate_uses_single_token_steps(self):
        for ref in self.models:
            split = build_cached_model(ref, Partition(2, 2, 2))
            expected, em = CachedSession(ref).generate(self.inputs(), self.generation, trace=True)
            actual, am = CachedSession(split).generate(self.inputs(), self.generation, trace=True)
            self.assertTrue(torch.equal(expected.sequences, actual.sequences))
            for left, right in zip(expected.logits, actual.logits):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(am["query_tokens_per_forward"], [4, 1, 1, 1])
            self.assertEqual(em["query_tokens_per_forward"], am["query_tokens_per_forward"])
            self.assertEqual(am["cached_tokens"], 7)

    def test_left_padding_prefill_and_decode(self):
        ids = torch.tensor([[0, 0, 5, 7], [5, 7, 9, 11]])
        mask = (ids != 0).long()
        for ref in self.models:
            split = build_cached_model(ref, Partition(1, 2, 3))
            rc, sc = DynamicCache(config=ref.config), split.new_cache()
            for step in range(2):
                positions = mask.cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 1)
                current = ids if step == 0 else ids[:, -1:]
                with torch.inference_mode():
                    left = ref(
                        input_ids=current, attention_mask=mask,
                        position_ids=positions[:, -current.shape[1]:], past_key_values=rc, use_cache=True,
                    )
                    right = split(
                        input_ids=current, attention_mask=mask,
                        position_ids=positions[:, -current.shape[1]:], past_key_values=sc,
                    )
                torch.testing.assert_close(left.logits, right.logits, atol=0, rtol=0)
                ids = torch.cat((ids, torch.tensor([[13], [15]])), dim=1)
                mask = torch.cat((mask, torch.ones(2, 1, dtype=torch.long)), dim=1)

    def test_multi_turn_prefix_reuse(self):
        for ref in self.models:
            split = build_cached_model(ref, Partition(1, 2, 3))
            full_session, split_session = CachedSession(ref), CachedSession(split)
            first, _ = full_session.generate(self.inputs(), self.generation)
            split_session.generate(self.inputs(), self.generation)
            ids = torch.cat((first.sequences, torch.tensor([[21, 23]])), dim=1)
            left, lm = full_session.generate(self.inputs(ids), self.generation, trace=True)
            right, rm = split_session.generate(self.inputs(ids), self.generation, trace=True)
            self.assertEqual(rm["reused_prefix_tokens"], 7)
            self.assertEqual(rm["prefill_tokens"], 3)
            self.assertEqual(rm["query_tokens_per_forward"], [3, 1, 1, 1])
            self.assertEqual(lm["reused_prefix_tokens"], rm["reused_prefix_tokens"])
            self.assertTrue(torch.equal(left.sequences, right.sequences))
            for a, b in zip(left.logits, right.logits):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            fresh, _ = CachedSession(ref).generate(self.inputs(ids), self.generation)
            self.assertTrue(torch.equal(fresh.sequences, right.sequences))

    def test_repeated_prompt_crops_before_last_token(self):
        session = CachedSession(build_cached_model(self.models[0], Partition(2, 2, 2)))
        first, _ = session.generate(self.inputs(), self.generation)
        second, metrics = session.generate(self.inputs(), self.generation)
        self.assertEqual(metrics["reused_prefix_tokens"], 3)
        self.assertEqual(metrics["query_tokens_per_forward"], [1, 1, 1, 1])
        self.assertTrue(torch.equal(first.sequences, second.sequences))

    def test_changed_prefix_rebuilds(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        session.generate(self.inputs(), self.generation)
        changed = self.ids[:, :4].clone()
        changed[0, 0] = 33
        output, metrics = session.generate(self.inputs(changed), self.generation)
        fresh, _ = CachedSession(split).generate(self.inputs(changed), self.generation)
        self.assertEqual(metrics["reused_prefix_tokens"], 0)
        self.assertTrue(torch.equal(output.sequences, fresh.sequences))

    def test_partial_prefix_edit_crops(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        session.generate(self.inputs(), self.generation)
        changed = self.ids[:, :4].clone()
        changed[0, 2] = 33
        output, metrics = session.generate(self.inputs(changed), self.generation)
        fresh, _ = CachedSession(split).generate(self.inputs(changed), self.generation)
        self.assertEqual(metrics["reused_prefix_tokens"], 2)
        self.assertTrue(torch.equal(output.sequences, fresh.sequences))

    def test_reset_releases_history(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        session.generate(self.inputs(), self.generation)
        session.cache.reset()
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertTrue(all(layer.keys is None for layer in session.cache.layers))
        session.reset()
        self.assertIsNone(session.cached_ids)
        self.assertTrue(all(row["kv_bytes"] == 0 for row in session.cache.describe().values()))
        _, metrics = session.generate(self.inputs(), self.generation)
        self.assertEqual(metrics["reused_prefix_tokens"], 0)

    def test_interleaved_sessions_are_isolated(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        a, b = CachedSession(split), CachedSession(split)
        first, _ = a.generate(self.inputs(), self.generation)
        pointers = [layer.keys.data_ptr() for layer in a.cache.layers]
        b.generate(self.inputs(self.ids[:, :6]), self.generation)
        self.assertEqual(pointers, [layer.keys.data_ptr() for layer in a.cache.layers])
        self.assertEqual(a.cache.get_seq_length(), 7)
        self.assertEqual(b.cache.get_seq_length(), 9)
        self.assertNotEqual(a.cache.layers[0].keys.data_ptr(), b.cache.layers[0].keys.data_ptr())
        again, _ = a.generate(self.inputs(), self.generation)
        self.assertTrue(torch.equal(first.sequences, again.sequences))

    def test_parameter_and_expert_objects_reused(self):
        for ref in self.models:
            split = build_cached_model(ref, Partition(2, 2, 2))
            self.assertEqual({id(p) for p in ref.parameters()}, {id(p) for p in split.parameters()})
            self.assertIs(split.model.stage_b.layers[0].mlp, ref.model.layers[2].mlp)
            self.assertIs(split.lm_head, ref.lm_head)
            tied = ref.lm_head.weight is ref.model.embed_tokens.weight
            self.assertEqual(tied, split.lm_head.weight is split.model.stage_a.embed_tokens.weight)
            self.assertEqual([layer.self_attn.layer_idx for layer in ref.model.layers], list(range(6)))

    def test_reference_forward_not_called(self):
        ref = self.models[0]
        split = build_cached_model(ref, Partition(2, 2, 2))
        with patch.object(ref, "forward", side_effect=AssertionError("reference shortcut")):
            with torch.inference_mode():
                split(**self.inputs())

    def test_cache_range_and_owner_enforced(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        cache = split.new_cache()
        tensor = torch.zeros(1, 2, 1, 8)
        with self.assertRaises(ValueError):
            cache.b.update(tensor, tensor, 0)
        with self.assertRaises(ValueError):
            cache.update(tensor, tensor, 0)
        other = build_cached_model(self.models[0], Partition(2, 2, 2))
        with self.assertRaises(ValueError):
            other(**self.inputs(), past_key_values=cache)
        with self.assertRaises(ValueError):
            split(**self.inputs(), past_key_values=DynamicCache())

    def test_replay_and_gap_rejected_before_mutation(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        cache = split.new_cache()
        with torch.inference_mode():
            split(**self.inputs(), past_key_values=cache)
        for position in (0, 3, 5):
            with self.assertRaises(ValueError):
                split(
                    input_ids=self.ids[:, :1], attention_mask=torch.ones(1, 5),
                    past_key_values=cache, cache_position=torch.tensor([position]),
                )
            self.assertEqual(cache.assert_consistent(), 4)

    def test_bad_input_shapes_and_masks(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        for inputs in (
            {"input_ids": self.ids[:, :0]},
            {"input_ids": self.ids.float()},
            {"input_ids": self.ids, "attention_mask": torch.ones(1, 8)},
            {"input_ids": self.ids, "attention_mask": torch.full_like(self.ids, 2)},
            {"input_ids": self.ids, "inputs_embeds": torch.zeros(1, 7, 32)},
        ):
            with self.subTest(inputs=list(inputs)), self.assertRaises(ValueError):
                split(**inputs)

    def test_context_overflow_and_generation_budget(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        with self.assertRaises(ValueError):
            split(input_ids=torch.ones(1, 129, dtype=torch.long))
        with self.assertRaises(ValueError):
            CachedSession(split).generate(self.inputs(torch.ones(1, 126, dtype=torch.long)), self.generation)

    def test_changed_cached_mask_rejected(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        cache = split.new_cache()
        with torch.inference_mode():
            split(**self.inputs(), past_key_values=cache)
        mask = torch.ones(1, 5, dtype=torch.long)
        mask[0, 1] = 0
        with self.assertRaises(ValueError):
            split(input_ids=self.ids[:, :1], attention_mask=mask, past_key_values=cache)

    def test_partial_failure_invalidates_cache(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        cache = split.new_cache()
        with patch.object(split.model.stage_b, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError), torch.inference_mode():
                split(**self.inputs(), past_key_values=cache)
        self.assertFalse(cache.valid)
        with self.assertRaises(ValueError):
            split(**self.inputs(), past_key_values=cache)
        cache.reset()
        with torch.inference_mode():
            split(**self.inputs(), past_key_values=cache)
        self.assertEqual(cache.assert_consistent(), 4)

    def test_failed_session_releases_partial_cache_and_hooks(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        hooks = len(split._forward_pre_hooks)
        with patch.object(split.model.stage_b, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                session.generate(self.inputs(), self.generation)
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertIsNone(session.cached_ids)
        self.assertEqual(len(split._forward_pre_hooks), hooks)

    def test_forced_eos_stops_and_leaves_last_token_pending(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        config = GenerationConfig(
            max_new_tokens=5, use_cache=True, do_sample=False, pad_token_id=0, bos_token_id=1,
            eos_token_id=2, forced_decoder_ids=None, suppress_tokens=None,
        )
        from transformers import LogitsProcessor, LogitsProcessorList

        class ForceEos(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores.fill_(-float("inf"))
                scores[:, 2] = 0
                return scores

        with torch.inference_mode():
            output = split.generate(
                **self.inputs(), generation_config=config, return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([ForceEos()]),
            )
        self.assertEqual(output.sequences.shape[1], 5)
        self.assertEqual(output.sequences[0, -1].item(), 2)
        self.assertEqual(output.past_key_values.assert_consistent(), 4)

    def test_unsupported_modes_and_wrong_adapter(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        with self.assertRaises(ValueError):
            split(**self.inputs(), use_cache=False)
        with self.assertRaises(ValueError):
            split(**self.inputs(), output_attentions=True)
        with self.assertRaises(ValueError):
            build_cached_model(self.models[0], Partition(2, 2, 2), "qwen3_moe")
        for option in ({"use_cache": False}, {"do_sample": True}, {"num_beams": 2},
                       {"cache_implementation": "static"}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                split.generate(**self.inputs(), generation_config=self.generation, **option)

    def test_inputs_embeds(self):
        for ref in self.models:
            split = build_cached_model(ref, Partition(2, 2, 2))
            with torch.inference_mode():
                embeds = ref.model.embed_tokens(self.ids)
                left = ref(inputs_embeds=embeds, use_cache=True)
                right = split(inputs_embeds=embeds)
            torch.testing.assert_close(left.logits, right.logits, atol=0, rtol=0)

    def test_prefix_crop_failure_resets_the_session(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        session.generate(self.inputs(), self.generation)
        with patch.object(session.cache, "crop", side_effect=RuntimeError("crop failed")):
            with self.assertRaisesRegex(RuntimeError, "crop failed"):
                session.generate(self.inputs(), self.generation)
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertIsNone(session.cached_ids)

    def test_initial_synchronize_failure_removes_hook_and_clears_cache(self):
        split = build_cached_model(self.models[0], Partition(2, 2, 2))
        session = CachedSession(split)
        before = len(split._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            session.generate(
                self.inputs(), self.generation,
                synchronize=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
            )
        self.assertEqual(len(split._forward_pre_hooks), before)
        self.assertEqual(session.cache.assert_consistent(), 0)


if __name__ == "__main__":
    unittest.main()
