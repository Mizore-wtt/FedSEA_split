"""Tiny random dense/MoE checkpoints; no download or real-model requirement."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import (
    GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM,
)
from transformers.cache_utils import DynamicCache

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from https_core.settings import ROOT
from fedsea.partition import Partition
from https_core.weights import RoleWeights
from https_core.engine import ServerState, build_client
from https_core.protocol import pack, unpack
from https_core.session import HttpsSession
sys.path.append(str(ROOT / "3_kv_cache"))
from session import CachedSession

NET = {
    "max_context_tokens": 128, "max_sessions": 4, "session_ttl_seconds": 600,
    "max_body_bytes": 1024 * 1024, "timeout_seconds": 3,
}


def tiny(moe=False, tied=True):
    config_class, model_class = (Qwen3MoeConfig, Qwen3MoeForCausalLM) if moe else (Qwen2Config, Qwen2ForCausalLM)
    config = config_class(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=tied,
        head_dim=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    )
    config._attn_implementation = "eager"
    return model_class(config).eval().requires_grad_(False)


class LocalRpc:
    """Same protocol bytes as HTTPS, deterministic transport for unit failure injection."""

    def __init__(self, state):
        self.state = state
        self.calls = []

    def call(self, path, metadata, tensors=None):
        self.calls.append(path)
        meta, data = unpack(pack(metadata, tensors))
        output, tensors = self.state.dispatch(path, meta, data)
        return unpack(pack(output, tensors))


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.examples = []
        for index, (moe, tied) in enumerate(((False, True), (False, False), (True, False), (True, True))):
            model = tiny(moe, tied)
            folder = Path(cls.temp.name) / str(index)
            model.save_pretrained(folder, max_shard_size="12KB" if moe else "5GB")
            cls.examples.append((model, folder))
        cls.ids = torch.tensor([[5, 7, 9, 11, 13, 15, 17]])
        cls.gen = GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=4, min_new_tokens=4,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, index=0, part=Partition(2, 2, 2)):
        reference, folder = self.examples[index]
        client = RoleWeights(folder, reference.config, part, "client", torch.device("cpu"), torch.float32)
        server = RoleWeights(folder, reference.config, part, "server", torch.device("cpu"), torch.float32)
        identity = {"test": index, "p": part.p, "k": part.k, "q": part.q}
        state = ServerState(server, identity, NET)
        rpc = LocalRpc(state)
        return build_client(client, rpc, identity, 128), state, rpc

    def inputs(self, ids=None):
        ids = self.ids[:, :4] if ids is None else ids
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def test_owned_parameter_keys_disjoint_and_complete(self):
        for index, (ref, _) in enumerate(self.examples):
            client, state, _ = self.build(index)
            left, right = client.model.weights, state.weights
            self.assertFalse(set(left.loaded_keys) & set(right.loaded_keys))
            self.assertEqual(left.parameter_count + right.parameter_count, sum(p.numel() for p in ref.parameters()))
            self.assertEqual(left.indices, [0, 1, 4, 5])
            self.assertEqual(right.indices, [2, 3])
            self.assertFalse(hasattr(right, "embed_tokens"))
            self.assertFalse(hasattr(right, "lm_head"))
            self.assertEqual(left.lm_head.weight is left.embed_tokens.weight, ref.config.tie_word_embeddings)
            self.assertEqual(left.rotary_emb.inv_freq.dtype, torch.float32)

    def test_prefill_decode_exact_logits_and_all_caches(self):
        for index, (ref, _) in enumerate(self.examples):
            for part in (Partition(2, 2, 2), Partition(1, 2, 3), Partition(3, 2, 1)):
                with self.subTest(index=index, part=part), torch.inference_mode():
                    model, state, _ = self.build(index, part)
                    rc, sc = DynamicCache(config=ref.config), model.new_cache()
                    start = 0
                    try:
                        for stop in (4, 5, 6, 7):
                            data = {"input_ids": self.ids[:, start:stop], "attention_mask": torch.ones_like(self.ids[:, :stop])}
                            expected = ref(**data, past_key_values=rc, use_cache=True)
                            actual = model(**data, past_key_values=sc, use_cache=True)
                            torch.testing.assert_close(expected.logits, actual.logits, atol=0, rtol=0)
                            all_layers = sc.a.layers + state.sessions[sc.sid]["cache"].layers + sc.c.layers
                            for a, b in zip(rc.layers, all_layers):
                                torch.testing.assert_close(a.keys, b.keys, atol=0, rtol=0)
                                torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
                            self.assertEqual(sc.assert_consistent(), stop)
                            start = stop
                    finally:
                        sc.close()
                    self.assertFalse(state.sessions)

    def test_generation_and_multiturn_exact(self):
        for index, (ref, _) in enumerate(self.examples):
            model, state, _ = self.build(index)
            session = HttpsSession(model)
            full_session = CachedSession(ref)
            try:
                inputs = self.inputs()
                for turn in range(2):
                    expected, _ = full_session.generate(inputs, self.gen, trace=True)
                    actual, metrics = session.generate(inputs, self.gen, trace=True)
                    self.assertTrue(torch.equal(expected.sequences, actual.sequences))
                    for a, b in zip(expected.logits, actual.logits):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                    self.assertEqual(metrics["query_tokens_per_forward"], [4 if turn == 0 else 3, 1, 1, 1])
                    self.assertEqual(metrics["reused_prefix_tokens"], 0 if turn == 0 else 7)
                    inputs = self.inputs(torch.cat((actual.sequences, torch.tensor([[21, 23]])), dim=1))
            finally:
                session.close()
            self.assertFalse(state.sessions)

    def test_repeated_prompt_and_edited_prefix_crop(self):
        model, state, rpc = self.build()
        session = HttpsSession(model)
        try:
            first, _ = session.generate(self.inputs(), self.gen)
            again, metrics = session.generate(self.inputs(), self.gen)
            self.assertTrue(torch.equal(first.sequences, again.sequences))
            self.assertEqual(metrics["reused_prefix_tokens"], 3)
            self.assertIn("/v1/crop", rpc.calls)
            for column in (2, 0):
                ids = self.ids[:, :4].clone()
                ids[0, column] = 33
                actual, metrics = session.generate(self.inputs(ids), self.gen)
                self.assertEqual(metrics["reused_prefix_tokens"], column)
                with torch.inference_mode():
                    expected = self.examples[0][0].generate(**self.inputs(ids), generation_config=self.gen)
                self.assertTrue(torch.equal(actual.sequences, expected))
        finally:
            session.close()
        self.assertFalse(state.sessions)

    def test_lost_forward_response_never_retried_and_cache_cleared(self):
        model, state, rpc = self.build()
        session = HttpsSession(model)
        original = rpc.call

        def lose(path, meta, tensors=None):
            result = original(path, meta, tensors)
            if path == "/v1/forward":
                raise RuntimeError("response lost AFTER B committed")
            return result

        with patch.object(rpc, "call", side_effect=lose), self.assertRaises(RuntimeError):
            session.generate(self.inputs(), self.gen)
        self.assertEqual(rpc.calls.count("/v1/forward"), 1)
        self.assertFalse(state.sessions)
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertIsNone(session.cached_ids)
        self.assertFalse(model._forward_pre_hooks)
        session.generate(self.inputs(), self.gen)
        session.close()

    def test_partial_server_failure_clears_session(self):
        model, state, _ = self.build()
        session = HttpsSession(model)
        with patch.object(state.weights.layers["3"], "forward", side_effect=RuntimeError("failed B")), self.assertRaises(RuntimeError):
            session.generate(self.inputs(), self.gen)
        self.assertFalse(state.sessions)
        self.assertEqual(session.cache.assert_consistent(), 0)

    def test_partial_client_failure_closes_remote_cache(self):
        model, state, _ = self.build()
        session = HttpsSession(model)
        with patch.object(model.model.weights.layers["5"], "forward", side_effect=RuntimeError("failed C")), self.assertRaises(RuntimeError):
            session.generate(self.inputs(), self.gen)
        self.assertFalse(state.sessions)
        self.assertEqual(session.cache.assert_consistent(), 0)

    def test_replay_invalidates_remote_session(self):
        model, state, rpc = self.build()
        session = HttpsSession(model)
        session.generate(self.inputs(), self.gen)
        with self.assertRaises(ValueError):
            rpc.call("/v1/crop", {"session": session.cache.sid, "seq": 0, "past": 7, "keep": 3})
        self.assertFalse(state.sessions)
        session.close()

    def test_capacity_ttl_and_identity(self):
        _, state, rpc = self.build()
        now = [0.0]
        state.clock = lambda: now[0]
        with self.assertRaises(ValueError):
            rpc.call("/v1/session", {"identity": {}})
        for _ in range(4):
            rpc.call("/v1/session", {"identity": state.identity})
        with self.assertRaises(ValueError):
            rpc.call("/v1/session", {"identity": state.identity})
        now[0] = 601
        state.expire()
        self.assertFalse(state.sessions)
        rpc.call("/v1/session", {"identity": state.identity})
        self.assertEqual(len(state.sessions), 1)

    def test_left_padding_exact(self):
        model, _, _ = self.build()
        inputs = {"input_ids": torch.tensor([[0, 0, 5, 7]]), "attention_mask": torch.tensor([[0, 0, 1, 1]])}
        session = HttpsSession(model)
        try:
            actual, _ = session.generate(inputs, self.gen, trace=True)
            with torch.inference_mode():
                expected = self.examples[0][0].generate(
                    **inputs, generation_config=self.gen, return_dict_in_generate=True, output_logits=True
                )
            for a, b in zip(actual.logits, expected.logits):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        finally:
            session.close()

    def test_bad_mask_position_and_context_fail_closed(self):
        model, state, _ = self.build()
        for data in (
            {"attention_mask": torch.ones(1, 5, dtype=torch.long)},
            {"cache_position": torch.tensor([0, 1, 2, 4])},
            {"position_ids": torch.tensor([[0, 1, 2, 128]])},
            {"input_ids": torch.ones(1, 129, dtype=torch.long), "attention_mask": torch.ones(1, 129, dtype=torch.long)},
        ):
            cache = model.new_cache()
            with torch.inference_mode(), self.assertRaises(ValueError):
                model(**{**self.inputs(), **data}, past_key_values=cache)
            self.assertFalse(state.sessions)
            self.assertFalse(cache.valid)

    def test_unsupported_modes_and_wrong_owner(self):
        model, _, _ = self.build()
        other, _, _ = self.build()
        with self.assertRaises(ValueError):
            model(**self.inputs(), past_key_values=other.new_cache())
        for options in ({"use_cache": False}, {"num_beams": 2}, {"do_sample": True},
                        {"cache_implementation": "static"}):
            with self.assertRaises(ValueError):
                model.generate(**self.inputs(), generation_config=self.gen, **options)

    def test_independent_sessions_and_reset(self):
        model, state, _ = self.build()
        left, right = HttpsSession(model), HttpsSession(model)
        left.generate(self.inputs(), self.gen)
        right.generate(self.inputs(self.ids[:, :6]), self.gen)
        self.assertEqual(len(state.sessions), 2)
        self.assertNotEqual(left.cache.sid, right.cache.sid)
        left.reset()
        self.assertEqual(len(state.sessions), 1)
        self.assertEqual(right.cache.assert_consistent(), 9)
        self.assertTrue(all(layer.keys is None for layer in left.cache.layers))
        right.close()

    def test_corrupt_or_missing_owned_weights_rejected(self):
        from safetensors.torch import load_file, save_file
        ref, folder = self.examples[0]
        with tempfile.TemporaryDirectory() as temp:
            state = load_file(folder / "model.safetensors")
            del state["model.layers.2.self_attn.q_proj.weight"]
            save_file(state, str(Path(temp) / "model.safetensors"))
            with self.assertRaises(ValueError):
                RoleWeights(temp, ref.config, Partition(2, 2, 2), "server", torch.device("cpu"), torch.float32)

    def test_initial_synchronize_failure_removes_hook_and_clears_cache(self):
        model, state, _ = self.build()
        session = HttpsSession(model)
        before = len(model._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            session.generate(
                self.inputs(), self.gen,
                synchronize=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
            )
        self.assertEqual(len(model._forward_pre_hooks), before)
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertFalse(state.sessions)

    def test_failed_prefix_crop_closes_remote_session(self):
        model, state, _ = self.build()
        session = HttpsSession(model)
        session.generate(self.inputs(), self.gen)
        self.assertTrue(state.sessions)
        with patch.object(session.cache, "crop", side_effect=RuntimeError("crop failed")):
            with self.assertRaisesRegex(RuntimeError, "crop failed"):
                session.generate(self.inputs(), self.gen)
        self.assertEqual(session.cache.assert_consistent(), 0)
        self.assertIsNone(session.cached_ids)
        self.assertFalse(state.sessions)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available; CPU tests remain required.")
    def test_cuda_rope_matches_cpu_initialized_native_at_long_positions(self):
        from transformers import AutoModelForCausalLM
        config = Qwen2Config(
            vocab_size=97, hidden_size=128, intermediate_size=256, num_hidden_layers=6,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=1000000.0,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"
        with tempfile.TemporaryDirectory() as temp:
            Qwen2ForCausalLM(config).save_pretrained(temp)
            reference = AutoModelForCausalLM.from_pretrained(
                temp, local_files_only=True, dtype=torch.float16, attn_implementation="eager"
            ).to("cuda").eval().requires_grad_(False)
            part = Partition(2, 2, 2)
            left = RoleWeights(temp, reference.config, part, "client", torch.device("cuda"), torch.float16)
            right = RoleWeights(temp, reference.config, part, "server", torch.device("cuda"), torch.float16)
            self.assertTrue(torch.equal(reference.model.rotary_emb.inv_freq, left.rotary_emb.inv_freq))
            self.assertTrue(torch.equal(reference.model.rotary_emb.inv_freq, right.rotary_emb.inv_freq))
            net = {**NET, "max_context_tokens": 1152}
            state = ServerState(right, {"test": "cuda-rope"}, net)
            model = build_client(left, LocalRpc(state), state.identity, 1152)
            rc, sc = DynamicCache(config=reference.config), model.new_cache()
            try:
                with torch.inference_mode():
                    start = 0
                    ids = torch.arange(1027, device="cuda").reshape(1, -1) % 80 + 3
                    for stop in (1024, 1025, 1026, 1027):
                        inputs = {"input_ids": ids[:, start:stop], "attention_mask": torch.ones_like(ids[:, :stop])}
                        expected = reference(**inputs, past_key_values=rc, use_cache=True)
                        actual = model(**inputs, past_key_values=sc, use_cache=True)
                        torch.testing.assert_close(expected.logits, actual.logits, atol=0, rtol=0)
                        start = stop
            finally:
                sc.close()


if __name__ == "__main__":
    unittest.main()
