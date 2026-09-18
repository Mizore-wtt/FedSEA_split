from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fixtures import build, tiny, inputs, runtime, GEN, Partition
from noisy_session import NoiseSession, NoisyRpc
from noise import GaussianNoise
from execution import run_trial, generate_reference
from https_core.session import HttpsSession

CONFIG = {"distribution": "gaussian", "sigma": 0.1, "seed": 42}


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.examples = []
        for index in range(2):
            reference = tiny(moe=bool(index))
            directory = Path(cls.temp.name) / str(index)
            reference.save_pretrained(directory)
            cls.examples.append((reference, directory))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, index=0, part=Partition(2, 2, 2)):
        reference, directory = self.examples[index]
        return build(directory, reference, part)

    def test_zero_matches_full_model_across_architectures_and_splits(self):
        for index, (reference, _) in enumerate(self.examples):
            for part in (Partition(2, 2, 2), Partition(1, 2, 3), Partition(3, 2, 1)):
                model, state, rpc = self.build(index, part)
                runner = runtime(model, state, rpc)
                expected_runner = runtime(reference, state, rpc)
                expected = generate_reference(expected_runner, inputs())
                actual = run_trial(runner, inputs(), {**CONFIG, "sigma": 0}, 42)
                self.assertEqual(expected["token_ids"], actual["token_ids"])
                for left, right in zip(expected["logits"], actual["logits"]):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                self.assertFalse(state.sessions)
                self.assertEqual(actual["noise"]["forward_calls"], 3)

    def test_only_A_output_is_changed_on_the_wire(self):
        model, state, rpc = self.build()
        clean = HttpsSession(model)
        clean.generate(inputs(), GEN)
        clean.close()
        original_uploads = rpc.uploads
        rpc.uploads = []
        noisy = NoiseSession(model, CONFIG, 42)
        noisy.generate(inputs(), GEN)
        noisy.close()
        expected = GaussianNoise(0.1, 42).apply(original_uploads[0][1]["hidden"])
        self.assertTrue(torch.equal(rpc.uploads[0][1]["hidden"], expected))
        for name in ("attention_mask", "position_ids", "cache_position"):
            self.assertTrue(torch.equal(original_uploads[0][1][name], rpc.uploads[0][1][name]))
        for meta, tensors in rpc.uploads:
            self.assertEqual(set(meta), {"session", "seq", "past"})
            self.assertEqual(set(tensors), {"hidden", "attention_mask", "position_ids", "cache_position"})
        self.assertEqual([data["hidden"].shape[1] for _, data in rpc.uploads], [4, 1, 1])

    def test_repeated_generations_rebuild_noisy_prefixes(self):
        model, state, rpc = self.build()
        session = NoiseSession(model, CONFIG, 42)
        try:
            first, _ = session.generate(inputs(), GEN, trace=True)
            second, metrics = session.generate(inputs(), GEN, trace=True)
            self.assertTrue(torch.equal(first.sequences, second.sequences))
            for left, right in zip(first.logits, second.logits):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(metrics["reused_prefix_tokens"], 0)
            self.assertNotIn("/v1/crop", rpc.calls)
        finally:
            session.close()
        self.assertFalse(state.sessions)

    def test_noisy_and_clean_sessions_do_not_change_the_shared_model(self):
        model, state, rpc = self.build()
        runner = runtime(model, state, rpc)
        expected = run_trial(runner, inputs(), {**CONFIG, "sigma": 0}, 1)
        first = run_trial(runner, inputs(), CONFIG, 42)
        different = run_trial(runner, inputs(), CONFIG, 43)
        again = run_trial(runner, inputs(), CONFIG, 42)
        clean = run_trial(runner, inputs(), {**CONFIG, "sigma": 0}, 2)
        self.assertIs(model.model.rpc, rpc)
        torch.testing.assert_close(first["logits"][0], again["logits"][0], atol=0, rtol=0)
        self.assertFalse(torch.equal(first["logits"][0], different["logits"][0]))
        torch.testing.assert_close(expected["logits"][0], clean["logits"][0], atol=0, rtol=0)
        self.assertFalse(state.sessions)

    def test_lost_response_closes_noisy_session_without_retry(self):
        model, state, rpc = self.build()
        original = rpc.call

        def fail(path, meta, tensors=None):
            result = original(path, meta, tensors)
            if path == "/v1/forward":
                raise RuntimeError("reply lost")
            return result

        session = NoiseSession(model, CONFIG, 42)
        with patch.object(rpc, "call", side_effect=fail), self.assertRaisesRegex(RuntimeError, "reply lost"):
            session.generate(inputs(), GEN)
        self.assertEqual(rpc.calls.count("/v1/forward"), 1)
        self.assertFalse(state.sessions)
        self.assertFalse(model._forward_pre_hooks)
        self.assertEqual(session.cache.assert_consistent(), 0)

    def test_invalid_noise_is_not_uploaded_and_session_is_closed(self):
        model, state, rpc = self.build()
        session = NoiseSession(model, CONFIG, 42)
        with patch.object(GaussianNoise, "apply", side_effect=ValueError("invalid noise")):
            with self.assertRaisesRegex(ValueError, "invalid noise"):
                session.generate(inputs(), GEN)
        self.assertNotIn("/v1/forward", rpc.calls)
        self.assertFalse(state.sessions)
        self.assertFalse(model._forward_pre_hooks)

    def test_rpc_adapter_does_not_mutate_input_dictionary(self):
        class Sink:
            def call(self, path, meta, tensors):
                return tensors

        hidden = torch.ones(1, 2, 32)
        data = {"hidden": hidden, "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "cache_position": torch.arange(2), "position_ids": torch.arange(2).reshape(1, -1)}
        result = NoisyRpc(Sink(), GaussianNoise(0.1, 42)).call("/v1/forward", {}, data)
        self.assertIs(data["hidden"], hidden)
        self.assertTrue(torch.equal(hidden, torch.ones_like(hidden)))
        self.assertIsNot(result, data)
        self.assertIs(result["attention_mask"], data["attention_mask"])

    def test_communication_counts_include_session_open_and_close(self):
        model, state, rpc = self.build()
        result = run_trial(runtime(model, state, rpc), inputs(), CONFIG, 42)
        self.assertEqual(result["communication"]["requests"], 5)
        self.assertEqual(rpc.calls[0], "/v1/session")
        self.assertEqual(rpc.calls[-1], "/v1/close")
        self.assertIsNone(result["metrics"]["client_peak_gpu_allocated_mib"])
        self.assertFalse(state.sessions)


if __name__ == "__main__":
    unittest.main()
