from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from fixtures import tiny, build, runtime, inputs, NET
from execution import run_trial
from https_core.engine import build_client
from https_core.transport import HttpsRpc, LoopbackServer
from prepare_tls import prepare_credentials


class HttpsNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.reference = tiny()
        cls.reference.save_pretrained(root / "model")
        cls.local_model, cls.state, cls.local_rpc = build(root / "model", cls.reference)
        cls.credentials = prepare_credentials(root / "credentials")
        cls.server = LoopbackServer(cls.state, cls.credentials, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05})
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(5)
        cls.server.server_close()
        cls.temp.cleanup()

    def test_actual_tls_matches_local_wire_computation_with_zero_and_positive_noise(self):
        port = self.server.server_address[1]
        rpc = HttpsRpc(f"https://localhost:{port}", self.credentials, timeout=3, limit=NET["max_body_bytes"])
        model = build_client(self.local_model.model.weights, rpc, self.state.identity, 128)
        for sigma in (0.0, 0.1):
            config = {"distribution": "gaussian", "sigma": sigma, "seed": 42}
            expected = run_trial(runtime(self.local_model, self.state, self.local_rpc), inputs(), config, 42)
            actual = run_trial(runtime(model, self.state, rpc), inputs(), config, 42)
            self.assertEqual(expected["token_ids"], actual["token_ids"])
            for a, b in zip(expected["logits"], actual["logits"]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(actual["communication"]["requests"], 5)
            self.assertFalse(self.state.sessions)

    def test_server_receives_no_noise_seed_or_client_text(self):
        port = self.server.server_address[1]
        rpc = HttpsRpc(f"https://localhost:{port}", self.credentials, timeout=3, limit=NET["max_body_bytes"])
        model = build_client(self.local_model.model.weights, rpc, self.state.identity, 128)
        original, observed = self.state.dispatch, []

        def inspect(path, metadata, tensors):
            if path == "/v1/forward":
                observed.append((set(metadata), set(tensors)))
            return original(path, metadata, tensors)

        with patch.object(self.state, "dispatch", side_effect=inspect):
            run_trial(runtime(model, self.state, rpc), inputs(),
                      {"distribution": "gaussian", "sigma": 0.02, "seed": 7}, 7)
        self.assertEqual(len(observed), 3)
        for metadata, tensors in observed:
            self.assertEqual(metadata, {"session", "seq", "past"})
            self.assertEqual(tensors, {"hidden", "attention_mask", "position_ids", "cache_position"})
        self.assertFalse(self.state.sessions)


if __name__ == "__main__":
    unittest.main()
