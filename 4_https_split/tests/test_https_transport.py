"""Real loopback TLS and strict protocol validation, using temporary credentials."""

import copy
import http.client
import json
from pathlib import Path
import socket
import ssl
import struct
import sys
import tempfile
import threading
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_tls import prepare_credentials
from https_core.protocol import pack, unpack, validate_step, CONTENT_TYPE
from https_core.transport import HttpsRpc, LoopbackServer, RpcError
from https_core.engine import ServerState, build_client
from https_core.weights import RoleWeights
from https_core.session import HttpsSession
from https_core.settings import validate_network, read_json, STAGE
from fedsea.partition import Partition
from test_https_engine import tiny, NET


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_all_supported_dtypes_and_aliases(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.long):
            data = torch.arange(24).reshape(1, 3, 8).to(dtype)
            meta, tensors = unpack(pack({"seq": 3}, {"left": data, "right": data}))
            self.assertEqual(meta, {"seq": 3})
            self.assertTrue(torch.equal(data, tensors["left"]))
            self.assertTrue(torch.equal(data, tensors["right"]))

    def test_malformed_frame_and_duplicate_json(self):
        for payload in (b"", b"123", struct.pack("!I", 999999) + b"{}",
                        struct.pack("!I", 13) + b'{"x":1,"x":2}', pack({}) + b"broken"):
            with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                unpack(payload)

    def test_oversize_body_header_and_tensor_shape(self):
        with self.assertRaises(ValueError):
            pack({"text": "x" * 17000})
        with self.assertRaises(ValueError):
            unpack(pack({}), limit=4)
        header = json.dumps({"hidden": {"dtype": "F32", "shape": [1, 1000000000, 896],
                                      "data_offsets": [0, 0]}}).encode()
        with self.assertRaises(ValueError):
            unpack(pack({}) + struct.pack("<Q", len(header)) + header)

    def test_no_text_tokens_or_nonfinite_hidden(self):
        config = tiny().config
        tensors = {
            "hidden": torch.zeros(1, 2, 32), "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]]), "cache_position": torch.tensor([0, 1]),
        }
        self.assertEqual(validate_step(tensors, config, torch.float32, 0, 128), 2)
        for bad in (
            {**tensors, "input_ids": torch.tensor([[1, 2]])},
            {**tensors, "hidden": torch.full((1, 2, 32), float("nan"))},
            {**tensors, "hidden": tensors["hidden"].half()},
            {**tensors, "attention_mask": torch.full((1, 2), 2, dtype=torch.long)},
            {**tensors, "position_ids": torch.tensor([[0, -1]])},
        ):
            with self.assertRaises(ValueError):
                validate_step(bad, config, torch.float32, 0, 128)

    def test_network_settings_reject_public_and_plaintext(self):
        net = read_json(STAGE / "config.json")["https"]
        validate_network(net)
        for override in ({"host": "0.0.0.0"}, {"url": "http://localhost:8443"},
                         {"url": "https://example.org:8443"}, {"port": True},
                         {"max_context_tokens": 100000}, {"credentials_dir": "../secrets"}):
            with self.assertRaises(ValueError):
                validate_network({**net, **override})


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.credentials = prepare_credentials(root / "credentials")
        cls.reference = tiny().eval()
        cls.reference.save_pretrained(root / "model")
        cls.part = Partition(2, 2, 2)
        cls.identity = {"test": "real-tls", "p": 2, "k": 2, "q": 2}
        cls.weights = RoleWeights(root / "model", cls.reference.config, cls.part,
                                  "client", torch.device("cpu"), torch.float32)
        server_weights = RoleWeights(root / "model", cls.reference.config, cls.part,
                                     "server", torch.device("cpu"), torch.float32)
        cls.state = ServerState(server_weights, cls.identity, NET)
        cls.server = LoopbackServer(cls.state, cls.credentials, 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05})
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(5)
        cls.server.server_close()
        cls.temp.cleanup()

    def setUp(self):
        with self.state.lock:
            self.state.sessions.clear()

    def rpc(self, host="localhost"):
        return HttpsRpc(f"https://{host}:{self.port}", self.credentials, timeout=3, limit=NET["max_body_bytes"])

    def raw(self, headers, body=b"", path="/v1/health"):
        rpc = self.rpc()
        connection = http.client.HTTPSConnection("localhost", self.port, context=rpc.context, timeout=3)
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_authenticated_health_both_sans(self):
        for host in ("localhost", "127.0.0.1"):
            result, tensors = self.rpc(host).call("/v1/health", {})
            self.assertEqual(result["identity"], self.identity)
            self.assertEqual(result["weights"]["layers_1based"], [3, 4])
            self.assertFalse(tensors)

    def test_real_https_generation_matches_complete(self):
        from transformers import GenerationConfig
        rpc = self.rpc()
        model = build_client(self.weights, rpc, self.identity, 128)
        session = HttpsSession(model)
        generation = GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=4, min_new_tokens=4,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        inputs = {"input_ids": torch.tensor([[5, 7, 9, 11]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        try:
            actual, metrics = session.generate(inputs, generation, trace=True)
            with torch.inference_mode():
                expected = self.reference.generate(**inputs, generation_config=generation,
                                                   return_dict_in_generate=True, output_logits=True)
            for a, b in zip(expected.logits, actual.logits):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertTrue(torch.equal(actual.sequences, expected.sequences))
            self.assertEqual(metrics["query_tokens_per_forward"], [4, 1, 1, 1])
        finally:
            session.close()
        health, _ = rpc.call("/v1/health", {})
        self.assertEqual(health["active_sessions"], 0)

    def test_missing_and_wrong_bearer_rejected(self):
        body = pack({})
        for auth in (None, "Bearer " + "0" * 64):
            headers = {"Content-Type": CONTENT_TYPE}
            if auth is not None:
                headers["Authorization"] = auth
            status, _ = self.raw(headers, body)
            self.assertEqual(status, 401)

    def test_untrusted_ca_rejected(self):
        context = ssl.create_default_context()
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as raw:
            with self.assertRaises(ssl.SSLCertVerificationError):
                context.wrap_socket(raw, server_hostname="localhost")

    def test_wrong_hostname_rejected(self):
        context = self.rpc().context
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as raw:
            with self.assertRaises(ssl.SSLCertVerificationError):
                context.wrap_socket(raw, server_hostname="not-localhost.invalid")

    def test_plain_http_not_accepted(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as raw:
            raw.sendall(b"GET / HTTP/1.0\r\n\r\n")
            try:
                reply = raw.recv(128)
            except (ConnectionResetError, ConnectionAbortedError):
                reply = b""
            self.assertNotIn(b"HTTP/1.", reply)

    def test_wrong_identity_and_replay_over_https(self):
        rpc = self.rpc()
        with self.assertRaises(RpcError):
            rpc.call("/v1/session", {"identity": {"test": "different"}})
        reply, _ = rpc.call("/v1/session", {"identity": self.identity})
        with self.assertRaises(RpcError):
            rpc.call("/v1/crop", {"session": reply["session"], "seq": 9, "past": 0, "keep": 0})
        health, _ = rpc.call("/v1/health", {})
        self.assertEqual(health["active_sessions"], 0)

    def test_oversized_and_malformed_requests_rejected(self):
        headers = {"Authorization": "Bearer " + self.rpc().token, "Content-Type": CONTENT_TYPE}
        status, _ = self.raw({**headers, "Content-Length": str(NET["max_body_bytes"] + 1)})
        self.assertEqual(status, 413)
        status, _ = self.raw(headers, b"broken frame")
        self.assertEqual(status, 400)
        status, _ = self.raw({**headers, "Transfer-Encoding": "chunked"}, pack({}))
        self.assertEqual(status, 400)
        status, _ = self.raw(headers, pack({}), "/v1/unknown")
        self.assertEqual(status, 404)

    def test_existing_credentials_never_overwritten(self):
        before = (self.credentials / "token.txt").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_credentials(self.credentials)
        self.assertEqual(before, (self.credentials / "token.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
