"""Comparison failures and the public, offline stage-3 command-line contract."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
sys.path.insert(0, str(STAGE))
from cached_model import Partition, build_cached_model
from checks import capture, check_steps, compare_generation, tensor_report
from run_cached import validate_stage_config


class CheckTests(unittest.TestCase):
    def test_numeric_failures(self):
        for left, right in (
            (torch.ones(2), torch.zeros(2)), (torch.ones(2), torch.ones(3)),
            (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
            (torch.tensor([float("inf")]), torch.tensor([float("inf")])),
        ):
            self.assertFalse(tensor_report(left, right, 1e-5, 1e-5)["passed"])

    def test_generation_mismatch(self):
        a = SimpleNamespace(sequences=torch.tensor([[1, 2]]), logits=[torch.zeros(1, 4)])
        b = SimpleNamespace(sequences=torch.tensor([[1, 3]]), logits=[torch.zeros(1, 4)])
        self.assertFalse(compare_generation(a, b, 1e-5, 1e-5)["passed"])
        b.sequences = a.sequences
        b.logits = []
        self.assertFalse(compare_generation(a, b, 1e-5, 1e-5)["passed"])

    def test_hook_cleanup_on_exception(self):
        layer = torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with capture({"A": layer}):
                layer(torch.ones(1))
                raise RuntimeError("test")
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_full_comparison_path(self):
        torch.set_num_threads(1)
        config = Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        full = Qwen2ForCausalLM(config).eval()
        full.requires_grad_(False)
        split = build_cached_model(full, Partition(1, 2, 1))
        result = check_steps(full, split, torch.tensor([[5, 6, 7, 8, 9, 10]]), 3, 3, 1e-5, 1e-5)
        self.assertTrue(result["passed"])
        self.assertEqual([row["query_tokens"] for row in result["steps"]], [3, 1, 1, 1])
        self.assertEqual([row["cached_tokens"] for row in result["steps"]], [3, 4, 5, 6])
        for row in result["steps"]:
            self.assertEqual(len(row["kv_cache"]["layers"]), 4)
            self.assertTrue(row["logits"]["exact"])

    def test_invalid_stage_config(self):
        config = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
        for key, value in (("decode_check_tokens", 0), ("length_checks", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_stage_config({**config, key: value}, 24)
        for value in (float("nan"), -1, float("inf")):
            with self.assertRaises(ValueError):
                validate_stage_config({**config, "comparison": {"atol": value, "rtol": 1e-5}}, 24)


class CliTests(unittest.TestCase):
    def run_cli(self, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(STAGE / "run_cached.py"), *args],
            cwd=ROOT.parent, text=True, encoding="utf-8", capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertNotIn("Loading ", result.stdout)
        return result

    def test_default_preview(self):
        data = json.loads(self.run_cli("--describe").stdout)
        self.assertEqual(data["partition"], {"p": 2, "k": 20, "q": 2})
        self.assertFalse(data["loads_weights"])
        self.assertFalse(data["network_inference"])

    def test_reserved_moe_preview(self):
        data = json.loads(self.run_cli(
            "--describe", "--model", "qwen3-30b-a3b-instruct-2507", "--p", "1", "--q", "3"
        ).stdout)
        self.assertEqual(data["partition"], {"p": 1, "k": 44, "q": 3})
        self.assertFalse(data["model"]["enabled"])

    def test_preset_and_override(self):
        data = json.loads(self.run_cli("--describe", "--split", "p1_q3", "--p", "2").stdout)
        self.assertEqual(data["partition"], {"p": 2, "k": 19, "q": 3})

    def test_reserved_execution_refused(self):
        result = self.run_cli("--model", "qwen3-30b-a3b", "--compare", success=False)
        self.assertIn("reserved, not deployed/enabled", result.stderr)

    def test_invalid_split_before_loading(self):
        self.assertIn("positive integers", self.run_cli(
            "--describe", "--p", "23", "--q", "1", success=False
        ).stderr)


if __name__ == "__main__":
    unittest.main()
