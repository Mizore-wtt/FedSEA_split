"""Offline CLI contracts; previews and reserved profiles must never load weights."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, entry, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(ROOT / entry), *args],
            cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertNotIn("Loading ", result.stdout)
        return result

    def test_list_models(self):
        result = self.run_cli("1_model_base/baseline.py", "--list-models")
        self.assertIn("qwen2.5-0.5b-instruct | enabled", result.stdout)
        self.assertIn("qwen3-30b-a3b | reserved", result.stdout)

    def test_default_partition_preview(self):
        result = self.run_cli("2_model_sep/run_split.py", "--describe")
        self.assertEqual(json.loads(result.stdout)["partition"], {"p": 2, "k": 20, "q": 2})

    def test_split_history_flag_preview(self):
        result = self.run_cli("2_model_sep/run_split.py", "--describe", "--stage1-run", "none")
        self.assertEqual(json.loads(result.stdout)["stage1_run"], "none")

    def test_missing_explicit_history_is_rejected_before_loading(self):
        result = self.run_cli(
            "2_model_sep/run_split.py", "--compare", "--stage1-run",
            "1_model_base/runs/__missing_cli_test_reference__", success=False,
        )
        self.assertIn("--stage1-run none", result.stderr)

    def test_reserved_moe_preview(self):
        result = self.run_cli(
            "2_model_sep/run_split.py", "--describe",
            "--model", "qwen3-30b-a3b-instruct-2507", "--split", "p1_q3",
        )
        data = json.loads(result.stdout)
        self.assertEqual(data["partition"], {"p": 1, "k": 44, "q": 3})
        self.assertFalse(data["loads_weights"])
        self.assertFalse(data["model"]["enabled"])

    def test_stage_config_model_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"model": "qwen3-30b-a3b"}), encoding="utf-8")
            result = self.run_cli("1_model_base/baseline.py", "--describe", "--config", str(config))
            self.assertEqual(json.loads(result.stdout)["name"], "qwen3-30b-a3b")
            result = self.run_cli(
                "1_model_base/baseline.py", "--describe", "--config", str(config),
                "--model", "qwen2.5-0.5b-instruct",
            )
            self.assertEqual(json.loads(result.stdout)["name"], "qwen2.5-0.5b-instruct")

    def test_reserved_inference_rejected(self):
        result = self.run_cli(
            "2_model_sep/run_split.py", "--model", "qwen3-30b-a3b",
            "--prompt", "Hello", success=False,
        )
        self.assertIn("reserved, not deployed/enabled", result.stderr)

    def test_reserved_preparation_rejected(self):
        result = self.run_cli(
            "scripts/prepare_model.py", "--model", "qwen3-30b-a3b", success=False,
        )
        self.assertIn("reserved, not deployed/enabled", result.stderr)

    def test_invalid_partition_rejected(self):
        result = self.run_cli(
            "2_model_sep/run_split.py", "--describe", "--p", "23", "--q", "1", success=False,
        )
        self.assertIn("positive integers", result.stderr)


if __name__ == "__main__":
    unittest.main()
