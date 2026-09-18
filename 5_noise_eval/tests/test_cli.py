import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import ROOT, STAGE, read_json, load_cases, validate_evaluation
import run_noise


class CliTests(unittest.TestCase):
    def cli(self, entry, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(STAGE / entry), *args],
            cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertNotIn("Loading ", result.stdout)
        self.assertNotIn("Private HTTPS B ready", result.stdout)
        return result

    def test_default_preview(self):
        data = json.loads(self.cli("run_noise.py", "--describe").stdout)
        self.assertEqual(data["partition"], {"p": 2, "k": 20, "q": 2})
        self.assertEqual(data["noise_std"], 0.02)
        self.assertFalse(data["cross_turn_cache_reuse"])
        self.assertFalse(data["opens_network"])

    def test_reserved_moe_and_custom_split_preview(self):
        data = json.loads(self.cli(
            "run_local.py", "--describe", "--model", "qwen3-30b-a3b", "--p", "1", "--q", "3",
            "--sigma", "0.05",
        ).stdout)
        self.assertEqual(data["partition"], {"p": 1, "k": 44, "q": 3})
        self.assertFalse(data["model"]["enabled"])
        self.assertEqual(data["noise_std"], 0.05)

    def test_reserved_execution_rejected_before_network(self):
        self.cli("run_noise.py", "--model", "qwen3-30b-a3b", "--prompt", "Hi", success=False)

    def test_eval_preview_and_cli_overrides_do_not_modify_config(self):
        before = (STAGE / "config.json").read_bytes()
        data = json.loads(self.cli(
            "run_noise.py", "--describe", "--evaluate", "--sigmas", "0", "0.5",
            "--seeds", "7", "--port", "8445",
        ).stdout)
        self.assertEqual(data["evaluation"]["sigmas"], [0, 0.5])
        self.assertEqual(data["evaluation"]["seeds"], [7])
        self.assertEqual(data["https"]["port"], 8445)
        self.assertEqual((STAGE / "config.json").read_bytes(), before)

    def test_invalid_flags_rejected_before_loading(self):
        for flags in (
            ["--prompt", ""], ["--sigma", "-1"], ["--sigma", "nan"],
            ["--noise-seed", "-1"], ["--evaluate", "--sigma", "0.1"],
            ["--sigmas", "0", "0.1"], ["--evaluate", "--sigmas", "0.1"],
            ["--evaluate", "--seeds", "42", "42"],
        ):
            with self.subTest(flags=flags), patch.object(run_noise, "ClientRuntime") as loader:
                with self.assertRaises(ValueError):
                    run_noise.main(flags)
                loader.assert_not_called()

    def test_unknown_distribution_rejected(self):
        config = read_json(STAGE / "config.json")
        config["noise"]["distribution"] = "unknown"
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "config.json"
            filename.write_text(json.dumps(config), encoding="utf-8")
            self.cli("run_local.py", "--describe", "--config", str(filename), success=False)

    def test_invalid_evaluation_grid(self):
        valid = read_json(STAGE / "config.json")["evaluation"]
        for patch_values in (
            {"sigmas": []}, {"sigmas": [0, 0]}, {"sigmas": [0, True]},
            {"seeds": [False]}, {"seeds": [2**63]}, {"rtol": float("nan")},
            {"warmup_new_tokens": 0}, {"prompts_file": "../outside"},
        ):
            with self.assertRaises(ValueError):
                validate_evaluation({**valid, **patch_values})

    def test_invalid_cases_and_duplicate_ids_rejected(self):
        valid = {"id": "case", "prompt": "Hi", "answers": ["OK"]}
        cases = (
            [], [valid, valid], [{**valid, "id": "../bad"}],
            [{**valid, "answers": []}], [{**valid, "prompt": ""}],
            [{**valid, "messages": []}],
            [{"id": "case", "messages": [{"role": "assistant", "content": "Hi"}]}],
        )
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "prompts.json"
            for data in cases:
                filename.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(filename)
        self.assertEqual(len(load_cases(STAGE / "prompts.json")), 6)


if __name__ == "__main__":
    unittest.main()
