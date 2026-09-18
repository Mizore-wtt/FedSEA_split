"""Configuration previews and verification assertions do not need a live server."""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from https_core.settings import parser, select
from verify import tensor_check


class CliTests(unittest.TestCase):
    def describe(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = select(parser("test").parse_args(["--describe", *arguments]))
        self.assertIsNone(result)
        return json.loads(output.getvalue())

    def test_default_and_custom_split_preview(self):
        self.assertEqual(self.describe()["partition"], {"p": 2, "k": 20, "q": 2})
        self.assertEqual(self.describe("--p", "1", "--q", "3")["partition"], {"p": 1, "k": 20, "q": 3})

    def test_reserved_moe_preview_does_not_enable_model(self):
        for name in ("qwen3-30b-a3b", "qwen3-30b-a3b-instruct-2507"):
            preview = self.describe("--model", name)
            self.assertFalse(preview["model"]["enabled"])
            self.assertFalse(preview["loads_weights"])
            self.assertFalse(preview["opens_network"])
            self.assertEqual(preview["partition"], {"p": 2, "k": 44, "q": 2})
            with self.assertRaises(ValueError):
                select(parser("test").parse_args(["--model", name]))

    def test_invalid_splits_rejected_before_loading(self):
        for options in (["--p", "0"], ["--q", "-1"], ["--p", "23", "--q", "2"], ["--split", "missing"]):
            with self.assertRaises(ValueError):
                self.describe(*options)

    def test_preset_override_and_port_do_not_persist(self):
        value = self.describe("--split", "p1_q3", "--p", "2", "--port", "8444")
        self.assertEqual(value["partition"], {"p": 2, "k": 19, "q": 3})
        self.assertEqual(value["https"]["url"], "https://localhost:8444")
        self.assertEqual(self.describe()["https"]["port"], 8443)

    def test_context_budget_must_fit_https(self):
        with self.assertRaises(ValueError):
            self.describe("--max-new-tokens", "1024")

    def test_tensor_checks_reject_nonfinite_and_real_errors(self):
        zero = torch.zeros(1, 3)
        self.assertTrue(tensor_check(zero, zero, 1e-5, 1e-5)["exact"])
        self.assertFalse(tensor_check(zero, torch.ones_like(zero), 1e-5, 1e-5)["passed"])
        self.assertFalse(tensor_check(zero, torch.zeros(2, 3), 1e-5, 1e-5)["passed"])
        for bad in (float("nan"), float("inf")):
            result = tensor_check(torch.full_like(zero, bad), torch.full_like(zero, bad), 1e-5, 1e-5)
            self.assertFalse(result["passed"])
            self.assertIsNone(result["max_abs_error"])


if __name__ == "__main__":
    unittest.main()
