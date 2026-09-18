import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison import capture_boundaries, compare_tensor, check_generation, check_historical
from run_split import validate_stage_config, project_path


class ComparisonTests(unittest.TestCase):
    def test_equal_tensor(self):
        tensor = torch.tensor([1.0, 2.0])
        result = compare_tensor(tensor, tensor, atol=1e-5, rtol=1e-5)
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact"])
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_value_difference_detected(self):
        result = compare_tensor(torch.tensor([1.0]), torch.tensor([2.0]), 1e-5, 1e-5)
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs_error"], 1.0)

    def test_shape_difference_detected(self):
        result = compare_tensor(torch.zeros(2), torch.zeros(3), 1e-5, 1e-5)
        self.assertFalse(result["passed"])

    def test_nonfinite_rejected_even_when_identical(self):
        for value in (float("nan"), float("inf")):
            tensor = torch.tensor([value])
            self.assertFalse(compare_tensor(tensor, tensor, 1e-5, 1e-5)["passed"])

    def test_generation_length_mismatch(self):
        logits = torch.zeros(1, 5)
        left = SimpleNamespace(sequences=torch.tensor([[1, 2]]), logits=(logits, logits))
        right = SimpleNamespace(sequences=torch.tensor([[1, 2]]), logits=(logits,))
        self.assertFalse(check_generation(left, right, 0, 0)["passed"])

    def test_generation_token_mismatch(self):
        logits = torch.zeros(1, 5)
        left = SimpleNamespace(sequences=torch.tensor([[1, 2]]), logits=(logits,))
        right = SimpleNamespace(sequences=torch.tensor([[1, 3]]), logits=(logits,))
        self.assertFalse(check_generation(left, right, 0, 0)["passed"])

    def test_hooks_removed_on_failure(self):
        module = torch.nn.Identity()
        before = len(module._forward_hooks)
        with self.assertRaises(ValueError):
            with capture_boundaries({"A": module}) as captured:
                module(torch.ones(1))
                self.assertIn("A", captured)
                raise ValueError("test")
        self.assertEqual(len(module._forward_hooks), before)

    def test_config_tolerance_and_lengths(self):
        config = {
            "partition": {"p": 2, "k": 20, "q": 2},
            "comparison": {"atol": 1e-5, "rtol": 1e-5},
            "length_checks": [1, 128],
        }
        validate_stage_config(config, 24)
        for value in (float("nan"), float("inf"), -1):
            config["comparison"]["atol"] = value
            with self.assertRaises(ValueError):
                validate_stage_config(config, 24)

    def test_outside_project_path_rejected(self):
        with self.assertRaises(ValueError):
            project_path("../../outside.json")

    def test_ambiguous_partition_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            validate_stage_config({
                "partition": {"p": 2, "k": 20, "q": 2}, "split": {"p": 1, "q": 3}
            }, 24)

    def test_historical_model_identity_isolated(self):
        result = check_historical(
            (Path("."), {"model": {"repo_id": "different"}}),
            "case", [], {}, {"repo_id": "current"}, {}, None, None, 0, 0,
        )
        self.assertEqual(result["status"], "not_applicable")

    def test_historical_template_settings_isolated(self):
        result = check_historical(
            (Path("."), {"model": {}, "model_profile": {
                "chat_template_kwargs": {"enable_thinking": True}
            }}), "case", [], {}, {}, {}, None, None, 0, 0,
            template_kwargs={"enable_thinking": False},
        )
        self.assertEqual(result["status"], "skipped")

    def test_historical_weight_digest_isolated(self):
        result = check_historical(
            (Path("."), {"model": {"weights_sha256": "old"}}),
            "case", [], {}, {"weights_sha256": "current"}, {}, None, None, 0, 0,
        )
        self.assertEqual(result["status"], "not_applicable")

    def test_historical_missing_artifacts_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            result = check_historical(
                (Path(folder), {"model": {}, "config": {}, "results": [
                    {"id": "case", "messages": []}
                ]}), "case", [], {}, {}, {}, None, None, 0, 0,
            )
        self.assertEqual(result["status"], "skipped")

    def test_historical_arrays_are_actually_compared(self):
        messages = [{"role": "user", "content": "Hi"}]
        ids = np.array([[1, 2]], dtype=np.int64)
        output = np.array([3, 4], dtype=np.int64)
        logits = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        record = {"model": {}, "config": {}, "results": [{"id": "case", "messages": messages}]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            artifacts = folder / "case_00"
            artifacts.mkdir()
            for name, data in (
                ("input_ids", ids), ("generated_ids", output), ("first_token_logits", logits),
            ):
                np.save(artifacts / f"{name}.npy", data, allow_pickle=False)
            for changed in ("none", "input", "output", "logits"):
                with self.subTest(changed=changed):
                    result = check_historical(
                        (folder, record), "case", messages, {}, {},
                        {"input_ids": torch.from_numpy(ids + (changed == "input"))},
                        output + (changed == "output"), logits + (changed == "logits"), 0, 0,
                    )
                    self.assertEqual(result["status"], "passed" if changed == "none" else "failed")


if __name__ == "__main__":
    unittest.main()
