from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import compare_outputs, tensor_metrics, task_score, normalize_answer


def result(ids, logits=None):
    return {"response": str(ids), "token_ids": ids,
            "logits": logits if logits is not None else (torch.tensor([[1.0, 2.0]]),)}


class MetricTests(unittest.TestCase):
    def test_normalized_task_exact_match_is_not_substring_search(self):
        self.assertEqual(task_score("  POSITIVE \n", ["positive"]), 1)
        self.assertEqual(task_score("positive but not really", ["positive"]), 0)
        self.assertEqual(task_score("42.", ["42"]), 0)
        self.assertIsNone(task_score("anything", None))
        self.assertEqual(normalize_answer("One   TWO"), "one two")

    def test_zero_gate_checks_logits_not_only_text(self):
        expected = result([1, 2])
        actual = result([1, 2], (torch.tensor([[1.0, 3.0]]),))
        compared = compare_outputs(expected, actual, zero_noise=True, atol=0, rtol=0)
        self.assertTrue(compared["reference_token_ids_equal"])
        self.assertFalse(compared["zero_noise_passed"])
        self.assertEqual(compared["first_token_logits"]["max_abs_error"], 1)

    def test_zero_gate_rejects_different_generation_lengths(self):
        compared = compare_outputs(result([1, 2]), result([1]), zero_noise=True, atol=0, rtol=0)
        self.assertFalse(compared["zero_noise_passed"])
        self.assertEqual(compared["generated_token_prefix_fraction"], 0.5)

    def test_noisy_rollout_differences_are_measurements_not_failures(self):
        compared = compare_outputs(result([1, 2, 3]), result([1, 7, 3]), zero_noise=False, atol=0, rtol=0)
        self.assertIsNone(compared["zero_noise_passed"])
        self.assertEqual(compared["generated_token_prefix_fraction"], 1 / 3)
        self.assertNotIn("per_step_logits", compared)

    def test_nonfinite_and_shape_mismatch_fail(self):
        for actual in (torch.tensor([float("nan")]), torch.zeros(2)):
            self.assertFalse(tensor_metrics(torch.zeros(1), actual, 0, 0)["passed"])

    def test_empty_logit_collection_is_not_a_pass(self):
        with self.assertRaisesRegex(RuntimeError, "at least one"):
            compare_outputs(result([1], ()), result([1]), zero_noise=True, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
