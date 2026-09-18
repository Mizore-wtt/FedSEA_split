import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fixtures import build, tiny, inputs, runtime
import evaluation
import run_noise
from execution import generate_reference
from reporting import write_json


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = tiny()
        self.reference.save_pretrained(self.root / "model")
        model, self.state, rpc = build(self.root / "model", self.reference)
        self.runner = runtime(model, self.state, rpc)
        self.full_runner = runtime(self.reference, self.state, rpc)
        self.full_runner.environment = {"test": True}
        text = generate_reference(self.full_runner, inputs())["response"]
        self.cases = [{"id": "sample", "prompt": "Hi", "answers": [text]},
                      {"id": "open", "prompt": "Explain"}]
        self.prompts = self.root / "prompts.json"
        self.prompts.write_text(json.dumps(self.cases), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, **patches):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, "STAGE", self.root))
            stack.enter_context(patch.object(evaluation, "project_path", return_value=self.prompts))
            stack.enter_context(patch.object(evaluation, "ModelRuntime", return_value=self.full_runner))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            for name, value in patches.items():
                stack.enter_context(patch.object(evaluation, name, side_effect=value))
            return evaluation.evaluate(self.runner, self.cases)

    def record(self):
        paths = list((self.root / "runs").glob("*/run.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_full_sweep_reports_all_trials_and_coverage(self):
        folder = self.execute()
        record = self.record()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["zero_noise_gate"], {"passed": True, "checked_trials": 4})
        self.assertEqual(len(record["references"]), 2)
        self.assertEqual(len(record["trials"]), 8)
        with (folder / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["task_accuracy"], "1.0")
        self.assertEqual(summary[0]["scored_trials"], "2")
        self.assertTrue((folder / "summary.md").is_file())
        self.assertFalse(self.state.sessions)
        self.assertNotIn("logits", record["trials"][0])

    def test_zero_numeric_failure_marks_report_failed(self):
        original = evaluation.run_trial

        def wrong(*args, **kwargs):
            result = original(*args, **kwargs)
            if "generation_config" not in kwargs:
                result["logits"] = tuple(value + 1 for value in result["logits"])
            return result

        with self.assertRaisesRegex(RuntimeError, "Zero-noise"):
            self.execute(run_trial=wrong)
        record = self.record()
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["zero_noise_gate"]["passed"])
        self.assertFalse(self.state.sessions)

    def test_poor_noisy_quality_is_not_hidden_as_an_execution_error(self):
        original = evaluation.run_trial

        def wrong_answer(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[2]["sigma"] > 0:
                result["response"] = "WRONG"
            return result

        self.execute(run_trial=wrong_answer)
        record = self.record()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["summary"][1]["task_accuracy"], 0)
        self.assertEqual(record["summary"][1]["accuracy_drop_vs_reference"], 1)

    def test_interrupted_warmup_records_incomplete_gate(self):
        with self.assertRaises(KeyboardInterrupt):
            self.execute(generate_reference=KeyboardInterrupt())
        record = self.record()
        self.assertEqual(record["status"], "interrupted")
        self.assertIsNone(record["zero_noise_gate"]["passed"])
        self.assertFalse(self.state.sessions)

    def test_empty_experiment_is_not_completed(self):
        self.cases = []
        with self.assertRaisesRegex(ValueError, "without cases"):
            self.execute()
        self.assertEqual(self.record()["status"], "error")

    def test_single_prompt_has_no_full_model_gate_or_logits(self):
        with patch.object(run_noise, "STAGE", self.root), contextlib.redirect_stdout(io.StringIO()):
            run_noise.single_prompt(self.runner, "Hi")
        record = self.record()
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["full_model_loaded_for_reference"])
        self.assertIsNone(record["zero_noise_gate"])
        self.assertNotIn("logits", record)
        self.assertFalse(self.state.sessions)

    def test_chat_reset_and_exit_do_not_create_transcripts(self):
        with patch("builtins.input", side_effect=["Hi", "/reset", "Hi", "/exit"]):
            with patch.object(run_noise, "STAGE", self.root), contextlib.redirect_stdout(io.StringIO()):
                run_noise.chat(self.runner)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse(self.state.sessions)

    def test_atomic_json_rejects_nan_without_replacing_previous_record(self):
        path = self.root / "run.json"
        write_json(path, {"status": "running"})
        with self.assertRaises(ValueError):
            write_json(path, {"metric": float("nan")})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "running"})


if __name__ == "__main__":
    unittest.main()
