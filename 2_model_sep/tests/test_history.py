"""Deleted historical runs must not prevent fresh full-versus-split comparisons."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import load_reference
import run_split
from fedsea.partition import Partition


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile = SimpleNamespace(settings={"baseline_run": "1_model_base/runs/deleted"})

    def tearDown(self):
        self.temp.cleanup()

    def saved_reference(self, name="available"):
        folder = self.root / "1_model_base/runs" / name
        folder.mkdir(parents=True)
        record = {
            "stage": "1_model_base", "model": {}, "config": {},
            "results": [{"id": "sample", "messages": [{"role": "user", "content": "Hi"}]}],
        }
        (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
        return folder, record

    def test_disabled_ignores_stale_registered_path(self):
        for choice in (None, "none"):
            history, info = load_reference(choice, self.profile, root=self.root)
            self.assertIsNone(history)
            self.assertEqual(info["status"], "skipped")
            self.assertIsNone(info["path"])

    def test_auto_without_registered_reference(self):
        self.profile.settings["baseline_run"] = None
        history, info = load_reference("auto", self.profile, root=self.root)
        self.assertIsNone(history)
        self.assertIn("No baseline_run", info["reason"])

    def test_deleted_auto_reference_skipped_without_substitution(self):
        self.saved_reference("newest_but_not_selected")
        history, info = load_reference("auto", self.profile, root=self.root)
        self.assertIsNone(history)
        self.assertEqual(info["path"], "1_model_base/runs/deleted")
        self.assertEqual(info["status"], "skipped")
        self.assertIn("no alternate run", info["reason"])

    def test_auto_existing_reference_records_exact_source_hash(self):
        folder, record = self.saved_reference()
        self.profile.settings["baseline_run"] = "1_model_base/runs/available"
        history, info = load_reference("auto", self.profile, root=self.root)
        self.assertEqual(history, (folder, record))
        self.assertEqual(info["status"], "loaded")
        self.assertEqual(info["run_json_sha256"], hashlib.sha256((folder / "run.json").read_bytes()).hexdigest())

    def test_windows_bom_reference_supported(self):
        folder, record = self.saved_reference()
        (folder / "run.json").write_text(json.dumps(record), encoding="utf-8-sig")
        history, info = load_reference("1_model_base/runs/available", self.profile, root=self.root)
        self.assertEqual(history, (folder, record))
        self.assertEqual(info["status"], "loaded")

    def test_explicit_relative_or_absolute_reference(self):
        folder, record = self.saved_reference()
        for selection in (str(folder), "1_model_base/runs/available"):
            history, _ = load_reference(selection, self.profile, root=self.root)
            self.assertEqual(history, (folder, record))

    def test_explicit_missing_reference_gives_recovery_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "--stage1-run none"):
            load_reference("1_model_base/runs/deleted", self.profile, root=self.root)

    def test_auto_invalid_json_is_skipped_but_explicit_is_error(self):
        folder, _ = self.saved_reference()
        (folder / "run.json").write_text("{broken", encoding="utf-8")
        self.profile.settings["baseline_run"] = "1_model_base/runs/available"
        history, info = load_reference("auto", self.profile, root=self.root)
        self.assertIsNone(history)
        self.assertEqual(info["status"], "skipped")
        with self.assertRaises(ValueError):
            load_reference("1_model_base/runs/available", self.profile, root=self.root)

    def test_bad_record_schema_rejected(self):
        folder, valid = self.saved_reference()
        for data in (
            [], {**valid, "stage": "2_model_sep"}, {**valid, "model": None},
            {**valid, "results": []}, {**valid, "results": valid["results"] * 2},
            {**valid, "results": [{"id": "x", "messages": None}]},
            {**valid, "model_profile": None},
            {**valid, "model_profile": {"chat_template_kwargs": None}},
        ):
            (folder / "run.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reference("1_model_base/runs/available", self.profile, root=self.root)

    def test_bad_selection_and_outside_path_rejected(self):
        for selection in (True, 0, [], {}, "", "../outside", "configs", "1_model_base/runs"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_reference(selection, self.profile, root=self.root)

    def test_explicit_missing_path_rejected_before_model_load(self):
        with patch.object(run_split, "ModelRuntime") as loader:
            with self.assertRaisesRegex(FileNotFoundError, "--stage1-run none"):
                run_split.main(["--compare", "--stage1-run", "1_model_base/runs/__missing_test_reference__"])
            loader.assert_not_called()

    def test_oversize_length_check_rejected_before_model_load(self):
        config = json.loads((run_split.STAGE / "config.json").read_text(encoding="utf-8"))
        config["runtime_overrides"] = {"max_input_tokens": 16}
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with patch.object(run_split, "ModelRuntime") as loader:
            with self.assertRaisesRegex(ValueError, "length check exceeds"):
                run_split.main(["--compare", "--config", str(path), "--stage1-run", "none"])
            loader.assert_not_called()


class ComparisonLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = self.root / "2_model_sep"
        self.stage.mkdir()
        for name in ("run_split.py", "split_model.py", "comparison.py", "history.py"):
            shutil.copyfile(run_split.STAGE / name, self.stage / name)
        profile = SimpleNamespace(settings={"baseline_run": "1_model_base/runs/__missing_test_reference__"},
                                  template_kwargs={}, describe=lambda: {})
        self.runner = SimpleNamespace(
            profile=profile, config={"system_prompt": "Test"}, spec={}, environment={}, model=object(),
            tokenize=lambda messages: {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)},
            tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "OK"),
        )
        self.split = SimpleNamespace(partition=Partition(1, 1, 1))
        self.config = {"stage1_run": "auto", "comparison": {"atol": 1e-5, "rtol": 1e-5}}
        self.prefill = ({"passed": True, "logits": {"max_abs_error": 0.0}}, (np.ones(3), np.ones(3)))
        self.output = SimpleNamespace(sequences=torch.tensor([[1, 2]]), logits=(torch.ones(1, 3),))

    def tearDown(self):
        self.temp.cleanup()

    def run_comparison(self, error=None, empty=False, mismatch=False):
        with patch.object(run_split, "STAGE", self.stage), patch.object(run_split, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            prefill = self.prefill if not mismatch else (
                {"passed": False, "logits": {"max_abs_error": 1.0}}, self.prefill[1]
            )
            with patch.object(run_split, "check_prefill", return_value=prefill):
                with patch.object(run_split, "generate", return_value=(self.output, 0.1), side_effect=error):
                    run_split.run_comparison(
                        self.runner, self.split, None, self.config,
                        [] if empty else [{"id": "test", "prompt": "Hi"}], include_lengths=False,
                    )

    def record(self):
        paths = list((self.stage / "runs").glob("*/run.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_live_comparison_works_when_auto_history_was_deleted(self):
        self.run_comparison()
        record = self.record()
        self.assertEqual(record["status"], "passed")
        self.assertEqual(record["historical_reference"]["status"], "skipped")
        self.assertEqual(record["cases"][0]["stage1_reference"]["status"], "skipped")

    def test_computation_error_saved(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.run_comparison(RuntimeError("test failure"))
        self.assertEqual(self.record()["status"], "error")

    def test_live_mismatch_still_fails_without_history(self):
        with self.assertRaisesRegex(RuntimeError, "comparison failed"):
            self.run_comparison(mismatch=True)
        record = self.record()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["historical_reference"]["status"], "skipped")
        self.assertFalse(record["cases"][0]["passed"])

    def test_interrupt_saved_instead_of_leaving_running(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_comparison(KeyboardInterrupt())
        self.assertEqual(self.record()["status"], "interrupted")

    def test_empty_comparison_is_not_a_pass(self):
        with self.assertRaisesRegex(RuntimeError, "comparison failed"):
            self.run_comparison(empty=True)
        self.assertEqual(self.record()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
