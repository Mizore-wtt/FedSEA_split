import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedsea.checkpoints import checkpoint_files, snapshot_files
from fedsea.model_registry import ModelRegistry
from fedsea.partition import resolve_partition
from fedsea.project import load_runtime_config, read_json
from fedsea.runtime import ModelRuntime, estimated_weight_bytes


class InterfaceTests(unittest.TestCase):
    def test_current_model_identity(self):
        profile = ModelRegistry().get()
        self.assertEqual(profile.spec["model_type"], "qwen2")
        self.assertEqual(profile.spec["num_hidden_layers"], 24)
        profile.require_enabled()

    def test_reserved_models_are_separate(self):
        registry = ModelRegistry()
        a = registry.get("qwen3-30b-a3b")
        b = registry.get("qwen3-30b-a3b-instruct-2507")
        self.assertNotEqual(a.spec["repo_id"], b.spec["repo_id"])
        self.assertNotEqual(a.spec["local_dir"], b.spec["local_dir"])
        self.assertEqual(a.spec["model_type"], "qwen3_moe")
        self.assertEqual(a.template_kwargs, {"enable_thinking": False})

    def test_reserved_loading_stops_before_weight_access(self):
        config = read_json(ROOT / "configs/inference.json")
        with self.assertRaisesRegex(ValueError, "reserved"):
            ModelRuntime(config, "qwen3-30b-a3b")

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            ModelRegistry().get("not-a-model")

    def test_middle_layers_follow_model(self):
        for total, expected in ((24, 20), (48, 44)):
            split = resolve_partition(total, {"p": 2, "q": 2})
            self.assertEqual(split.k, expected)

    def test_presets_and_overrides(self):
        split = resolve_partition(48, {"preset": "p2_q2"}, preset="p1_q3")
        self.assertEqual((split.p, split.k, split.q), (1, 44, 3))
        custom = resolve_partition(48, {"preset": "p1_q3"}, p=2, q=4)
        self.assertEqual((custom.p, custom.k, custom.q), (2, 42, 4))

    def test_all_presets_are_valid_for_both_models(self):
        presets = read_json(ROOT / "configs/splits.json")
        for total in (24, 48):
            for name in presets["presets"]:
                resolve_partition(total, preset=name).validate(total)

    def test_cli_preset_replaces_config_custom_values(self):
        split = resolve_partition(24, {"p": 3, "q": 3}, preset="p1_q3")
        self.assertEqual((split.p, split.k, split.q), (1, 20, 3))
        split = resolve_partition(24, {"p": 3, "q": 3}, preset="p1_q3", p=2)
        self.assertEqual((split.p, split.k, split.q), (2, 19, 3))

    def test_bad_splits(self):
        for choice in ({"p": 0, "q": 2}, {"p": 23, "q": 1}, {"p": True, "q": 2},
                       {"p": 2, "q": 2, "k": 20}, {"preset": "missing"}):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                resolve_partition(24, choice)

    def test_runtime_overrides_do_not_modify_shared_file(self):
        stage = {"runtime_config": "configs/inference.json", "runtime_overrides": {"max_new_tokens": 7}}
        self.assertEqual(load_runtime_config(stage)["max_new_tokens"], 7)
        self.assertEqual(read_json(ROOT / "configs/inference.json")["max_new_tokens"], 128)

    def test_unknown_runtime_override(self):
        with self.assertRaises(ValueError):
            load_runtime_config({"runtime_config": "configs/inference.json", "runtime_overrides": {"x": 1}})

    def test_large_model_weight_estimate_is_total_not_active_parameters(self):
        profile = ModelRegistry().get("qwen3-30b-a3b")
        self.assertGreater(estimated_weight_bytes(profile, "float16"), 50 * 1024**3)

    def test_duplicate_model_directories_rejected(self):
        data = read_json(ROOT / "configs/model.json")
        data["models"]["qwen3-30b-a3b"]["local_dir"] = data["models"][data["default_model"]]["local_dir"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "registry.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "share one model"):
                ModelRegistry(path)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self, names):
        path = self.folder / "model.safetensors.index.json"
        path.write_text(json.dumps({"weight_map": {str(i): name for i, name in enumerate(names)}}))

    def test_single_file(self):
        (self.folder / "model.safetensors").touch()
        self.assertEqual(checkpoint_files(self.folder), ["model.safetensors"])

    def test_indexed_files(self):
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        self.write_index(names + names)
        for name in names:
            (self.folder / name).touch()
        self.assertEqual(checkpoint_files(self.folder), ["model.safetensors.index.json", *names])

    def test_missing_shard(self):
        self.write_index(["missing.safetensors"])
        with self.assertRaises(FileNotFoundError):
            checkpoint_files(self.folder)

    def test_traversal_rejected(self):
        for name in ("../escape.safetensors", "..\\escape.safetensors", "C:escape.safetensors"):
            self.write_index([name])
            with self.assertRaises(ValueError):
                checkpoint_files(self.folder)

    def test_ambiguous_checkpoint_rejected(self):
        (self.folder / "model.safetensors").touch()
        self.write_index(["model.safetensors"])
        with self.assertRaises(ValueError):
            checkpoint_files(self.folder)

    def test_separate_chat_template_preserved(self):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "model.safetensors", "chat_template.jinja", "special_tokens_map.json"):
            (self.folder / name).touch()
        files = snapshot_files(self.folder)
        self.assertIn("chat_template.jinja", files)
        self.assertIn("special_tokens_map.json", files)


if __name__ == "__main__":
    unittest.main()
