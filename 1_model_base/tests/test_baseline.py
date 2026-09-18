import copy
import importlib.util
import unittest
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", STAGE / "baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = baseline.load_runtime_config(baseline.read_json(STAGE / "config.json"))

    def test_default_config(self):
        baseline.validate_config(self.config)
        self.assertEqual(self.config["device"], "cuda")

    def test_positive_token_limits(self):
        for field in ("max_input_tokens", "max_new_tokens"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value):
                    bad = copy.deepcopy(self.config)
                    bad[field] = value
                    with self.assertRaises(ValueError):
                        baseline.validate_config(bad)

    def test_invalid_device(self):
        self.config["device"] = "automatic"
        with self.assertRaises(ValueError):
            baseline.validate_config(self.config)

    def test_invalid_dtype(self):
        self.config["dtype"] = "int8"
        with self.assertRaises(ValueError):
            baseline.validate_config(self.config)

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            baseline.make_messages(" \n", "System")

    def test_message_roles_and_content(self):
        messages = baseline.make_messages("Hello", "System")
        self.assertEqual(messages, [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ])

    def test_length_boundary(self):
        baseline.validate_length(self.config["max_input_tokens"], self.config, 32768)
        with self.assertRaises(ValueError):
            baseline.validate_length(self.config["max_input_tokens"] + 1, self.config, 32768)

    def test_context_overflow(self):
        with self.assertRaises(ValueError):
            baseline.validate_length(100, self.config, 100)

    def test_model_spec(self):
        model = baseline.ModelRegistry().get().spec
        self.assertEqual(model["num_hidden_layers"], 24)
        self.assertEqual(len(model["weights_sha256"]), 64)
        self.assertEqual(len(model["revision"]), 40)
        self.assertEqual(model["local_dir"], "models/Qwen2.5-0.5B-Instruct")

    def test_benchmark_cases(self):
        cases = baseline.read_json(STAGE / "prompts.json")
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            baseline.make_messages(case["prompt"], self.config["system_prompt"])


if __name__ == "__main__":
    unittest.main()
