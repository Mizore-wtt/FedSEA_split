"""Stage-5 CLI/configuration, reusing the established model, split and HTTPS selectors."""

import math
from pathlib import Path
import re
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
for directory in (ROOT, ROOT / "4_https_split"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from https_core.settings import parser as https_parser, select as https_select
from fedsea.project import project_path, read_json
from noise import validate_noise, validate_seed, validate_sigma


def parser(description):
    result = https_parser(description)
    result.set_defaults(config=STAGE / "config.json")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--evaluate", action="store_true", help="Full-model zero gate and sigma/seed sweep.")
    modes.add_argument("--prompt", help="One noisy answer; saves a local result, without a full-model control.")
    modes.add_argument("--chat", action="store_true", help="Chat (default); no transcripts saved.")
    result.add_argument("--sigma", type=float, help="Absolute Gaussian std for --prompt/--chat.")
    result.add_argument("--noise-seed", type=int, help="Experiment base seed for --prompt/--chat.")
    result.add_argument("--sigmas", nargs="+", type=float, help="Evaluation grid; must include zero.")
    result.add_argument("--seeds", nargs="+", type=int, help="Evaluation base seeds.")
    return result


def validate_evaluation(config):
    required = {"sigmas", "seeds", "prompts_file", "atol", "rtol", "warmup_new_tokens"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"evaluation requires exactly: {', '.join(sorted(required))}.")
    for field, validate, maximum in (("sigmas", validate_sigma, 32), ("seeds", validate_seed, 16)):
        values = config[field]
        if not isinstance(values, list) or not 1 <= len(values) <= maximum:
            raise ValueError(f"{field} must be a nonempty list of at most {maximum} entries.")
        for value in values:
            validate(value)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field} would distort aggregate results.")
    if 0 not in config["sigmas"]:
        raise ValueError("Evaluation sigmas must include 0 for the full-model correctness gate.")
    for field in ("atol", "rtol"):
        value = config[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Comparison tolerances must be finite and nonnegative.")
    if type(config["warmup_new_tokens"]) is not int or not 1 <= config["warmup_new_tokens"] <= 16:
        raise ValueError("warmup_new_tokens must be an integer from 1 to 16.")
    if not isinstance(config["prompts_file"], str) or not config["prompts_file"].strip():
        raise ValueError("prompts_file must select a project-local JSON file.")
    project_path(config["prompts_file"])


def select(args):
    if args.list_models or args.list_splits:
        return https_select(args)
    stage = read_json(args.config)
    if args.evaluate and (args.sigma is not None or args.noise_seed is not None):
        raise ValueError("--evaluate uses --sigmas/--seeds, not --sigma/--noise-seed.")
    if not args.evaluate and (args.sigmas is not None or args.seeds is not None):
        raise ValueError("--sigmas/--seeds require --evaluate.")
    if args.sigma is not None:
        stage["noise"]["sigma"] = args.sigma
    if args.noise_seed is not None:
        stage["noise"]["seed"] = args.noise_seed
    if args.sigmas is not None:
        stage["evaluation"]["sigmas"] = args.sigmas
    if args.seeds is not None:
        stage["evaluation"]["seeds"] = args.seeds
    validate_noise(stage["noise"])
    validate_evaluation(stage["evaluation"])
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("The prompt must not be empty.")
    return https_select(args, stage_config=stage, describe_extra={
        "stage": "5_noise_eval", "noise_std": stage["noise"]["sigma"],
        "noise": stage["noise"], "evaluation": stage["evaluation"],
        "noise_boundary": "A output before HTTPS upload, every prefill/decode forward",
        "cross_turn_cache_reuse": False, "formal_privacy_guarantee": False,
    })


def load_cases(path):
    cases = read_json(path)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires a nonempty prompt list.")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"id", "prompt", "messages", "answers"}:
            raise ValueError("Each case supports id, prompt OR messages, and optional answers.")
        name = case.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in identifiers:
            raise ValueError("Case IDs must be unique, short ASCII identifiers.")
        identifiers.add(name)
        if ("prompt" in case) == ("messages" in case):
            raise ValueError("Each case needs exactly one of prompt or messages.")
        if "prompt" in case:
            if not isinstance(case["prompt"], str) or not case["prompt"].strip():
                raise ValueError("Case prompts must be nonempty strings.")
        else:
            messages = case["messages"]
            if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
                raise ValueError("Case messages must start and end with a user turn.")
            for index, message in enumerate(messages):
                if (not isinstance(message, dict) or set(message) != {"role", "content"}
                        or message["role"] != ("user" if index % 2 == 0 else "assistant")
                        or not isinstance(message["content"], str) or not message["content"].strip()):
                    raise ValueError("Case messages must alternate user/assistant with nonempty content.")
        if "answers" in case and (
            not isinstance(case["answers"], list) or not case["answers"]
            or any(not isinstance(answer, str) or not answer.strip() for answer in case["answers"])
        ):
            raise ValueError("answers must be a nonempty list of accepted exact answers.")
    return cases
