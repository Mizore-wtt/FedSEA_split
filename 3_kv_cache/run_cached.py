"""Stage 3: A/B/C dynamic KV cache, local chat, and cached-reference comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedsea.model_registry import ModelRegistry
from fedsea.partition import resolve_partition
from fedsea.project import load_runtime_config, project_path, read_json
from fedsea.runtime import ModelRuntime, make_messages, validate_config
from cached_model import build_cached_model
from checks import check_steps, compare_caches, compare_generation
from session import CachedSession


def validate_stage_config(config, total, *, preset=None, p=None, q=None):
    partition = resolve_partition(total, config.get("split"), preset=preset, p=p, q=q)
    for key in ("atol", "rtol"):
        value = config["comparison"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Comparison tolerances must be finite and nonnegative.")
    if type(config["decode_check_tokens"]) is not int or config["decode_check_tokens"] < 1:
        raise ValueError("decode_check_tokens must be a positive integer.")
    if not config["length_checks"] or any(
        type(length) is not int or length < 1 for length in config["length_checks"]
    ):
        raise ValueError("length_checks must contain positive integer lengths.")
    return partition


def chat(runner, split):
    session = CachedSession(split)
    messages = [{"role": "system", "content": runner.config["system_prompt"]}]
    print("A -> B -> C chat. Segment KV caches ON. No HTTPS/no noise. Transcripts not saved.")
    print("/reset clears BOTH conversation and A/B/C caches; /exit quits.")
    try:
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt == "/exit":
                break
            if prompt == "/reset":
                messages = messages[:1]
                session.reset()
                print("Conversation and all three caches cleared.")
                continue
            if not prompt:
                continue
            candidate = messages + [{"role": "user", "content": prompt}]
            try:
                inputs = runner.tokenize(candidate)
                output, metrics = session.generate(
                    inputs, runner.generation_config, synchronize=runner.synchronize
                )
            except (ValueError, RuntimeError) as error:
                print(f"Request failed: {error}")
                continue
            except KeyboardInterrupt:
                print("\nGeneration cancelled; cache cleared.")
                continue
            ids = output.sequences[0, inputs["input_ids"].shape[1]:]
            response = runner.tokenizer.decode(ids, skip_special_tokens=True)
            del output
            print(f"\nAssistant (A/B/C + cache)> {response}")
            print(
                f"[new tokens={len(ids)}, reused prefix={metrics['reused_prefix_tokens']}, "
                f"prefill={metrics['prefill_tokens']}, cached tokens={metrics['cached_tokens']}]"
            )
            messages = candidate + [{"role": "assistant", "content": response}]
    finally:
        session.reset()


def run_comparison(runner, split, config, cases, comprehensive):
    import numpy as np
    import torch

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = STAGE / "runs" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    tolerance = config["comparison"]
    record = {
        "stage": "3_kv_cache", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "single_process_shared_weights_partitioned_cache",
        "reference_use_cache": True, "split_use_cache": True,
        "noise_std": 0.0, "network_inference": False,
        "model": runner.spec, "model_profile": runner.profile.describe(),
        "runtime_config": runner.config, "stage_config": config,
        "partition_map": split.partition.describe(), "environment": runner.environment,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*sorted(STAGE.glob("*.py")), *sorted((ROOT / "fedsea").glob("*.py")))
        },
        "cases": [], "length_checks": [], "multi_turn": [], "status": "running",
    }

    def save():
        (folder / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def compare_turn(messages, full_session, split_session, artifact_name):
        inputs = runner.tokenize(messages)
        expected, em = full_session.generate(
            inputs, runner.generation_config, trace=True, synchronize=runner.synchronize
        )
        actual, am = split_session.generate(
            inputs, runner.generation_config, trace=True, synchronize=runner.synchronize
        )
        generation = compare_generation(expected, actual, **tolerance)
        cache = compare_caches(full_session.cache, split_session.cache, **tolerance)
        offset = inputs["input_ids"].shape[1]
        left_ids = expected.sequences[0, offset:].cpu().numpy()
        right_ids = actual.sequences[0, offset:].cpu().numpy()
        left_text = runner.tokenizer.decode(left_ids, skip_special_tokens=True)
        right_text = runner.tokenizer.decode(right_ids, skip_special_tokens=True)
        expected_schedule = [am["prefill_tokens"]] + [1] * (len(right_ids) - 1)
        schedule_ok = (
            am["query_tokens_per_forward"] == em["query_tokens_per_forward"] == expected_schedule
            and am["reused_prefix_tokens"] == em["reused_prefix_tokens"]
        )
        passed = (
            generation["passed"] and cache["passed"] and schedule_ok
            and left_text == right_text and bool(right_text.strip())
        )
        artifacts = folder / artifact_name
        artifacts.mkdir()
        for name, data in (
            ("input_ids", inputs["input_ids"].cpu().numpy()),
            ("full_generated_ids", left_ids), ("split_generated_ids", right_ids),
            ("full_first_token_logits", expected.logits[0][0].float().cpu().numpy()),
            ("split_first_token_logits", actual.logits[0][0].float().cpu().numpy()),
        ):
            np.save(artifacts / f"{name}.npy", data, allow_pickle=False)
        return {
            "messages": messages, "full_response": left_text, "split_response": right_text,
            "generation": generation, "final_kv_cache": cache,
            "full_metrics": em, "split_metrics": am, "incremental_schedule_passed": schedule_ok,
            "passed": passed,
        }, inputs

    def show(label, row):
        status = "PASS" if row["passed"] else "FAIL"
        metrics = row["split_metrics"]
        print(
            f"[{status}] {label}: {metrics['generated_tokens_including_eos']} tokens; "
            f"logits exact={row['generation']['all_logits_exact']}; "
            f"reuse={metrics['reused_prefix_tokens']}; prefill={metrics['prefill_tokens']}",
            flush=True,
        )
        print(f"  Full: {row['full_response']}\n  Split: {row['split_response']}", flush=True)

    save()
    try:
        for index, case in enumerate(cases):
            messages = (
                [{"role": "system", "content": runner.config["system_prompt"]}] + case["messages"]
                if "messages" in case else make_messages(case["prompt"], runner.config["system_prompt"])
            )
            row, inputs = compare_turn(
                messages, CachedSession(runner.model), CachedSession(split), f"case_{index:02d}"
            )
            n = config["decode_check_tokens"]
            tail = (torch.arange(n, device=runner.device).reshape(1, -1) + 10)
            ids = torch.cat((inputs["input_ids"], tail), dim=1)
            row["forward_checks"] = check_steps(
                runner.model, split, ids, inputs["input_ids"].shape[1], n, **tolerance
            )
            row["passed"] = row["passed"] and row["forward_checks"]["passed"]
            row["id"] = case["id"]
            record["cases"].append(row)
            save()
            show(case["id"], row)
        if comprehensive:
            for length in config["length_checks"]:
                if length > runner.config["max_input_tokens"]:
                    raise ValueError("A length check exceeds the configured max_input_tokens.")
                n = config["decode_check_tokens"]
                ids = (
                    torch.arange(length + n, device=runner.device).unsqueeze(0)
                    % (runner.model.config.vocab_size - 20)
                ) + 10
                checked = check_steps(runner.model, split, ids, length, n, **tolerance)
                record["length_checks"].append({"prefill_tokens": length, **checked})
                save()
                print(f"[{'PASS' if checked['passed'] else 'FAIL'}] prefill={length}, decode={n}", flush=True)
            full_session, split_session = CachedSession(runner.model), CachedSession(split)
            history = [{"role": "system", "content": runner.config["system_prompt"]}]
            for index, case in enumerate(read_json(STAGE / "prompts.json")):
                if case.get("reset_before"):
                    full_session.reset()
                    split_session.reset()
                    history = history[:1]
                messages = history + [{"role": "user", "content": case["prompt"]}]
                row, _ = compare_turn(messages, full_session, split_session, f"turn_{index:02d}")
                reuse = row["split_metrics"]["reused_prefix_tokens"]
                row["id"] = case["id"]
                row["reset_before"] = bool(case.get("reset_before"))
                row["reuse_policy_passed"] = (
                    reuse > 0 if case.get("expect_reuse") else reuse == 0
                )
                row["passed"] = row["passed"] and row["reuse_policy_passed"]
                record["multi_turn"].append(row)
                history = messages + [{"role": "assistant", "content": row["full_response"]}]
                save()
                show(f"session/{case['id']}", row)
        all_rows = record["cases"] + record["length_checks"] + record["multi_turn"]
        passed = bool(all_rows) and all(row["passed"] for row in all_rows)
        record["status"] = "passed" if passed else "failed"
        save()
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        print(f"Comparison record (contains prompts/outputs): {folder / 'run.json'}", flush=True)
    if not passed:
        raise RuntimeError("Cached split comparison failed; inspect run.json without relaxing tolerances.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--chat", action="store_true", help="Persistent local chat (default); no transcript.")
    modes.add_argument("--compare", action="store_true", help="Fixed prompts, lengths, and live multi-turn cache.")
    modes.add_argument("--prompt", help="Compare one prompt and save results.")
    modes.add_argument("--describe", action="store_true", help="Preview configuration without weights.")
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--list-splits", action="store_true")
    parser.add_argument("--config", type=Path, default=STAGE / "config.json")
    parser.add_argument("--model")
    parser.add_argument("--split", dest="preset")
    parser.add_argument("--p", type=int)
    parser.add_argument("--q", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args(argv)
    registry = ModelRegistry()
    if args.list_models:
        registry.list_models()
        return
    if args.list_splits:
        print(json.dumps(read_json(ROOT / "configs/splits.json"), indent=2))
        return
    config = read_json(args.config)
    profile = registry.get(args.model or config.get("model"))
    part = validate_stage_config(
        config, profile.spec["num_hidden_layers"], preset=args.preset, p=args.p, q=args.q
    )
    config["resolved_partition"] = {"p": part.p, "k": part.k, "q": part.q}
    if args.describe:
        print(json.dumps({
            "model": profile.describe(), "partition": config["resolved_partition"],
            "partition_map": part.describe(), "cache": "separate_A_B_C_dynamic_full_attention",
            "loads_weights": False, "network_inference": False, "noise_std": 0.0,
        }, indent=2, ensure_ascii=False))
        return
    runtime = load_runtime_config(config)
    if args.device is not None:
        runtime["device"] = args.device
    if args.max_new_tokens is not None:
        runtime["max_new_tokens"] = args.max_new_tokens
    validate_config(runtime)
    if args.prompt is not None:
        make_messages(args.prompt, runtime["system_prompt"])
    runner = ModelRuntime(runtime, profile.name)
    split = build_cached_model(runner.model, part, profile.settings["adapter"])
    print(
        f"Cached split ready: A={part.p}, B={part.k}, C={part.q}. "
        "One weight copy; three owned caches; no network/no noise.", flush=True
    )
    if args.compare:
        cases = []
        for filename in config["prompt_files"]:
            cases.extend(read_json(project_path(filename)))
        run_comparison(runner, split, config, cases, comprehensive=True)
    elif args.prompt is not None:
        run_comparison(
            runner, split, config, [{"id": "custom_prompt", "prompt": args.prompt}], comprehensive=False
        )
    else:
        chat(runner, split)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; session caches released.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
