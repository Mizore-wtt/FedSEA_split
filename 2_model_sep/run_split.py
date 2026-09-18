"""Stage 2: local A/B/C chat or comparison against the unchanged full model."""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fedsea.model_registry import ModelRegistry
from fedsea.project import load_runtime_config, project_path, read_json
from fedsea.partition import Partition, resolve_partition
from fedsea.runtime import ModelRuntime, make_messages, validate_config
from split_model import build_split_model
from comparison import check_generation, check_historical, check_prefill
from history import load_reference, validate_selection

STAGE = Path(__file__).resolve().parent


def validate_stage_config(config, total, *, preset=None, p=None, q=None):
    validate_selection(config.get("stage1_run", "auto"))
    if "partition" in config and "split" in config:
        raise ValueError("Use split or legacy partition, not both.")
    if "partition" in config:
        original = Partition(**config["partition"])
        original.validate(total)
        selection = {"p": original.p, "q": original.q}
    else:
        selection = config.get("split")
    partition = resolve_partition(total, selection, preset=preset, p=p, q=q)
    for key in ("atol", "rtol"):
        value = config["comparison"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number.")
    if not config["length_checks"]:
        raise ValueError("At least one synthetic sequence length is required.")
    for length in config["length_checks"]:
        if type(length) is not int or length <= 0:
            raise ValueError("Synthetic sequence lengths must be positive integers.")
    return partition


def tokenize(runner, messages):
    return runner.tokenize(messages)


def generate(runner, model, inputs, config, trace=False):
    # trace collects raw logits for numerical checks; normal chat only needs token IDs.
    runner.synchronize()
    start = time.perf_counter()
    with runner.torch.inference_mode():
        output = model.generate(
            **inputs, generation_config=config,
            return_dict_in_generate=trace, output_logits=trace,
        )
    runner.synchronize()
    return output, time.perf_counter() - start


def chat(runner, split, generation_config):
    print("A -> B -> C local chat. KV cache OFF; no noise; no HTTPS yet.")
    print("/reset clears history; /exit quits. Transcripts are not saved.")
    messages = [{"role": "system", "content": runner.config["system_prompt"]}]
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if prompt == "/exit":
            return
        if prompt == "/reset":
            messages = messages[:1]
            print("History cleared.")
            continue
        if not prompt:
            continue
        candidate = messages + [{"role": "user", "content": prompt}]
        try:
            inputs = tokenize(runner, candidate)
            sequence, seconds = generate(runner, split, inputs, generation_config)
        except ValueError as error:
            print(f"Input rejected: {error}")
            continue
        output_ids = sequence[0, inputs["input_ids"].shape[-1]:]
        response = runner.tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"\nAssistant (A/B/C)> {response}\n[{len(output_ids)} tokens, {seconds:.2f}s, cache OFF]")
        messages = candidate + [{"role": "assistant", "content": response}]


def run_comparison(runner, split, generation_config, config, cases, include_lengths, historical=None):
    import numpy as np
    import torch

    # Resolve optional history before creating a result folder or starting any comparison.
    history, history_info = historical if historical is not None else load_reference(
        config.get("stage1_run", "auto"), runner.profile
    )
    print(f"Stage-1 history: {history_info['status']}. {history_info['reason']}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = STAGE / "runs" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    tolerance = config["comparison"]
    record = {
        "stage": "2_model_sep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "single_process_shared_weights",
        "use_cache": False, "noise_std": 0.0, "network_inference": False,
        "model": runner.spec, "model_profile": runner.profile.describe(), "runtime_config": runner.config,
        "stage_config": config, "partition_map": split.partition.describe(),
        "historical_reference": history_info,
        "environment": runner.environment,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                STAGE / "split_model.py", STAGE / "comparison.py", STAGE / "run_split.py", STAGE / "history.py",
                *sorted((ROOT / "fedsea").glob("*.py")),
            )
        },
        "cases": [], "length_checks": [], "status": "running",
    }

    def save():
        (folder / "run.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    save()
    try:
        for index, case in enumerate(cases):
            messages = (
                [{"role": "system", "content": runner.config["system_prompt"]}] + case["messages"]
                if "messages" in case else
                make_messages(case["prompt"], runner.config["system_prompt"])
            )
            inputs = tokenize(runner, messages)
            # Main verdict: two fresh executions of the SAME weights, both with cache disabled.
            prefill, first_logits = check_prefill(runner.model, split, inputs, **tolerance)
            ref_output, ref_seconds = generate(runner, runner.model, inputs, generation_config, trace=True)
            split_output, split_seconds = generate(runner, split, inputs, generation_config, trace=True)
            comparison = check_generation(ref_output, split_output, **tolerance)
            offset = inputs["input_ids"].shape[-1]
            ref_ids = ref_output.sequences[0, offset:].cpu().numpy()
            split_ids = split_output.sequences[0, offset:].cpu().numpy()
            ref_text = runner.tokenizer.decode(ref_ids, skip_special_tokens=True)
            split_text = runner.tokenizer.decode(split_ids, skip_special_tokens=True)
            historical_result = check_historical(
                history, case["id"], messages, runner.config, runner.spec,
                inputs, split_ids, first_logits[1], template_kwargs=runner.profile.template_kwargs,
                **tolerance,
            )
            if history is None:
                historical_result = {"status": "skipped", "reason": history_info["reason"]}
            artifact_dir = folder / f"case_{index:02d}"
            artifact_dir.mkdir()
            for name, array in (
                ("input_ids", inputs["input_ids"].cpu().numpy()),
                ("full_generated_ids", ref_ids), ("split_generated_ids", split_ids),
                ("full_first_token_logits", first_logits[0]),
                ("split_first_token_logits", first_logits[1]),
            ):
                np.save(artifact_dir / f"{name}.npy", array, allow_pickle=False)
            passed = (prefill["passed"] and comparison["passed"]
                      and historical_result["status"] != "failed" and ref_text == split_text
                      and bool(split_text.strip()))
            result = {
                "id": case["id"], "messages": messages,
                "input_tokens": offset,
                "full_response": ref_text, "split_response": split_text,
                "response_text_equal": ref_text == split_text,
                "prefill": prefill, "generation": comparison, "stage1_reference": historical_result,
                "full_generation_seconds": ref_seconds, "split_generation_seconds": split_seconds,
                "timing_note": "Includes raw-logit collection; not a throughput benchmark.",
                "passed": passed,
            }
            record["cases"].append(result)
            save()
            print(
                f"[{'PASS' if passed else 'FAIL'}] {case['id']}: "
                f"{len(split_ids)} tokens; prefill max error={prefill['logits']['max_abs_error']}; "
                f"generation logits exact={comparison['all_logits_exact']}; "
                f"stage1={historical_result['status']}",
                flush=True,
            )
            print(f"  Full:  {ref_text}\n  Split: {split_text}", flush=True)
            del ref_output, split_output
        if include_lengths:
            for length in config["length_checks"]:
                if length > runner.config["max_input_tokens"]:
                    raise ValueError("A synthetic length check exceeds max_input_tokens.")
                ids = (torch.arange(length, device=runner.device).unsqueeze(0)
                       % (runner.model.config.vocab_size - 20)) + 10
                inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                result, _ = check_prefill(runner.model, split, inputs, **tolerance, logits_to_keep=1)
                record["length_checks"].append({"input_tokens": length, **result})
                save()
                print(
                    f"[{'PASS' if result['passed'] else 'FAIL'}] synthetic length={length}, "
                    f"last-position max error={result['logits']['max_abs_error']}", flush=True
                )
        all_rows = record["cases"] + record["length_checks"]
        passed = bool(all_rows) and all(row["passed"] for row in all_rows)
        record["status"] = "passed" if passed else "failed"
        save()
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        print(f"Comparison record (includes prompts and outputs): {folder / 'run.json'}", flush=True)
    if not passed:
        raise RuntimeError("Split comparison failed. Inspect run.json before changing tolerances.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compare", action="store_true", help="Compare fixed prompts and sequence lengths.")
    modes.add_argument("--prompt", help="Compare full and split inference on one prompt; saves results.")
    modes.add_argument("--chat", action="store_true", help="Split-model chat (default), without transcripts.")
    modes.add_argument("--list-models", action="store_true", help="List registered model profiles.")
    modes.add_argument("--list-splits", action="store_true", help="List named p/q combinations.")
    modes.add_argument("--describe", action="store_true", help="Preview model and partition without weights.")
    parser.add_argument("--config", type=Path, default=STAGE / "config.json")
    parser.add_argument("--model", help="Model profile from configs/model.json.")
    parser.add_argument("--split", dest="split_preset", help="Preset name from configs/splits.json.")
    parser.add_argument("--p", type=int, help="Front decoder blocks; middle count is derived.")
    parser.add_argument("--q", type=int, help="Back decoder blocks; middle count is derived.")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--stage1-run",
        help="Optional history: none (disable), auto (model's baseline), or a stage-1 run directory.",
    )
    args = parser.parse_args(argv)
    registry = ModelRegistry()
    if args.list_models:
        registry.list_models()
        return
    if args.list_splits:
        print(json.dumps(read_json(ROOT / "configs/splits.json"), indent=2))
        return
    config = read_json(args.config)
    if args.stage1_run is not None:
        config["stage1_run"] = args.stage1_run
    profile = registry.get(args.model or config.get("model"))
    partition = validate_stage_config(
        config, profile.spec["num_hidden_layers"], preset=args.split_preset, p=args.p, q=args.q
    )
    config["resolved_partition"] = {"p": partition.p, "k": partition.k, "q": partition.q}
    if args.describe:
        print(json.dumps({
            "model": profile.describe(), "partition": config["resolved_partition"],
            "partition_map": partition.describe(), "loads_weights": False,
            "stage1_run": config.get("stage1_run", "auto"),
        }, indent=2, ensure_ascii=False))
        return
    runtime_config = load_runtime_config(config)
    if args.device is not None:
        runtime_config["device"] = args.device
    if args.max_new_tokens is not None:
        runtime_config["max_new_tokens"] = args.max_new_tokens
    validate_config(runtime_config)
    if args.prompt is not None:
        make_messages(args.prompt, runtime_config["system_prompt"])
    historical = None
    if args.compare or args.prompt is not None:
        # A bad explicit path is a configuration error, so fail before allocating GPU weights.
        historical = load_reference(config.get("stage1_run", "auto"), profile)
    if args.compare and any(n > runtime_config["max_input_tokens"] for n in config["length_checks"]):
        raise ValueError("A synthetic length check exceeds max_input_tokens; fix the stage config first.")
    runner = ModelRuntime(runtime_config, profile.name)
    split = build_split_model(runner.model, partition, profile.settings["adapter"])
    generation_config = copy.deepcopy(runner.generation_config)
    generation_config.use_cache = False
    print(
        f"Split ready: A={partition.p}, B={partition.k}, C={partition.q}. "
        "One weight copy; cache OFF; no noise; no network.", flush=True
    )
    if args.compare:
        cases = read_json(ROOT / "1_model_base/prompts.json")
        cases += read_json(STAGE / "prompts.json")
        run_comparison(runner, split, generation_config, config, cases, include_lengths=True, historical=historical)
    elif args.prompt is not None:
        run_comparison(
            runner, split, generation_config, config,
            [{"id": "custom_prompt", "prompt": args.prompt}], include_lengths=False, historical=historical,
        )
    else:
        chat(runner, split, generation_config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Comparison interrupted; any started run is marked interrupted.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
