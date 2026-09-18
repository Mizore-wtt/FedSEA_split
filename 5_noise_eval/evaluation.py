"""Live full-model reference, zero-noise correctness gate, then paired sigma/seed trials."""

import copy
import hashlib

from settings import STAGE, project_path
from fedsea.runtime import ModelRuntime, make_messages
from execution import generate_reference, run_trial
from metrics import compare_outputs, task_score
from noise import derive_seed
from reporting import metadata, new_run_dir, save_evaluation


def case_messages(case, system_prompt):
    return (
        [{"role": "system", "content": system_prompt}] + case["messages"]
        if "messages" in case else make_messages(case["prompt"], system_prompt)
    )


def evaluate(runner, cases):
    config = runner.stage["evaluation"]
    folder = new_run_dir(STAGE)
    record = metadata(runner, "noise_sweep")
    record.update(
        full_model_loaded_for_reference=True, references=[], trials=[],
        zero_noise_gate={"passed": None, "checked_trials": 0},
        prompts_sha256=hashlib.sha256(project_path(config["prompts_file"]).read_bytes()).hexdigest(),
    )
    save_evaluation(folder, record)
    try:
        if not cases:
            raise ValueError("Evaluation cannot pass without cases.")
        print("Evaluation only: loading a full cached reference model in this client process.", flush=True)
        reference = ModelRuntime(runner.config, runner.profile.name)
        record["reference_environment"] = reference.environment
        inputs_by_case = [
            (case, case_messages(case, runner.config["system_prompt"])) for case in cases
        ]
        # Tokenize/check every prompt before a costly sweep, with no silent truncation.
        tokenized = [(case, messages, runner.tokenize(messages)) for case, messages in inputs_by_case]
        warmup = copy.deepcopy(runner.generation_config)
        warmup.max_new_tokens = min(config["warmup_new_tokens"], runner.generation_config.max_new_tokens)
        warmup.min_new_tokens = 0
        warmup.min_length = 0
        print("Unmeasured warmup: full model and clean HTTPS, with fresh caches.", flush=True)
        generate_reference(reference, tokenized[0][2], generation_config=warmup)
        run_trial(
            runner, tokenized[0][2], {**runner.stage["noise"], "sigma": 0.0},
            derive_seed(config["seeds"][0], "__warmup__"), generation_config=warmup,
        )
        # Always execute zero first, even when the user's grid is written in a different order.
        sigmas = [0.0] + [float(sigma) for sigma in config["sigmas"] if sigma != 0]
        total = len(cases) * len(sigmas) * len(config["seeds"])
        for case, messages, inputs in tokenized:
            full = generate_reference(reference, inputs)
            full_score = task_score(full["response"], case.get("answers"))
            record["references"].append({
                "id": case["id"], "messages": messages, "answers": case.get("answers"),
                "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
                "response": full["response"], "token_ids": full["token_ids"],
                "task_score": full_score, "generation_seconds": full["generation_seconds"],
            })
            for sigma in sigmas:
                for seed in config["seeds"]:
                    effective_seed = derive_seed(seed, case["id"])
                    actual = run_trial(
                        runner, inputs, {**runner.stage["noise"], "sigma": sigma}, effective_seed,
                    )
                    comparison = compare_outputs(
                        full, actual, zero_noise=sigma == 0, atol=config["atol"], rtol=config["rtol"],
                    )
                    row = {
                        "case_id": case["id"], "sigma": sigma, "base_seed": seed,
                        "effective_seed": effective_seed,
                        **{key: value for key, value in actual.items() if key != "logits"},
                        "task_score": task_score(actual["response"], case.get("answers")),
                        "reference_task_score": full_score, "comparison": comparison,
                    }
                    record["trials"].append(row)
                    if sigma == 0:
                        record["zero_noise_gate"]["checked_trials"] += 1
                        if not comparison["zero_noise_passed"]:
                            record["zero_noise_gate"]["passed"] = False
                            record["status"] = "failed"
                            save_evaluation(folder, record)
                            raise RuntimeError("Zero-noise/full-model comparison failed; do not relax tolerances.")
                    save_evaluation(folder, record)
                    label = "ZERO PASS" if sigma == 0 else "MEASURED"
                    print(
                        f"[{len(record['trials'])}/{total} {label}] {case['id']} "
                        f"sigma={sigma:g} seed={seed}; score={row['task_score']}; "
                        f"reference_match={comparison['reference_token_ids_equal']}; "
                        f"effective_noise_rms={actual['noise']['effective_noise_rms']:.6f}",
                        flush=True,
                    )
                    del actual
            del full
        expected_zeros = len(cases) * len(config["seeds"])
        if record["zero_noise_gate"]["checked_trials"] != expected_zeros:
            raise RuntimeError("Zero-noise coverage incomplete.")
        record["zero_noise_gate"]["passed"] = True
        record["status"] = "completed"
        save_evaluation(folder, record)
        print("Experiment completed. Noisy answer quality is measured separately from the zero gate.", flush=True)
        return folder
    except BaseException as error:
        if record["status"] != "failed":
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save_evaluation(folder, record)
        raise
    finally:
        print(f"Experiment record (contains prompts/answers/seeds): {folder / 'run.json'}", flush=True)
