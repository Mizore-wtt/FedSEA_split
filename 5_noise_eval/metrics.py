"""Separate task quality from output stability and zero-noise numerical correctness."""

import math
import unicodedata

import torch


def normalize_answer(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def task_score(response, answers):
    if answers is None:
        return None
    return int(normalize_answer(response) in {normalize_answer(answer) for answer in answers})


def tensor_metrics(expected, actual, atol, rtol):
    same_shape = expected.shape == actual.shape and expected.numel() > 0
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    result = {"passed": False, "exact": False, "finite": finite, "max_abs_error": None, "rmse": None}
    if not same_shape or not finite:
        return result
    left, right = expected.detach().double().cpu(), actual.detach().double().cpu()
    delta = left - right
    result.update(
        passed=torch.allclose(left, right, atol=atol, rtol=rtol),
        exact=torch.equal(left, right), max_abs_error=delta.abs().max().item(),
        rmse=delta.square().mean().sqrt().item(),
    )
    return result


def compare_outputs(reference, actual, *, zero_noise, atol, rtol):
    left, right = reference["token_ids"], actual["token_ids"]
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    if not reference["logits"] or not actual["logits"]:
        raise RuntimeError("Generation must provide at least one raw-logit step.")
    first = tensor_metrics(reference["logits"][0], actual["logits"][0], atol, rtol)
    result = {
        "reference_token_ids_equal": left == right,
        "reference_text_equal": reference["response"] == actual["response"],
        "generated_token_prefix_fraction": common / max(len(left), len(right), 1),
        "first_token_logits": first,
        "zero_noise_passed": None,
    }
    if zero_noise:
        steps = [tensor_metrics(a, b, atol, rtol) for a, b in zip(reference["logits"], actual["logits"])]
        equal_length = len(reference["logits"]) == len(actual["logits"])
        result.update(
            zero_noise_passed=left == right and result["reference_text_equal"]
            and equal_length and all(step["passed"] for step in steps),
            all_generation_logits_exact=equal_length and all(step["exact"] for step in steps),
            per_step_logits=steps,
        )
    # After the first changed token, rollout contexts differ: do not call later logits aligned.
    return result


def aggregate(rows):
    summaries = []
    for sigma in sorted({row["sigma"] for row in rows}):
        group = [row for row in rows if row["sigma"] == sigma]
        scored = [row for row in group if row["task_score"] is not None]

        def mean(values):
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        accuracy = mean(row["task_score"] for row in scored)
        reference_accuracy = mean(row["reference_task_score"] for row in scored)
        peaks = [row["metrics"]["client_peak_gpu_allocated_mib"] for row in group
                 if row["metrics"]["client_peak_gpu_allocated_mib"] is not None]
        summary = {
            "sigma": sigma, "trials": len(group), "scored_trials": len(scored),
            "task_accuracy": accuracy, "reference_task_accuracy": reference_accuracy,
            "accuracy_drop_vs_reference": reference_accuracy - accuracy if scored else None,
            "reference_exact_match_rate": mean(row["comparison"]["reference_token_ids_equal"] for row in group),
            "nonempty_rate": mean(bool(row["response"].strip()) for row in group),
            "mean_first_logit_rmse": mean(row["comparison"]["first_token_logits"]["rmse"] for row in group),
            "mean_effective_noise_rms": mean(row["noise"]["effective_noise_rms"] for row in group),
            "mean_generation_seconds": mean(row["metrics"]["generation_seconds"] for row in group),
            "mean_generated_tokens": mean(len(row["token_ids"]) for row in group),
            "mean_sent_body_bytes": mean(row["communication"]["sent_body_bytes"] for row in group),
            "mean_received_body_bytes": mean(row["communication"]["received_body_bytes"] for row in group),
            "max_client_peak_gpu_allocated_mib_including_reference": max(peaks) if peaks else None,
        }
        if any(isinstance(value, float) and not math.isfinite(value) for value in summary.values()):
            raise RuntimeError("Aggregate metrics must be finite.")
        summaries.append(summary)
    return summaries
