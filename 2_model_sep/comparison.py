"""Numerical, boundary, generated-sequence, and historical-reference checks."""

from contextlib import contextmanager

import numpy as np
import torch


def compare_tensor(expected, actual, atol, rtol):
    # Equal text is insufficient: nonfinite tensors and numerical differences must also fail.
    result = {
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "exact": False, "passed": False, "finite": False,
        "max_abs_error": None, "mean_abs_error": None,
    }
    if expected.shape != actual.shape or expected.numel() == 0:
        return result
    a, b = expected.detach().cpu(), actual.detach().cpu()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return result
    difference = (a.float() - b.float()).abs()
    result.update(
        finite=True,
        exact=torch.equal(a, b),
        passed=torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol),
        max_abs_error=difference.max().item(),
        mean_abs_error=difference.mean().item(),
    )
    return result


@contextmanager
def capture_boundaries(modules):
    captured, handles = {}, []
    for name, module in modules.items():
        def hook(_module, _inputs, output, key=name):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[key] = tensor.detach().cpu().clone()
        handles.append(module.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def check_prefill(reference, split, inputs, atol, rtol, logits_to_keep=0):
    # Observe matching boundaries to locate the first divergent segment, not just final output.
    partition = split.partition
    reference_modules = {
        "A": reference.model.layers[partition.p - 1],
        "B": reference.model.layers[partition.p + partition.k - 1],
        "C_after_norm": reference.model.norm,
    }
    split_modules = {
        "A": split.model.stage_a,
        "B": split.model.stage_b,
        "C_after_norm": split.model.stage_c,
    }
    with torch.inference_mode():
        with capture_boundaries(reference_modules) as expected:
            ref_logits = reference(
                **inputs, use_cache=False, logits_to_keep=logits_to_keep, return_dict=True
            ).logits
        with capture_boundaries(split_modules) as actual:
            split_logits = split(
                **inputs, use_cache=False, logits_to_keep=logits_to_keep, return_dict=True
            ).logits
    boundaries = {
        name: compare_tensor(expected[name], actual[name], atol, rtol)
        for name in reference_modules
    }
    logits = compare_tensor(ref_logits, split_logits, atol, rtol)
    result = {
        "boundaries": boundaries,
        "logits": logits,
        "passed": logits["passed"] and all(row["passed"] for row in boundaries.values()),
    }
    first = (
        ref_logits[0, -1].float().cpu().numpy(),
        split_logits[0, -1].float().cpu().numpy(),
    )
    return result, first


def check_generation(expected, actual, atol, rtol):
    sequence_match = torch.equal(expected.sequences, actual.sequences)
    length_match = len(expected.logits) == len(actual.logits)
    steps = [
        compare_tensor(left, right, atol, rtol)
        for left, right in zip(expected.logits, actual.logits)
    ]
    return {
        "token_ids_equal": sequence_match,
        "step_count_equal": length_match,
        "expected_steps": len(expected.logits),
        "actual_steps": len(actual.logits),
        "per_step_logits": steps,
        "all_logits_exact": length_match and all(row["exact"] for row in steps),
        "passed": sequence_match and length_match and bool(steps)
                  and all(row["passed"] for row in steps),
    }


def check_historical(history, case_id, messages, runtime_config, model_spec,
                     inputs, generated_ids, first_logits, atol, rtol, template_kwargs=None):
    if history is None:
        return {"status": "skipped", "reason": "No stage1_run configured."}
    folder, record = history
    # Only compare saved arrays after confirming the model, prompt and runtime are compatible.
    identity_keys = ("repo_id", "revision", "model_type", "num_hidden_layers", "hidden_size", "weights_sha256")
    if any(record["model"].get(key) != model_spec.get(key) for key in identity_keys):
        return {"status": "not_applicable", "reason": "Historical record belongs to a different model."}
    old_template = record.get("model_profile", {}).get("chat_template_kwargs", {})
    if old_template != (template_kwargs or {}):
        return {"status": "skipped", "reason": "Chat-template options differ from stage 1."}
    matches = [(i, row) for i, row in enumerate(record["results"]) if row["id"] == case_id]
    if not matches:
        return {"status": "not_applicable", "reason": "No historical case with this ID."}
    index, row = matches[0]
    if row["messages"] != messages or record["config"] != runtime_config:
        return {"status": "skipped", "reason": "Prompt or runtime settings differ from stage 1."}
    artifacts = folder / f"case_{index:02d}"
    if not all((artifacts / name).is_file() for name in (
        "input_ids.npy", "generated_ids.npy", "first_token_logits.npy"
    )):
        return {"status": "skipped", "reason": "Historical artifacts missing; use --save-artifacts in stage 1."}
    old_input = np.load(artifacts / "input_ids.npy", allow_pickle=False)
    old_output = np.load(artifacts / "generated_ids.npy", allow_pickle=False)
    old_logits = np.load(artifacts / "first_token_logits.npy", allow_pickle=False)
    input_match = np.array_equal(old_input, inputs["input_ids"].cpu().numpy())
    output_match = np.array_equal(old_output, generated_ids)
    logits = compare_tensor(torch.from_numpy(old_logits), torch.from_numpy(first_logits), atol, rtol)
    passed = input_match and output_match and logits["passed"]
    return {
        "status": "passed" if passed else "failed",
        "reference_used_cache": True,
        "split_used_cache": False,
        "input_ids_equal": input_match,
        "generated_ids_equal": output_match,
        "first_token_logits": logits,
    }
