"""Cached-vs-cached checks for boundaries, per-layer K/V, and generated logits."""

from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache


def tensor_report(expected, actual, atol, rtol):
    result = {
        "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
        "finite": False, "exact": False, "passed": False,
        "max_abs_error": None, "mean_abs_error": None,
    }
    if expected.shape != actual.shape or not expected.numel():
        return result
    left, right = expected.detach().float().cpu(), actual.detach().float().cpu()
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return result
    difference = (left - right).abs()
    result.update(
        finite=True, exact=torch.equal(expected.detach().cpu(), actual.detach().cpu()),
        passed=torch.allclose(left, right, atol=atol, rtol=rtol),
        max_abs_error=difference.max().item(), mean_abs_error=difference.mean().item(),
    )
    return result


@contextmanager
def capture(modules):
    states, handles = {}, []
    try:
        for name, module in modules.items():
            def record(_module, _args, output, key=name):
                value = output[0] if isinstance(output, tuple) else output
                states[key] = value.detach().cpu().clone()
            handles.append(module.register_forward_hook(record))
        yield states
    finally:
        for handle in handles:
            handle.remove()


def compare_caches(reference_cache, split_cache, atol, rtol):
    expected_length = reference_cache.get_seq_length()
    actual_length = split_cache.assert_consistent()
    rows = []
    if len(reference_cache.layers) != len(split_cache.layers):
        return {"passed": False, "reason": "Different cache layer counts."}
    for index, (left, right) in enumerate(zip(reference_cache.layers, split_cache.layers)):
        rows.append({
            "layer_1based": index + 1,
            "keys": tensor_report(left.keys, right.keys, atol, rtol),
            "values": tensor_report(left.values, right.values, atol, rtol),
        })
    return {
        "expected_tokens": expected_length, "actual_tokens": actual_length,
        "layers": rows, "layout": split_cache.describe(),
        "passed": expected_length == actual_length and all(
            row["keys"]["passed"] and row["values"]["passed"] for row in rows
        ),
    }


def check_steps(reference, split, ids, prefill_tokens, decode_steps, atol, rtol):
    """Identical chunk schedules; prefill followed by one-token decoder calls."""
    rc, sc = DynamicCache(config=reference.config), split.new_cache()
    part = split.partition
    ref_modules = {
        "A": reference.model.layers[part.p - 1],
        "B": reference.model.layers[part.p + part.k - 1],
        "C_after_norm": reference.model.norm,
    }
    split_modules = {
        "A": split.model.stage_a, "B": split.model.stage_b, "C_after_norm": split.model.stage_c,
    }
    results = []
    start = 0
    for stop in range(prefill_tokens, prefill_tokens + decode_steps + 1):
        current = ids[:, start:stop]
        inputs = {
            "input_ids": current,
            "attention_mask": torch.ones_like(ids[:, :stop]),
            "cache_position": torch.arange(start, stop, device=ids.device),
            "use_cache": True, "logits_to_keep": 1,
        }
        with torch.inference_mode():
            with capture(ref_modules) as expected:
                left = reference(**inputs, past_key_values=rc)
            with capture(split_modules) as actual:
                right = split(**inputs, past_key_values=sc)
        boundaries = {
            name: tensor_report(expected[name], actual[name], atol, rtol) for name in ref_modules
        }
        logits = tensor_report(left.logits, right.logits, atol, rtol)
        cache = compare_caches(rc, sc, atol, rtol)
        results.append({
            "phase": "prefill" if start == 0 else "decode",
            "query_tokens": current.shape[1], "cached_tokens": stop,
            "boundaries": boundaries, "logits": logits, "kv_cache": cache,
            "passed": logits["passed"] and cache["passed"] and all(
                row["passed"] for row in boundaries.values()
            ),
        })
        start = stop
    return {"steps": results, "passed": all(row["passed"] for row in results)}


def compare_generation(expected, actual, atol, rtol):
    length_match = len(expected.logits) == len(actual.logits)
    steps = [tensor_report(a, b, atol, rtol) for a, b in zip(expected.logits, actual.logits)]
    tokens_equal = torch.equal(expected.sequences, actual.sequences)
    return {
        "generated_steps": len(actual.logits), "step_count_equal": length_match,
        "token_ids_equal": tokens_equal, "per_step_logits": steps,
        "all_logits_exact": length_match and bool(steps) and all(row["exact"] for row in steps),
        "passed": length_match and tokens_equal and bool(steps) and all(row["passed"] for row in steps),
    }
