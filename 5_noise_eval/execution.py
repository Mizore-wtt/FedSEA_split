"""Fresh-cache reference/trial execution and comparable, explicitly scoped measurements."""

import copy
import time

import torch

from noisy_session import NoiseSession


def snapshot(output, input_tokens, tokenizer):
    ids = output.sequences[0, input_tokens:].detach().cpu().tolist()
    logits = tuple(value.detach().float().cpu() for value in output.logits) if output.logits else ()
    if any(not torch.isfinite(value).all() for value in logits):
        raise RuntimeError("Model produced nonfinite logits.")
    return {"token_ids": ids, "response": tokenizer.decode(ids, skip_special_tokens=True), "logits": logits}


def peak_memory(runner):
    return torch.cuda.max_memory_allocated(runner.device) / 1024**2 if runner.device.type == "cuda" else None


def generate_reference(runner, inputs, generation_config=None):
    config = copy.deepcopy(generation_config or runner.generation_config)
    config.use_cache = True
    runner.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = runner.model.generate(
            **inputs, generation_config=config, return_dict_in_generate=True, output_logits=True,
        )
    runner.synchronize()
    seconds = time.perf_counter() - start
    result = snapshot(output, inputs["input_ids"].shape[1], runner.tokenizer)
    result["generation_seconds"] = seconds
    return result


def run_trial(runner, inputs, noise_config, seed, *, trace=True, generation_config=None):
    session = NoiseSession(runner.model, noise_config, seed)
    before = dict(runner.rpc.stats)
    runner.synchronize()
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    start = time.perf_counter()
    try:
        output, metrics = session.generate(
            inputs, generation_config or runner.generation_config,
            trace=trace, synchronize=runner.synchronize,
        )
        metrics["client_peak_gpu_allocated_mib"] = peak_memory(runner)
        result = snapshot(output, inputs["input_ids"].shape[1], runner.tokenizer)
        expected = [inputs["input_ids"].shape[1]] + [1] * (len(result["token_ids"]) - 1)
        if metrics["query_tokens_per_forward"] != expected or metrics["reused_prefix_tokens"] != 0:
            raise RuntimeError("Noisy trials must start fresh and decode one new token per forward.")
        result.update(metrics=metrics, noise=session.noise.describe())
    finally:
        session.close()
    result["metrics"]["trial_wall_seconds_including_snapshot_and_close"] = time.perf_counter() - start
    result["communication"] = {key: runner.rpc.stats[key] - value for key, value in before.items()}
    return result
