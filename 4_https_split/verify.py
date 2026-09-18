"""Real HTTPS versus a complete cached reference; saves reproducible local artifacts."""

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from https_core.settings import ROOT, STAGE, project_path, read_json
from https_core.session import HttpsSession
from fedsea.runtime import ModelRuntime, make_messages

# Verification reuses the independently tested full-model prefix policy from stage 3.
# Normal HTTPS client/server execution does not import previous-stage implementations.
sys.path.append(str(ROOT / "3_kv_cache"))
from session import CachedSession


def tensor_check(expected, actual, atol, rtol):
    left, right = expected.detach().float().cpu(), actual.detach().float().cpu()
    shape = left.shape == right.shape
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    error = float((left - right).abs().max()) if shape and finite and left.numel() else None
    return {
        "passed": shape and finite and torch.allclose(left, right, atol=atol, rtol=rtol),
        "exact": shape and finite and torch.equal(left, right), "finite": finite,
        "max_abs_error": error, "shape": list(left.shape),
    }


def local_cache_checks(full, remote, tolerance):
    length = remote.assert_consistent()
    checked = []
    for segment in (remote.a, remote.c):
        for offset, local in enumerate(segment.layers):
            expected = full.layers[segment.start + offset]
            for name in ("keys", "values"):
                checked.append(tensor_check(getattr(expected, name), getattr(local, name), **tolerance))
    return {"length": length, "matches_full_length": length == full.get_seq_length(), "tensors": checked}


def step_check(runner, reference, ids, prefill, decode, tolerance):
    full_cache, remote_cache = DynamicCache(config=reference.config), runner.model.new_cache()
    boundaries, handles, rows = {}, [], []
    part = runner.partition

    def hook(name):
        def store(_module, _args, value):
            boundaries[name] = value.detach().clone()
        return store

    for layer, name in (
        (reference.model.layers[part.p - 1], "full_a"),
        (reference.model.layers[part.p + part.k - 1], "full_b"),
        (reference.model.norm, "full_c"),
        (runner.model.model.weights.norm, "split_c"),
    ):
        handles.append(layer.register_forward_hook(hook(name)))
    native_call = runner.rpc.call

    def traced_call(path, meta, tensors=None):
        if path == "/v1/forward":
            boundaries["split_a"] = tensors["hidden"].detach().clone()
        reply, output = native_call(path, meta, tensors)
        if path == "/v1/forward":
            boundaries["split_b"] = output["hidden"].clone()
        return reply, output

    try:
        with torch.inference_mode(), patch.object(runner.rpc, "call", side_effect=traced_call):
            start = 0
            for stop in [prefill, *range(prefill + 1, prefill + decode + 1)]:
                inputs = {"input_ids": ids[:, start:stop], "attention_mask": torch.ones_like(ids[:, :stop])}
                expected = reference(**inputs, past_key_values=full_cache, use_cache=True, logits_to_keep=1)
                actual = runner.model(**inputs, past_key_values=remote_cache, use_cache=True, logits_to_keep=1)
                row = {
                    "new_tokens": stop - start, "cached_tokens": stop,
                    "logits": tensor_check(expected.logits, actual.logits, **tolerance),
                    "boundaries": {
                        name: tensor_check(boundaries["full_" + name], boundaries["split_" + name], **tolerance)
                        for name in ("a", "b", "c")
                    },
                    "local_kv": local_cache_checks(full_cache, remote_cache, tolerance),
                }
                row["passed"] = (
                    row["logits"]["passed"] and all(x["passed"] for x in row["boundaries"].values())
                    and row["local_kv"]["matches_full_length"]
                    and all(x["passed"] for x in row["local_kv"]["tensors"])
                )
                rows.append(row)
                start = stop
    finally:
        for handle in handles:
            handle.remove()
        remote_cache.close()
    return {"passed": bool(rows) and all(row["passed"] for row in rows), "steps": rows}


def compare(runner):
    stage = runner.stage
    tolerance = stage["comparison"]
    for name in ("atol", "rtol"):
        if type(tolerance[name]) not in (int, float) or not math.isfinite(tolerance[name]) or tolerance[name] < 0:
            raise ValueError("Comparison tolerances must be finite and nonnegative.")
    if (type(stage["decode_check_tokens"]) is not int or stage["decode_check_tokens"] < 1
            or not stage["length_checks"] or any(
                type(n) is not int or not 1 <= n <= runner.config["max_input_tokens"] for n in stage["length_checks"]
            )):
        raise ValueError("Invalid verification lengths.")
    if runner.health["pid"] == os.getpid():
        raise RuntimeError("Real comparison requires a separate server process.")
    print("Verification only: also loading a complete reference model. Normal chat does not do this.", flush=True)
    reference = ModelRuntime(runner.config, runner.profile.name)
    folder = STAGE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    source_paths = sorted(STAGE.rglob("*.py")) + sorted((ROOT / "fedsea").glob("*.py"))
    source_paths += [ROOT / "3_kv_cache/session.py"]
    record = {
        "stage": "4_https_split", "mode": "two_process_https_comparison", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "noise_std": 0.0,
        "identity": runner.identity, "runtime_config": runner.config, "stage_config": stage,
        "reference_environment": reference.environment,
        "client_pid": os.getpid(), "server_pid": runner.health["pid"],
        "client_weights": runner.model.model.weights.describe(),
        "server_weights": runner.health["weights"],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "cases": [], "length_checks": [], "multi_turn": [],
        "notes": [
            "HTTPS verification is enabled; hidden states/masks/positions only; no token IDs sent.",
            "B cache tensors are not exported by the production API. Their lengths and B boundary outputs are checked.",
            "Tiny-checkpoint unit tests directly compare all A/B/C KV tensors.",
            "Traced timings include checks, CPU copies and TLS handshakes; this is not a speed benchmark.",
            "Body byte metrics exclude HTTP headers and TLS/TCP overhead.",
        ],
    }

    def save():
        record["communication"] = dict(runner.rpc.stats)
        (folder / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def turn(messages, full_session, remote_session, name):
        inputs = runner.tokenize(messages)
        expected, em = full_session.generate(inputs, runner.generation_config, trace=True,
                                             synchronize=runner.synchronize)
        actual, am = remote_session.generate(inputs, runner.generation_config, trace=True,
                                             synchronize=runner.synchronize)
        logits = [tensor_check(a, b, **tolerance) for a, b in zip(expected.logits, actual.logits)]
        cache = local_cache_checks(full_session.cache, remote_session.cache, tolerance)
        offset = inputs["input_ids"].shape[1]
        left, right = expected.sequences[0, offset:], actual.sequences[0, offset:]
        full_text = runner.tokenizer.decode(left, skip_special_tokens=True)
        remote_text = runner.tokenizer.decode(right, skip_special_tokens=True)
        schedule = [am["prefill_tokens"]] + [1] * (len(right) - 1)
        row = {
            "id": name, "messages": messages, "full_response": full_text, "https_response": remote_text,
            "token_ids_equal": torch.equal(expected.sequences, actual.sequences),
            "logits": logits, "local_kv": cache, "full_metrics": em, "https_metrics": am,
            "incremental_schedule_passed": (
                em["query_tokens_per_forward"] == am["query_tokens_per_forward"] == schedule
                and em["reused_prefix_tokens"] == am["reused_prefix_tokens"]
            ),
        }
        row["passed"] = (
            row["token_ids_equal"] and len(expected.logits) == len(actual.logits)
            and bool(logits) and all(x["passed"] for x in logits) and cache["matches_full_length"]
            and all(x["passed"] for x in cache["tensors"]) and row["incremental_schedule_passed"]
            and full_text == remote_text and bool(remote_text.strip())
        )
        artifacts = folder / name
        artifacts.mkdir()
        for filename, tensor in (
            ("input_ids", inputs["input_ids"]), ("full_generated_ids", left), ("https_generated_ids", right),
            ("full_first_token_logits", expected.logits[0]), ("https_first_token_logits", actual.logits[0]),
        ):
            np.save(artifacts / (filename + ".npy"), tensor.detach().cpu().numpy(), allow_pickle=False)
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {name}: tokens={len(right)}, "
              f"reuse={am['reused_prefix_tokens']}, logits_exact={all(x['exact'] for x in logits)}", flush=True)
        print(f"  HTTPS: {remote_text}", flush=True)
        return row

    save()
    live = None
    try:
        cases = [case for name in stage["prompt_files"] for case in read_json(project_path(name))]
        for index, case in enumerate(cases):
            messages = (
                [{"role": "system", "content": runner.config["system_prompt"]}] + case["messages"]
                if "messages" in case else make_messages(case["prompt"], runner.config["system_prompt"])
            )
            live = HttpsSession(runner.model)
            try:
                row = turn(messages, CachedSession(reference.model), live, f"case_{index:02d}")
                row["prompt_id"] = case["id"]
                record["cases"].append(row)
            finally:
                live.close()
                live = None
            save()
        for length in stage["length_checks"]:
            count = stage["decode_check_tokens"]
            ids = torch.arange(length + count, device=runner.device).reshape(1, -1) % (runner.model.config.vocab_size - 20) + 10
            row = {"prefill_tokens": length, **step_check(runner, reference.model, ids, length, count, tolerance)}
            record["length_checks"].append(row)
            save()
            print(f"[{'PASS' if row['passed'] else 'FAIL'}] HTTPS prefill={length}, decode={count}", flush=True)
        full_session, live = CachedSession(reference.model), HttpsSession(runner.model)
        history = [{"role": "system", "content": runner.config["system_prompt"]}]
        for index, case in enumerate(read_json(STAGE / "prompts.json")):
            if case.get("reset_before"):
                full_session.reset()
                live.reset()
                history = history[:1]
            messages = history + [{"role": "user", "content": case["prompt"]}]
            row = turn(messages, full_session, live, f"turn_{index:02d}")
            row["reuse_policy_passed"] = (
                row["https_metrics"]["reused_prefix_tokens"] > 0
                if case.get("expect_reuse") else row["https_metrics"]["reused_prefix_tokens"] == 0
            )
            row["passed"] = row["passed"] and row["reuse_policy_passed"]
            record["multi_turn"].append(row)
            history = messages + [{"role": "assistant", "content": row["full_response"]}]
            save()
        live.close()
        live = None
        rows = record["cases"] + record["length_checks"] + record["multi_turn"]
        record["status"] = "passed" if rows and all(x["passed"] for x in rows) else "failed"
        health, _ = runner.rpc.call("/v1/health", {})
        record["server_sessions_after_comparison"] = health["active_sessions"]
        save()
        if record["status"] != "passed":
            raise RuntimeError("HTTPS comparison failed; inspect results, do not relax tolerances.")
    except BaseException as error:
        if record["status"] != "failed":
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if live is not None:
            live.close()
        print(f"Verification record (contains prompts/outputs): {folder / 'run.json'}", flush=True)
