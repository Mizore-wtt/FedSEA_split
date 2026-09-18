"""Local-only experiment records; no activation dumps, session IDs or credentials."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from settings import ROOT, STAGE
from metrics import aggregate


def new_run_dir(stage=STAGE, prefix=""):
    folder = Path(stage) / "runs" / (
        prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True)
    return folder


def write_json(path, record):
    # Replace a complete temporary JSON so interruption cannot leave a half-written run.json.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def metadata(runner, mode):
    sources = [
        *sorted(STAGE.glob("*.py")), *sorted((ROOT / "fedsea").glob("*.py")),
        *sorted((ROOT / "4_https_split/https_core").glob("*.py")),
        ROOT / "4_https_split/server.py",
    ]
    return {
        "stage": "5_noise_eval", "mode": mode, "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_profile": runner.profile.describe(), "identity": runner.identity,
        "stage_config": runner.stage, "runtime_config": runner.config,
        "client_weights": runner.model.model.weights.describe(),
        "server_weights": runner.health["weights"],
        "noise_boundary": "client A output before serialization; no B-to-C noise",
        "cross_turn_cache_reuse": False, "network_inference": "loopback HTTPS",
        "formal_privacy_guarantee": False,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "measurement_notes": [
            "Gaussian sigma is absolute per-element standard deviation, not the paper's mean absolute noise scale.",
            "Fixed seeds support experiments, not cryptographic or differential privacy guarantees.",
            "Fresh A/B/C caches and a private per-case RNG for every trial; no cross-turn reuse.",
            "Generation time includes TLS, CPU noise sampling, device copies and noise statistics.",
            "Evaluation also collects logits; this is not a throughput or speedup benchmark.",
            "Body bytes include session open/forward/close, but exclude HTTP/TLS/TCP headers and framing.",
            "GPU memory is this client's allocated peak; evaluation includes the full reference model.",
            "Server peak memory is not measured; only owned server weight bytes are reported.",
            "Reference output agreement measures stability, not answer correctness.",
            "Only first-token logits share an identical prompt after noisy rollout divergence.",
        ],
    }


def save_evaluation(folder, record):
    summaries = aggregate(record["trials"])
    record["summary"] = summaries
    write_json(folder / "run.json", record)
    if not summaries:
        return
    temporary = folder / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    temporary.replace(folder / "summary.csv")
    lines = [
        "# Noise Evaluation", "", f"Status: {record['status']}",
        f"Zero-noise full-model gate: {record['zero_noise_gate']['passed']}", "",
        "| Sigma | Trials | Task Accuracy | Reference Match | Mean Effective Noise RMS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        accuracy = "N/A" if row["task_accuracy"] is None else f"{row['task_accuracy']:.3f}"
        lines.append(
            f"| {row['sigma']:g} | {row['trials']} | {accuracy} | "
            f"{row['reference_exact_match_rate']:.3f} | {row['mean_effective_noise_rms']:.6f} |"
        )
    lines += [
        "", "Task accuracy uses normalized exact answers for the labeled toy cases only.",
        "Rows average over case/seed trials, not a benchmark dataset or confidence interval.",
        "Completed means the experiment ran and its zero-noise gate passed, not that noise preserved quality.",
        "See run.json for every prompt, answer, seed, first-token difference, and measurement scope.",
        "No formal privacy, reconstruction resistance, or speedup claim is made.", "",
    ]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
