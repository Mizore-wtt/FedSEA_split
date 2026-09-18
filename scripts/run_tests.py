"""Run stage suites in separate Python processes so similarly named modules cannot collide."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "shared": "tests", "1": "1_model_base/tests", "2": "2_model_sep/tests",
    "3": "3_kv_cache/tests", "4": "4_https_split/tests",
    "5": "5_noise_eval/tests",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", nargs="+", choices=SUITES, help="Selected suites; default: all.")
    parser.add_argument("--report", action="store_true", help="Save logs and summary under tests/runs/.")
    args = parser.parse_args(argv)
    selected = list(dict.fromkeys(args.suite or SUITES))
    folder = None
    if args.report:
        folder = ROOT / "tests/runs" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        )
        folder.mkdir(parents=True)
    record = {"status": "running", "python": sys.executable, "suites": []}
    try:
        for name in selected:
            print(f"\nRunning suite {name}: {SUITES[name]}", flush=True)
            start = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", "-W", "error::ResourceWarning",
                 "-m", "unittest", "discover", "-s", str(ROOT / SUITES[name]), "-v"],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            )
            log = result.stdout + result.stderr
            print(log, end="" if log.endswith("\n") else "\n", flush=True)
            record["suites"].append({
                "suite": name, "directory": SUITES[name], "exit_code": result.returncode,
                "seconds": time.perf_counter() - start,
            })
            if folder is not None:
                (folder / f"suite_{name}.log").write_text(log, encoding="utf-8")
        record["status"] = "passed" if all(row["exit_code"] == 0 for row in record["suites"]) else "failed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if folder is not None:
            (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"Test report: {folder / 'run.json'}", flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
