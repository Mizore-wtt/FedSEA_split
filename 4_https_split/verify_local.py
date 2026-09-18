"""Start a private test server, run --compare in another process, then stop both."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from https_core.settings import STAGE, ROOT, parser, select, project_path
from https_core.transport import HttpsRpc, RpcError


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    selected = select(args)
    if selected is None:
        return
    stage, runtime, profile, part = selected
    net = stage["https"]
    port = args.port
    if port is None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
    directory = project_path(net["credentials_dir"])
    rpc = HttpsRpc(f"https://localhost:{port}", directory, timeout=2, limit=net["max_body_bytes"])
    options = [
        "--config", str(args.config.resolve()), "--model", profile.name, "--p", str(part.p),
        "--q", str(part.q), "--device", runtime["device"],
        "--max-new-tokens", str(runtime["max_new_tokens"]), "--port", str(port),
    ]
    folder = STAGE / "runs" / ("launcher_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    record = {
        "status": "running", "port": port, "partition": {"p": part.p, "k": part.k, "q": part.q},
        "server_stopped": False, "client_returncode": None,
    }
    server = client = None
    instance_id = uuid.uuid4().hex
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with (folder / "server.log").open("w", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                [sys.executable, "-X", "utf8", str(STAGE / "server.py"), *options,
                 "--instance-id", instance_id],
                cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, creationflags=flags,
            )
            record["server_pid"] = server.pid
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Test server exited; inspect {folder / 'server.log'}.")
                try:
                    health, _ = rpc.call("/v1/health", {})
                    if health.get("instance_id") != instance_id:
                        raise RuntimeError("Port belongs to another process; refusing to use it.")
                    record["server_worker_pid"] = health["pid"]
                    break
                except RpcError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Test server did not become ready within 120 seconds.")
                    time.sleep(0.5)
            print(f"Private HTTPS server ready: PID={health['pid']}, port={port}, p/k/q={part.p}/{part.k}/{part.q}", flush=True)
            with (folder / "client.log").open("w", encoding="utf-8") as client_log:
                client = subprocess.Popen(
                    [sys.executable, "-X", "utf8", str(STAGE / "client.py"), "--compare", *options],
                    cwd=ROOT, stdout=client_log, stderr=subprocess.STDOUT, creationflags=flags,
                )
                record["client_pid"] = client.pid
                record["client_returncode"] = client.wait(timeout=1800)
            print((folder / "client.log").read_text(encoding="utf-8"), flush=True)
            if record["client_returncode"] != 0:
                raise RuntimeError("HTTPS comparison process failed.")
            health, _ = rpc.call("/v1/health", {})
            record["remaining_sessions"] = health["active_sessions"]
            if health["active_sessions"] != 0:
                raise RuntimeError("Comparison left server sessions allocated.")
            record["status"] = "passed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # These are exactly the child processes created above, never other port owners.
        for process in (client, server):
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    # Windows venv python.exe may be a redirector with a worker child.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=True, capture_output=True, creationflags=flags,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        record["server_stopped"] = server is not None and server.poll() is not None
        (folder / "launcher.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Test launcher record: {folder / 'launcher.json'}", flush=True)


if __name__ == "__main__":
    main()
