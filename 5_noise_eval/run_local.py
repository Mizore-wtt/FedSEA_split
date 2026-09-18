"""One-command stage 5: launch a private HTTPS B server, run the client, then stop it."""

import os
import socket
import subprocess
import sys
import time
import uuid

from settings import ROOT, STAGE, parser, select, load_cases, project_path
from https_core.transport import HttpsRpc, RpcError
from reporting import new_run_dir, write_json

HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def stop_owned(process):
    """Only stop a Popen child created by this launcher, including Windows venv workers."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True, creationflags=HIDDEN, timeout=15,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser(__doc__).parse_args(arguments)
    selected = select(args)
    if selected is None:
        return
    stage, runtime, profile, part = selected
    if args.evaluate:
        load_cases(project_path(stage["evaluation"]["prompts_file"]))
    net = stage["https"]
    port = args.port
    if port is None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
    rpc = HttpsRpc(
        f"https://localhost:{port}", project_path(net["credentials_dir"]),
        timeout=2, limit=net["max_body_bytes"],
    )
    folder = new_run_dir(STAGE, prefix="launcher_")
    record = {
        "status": "running", "port": port, "server_stopped": False,
        "partition": {"p": part.p, "k": part.k, "q": part.q},
        "mode": "evaluate" if args.evaluate else "prompt" if args.prompt is not None else "chat",
        "client_transcript_logged_by_launcher": False,
    }
    server = client = None
    instance = uuid.uuid4().hex
    options = [
        "--config", str(args.config.resolve()), "--model", profile.name,
        "--p", str(part.p), "--q", str(part.q), "--device", runtime["device"],
        "--max-new-tokens", str(runtime["max_new_tokens"]), "--port", str(port),
    ]
    try:
        with (folder / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [sys.executable, "-X", "utf8", str(ROOT / "4_https_split/server.py"),
                 *options, "--instance-id", instance],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=HIDDEN,
            )
            record["server_pid"] = server.pid
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Private server exited; inspect {folder / 'server.log'}.")
                try:
                    health, _ = rpc.call("/v1/health", {})
                    if health.get("instance_id") != instance:
                        raise RuntimeError("Port belongs to another server; refusing to use it.")
                    record["server_worker_pid"] = health["pid"]
                    break
                except RpcError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Private server did not become ready in 120 seconds.")
                    time.sleep(0.5)
            print(f"Private HTTPS B ready: PID={health['pid']}, port={port}; p/k/q={part.p}/{part.k}/{part.q}", flush=True)
            # Inherit the terminal for interactive chat. Never tee private chat into a log.
            client = subprocess.Popen(
                [sys.executable, "-X", "utf8", str(STAGE / "run_noise.py"), *arguments, *options],
                cwd=ROOT,
            )
            record["client_pid"] = client.pid
            record["client_returncode"] = client.wait(timeout=1800 if args.evaluate or args.prompt is not None else None)
            if record["client_returncode"] != 0:
                raise RuntimeError(f"Stage-5 client exited with code {record['client_returncode']}.")
            health, _ = rpc.call("/v1/health", {})
            record["remaining_sessions"] = health["active_sessions"]
            if health["active_sessions"] != 0:
                raise RuntimeError("Client left a session allocated on its private test server.")
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            stop_owned(client)
        finally:
            try:
                stop_owned(server)
            finally:
                record["server_stopped"] = server is not None and server.poll() is not None
                write_json(folder / "launcher.json", record)
                print(f"Launcher record: {folder / 'launcher.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; private launcher children stopped.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
