import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_local


class LauncherTests(unittest.TestCase):
    def test_stop_only_owned_child_and_wait_for_exit(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            creationflags=run_local.HIDDEN,
        )
        try:
            run_local.stop_owned(child)
            self.assertIsNotNone(child.poll())
            run_local.stop_owned(child)
            run_local.stop_owned(None)
        finally:
            if child.poll() is None:
                run_local.stop_owned(child)

    def test_wrong_server_instance_is_refused_and_only_owned_child_is_cleaned(self):
        server = SimpleNamespace(pid=12345, poll=lambda: None)
        rpc = SimpleNamespace(call=Mock(return_value=({"instance_id": "someone-else", "pid": 99999}, {})))
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(run_local, "STAGE", Path(temp)))
                stack.enter_context(patch.object(run_local, "HttpsRpc", return_value=rpc))
                launch = stack.enter_context(patch.object(run_local.subprocess, "Popen", return_value=server))
                stop = stack.enter_context(patch.object(run_local, "stop_owned"))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, "another server"):
                    run_local.main(["--evaluate", "--port", "8447"])
                self.assertEqual(launch.call_count, 1)
                stop.assert_any_call(server)
                self.assertNotIn(99999, [args[0][0] for args in stop.call_args_list])
            files = list((Path(temp) / "runs").glob("*/launcher.json"))
            self.assertEqual(json.loads(files[0].read_text())["status"], "error")


if __name__ == "__main__":
    unittest.main()
