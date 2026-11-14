"""实时界面的状态、队列、HTTP 边界和 Event 查询；不加载模型。"""

from concurrent.futures import CancelledError
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vidinfer import DenoiseRequest, ExecutionPlan
from vidinfer.types import RequestContext
from vidinfer.web.backend import WebBackend
from vidinfer.web.progress import atomic_json, handle_progress
from vidinfer.web.server import create_server


def context():
    return RequestContext(
        DenoiseRequest("test"), ExecutionPlan("test", "m", "r", {}, None)
    )


class ProgressTests(unittest.TestCase):
    def test_nested_live_stages_and_exception_status(self):
        ctx = context()
        with ctx.stage("parent"):
            with ctx.stage("child"):
                s = ctx.snapshot()
                self.assertEqual(s["stage"], "child")
                self.assertEqual(s["stage_history"][1]["parent"], "parent")
                self.assertGreaterEqual(s["stage_history"][0]["elapsed_s"], 0)
        self.assertTrue(
            all(s["status"] == "completed" for s in ctx.snapshot()["stage_history"])
        )
        with self.assertRaises(CancelledError):
            with ctx.stage("cancel"):
                raise CancelledError()
        self.assertEqual(ctx.snapshot()["stage_history"][-1]["status"], "cancelled")

    def test_gpu_events_are_queried_without_wait_and_never_regress(self):
        class Event:
            def __init__(self, ready):
                self.ready = ready

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                return 2.5

            def synchronize(self):
                raise AssertionError("Polling must not synchronize")

        ctx = context()
        ctx.steps_done = 2
        ctx.resources["forward_events"] = [
            (Event(True), Event(True)),
            (Event(True), Event(False)),
        ]
        handle = SimpleNamespace(context=ctx, status=ctx.snapshot)
        self.assertEqual(handle_progress(handle)["gpu_forward_ms"], [2.5])
        ctx.resources.clear()
        self.assertEqual(handle_progress(handle)["gpu_completed_forwards"], 1)
        ctx.metadata["forward_ms"] = [2.5, 3]
        self.assertEqual(handle_progress(handle)["gpu_completed_forwards"], 2)

    def test_demo_queue_limit_and_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = WebBackend(folder, demo=True)
            try:
                first = backend.submit({"prompt": "demo"})
                second = backend.submit({"prompt": "demo"})
                with self.assertRaises(RuntimeError):
                    backend.submit({"prompt": "demo"})
                backend.cancel(first)
                backend.cancel(second)
            finally:
                backend.close()
            self.assertEqual(backend.status(first)["phase"], "cancelled")
            self.assertEqual(backend.status(second)["phase"], "cancelled")

    def test_bad_request_does_not_enter_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = WebBackend(folder, demo=True)
            try:
                for data in (
                    {"prompt": "x", "frames": 22},
                    {"prompt": "x", "checkpoint": "/other"},
                    {"prompt": ""},
                ):
                    with self.assertRaises(ValueError):
                        backend.submit(data)
                self.assertEqual(list(backend.jobs.iterdir()), [])
                with self.assertRaises(ValueError):
                    backend.folder("../../x")
            finally:
                backend.close()

    def test_resident_submission_uses_existing_wire_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = root / "control"
            atomic_json(control / "status.json", {"pid": os.getpid(), "phase": "ready"})
            cfg = root / "bridge.json"
            atomic_json(
                cfg,
                dict(
                    control=str(control),
                    checkpoint=str(root / "weights"),
                    package_path=str(root / "src"),
                ),
            )
            backend = WebBackend(root / "runtime", resident_config=cfg)
            try:
                job = backend.submit({"prompt": "x", "output_type": "latent"})
                record = json.loads(
                    (control / "inbox" / ("web-" + job + ".json")).read_text()
                )
                self.assertEqual(record["experiment"], "qk_abba")
                self.assertEqual(
                    json.loads((backend.folder(job) / "experiment.json").read_text())[
                        "runner"
                    ],
                    "web",
                )
                backend.cancel(job)
                self.assertTrue((backend.folder(job) / "cancel.request").exists())
            finally:
                backend.close()

    def test_web_import_does_not_load_torch(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import vidinfer.web.server; assert 'torch' not in sys.modules",
            ],
            check=True,
        )

    def test_terminal_status_waits_for_future_and_device_cleanup(self):
        ctx = context()
        ctx.phase = "cancelled"
        handle = SimpleNamespace(context=ctx, status=ctx.snapshot, ready=lambda: False)
        self.assertEqual(handle_progress(handle)["phase"], "cancelling")
        handle.ready = lambda: True
        self.assertEqual(handle_progress(handle)["phase"], "cancelled")

    def test_standalone_session_is_loaded_only_once(self):
        from concurrent.futures import Future
        from vidinfer.engine import RequestHandle
        from vidinfer.types import InferenceResult

        class Session:
            def __init__(self):
                self.submissions = 0

            def submit(self, **options):
                self.submissions += 1
                ctx = context()
                ctx.phase = "completed"
                f = Future()
                f.set_result(InferenceResult(ctx.request.request_id, {}, {}))
                return RequestHandle(f, ctx)

            def close(self):
                pass

        session = Session()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                "vidinfer.session.H3Session.from_pretrained", return_value=session
            ) as load,
        ):
            backend = WebBackend(folder, checkpoint=folder)
            try:
                first = backend.submit({"prompt": "x"})
                second = backend.submit({"prompt": "y"})
            finally:
                backend.close()
            self.assertEqual(backend.status(first)["phase"], "completed")
            self.assertEqual(backend.status(second)["phase"], "completed")
            self.assertEqual(load.call_count, 1)
            self.assertEqual(session.submissions, 2)


class HTTPTests(unittest.TestCase):
    def test_tokens_origin_media_ranges_and_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = WebBackend(folder, demo=True)
            server = create_server(backend, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def request(method, path, body=None, headers=None):
                client = http.client.HTTPConnection(
                    "127.0.0.1", server.server_port, timeout=5
                )
                client.request(method, path, body=body, headers=headers or {})
                response = client.getresponse()
                data = response.read()
                status = response.status
                client.close()
                return status, data

            try:
                status, html = request("GET", "/")
                self.assertEqual(status, 200)
                token = re.search(r"const token='([A-Za-z0-9_-]+)'", html.decode())[1]
                good = {"Content-Type": "application/json", "X-VidInfer-Token": token}
                self.assertEqual(
                    request(
                        "POST", "/api/jobs", "{}", {"Content-Type": "application/json"}
                    )[0],
                    403,
                )
                self.assertEqual(
                    request("GET", "/api/info", headers={"Host": "evil.invalid"})[0],
                    403,
                )
                self.assertEqual(
                    request(
                        "POST",
                        "/api/jobs",
                        "{}",
                        dict(good, Origin="http://evil.invalid"),
                    )[0],
                    403,
                )
                status, data = request(
                    "POST", "/api/jobs", json.dumps({"prompt": "demo"}), good
                )
                self.assertEqual(status, 202)
                job = json.loads(data)["job_id"]
                self.assertEqual(
                    request("POST", "/api/jobs/" + job + "/cancel", "{}", good)[0], 202
                )
                media = Path(folder) / "outputs/test/video.mp4"
                media.parent.mkdir(parents=True)
                media.write_bytes(b"0123456789")
                fake = "f" * 32
                out = backend.folder(fake)
                out.mkdir()
                atomic_json(
                    out / "input.json",
                    dict(parameters={"prompt": "x"}, created_at=time.time()),
                )
                atomic_json(out / "live.json", dict(phase="completed"))
                atomic_json(
                    out / "result.json", dict(outputs={"video_path": str(media)})
                )
                path = "/api/jobs/" + fake + "/media/video.mp4"
                self.assertEqual(
                    request("GET", path, headers={"Range": "bytes=2-5"}), (206, b"2345")
                )
                self.assertEqual(
                    request("GET", path, headers={"Range": "bytes=-2"}), (206, b"89")
                )
                self.assertEqual(
                    request("GET", path, headers={"Range": "bytes=20-"})[0], 416
                )
                outside = Path(folder) / "secret.mp4"
                outside.write_bytes(b"private")
                atomic_json(
                    out / "result.json", dict(outputs={"video_path": str(outside)})
                )
                self.assertEqual(request("GET", path)[0], 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                backend.close()

    def test_public_binding_is_rejected(self):
        with self.assertRaises(ValueError):
            create_server(None, host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
