"""界面任务后端：演示、本机 Session 或现有 H20 的串行文件队列。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid
import os

from ..runtime import RuntimeConfig
from ..types import DenoiseRequest
from ..validation import validate_generation_request
from .progress import atomic_json, read_json, monitor

TERMINAL = {"completed", "failed", "cancelled"}
FIELDS = {"prompt", "seed", "height", "width", "frames", "steps", "plan", "output_type"}


def parameters(value, config):
    if not isinstance(value, dict) or set(value) - FIELDS:
        raise ValueError("Unknown request parameters")
    p = dict(
        seed=42,
        height=544,
        width=960,
        frames=124,
        steps=5,
        plan="fully_fused",
        output_type="video",
    )
    p.update(value)
    if p.get("plan") not in ("reference", "qk_fused", "fully_fused"):
        raise ValueError("Unknown execution plan")
    validate_generation_request(
        DenoiseRequest(**{k: v for k, v in p.items() if k != "plan"}), config
    )
    return p


class WebBackend:
    def __init__(
        self,
        workspace,
        *,
        checkpoint=None,
        resident_config=None,
        demo=False,
        text_cache=None,
    ):
        self.config = RuntimeConfig(workspace=workspace)
        self.workspace = self.config.workspace
        self.jobs = self.workspace / "web/jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.bridge = (
            json.loads(Path(resident_config).read_text(encoding="utf-8"))
            if resident_config
            else None
        )
        self.demo = demo
        self.text_cache = text_cache
        self.session = None
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vidinfer-web"
        )
        if self.bridge:
            for key in ("control", "checkpoint", "package_path"):
                if not Path(self.bridge[key]).is_absolute():
                    raise ValueError("Bridge paths must be absolute")
            if not Path(self.bridge["control"], "status.json").exists():
                raise ValueError("Resident worker status missing")

    def info(self):
        status = (
            read_json(Path(self.bridge["control"]) / "status.json", {})
            if self.bridge
            else {}
        )
        if self.bridge:
            try:
                os.kill(int(status.get("pid", 0)), 0)
                status["process_alive"] = bool(status.get("pid"))
            except (OSError, ValueError, TypeError):
                status["process_alive"] = False
        return dict(
            mode="demo" if self.demo else "resident" if self.bridge else "standalone",
            resident=status,
            model_ready=status.get("phase") in ("ready", "running")
            if self.bridge
            else self.session is not None,
            cached_prompt=self.bridge.get("cached_prompt", "") if self.bridge else "",
            message="演示模式：阶段与时间为模拟数据，不是真实推理。"
            if self.demo
            else "首次文本编码/解码组件加载可能较慢；当前阶段会持续计时。",
        )

    def folder(self, job):
        if (
            not isinstance(job, str)
            or len(job) != 32
            or any(c not in "0123456789abcdef" for c in job)
        ):
            raise ValueError("Invalid job id")
        return self.jobs / job

    def submit(self, value):
        p = parameters(value, self.config)
        with self.lock:
            active = [
                j
                for j in self.jobs.iterdir()
                if j.is_dir() and self.status(j.name).get("phase") not in TERMINAL
            ]
            if len(active) >= 2:
                raise RuntimeError("最多同时保留两个运行中或排队任务，请等待或取消。")
            job = uuid.uuid4().hex
            folder = self.folder(job)
            folder.mkdir()
            atomic_json(
                folder / "input.json", dict(parameters=p, created_at=time.time())
            )
            atomic_json(
                folder / "live.json",
                dict(
                    phase="queued",
                    steps_done=0,
                    stage_history=[],
                    gpu_forward_ms=[],
                    updated_at=time.time(),
                ),
            )
            try:
                if self.bridge:
                    self._enqueue(job, p)
                else:
                    self.executor.submit(self._run, job, p)
            except Exception as error:
                atomic_json(
                    folder / "live.json",
                    dict(
                        phase="failed",
                        error=str(error),
                        stage_history=[],
                        updated_at=time.time(),
                    ),
                )
                raise
            return job

    def _enqueue(self, job, p):
        if not self.info()["resident"].get("process_alive"):
            raise RuntimeError("Resident worker is not alive")
        control = Path(self.bridge["control"])
        wire = "web-" + job
        folder = self.folder(job)
        # 此适配器专用于已有 H3 resident worker 的 qk_abba 热加载入口。
        spec = dict(
            runner="web",
            package_path=self.bridge["package_path"],
            checkpoint=self.bridge["checkpoint"],
            workspace=str(self.workspace),
            text_cache=self.bridge.get("text_cache"),
            adaln_cache=self.bridge.get("adaln_cache"),
            parameters=p,
        )
        atomic_json(folder / "experiment.json", spec)
        request = dict(
            id=wire,
            experiment="qk_abba",
            requested_experiment="web",
            steps=p["steps"],
            output=str(folder),
            skip_warmup=True,
            submitted_at=time.time(),
        )
        atomic_json(control / "jobs" / wire / "request.json", request)
        atomic_json(
            control / "jobs" / wire / "status.json",
            dict(phase="queued", request=request),
        )
        atomic_json(control / "inbox" / (wire + ".json"), request)

    def _run(self, job, p):
        folder = self.folder(job)
        atomic_json(folder / "started.json", dict(started_at=time.time()))
        try:
            if (folder / "cancel.request").exists():
                atomic_json(
                    folder / "live.json", dict(phase="cancelled", stage_history=[])
                )
                return
            if self.demo:
                self._demonstrate(folder, p)
                return
            if self.session is None:
                atomic_json(
                    folder / "live.json",
                    dict(
                        phase="running",
                        stage="load_model",
                        stage_history=[],
                        initializing_at=time.time(),
                    ),
                )
                from ..session import H3Session

                begin = time.perf_counter()
                self.session = H3Session.from_pretrained(
                    self.checkpoint, self.workspace, text_cache=self.text_cache
                )
                atomic_json(
                    folder / "setup.json",
                    [
                        dict(
                            name="load_model",
                            parent=None,
                            status="completed",
                            elapsed_s=time.perf_counter() - begin,
                        )
                    ],
                )
            monitor(self.session.submit(**p), folder)
        except BaseException as error:
            state = read_json(folder / "live.json", {})
            if state.get("phase") != "cancelled":
                state.update(phase="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(folder / "live.json", state)

    def _demonstrate(self, folder, p):
        start = time.perf_counter()
        history = []
        values = []
        for name, duration in (
            ("text_condition", 0.8),
            ("denoise", 2),
            ("decode", 0.8),
            ("export", 0.5),
        ):
            begin = time.perf_counter()
            record = dict(name=name, parent=None, status="running", elapsed_s=0)
            history.append(record)
            while time.perf_counter() - begin < duration:
                if (folder / "cancel.request").exists():
                    record["status"] = "cancelled"
                    atomic_json(
                        folder / "live.json",
                        dict(phase="cancelled", stage_history=history),
                    )
                    return
                record["elapsed_s"] = time.perf_counter() - begin
                if name == "denoise":
                    values = [duration * 1000 / (p["steps"] - 1)] * min(
                        p["steps"] - 1,
                        int(record["elapsed_s"] / duration * (p["steps"] - 1)),
                    )
                atomic_json(
                    folder / "live.json",
                    dict(
                        phase="running",
                        stage=name,
                        elapsed_s=time.perf_counter() - start,
                        stage_history=history,
                        gpu_forward_ms=values,
                        gpu_completed_forwards=len(values),
                        steps_done=len(values),
                        total_forwards=p["steps"] - 1,
                        updated_at=time.time(),
                    ),
                )
                time.sleep(0.15)
            record.update(status="completed", elapsed_s=time.perf_counter() - begin)
            if name == "denoise":
                values = [duration * 1000 / (p["steps"] - 1)] * (p["steps"] - 1)
        atomic_json(
            folder / "live.json",
            dict(
                phase="completed",
                elapsed_s=time.perf_counter() - start,
                stage_history=history,
                gpu_forward_ms=values,
                gpu_completed_forwards=p["steps"] - 1,
                total_forwards=p["steps"] - 1,
                updated_at=time.time(),
            ),
        )

    def status(self, job):
        folder = self.folder(job)
        request = read_json(folder / "input.json")
        if request is None:
            raise FileNotFoundError(job)
        state = read_json(folder / "live.json", {})
        if self.bridge:
            worker = read_json(
                Path(self.bridge["control"]) / "jobs" / ("web-" + job) / "status.json",
                {},
            )
            started_at = worker.get("started_at")
            if state.get("phase") not in TERMINAL and worker.get("phase") == "failed":
                state.update(phase="failed", error=worker.get("error", "Worker failed"))
        else:
            started_at = read_json(folder / "started.json", {}).get("started_at")
        queue_end = started_at or (
            state.get("updated_at", request["created_at"])
            if state.get("phase") in TERMINAL
            else time.time()
        )
        queue_s = max(0, queue_end - request["created_at"])
        state.update(
            job_id=job,
            parameters=request["parameters"],
            created_at=request["created_at"],
            server_now=time.time(),
        )
        if state.get("initializing_at") and not state.get("stage_history"):
            state["stage_history"] = [
                dict(
                    name=state.get("stage", "prepare_session"),
                    status="running",
                    parent=None,
                    elapsed_s=time.time() - state["initializing_at"],
                )
            ]
            state["elapsed_s"] = time.time() - state["initializing_at"]
        setup = read_json(folder / "setup.json", [])
        if setup:
            state["stage_history"] = setup + state.get("stage_history", [])
            state["elapsed_s"] = state.get("elapsed_s", 0) + sum(
                s["elapsed_s"] for s in setup
            )
        state["engine_queue_s"] = state.get("queue_s")
        state["queue_s"] = queue_s
        state["stage_history"] = [
            dict(
                name="queued",
                parent=None,
                status="running" if state.get("phase") == "queued" else "completed",
                elapsed_s=queue_s,
            )
        ] + state.get("stage_history", [])
        state["result"] = read_json(folder / "result.json")
        return state

    def cancel(self, job):
        state = self.status(job)
        requested = state["phase"] not in TERMINAL
        if requested:
            (self.folder(job) / "cancel.request").touch()
        return dict(requested=requested, job_id=job)

    def recent(self):
        folders = sorted(
            (
                j
                for j in self.jobs.iterdir()
                if j.is_dir() and (j / "input.json").exists()
            ),
            key=lambda j: j.stat().st_mtime,
            reverse=True,
        )[:20]
        return [
            dict(
                job_id=j.name,
                phase=self.status(j.name).get("phase"),
                output_type=read_json(j / "input.json", {})
                .get("parameters", {})
                .get("output_type"),
                created_at=read_json(j / "input.json", {}).get("created_at"),
            )
            for j in folders
        ]

    def media(self, job, name):
        if name not in ("video.mp4", "audio.wav"):
            raise ValueError("Unknown media")
        state = self.status(job)
        key = "video_path" if name == "video.mp4" else "audio_path"
        value = (state.get("result") or {}).get("outputs", {}).get(key)
        if not value:
            raise FileNotFoundError(name)
        path = Path(value).resolve()
        if (
            not path.is_relative_to((self.workspace / "outputs").resolve())
            or not path.is_file()
        ):
            raise ValueError("Media outside workspace")
        return path

    def close(self):
        self.executor.shutdown(wait=True)
        if self.session:
            self.session.close()
