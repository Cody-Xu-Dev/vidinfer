"""框架共用的数据对象：区分任务参数、共享模型、执行计划和每请求状态。

阅读顺序：DenoiseRequest → PreparedModel → ExecutionPlan → RequestContext。
本文件不导入 PyTorch，因此核心调度可以在没有 GPU 的环境中学习和测试。
"""

from dataclasses import dataclass, field
from concurrent.futures import CancelledError
from threading import Event, RLock
from types import MappingProxyType
from typing import Any, Mapping
import time
import uuid


class Cancelled(CancelledError):
    """协作式取消异常，由 Engine 接住并转成 cancelled 状态。"""

    pass


@dataclass(frozen=True)
class DenoiseRequest:
    """一次去噪的任务单；frozen=True 防止提交后改写字段。"""

    prompt: str
    seed: int = 42
    width: int = 960
    height: int = 544
    frames: int = 124
    steps: int = 5
    output_type: str = "latent"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class PreparedModel:
    """常驻模型的包装；payload 持有模型对象，不复制已有的设备权重。

    revision 当前是模型配置的哈希，不是权重文件的内容哈希。
    同一个对象可被多个 Engine 借用，它们共用下面的执行锁。
    """

    model_id: str
    revision: str
    backend: str
    payload: Any = field(repr=False, compare=False)
    # 锁属于这份 PreparedModel，而不是某一次请求。v0 对共享模型串行执行。
    execution_lock: Any = field(default_factory=RLock, repr=False, compare=False)


@dataclass(frozen=True)
class ExecutionPlan:
    """准备好的算子选择与模型视图；多次请求可以复用同一个计划。

    implementations 保存实现名称，payload 保存 H3 的私有模块树。
    frozen 只限制字段重新赋值，不会让 payload 内的 Tensor 自动变成只读。
    """

    name: str
    model_id: str
    model_revision: str
    implementations: Mapping[str, str]
    payload: Any = field(repr=False, compare=False)
    decisions: tuple[str, ...] = ()

    def __post_init__(self):
        # 同时禁止改写算子名称映射；共享权重只读仍靠调用约定保证。
        object.__setattr__(
            self, "implementations", MappingProxyType(dict(self.implementations))
        )


@dataclass(frozen=True)
class InferenceResult:
    """交给调用者的输出与诊断信息；当前 H3 adapter 返回 CPU latent。"""

    request_id: str
    outputs: Mapping[str, Any]
    metadata: Mapping[str, Any]


class RequestContext:
    """一次请求的可变状态，与共享的模型和计划分开管理。"""

    def __init__(self, request, plan, on_step=None):
        self.request = request
        self.plan = plan
        self.on_step = on_step
        self.cancel_event = Event()
        self._lock = RLock()
        self.phase = "queued"
        self.submitted_at = time.perf_counter()
        self.started_at = None
        self.finished_at = None
        self.steps_done = 0
        # keepalive 留住资源的拥有者；retain 增加引用，不复制 Tensor。
        self.keepalive = []
        # metadata 放可报告的数据；resources 放事件和临时设备 Tensor。
        self.metadata = {}
        self.resources = {}
        self.device_context = None
        self.completion = None
        self.cleaned = False

    def retain(self, *objects):
        """让这些对象至少活到请求确认设备完成并执行 release() 为止。"""
        self.keepalive.extend(objects)

    def check_cancelled(self):
        """只在安全边界检查标记，不会强行中断正在运行的 CUDA Kernel。"""
        if self.cancel_event.is_set():
            raise Cancelled("Request cancelled at a safe execution boundary")

    def step_finished(self):
        """统计前向已从 Python 返回的次数，此时 GPU 可能尚未完成。"""
        with self._lock:
            self.steps_done += 1
        if self.on_step:
            self.on_step(self)

    def snapshot(self):
        """给调用者一份轻量进度快照，不在这里等待 GPU。"""
        with self._lock:
            return dict(
                request_id=self.request.request_id,
                phase=self.phase,
                steps_done=self.steps_done,
                cleaned=self.cleaned,
                cancel_requested=self.cancel_event.is_set(),
                stage=self.metadata.get("active_stage"),
            )

    def release(self):
        """清理本请求临时引用；调用者必须先确认设备工作完成。

        request/plan/completion 等诊断与计划引用仍可能保留。
        清空 keepalive 不表示卸载外部常驻模型，也不等于清空显存缓存。
        """
        self.keepalive.clear()
        self.resources.clear()
        self.device_context = None
        self.cleaned = True
