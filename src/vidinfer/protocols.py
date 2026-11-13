"""扩展接口约定：新模型与新设备需要提供哪些方法。

Protocol 用于表达结构化类型；只要对象具有这些成员，就能按约定使用。
当前 Engine 按实际对象调用这些方法，没有通过继承来强制实现接口。
本文件不导入 PyTorch，可先在 CPU 环境理解边界，再阅读真实后端。
"""

from typing import Any, ContextManager, Protocol
from .types import PreparedModel, ExecutionPlan, DenoiseRequest, RequestContext
from .registry import OperatorRegistry


class Completion(Protocol):
    """已提交设备工作的完成凭证：阻塞等待，或查询是否已经完成。"""

    def wait(self) -> None: ...
    def ready(self) -> bool: ...


class Backend(Protocol):
    """设备环境、算子实现集合与完成确认；可选 completed_metadata 不在必需接口中。"""

    name: str
    registry: OperatorRegistry

    def activate(self, context: RequestContext) -> ContextManager[None]: ...
    def record_completion(self, context: RequestContext) -> Completion: ...
    def drain(self, context: RequestContext) -> None: ...


class ModelAdapter(Protocol):
    """模型语义边界：检查请求、准备计划、执行模型流程、整理输出。"""

    def validate(self, request: DenoiseRequest) -> None: ...
    def prepare_plan(
        self, prepared: PreparedModel, name: str, backend: Backend
    ) -> ExecutionPlan: ...
    def execute(
        self, prepared: PreparedModel, plan: ExecutionPlan, context: RequestContext
    ) -> Any: ...
    def finalize(self, raw: Any, context: RequestContext) -> dict[str, Any]: ...
