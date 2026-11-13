"""公共 Python API；导入本包不会自动加载权重、PyTorch 或 CUDA 扩展。"""

from .types import (
    DenoiseRequest,
    ExecutionPlan,
    PreparedModel,
    InferenceResult,
    Cancelled,
)
from .engine import Engine, QueueFull
from .runtime import RuntimeConfig
from .session import H3Session
from .registry import (
    OperatorRegistry,
    OperatorSpec,
    Implementation,
    UnsupportedOperator,
)

__all__ = [
    "Engine",
    "DenoiseRequest",
    "ExecutionPlan",
    "PreparedModel",
    "InferenceResult",
    "OperatorRegistry",
    "OperatorSpec",
    "Implementation",
    "UnsupportedOperator",
    "Cancelled",
    "QueueFull",
    "RuntimeConfig",
    "H3Session",
]
