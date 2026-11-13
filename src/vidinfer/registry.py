"""算子选择：先检查设备、布局与数值约束，再返回可用实现。

注册表只做选择，不执行推理、不编译计算图，也不会偷偷转换精度。
"""

from dataclasses import dataclass, field
from typing import Any, Callable


class UnsupportedOperator(RuntimeError):
    """所有候选实现都无法满足当前算子规格。"""

    pass


@dataclass(frozen=True)
class OperatorSpec:
    """描述模型需要的算子；attributes 补充 head_dim 等专用属性。"""

    name: str
    backend: str
    dtype: str
    layout: str
    attributes: tuple[tuple[str, Any], ...] = ()
    numerical_contract: str = "reference"

    def get(self, key, default=None):
        """按名称读取附加属性；tuple 便于保持规格对象不可变。"""
        return dict(self.attributes).get(key, default)


@dataclass(frozen=True)
class Implementation:
    """一个具体实现：supports 检查能力，run 才真正执行计算。

    contract 是实现声明的数值语义标签，不代表注册表已经测过误差。
    对应的误差验证仍由独立算子与模型回归负责。
    """

    name: str
    operator: str
    contract: str
    supports: Callable[[OperatorSpec], bool] = field(repr=False, compare=False)
    run: Callable = field(repr=False, compare=False)


class OperatorRegistry:
    """准备阶段注册，运行前冻结；按优先顺序选择实现并保留拒绝原因。"""

    def __init__(self):
        self._items = {}
        self._sealed = False

    def register(self, implementation):
        """登记一个唯一命名的实现，不会在这里调用它的 run。"""
        if self._sealed:
            raise RuntimeError("Registry is frozen")
        if implementation.name in self._items:
            raise ValueError("Duplicate implementation")
        self._items[implementation.name] = implementation

    def seal(self):
        """停止新增实现，避免执行期间注册表发生变化。"""
        self._sealed = True

    def resolve(self, spec, preferred, fallback=()):
        """先试 preferred，再试 fallback；数值约束不一致时禁止回退。"""
        rejected = []
        for name in tuple(preferred) + tuple(fallback):
            # 设备/精度等能力不满足时继续尝试，但不改变请求的规格。
            impl = self._items.get(name)
            if impl is None or impl.operator != spec.name:
                rejected.append(name + ": not registered for operator")
            elif impl.contract != spec.numerical_contract:
                rejected.append(name + ": numerical contract mismatch")
            elif not impl.supports(spec):
                rejected.append(name + ": unsupported shape/dtype/layout/device")
            else:
                return impl, tuple(rejected)
        raise UnsupportedOperator(f"{spec.name}: " + "; ".join(rejected))
