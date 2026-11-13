"""第一层：实验描述。它只描述要比较什么，不分配显存。"""

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class ExperimentSpec:
    operation: str
    source: dict = field(default_factory=lambda: {"kind": "synthetic", "seed": 42})
    candidates: tuple = ()
    reference: str = ""
    mode: str = "allclose"
    atol: float = 1e-3
    rtol: float = 1e-2
    warmup: int = 3
    cycles: int = 3

    def __post_init__(self):
        defaults = {
            "attention": (
                "flash_view",
                "flash_contiguous",
                "cudnn_view",
                "cudnn_contiguous",
            ),
            "swiglu": ("torch", "cuda"),
            "qk_norm_rope": ("torch", "cuda_w4", "cuda_w8"),
        }
        if self.operation not in defaults:
            raise ValueError(f"Unknown operation: {self.operation}")
        names = tuple(self.candidates or defaults[self.operation])
        allowed = set(defaults[self.operation])
        if self.operation == "attention":
            allowed.update(("efficient_view", "efficient_contiguous"))
        if len(names) != len(set(names)) or not set(names) <= allowed:
            raise ValueError("Duplicate or unknown candidates")
        reference = self.reference or names[0]
        if reference not in names:
            raise ValueError("Reference must be one of the candidates")
        object.__setattr__(self, "candidates", names)
        object.__setattr__(self, "reference", reference)
        if self.mode not in ("exact", "allclose"):
            raise ValueError("mode must be exact or allclose")
        for value in (self.atol, self.rtol):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Tolerances must be finite nonnegative numbers")
        for name in ("warmup", "cycles"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f"{name} must be an integer in 1..100")
        if not isinstance(self.source, dict):
            raise ValueError("source must be an object")
        kind = self.source.get("kind", "synthetic")
        if kind not in ("synthetic", "tensor_file"):
            raise ValueError("source.kind must be synthetic or tensor_file")
        if kind == "tensor_file":
            if not isinstance(self.source.get("path"), str) or not self.source["path"]:
                raise ValueError("tensor_file requires a path")
            if set(self.source) - {"kind", "path"}:
                raise ValueError("Unknown tensor_file source option")
        else:
            fields = {"kind", "seed"}
            fields.update(
                ("rows", "width")
                if self.operation == "swiglu"
                else ("batch", "length", "heads", "head_dim")
            )
            if set(self.source) - fields:
                raise ValueError("Unknown synthetic source option")
            for name, value in self.source.items():
                if name == "kind":
                    continue
                if type(value) is not int or (
                    value < 0 if name == "seed" else value < 1
                ):
                    raise ValueError(f"Invalid synthetic {name}")
            if self.source.get("seed", 42) >= 2**63:
                raise ValueError("seed must be below 2**63")
            if (
                self.operation == "qk_norm_rope"
                and self.source.get("head_dim", 128) != 128
            ):
                raise ValueError("H3 QK kernel requires head_dim=128")

    def to_dict(self):
        data = asdict(self)
        data["candidates"] = list(self.candidates)
        return data


def balanced_orders(names, cycles):
    """每个候选均等地占据各位置，并以相反顺序重复；避免总让基准先跑。"""
    names = list(names)
    if not names:
        return []
    orders = []
    for _ in range(cycles):
        for offset in range(len(names)):
            order = names[offset:] + names[:offset]
            orders.extend((order, list(reversed(order))))
    return orders
