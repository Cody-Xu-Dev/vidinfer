"""由已有 resident worker 热加载执行；HTTP 进程从不直接持有设备模型。"""

import json
from pathlib import Path
import time
from concurrent.futures import CancelledError

from .progress import atomic_json, monitor, read_json

_SESSION = None
_IDENTITY = None


def main(args, model, resident):
    global _SESSION, _IDENTITY
    folder = Path(args.out)
    spec = json.loads((folder / "experiment.json").read_text())
    if (folder / "cancel.request").exists():
        atomic_json(folder / "live.json", dict(phase="cancelled", stage_history=[]))
        return dict(summary=dict(cancelled=True))
    identity = (
        id(model),
        spec["workspace"],
        spec["checkpoint"],
        spec.get("text_cache"),
    )
    try:
        if _SESSION is None or _IDENTITY != identity:
            if _SESSION is not None:
                _SESSION.close()
            atomic_json(
                folder / "live.json",
                dict(
                    phase="running",
                    stage="prepare_session",
                    stage_history=[],
                    initializing_at=time.time(),
                ),
            )
            from ..session import H3Session

            begin = time.perf_counter()
            options = dict(
                model_id=resident["model_instance_id"],
                text_cache=spec.get("text_cache"),
            )
            if spec.get("adaln_cache"):
                options["adaln_cache"] = spec["adaln_cache"]
            _SESSION = H3Session.from_model(
                model, spec["checkpoint"], spec["workspace"], **options
            )
            _IDENTITY = identity
            atomic_json(
                folder / "setup.json",
                [
                    dict(
                        name="prepare_session",
                        parent=None,
                        status="completed",
                        elapsed_s=time.perf_counter() - begin,
                    )
                ],
            )
        result = monitor(_SESSION.submit(**spec["parameters"]), folder)
        return dict(
            summary=dict(
                completed=True, request_id=result.request_id, resident=resident
            )
        )
    except CancelledError:
        state = read_json(folder / "live.json", {})
        state["phase"] = "cancelled"
        atomic_json(folder / "live.json", state)
        return dict(summary=dict(cancelled=True, resident=resident))
    except BaseException as error:
        state = read_json(folder / "live.json", {})
        if state.get("phase") != "cancelled":
            state.update(phase="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(folder / "live.json", state)
        raise
