"""通用请求执行器：排队 → 执行 → 确认设备完成 → 整理输出 → 清理资源。

本模块不实现 H3 的网络，也不直接启动 CUDA Kernel。
模型语义委托给 Adapter，设备环境与完成事件委托给 Backend。
"""

from concurrent.futures import ThreadPoolExecutor, CancelledError
import time
import traceback
from threading import RLock
from .types import RequestContext, InferenceResult

# 无法确认设备完成时，进程级引用阻止资源随 Engine 的回收而释放。
# 这是故障隔离，不是正常缓存；这类故障需要上层处理并重启进程。
_DEVICE_QUARANTINE = []


class QueueFull(RuntimeError):
    """运行及排队请求达到配置上限，调用者应稍后重试。"""


class RequestHandle:
    """调用者持有的凭证：Future 保存结果，context 保存进度与取消标记。"""

    def __init__(self, future, context):
        self._future = future
        self.context = context

    def wait(self, timeout=None):
        """等待整个请求结束；超时只结束等待，不会自动取消后台任务。"""
        return self._future.result(timeout)

    def ready(self):
        """查询 Future 是否结束；成功、失败和取消都属于已结束。"""
        return self._future.done()

    def cancel(self):
        """排队任务可直接取消；运行中的任务只设置协作式取消标记。"""
        self.context.cancel_event.set()
        if self._future.cancel():
            self.context.phase = "cancelled"
            self.context.release()
        return not self._future.done() or self._future.cancelled()

    def status(self):
        """读取进度，不改变队列顺序。"""
        return self.context.snapshot()


class Engine:
    """按 PreparedModel 串行执行请求，确认设备完成后才释放临时资源。

    Adapter 提供 validate/prepare_plan/execute/finalize；
    Backend 提供 activate/record_completion/drain。
    CPU 测试用替身实现这些方法，因此不需要安装 PyTorch。
    """

    def __init__(self, prepared, adapter, backend, max_pending=4):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if prepared.backend != backend.name:
            raise ValueError("PreparedModel/backend mismatch")
        self.prepared = prepared
        self.adapter = adapter
        self.backend = backend
        self._plans = {}
        # 一个后台线程处理请求；Kernel 内部的 GPU 并行不受这个数量限制。
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vidinfer"
        )
        self._lock = RLock()
        self._closed = False
        self._healthy = True
        self._quarantine = []
        self._max_pending = max_pending
        self._inflight = set()

    def plan(self, name):
        """取得或构建执行计划；在请求执行前完成算子选择与模型视图准备。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed")
            cached = self._plans.get(name)
        if cached is not None:
            return cached
        with self.prepared.execution_lock:
            p = self.adapter.prepare_plan(self.prepared, name, self.backend)
        if (
            p.model_id != self.prepared.model_id
            or p.model_revision != self.prepared.revision
        ):
            raise ValueError("ExecutionPlan belongs to another model revision")
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed")
            return self._plans.setdefault(name, p)

    def submit(self, request, plan="reference", on_step=None):
        """校验并入队，返回句柄；返回时不保证请求或 GPU 已完成。"""
        self.adapter.validate(request)
        execution_plan = self.plan(plan) if isinstance(plan, str) else plan
        with self._lock:
            if self._closed or not self._healthy:
                raise RuntimeError("Engine closed or device unhealthy")
            self._inflight = {future for future in self._inflight if not future.done()}
            if len(self._inflight) >= self._max_pending:
                raise QueueFull(
                    f"At most {self._max_pending} pending requests are allowed"
                )
            if (execution_plan.model_id, execution_plan.model_revision) != (
                self.prepared.model_id,
                self.prepared.revision,
            ):
                raise ValueError("ExecutionPlan belongs to another model revision")
            ctx = RequestContext(request, execution_plan, on_step)
            future = self._executor.submit(self._execute, ctx)
            self._inflight.add(future)
            future.add_done_callback(lambda done: self._request_done(done, ctx))
            handle = RequestHandle(future, ctx)
            return handle

    def _request_done(self, future, context):
        with self._lock:
            self._inflight.discard(future)
            if future.cancelled():
                # Future.cancel 会立即执行回调；先完成排队取消的状态清理，
                # 让后续日志观察者不会看到仍是 queued 的最终记录。
                context.phase = "cancelled"
                context.finished_at = time.perf_counter()
                context.release()

    def run(self, request, plan="reference", on_step=None):
        """同步便捷接口，相当于 submit(...).wait()。"""
        return self.submit(request, plan, on_step).wait()

    def _execute(self, ctx):
        """由后台线程调用；正常返回或抛出异常前都会经过 finally。"""
        raw = None
        error = None
        result = None
        outputs = None
        failure = None
        device_done = False
        # 1. 取得共享模型的执行权，再开始计算请求的运行耗时。
        with self.prepared.execution_lock:
            ctx.started_at = time.perf_counter()
            ctx.phase = "running"
            ctx.metadata["queue_s"] = ctx.started_at - ctx.submitted_at
            ctx.retain(self.prepared, ctx.plan)
            try:
                if not self._healthy:
                    raise RuntimeError("Device unhealthy; no further execution")
                ctx.check_cancelled()
                # 2. 后端设置当前设备、Stream 和推理模式。
                with self.backend.activate(ctx):
                    try:
                        raw = self.adapter.execute(self.prepared, ctx.plan, ctx)
                        ctx.retain(raw)
                        ctx.check_cancelled()
                    except BaseException as exc:
                        # 先记住异常；之前排入设备的工作仍需要等待。
                        error = exc
                    finally:
                        # 3. 即使取消或报错，也为已经提交的工作记录完成事件。
                        ctx.completion = self.backend.record_completion(ctx)
                        try:
                            ctx.completion.wait()
                            device_done = True
                        except BaseException:
                            self._healthy = False
                            raise
                    if error:
                        raise error
                    if hasattr(self.backend, "completed_metadata"):
                        ctx.metadata.update(self.backend.completed_metadata(ctx))
                    # 4. 回传/整理也可能提交设备工作，必须单独确认完成。
                    device_done = False
                    with ctx.stage("finalize"):
                        outputs = self.adapter.finalize(raw, ctx)
                    self.backend.drain(ctx)
                    device_done = True
                    ctx.metadata.update(
                        model_id=self.prepared.model_id,
                        revision=self.prepared.revision,
                        plan=ctx.plan.name,
                        implementations=dict(ctx.plan.implementations),
                        decisions=list(ctx.plan.decisions),
                        steps=ctx.steps_done,
                        wall_s=time.perf_counter() - ctx.started_at,
                    )
                    result = InferenceResult(
                        ctx.request.request_id, outputs, dict(ctx.metadata)
                    )
                ctx.phase = "completed"
                return result
            except CancelledError as exc:
                failure = exc
                ctx.phase = "cancelled"
                ctx.metadata["error_traceback"] = "".join(
                    traceback.format_exception(exc)
                )
                raise
            except BaseException as exc:
                failure = exc
                ctx.phase = "failed"
                ctx.metadata["error_traceback"] = "".join(
                    traceback.format_exception(exc)
                )
                raise
            finally:
                # 5. 若正常事件等待没能确认完成，再尝试同步排空 Stream。
                if ctx.device_context is not None and not device_done:
                    try:
                        self.backend.drain(ctx)
                        device_done = True
                    except BaseException:
                        self._healthy = False
                        self._quarantine.append(ctx)
                        _DEVICE_QUARANTINE.append(ctx)
                # 只有确认完成，或根本未进入设备环境，才清理请求引用。
                ctx.finished_at = time.perf_counter()
                if ctx.device_context is None or device_done:
                    ctx.release()
                else:
                    # 调用者可能清空异常堆栈，不能仅靠保留异常来保护 Tensor。
                    ctx.retain(raw, outputs, result)
                    self._retain_error_owners(ctx, failure, error)
                self._clear_tracebacks(failure, error)
                raw = outputs = result = failure = error = None

    @staticmethod
    def _exception_nodes(*errors):
        pending = [exc for exc in errors if exc is not None]
        seen = set()
        while pending:
            exc = pending.pop()
            if id(exc) in seen:
                continue
            seen.add(id(exc))
            yield exc
            pending.extend(
                item for item in (exc.__cause__, exc.__context__) if item is not None
            )
            pending.extend(getattr(exc, "exceptions", ()))

    @classmethod
    def _retain_error_owners(cls, ctx, *errors):
        seen = set()
        for exc in cls._exception_nodes(*errors):
            frame = exc.__traceback__
            while frame is not None:
                if id(frame.tb_frame) not in seen:
                    seen.add(id(frame.tb_frame))
                    ctx.retain(*tuple(frame.tb_frame.f_locals.values()))
                frame = frame.tb_next

    @classmethod
    def _clear_tracebacks(cls, *errors):
        # Future 会保存异常。异常堆栈里的局部变量可能继续引用 GPU Tensor；
        # 先把堆栈转成文本保存在 metadata，再清理这些框架调用帧。
        for exc in cls._exception_nodes(*errors):
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None

    def close(self):
        """停止接收任务，等待已提交任务完成，再清空计划缓存。不会强制卸载模型。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._plans.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
