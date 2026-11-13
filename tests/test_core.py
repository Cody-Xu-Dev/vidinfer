"""无 GPU 的核心回归：用替身检查调度和资源生命周期，不验证 CUDA 数值。

学习时可先看 FakeAdapter/FakeBackend，再读每个 test 的中文说明。
真实 H3 数值和设备行为由 bridge.py 的集成验证另行检查。
"""

import unittest
import gc
import weakref
from contextlib import contextmanager
from concurrent.futures import TimeoutError, CancelledError
from threading import Event
from vidinfer import Engine, PreparedModel, ExecutionPlan, DenoiseRequest
from vidinfer.registry import (
    OperatorRegistry,
    OperatorSpec,
    Implementation,
    UnsupportedOperator,
)


class FakeBackend:
    """只模拟激活、等待和失败，不会分配显存或执行 CUDA。"""

    name = "fake"

    def __init__(self):
        self.events = []
        self.fail_completion = False

    @contextmanager
    def activate(self, ctx):
        ctx.device_context = object()
        yield

    def record_completion(self, ctx):
        backend = self

        class Completion:
            def wait(self):
                backend.events.append(("wait", len(ctx.keepalive)))
                if backend.fail_completion:
                    raise RuntimeError("device fault")

        return Completion()

    def drain(self, ctx):
        if self.fail_completion:
            raise RuntimeError("device fault")
        self.events.append(("drain", len(ctx.keepalive)))


class FakeAdapter:
    """用事件控制后台线程，用 seed 作为输出，方便观察请求隔离。"""

    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.block = False
        self.fail = False

    def validate(self, r):
        if r.steps <= 0:
            raise ValueError("bad steps")

    def prepare_plan(self, p, name, b):
        return ExecutionPlan(name, p.model_id, p.revision, {"test": "fake"}, None)

    def execute(self, p, plan, ctx):
        ctx.retain(object())
        self.entered.set()
        if self.block:
            self.release.wait(5)
        ctx.check_cancelled()
        if self.fail:
            raise ValueError("model failure")
        ctx.step_finished()
        return {"value": ctx.request.seed}

    def finalize(self, raw, ctx):
        return raw.copy()


class EngineTests(unittest.TestCase):
    def make(self):
        a, b = FakeAdapter(), FakeBackend()
        e = Engine(PreparedModel("model", "revision", "fake", object()), a, b)
        self.addCleanup(e.close)
        return e, a, b

    def test_request_state_and_completion_ownership(self):
        """等待完成时资源仍被持有，完成后清理；后续请求使用自己的 seed。"""
        e, a, b = self.make()
        h = e.submit(DenoiseRequest("p", seed=7))
        self.assertEqual(h.wait().outputs["value"], 7)
        self.assertTrue(h.context.cleaned)
        self.assertEqual(h.context.keepalive, [])
        self.assertGreater(b.events[0][1], 0)
        self.assertEqual(e.run(DenoiseRequest("p", seed=8)).outputs["value"], 8)
        self.assertEqual(h.context.steps_done, 1)

    def test_queue_budget_rejects_excess_and_reclaims_finished_slots(self):
        """达到容量时拒绝新请求；取消排队请求或完成任务后归还容量。"""
        from vidinfer import QueueFull

        a, b = FakeAdapter(), FakeBackend()
        a.block = True
        e = Engine(PreparedModel("bounded", "r", "fake", object()), a, b, max_pending=2)
        self.addCleanup(e.close)
        first = e.submit(DenoiseRequest("p"))
        self.assertTrue(a.entered.wait(1))
        second = e.submit(DenoiseRequest("p"))
        with self.assertRaises(QueueFull):
            e.submit(DenoiseRequest("p"))
        second.cancel()
        replacement = e.submit(DenoiseRequest("p", seed=9))
        a.block = False
        a.release.set()
        first.wait()
        self.assertEqual(replacement.wait().outputs["value"], 9)

    def test_cancel_running_waits_before_cleanup_and_engine_recovers(self):
        """运行中取消不能提前清理，等待后正常结束取消并允许后续请求。"""
        e, a, b = self.make()
        a.block = True
        h = e.submit(DenoiseRequest("p"))
        self.assertTrue(a.entered.wait(1))
        h.cancel()
        self.assertFalse(h.context.cleaned)
        a.release.set()
        with self.assertRaises(CancelledError):
            h.wait()
        self.assertTrue(h.context.cleaned)
        self.assertEqual(b.events[0][0], "wait")
        a.block = False
        self.assertEqual(e.run(DenoiseRequest("p")).outputs["value"], 42)

    def test_queued_cancel_and_timeout_dont_release_running_work(self):
        """取消排队任务与等待超时，都不能释放另一个运行请求的资源。"""
        e, a, b = self.make()
        a.block = True
        h = e.submit(DenoiseRequest("p"))
        self.assertTrue(a.entered.wait(1))
        q = e.submit(DenoiseRequest("p"))
        q.cancel()
        with self.assertRaises(CancelledError):
            q.wait()
        with self.assertRaises(TimeoutError):
            h.wait(0.01)
        self.assertFalse(h.context.cleaned)
        a.release.set()
        h.wait()

    def test_failure_drains_and_foreign_plan_rejected(self):
        """模型异常先处理完成等待；拒绝属于其他模型的执行计划。"""
        e, a, b = self.make()
        a.fail = True
        with self.assertRaises(ValueError):
            e.run(DenoiseRequest("p"))
        self.assertEqual(b.events[0][0], "wait")
        a.fail = False
        self.assertEqual(e.run(DenoiseRequest("p")).outputs["value"], 42)
        p = ExecutionPlan("bad", "another", "revision", {}, None)
        with self.assertRaises(ValueError):
            e.submit(DenoiseRequest("p"), p)

    def test_failed_future_does_not_pin_adapter_temporaries(self):
        """即使调用者继续持有失败的 Future，异常堆栈也不应留住临时对象。"""
        e, a, b = self.make()

        class Owner:
            pass

        refs = []

        def execute(prepared, plan, ctx):
            temporary = Owner()
            refs.append(weakref.ref(temporary))
            ctx.retain(temporary)
            raise ValueError("failure with temporary memory owners")

        a.execute = execute
        handle = e.submit(DenoiseRequest("p"))
        with self.assertRaises(ValueError):
            handle.wait()
        gc.collect()
        self.assertTrue(handle.context.cleaned)
        self.assertIsNone(refs[0]())

    def test_device_fault_quarantines_owners_and_disables_execution(self):
        """无法确认设备完成时保留引用，并拒绝在该 Engine 上继续执行。"""
        e, a, b = self.make()
        b.fail_completion = True
        h = e.submit(DenoiseRequest("p"))
        with self.assertRaises(RuntimeError):
            h.wait()
        self.assertFalse(h.context.cleaned)
        from vidinfer.engine import _DEVICE_QUARANTINE

        self.assertIn(h.context, _DEVICE_QUARANTINE)
        e.close()
        self.assertTrue(h.context.keepalive)
        self.assertTrue(h.context.keepalive)
        with self.assertRaises(RuntimeError):
            e.submit(DenoiseRequest("p"))

    def test_distinct_cause_and_context_do_not_pin_temporaries(self):
        """异常的显式 cause 与隐式 context 都要清理，不能只遍历一条链。"""
        e, a, b = self.make()
        class Owner: pass
        refs = []
        def inner():
            temporary = Owner()
            refs.append(weakref.ref(temporary))
            raise ValueError("inner failure")
        def execute(prepared, plan, ctx):
            try:
                inner()
            except ValueError:
                raise RuntimeError("outer failure") from KeyError("separate cause")
        a.execute = execute
        handle = e.submit(DenoiseRequest("p"))
        with self.assertRaises(RuntimeError): handle.wait()
        gc.collect()
        self.assertIsNone(refs[0]())

    def test_finalize_failure_retains_owners_if_completion_is_unknown(self):
        """输出回传阶段也可能失败；未确认设备完成时不能清理异常帧中的拥有者。"""
        e, a, b = self.make()
        class Owner: pass
        refs = []
        def finalize(raw, ctx):
            temporary = Owner()
            refs.append(weakref.ref(temporary))
            b.fail_completion = True
            raise ValueError("copy-back failure")
        a.finalize = finalize
        handle = e.submit(DenoiseRequest("p"))
        with self.assertRaises(ValueError): handle.wait()
        e.close()
        gc.collect()
        self.assertFalse(handle.context.cleaned)
        self.assertIsNotNone(refs[0]())
        with self.assertRaises(RuntimeError): e.submit(DenoiseRequest("p"))

    def test_closed_and_invalid_request(self):
        """关闭 Engine 后不再接受任务；非法参数在提交阶段被拒绝。"""
        e, a, b = self.make()
        with self.assertRaises(ValueError):
            e.submit(DenoiseRequest("p", steps=0))
        e.close()
        with self.assertRaises(RuntimeError):
            e.submit(DenoiseRequest("p"))


class RegistryTests(unittest.TestCase):
    def test_capability_fallback_and_precision_contract(self):
        """快速实现不满足规格时回退，但禁止换到数值约束不同的实现。"""
        r = OperatorRegistry()
        r.register(
            Implementation(
                "special", "norm", "exact", lambda s: s.get("d") == 128, lambda: None
            )
        )
        r.register(
            Implementation(
                "reference",
                "norm",
                "exact",
                lambda s: s.backend == "cuda",
                lambda: None,
            )
        )
        r.register(
            Implementation("approx", "norm", "approx", lambda s: True, lambda: None)
        )
        s = OperatorSpec("norm", "cuda", "bf16", "LHD", (("d", 64),), "exact")
        impl, reason = r.resolve(s, ("special",), ("reference",))
        self.assertEqual(impl.name, "reference")
        self.assertTrue(reason)
        with self.assertRaises(UnsupportedOperator):
            r.resolve(s, ("approx",))
        r.seal()
        with self.assertRaises(RuntimeError):
            r.register(impl)


if __name__ == "__main__":
    unittest.main()
