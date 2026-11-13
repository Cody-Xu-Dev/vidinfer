"""用真实 Engine 和 CPU 替身验证公开 Session 门面及最终请求记录。"""

import json
from pathlib import Path
import tempfile
import unittest
from vidinfer import Engine, H3Session, PreparedModel, RuntimeConfig
from test_core import FakeAdapter, FakeBackend


class SessionRecordsTests(unittest.TestCase):
    def make(self, folder):
        session = H3Session.__new__(H3Session)
        session.config = RuntimeConfig(Path(folder))
        adapter = FakeAdapter()
        session.engine = Engine(
            PreparedModel("test", "r", "fake", object()), adapter, FakeBackend()
        )
        return session, adapter

    def test_success_and_failure_records_are_written_without_tensor_state(self):
        with tempfile.TemporaryDirectory() as folder:
            session, adapter = self.make(folder)
            success = session.submit("p", steps=5, output_type="latent", seed=7)
            self.assertEqual(success.wait().outputs["value"], 7)
            adapter.fail = True
            failure = session.submit("p", steps=5, output_type="latent")
            with self.assertRaises(ValueError):
                failure.wait()
            session.engine.close()  # 等待所有完成回调写入记录。
            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (Path(folder) / "requests").glob("*.json")
            ]
            self.assertEqual(
                {record["status"]["phase"] for record in records},
                {"completed", "failed"},
            )
            failed = next(
                record for record in records if record["status"]["phase"] == "failed"
            )
            self.assertIn("model failure", failed["metadata"]["error_traceback"])
            self.assertTrue(all(record["status"]["cleaned"] for record in records))

    def test_queued_cancellation_record_has_final_cleaned_state(self):
        with tempfile.TemporaryDirectory() as folder:
            session, adapter = self.make(folder)
            adapter.block = True
            first = session.submit("p", steps=5, output_type="latent")
            self.assertTrue(adapter.entered.wait(1))
            queued = session.submit("p", steps=5, output_type="latent")
            queued.cancel()
            adapter.block = False
            adapter.release.set()
            first.wait()
            session.engine.close()
            path = (
                Path(folder)
                / "requests"
                / (queued.context.request.request_id + ".json")
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"]["phase"], "cancelled")
            self.assertTrue(record["status"]["cleaned"])
