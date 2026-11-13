"""验证实验协议的边界，而不是用重复实现检查被测 Kernel。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vidinfer.experiments.spec import ExperimentSpec, balanced_orders
from vidinfer.experiments.metrics import summarize
from vidinfer.experiments.report import write_reports


class ProtocolTests(unittest.TestCase):
    def test_every_candidate_occupies_every_position_equally(self):
        for size in (1, 2, 3, 4, 6):
            names = list(range(size))
            orders = balanced_orders(names, 2)
            self.assertEqual(len(orders), 4 * size)
            for order in orders:
                self.assertEqual(set(order), set(names))
            for candidate in names:
                for position in range(size):
                    self.assertEqual(
                        sum(order[position] == candidate for order in orders), 4
                    )

    def test_bad_configuration_fails_before_cuda(self):
        for options in (
            dict(operation="unknown"),
            dict(operation="attention", reference="missing"),
            dict(operation="swiglu", candidates=("torch", "torch")),
            dict(operation="swiglu", atol=float("nan")),
            dict(operation="swiglu", cycles=True),
            dict(operation="attention", source={"kind": "tensor_file"}),
            dict(operation="qk_norm_rope", source={"head_dim": 64}),
            dict(operation="swiglu", source={"seed": -1}),
            dict(operation="swiglu", source={"length": 100}),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExperimentSpec(**options)

    def test_samples_and_quantiles_are_preserved(self):
        result = summarize([4, 1, 3, 2])
        self.assertEqual(result["samples_ms"], [4, 1, 3, 2])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["p90_ms"], 3.7)
        with self.assertRaises(ValueError):
            summarize([float("inf")])

    def test_failed_report_is_renderable_and_escapes_input(self):
        spec = ExperimentSpec(operation="attention").to_dict()
        report = dict(
            schema_version=1,
            spec=spec,
            created_at="2026-11-13",
            completed=False,
            candidates={
                "<script>alert(1)</script>": dict(
                    status="probe_error", error="bad | backend"
                )
            },
            fatal_error="<img src=x onerror=alert(1)>",
        )
        with tempfile.TemporaryDirectory() as directory:
            write_reports(report, directory)
            folder = Path(directory)
            self.assertEqual(
                json.loads((folder / "results.json").read_text(encoding="utf-8")),
                report,
            )
            html = (folder / "report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertNotIn("<img src", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("False", html)

    def test_report_import_does_not_load_torch(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import vidinfer.experiments.report; assert 'torch' not in sys.modules",
            ],
            check=True,
        )


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "Requires NumPy for numerical metrics"
)
class MetricTests(unittest.TestCase):
    def compare(self, a, b, **kwargs):
        from vidinfer.experiments.metrics import compare_chunks

        return compare_chunks(
            zip(a, b),
            mode=kwargs.get("mode", "allclose"),
            atol=kwargs.get("atol", 1e-3),
            rtol=kwargs.get("rtol", 1e-2),
        )

    def test_same_direction_wrong_magnitude_is_rejected(self):
        result = self.compare([[1.0, 2.0]], [[10.0, 20.0]])
        self.assertAlmostEqual(result["cosine_similarity"], 1)
        self.assertAlmostEqual(result["relative_l2"], 9)
        self.assertFalse(result["passes"])

    def test_zero_reference_and_nonfinite_are_explicit(self):
        zero = self.compare([[0.0, 0.0]], [[0.0, 0.0]])
        self.assertTrue(zero["passes"])
        self.assertEqual(zero["relative_l2"], 0)
        self.assertIsNone(zero["cosine_similarity"])
        nonzero = self.compare([[0.0]], [[1.0]])
        self.assertIsNone(nonzero["relative_l2"])
        self.assertFalse(nonzero["passes"])
        bad = self.compare([[float("nan")]], [[float("nan")]], mode="exact")
        self.assertFalse(bad["passes"])
        json.dumps(bad, allow_nan=False)

    def test_exact_distinguishes_signed_zero_and_tolerance(self):
        self.assertFalse(self.compare([[0.0]], [[-0.0]], mode="exact")["passes"])
        self.assertTrue(self.compare([[0.0]], [[-0.0]])["passes"])
        self.assertTrue(self.compare([[1.0]], [[1.001]])["passes"])
        self.assertFalse(self.compare([[1.0]], [[1.001]], mode="exact")["passes"])

    def test_chunk_boundaries_do_not_change_aggregate(self):
        full = self.compare([[1.0, 2.0, 3.0, 4.0]], [[1.01, 2.0, 3.1, 4.0]])
        split = self.compare([[1.0, 2.0], [3.0, 4.0]], [[1.01, 2.0], [3.1, 4.0]])
        for key in (
            "elements",
            "different_elements",
            "failed_elements",
            "max_absolute_error",
            "mean_absolute_error",
            "relative_l2",
            "cosine_similarity",
        ):
            self.assertAlmostEqual(full[key], split[key])
        with self.assertRaises(ValueError):
            self.compare([], [])


if __name__ == "__main__":
    unittest.main()
