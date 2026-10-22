"""
Unit tests for Module 3 (Workload), Module 5 (Metrics), and Module 6 (Display).
Project 31: Thread Affinity Management for Multi-Core Processors (Phase 2)

Person 2 Focus:
- Metrics collection: timing (time.perf_counter), CPU utilization sampling
- Statistical aggregation: mean, min, max, standard deviation across repeated runs
- Speedup calculation vs baseline
- CSV export & formal text report generation
- Console visualization: ASCII per-core load bars & comparative summaries

Run with:
    python -m unittest test_modules_5_and_6.py -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

from display import (
    _render_bar,
    display_comparison_summary,
    display_configuration_summary,
    display_single_run_result,
    export_results_to_csv,
    export_summary_to_csv,
    export_results_to_txt,
    generate_benchmark_charts,
)
from metrics import (
    ConfigurationSummary,
    PerformanceMonitor,
    RunMetrics,
    aggregate_runs,
    calculate_confidence_interval_95,
    calculate_percentile,
)
from workload import (
    WorkloadType,
    get_workload_function,
    run_crypto_hash_workload,
    run_matrix_mult_workload,
    run_prime_sieve_workload,
)


class TestModule3Workload(unittest.TestCase):
    """Module 3 unit tests: Deterministic CPU-intensive workload execution."""

    def test_crypto_hash_deterministic(self):
        """Crypto hash workload returns identical operation count for same iterations."""
        r1 = run_crypto_hash_workload(0, iterations=50, chunk_size_kb=16)
        r2 = run_crypto_hash_workload(1, iterations=50, chunk_size_kb=16)
        self.assertEqual(r1, 50)
        self.assertEqual(r2, 50)

    def test_matrix_mult_deterministic(self):
        """Matrix multiplication produces deterministic checksums across threads."""
        c1 = run_matrix_mult_workload(0, matrix_size=20)
        c2 = run_matrix_mult_workload(1, matrix_size=20)
        self.assertEqual(c1, c2)
        self.assertGreater(c1, 0)

    def test_prime_sieve_deterministic(self):
        """Prime sieve finds identical prime counts (168 primes under 1000)."""
        p1 = run_prime_sieve_workload(0, limit=1000)
        p2 = run_prime_sieve_workload(1, limit=1000)
        self.assertEqual(p1, 168)
        self.assertEqual(p2, 168)

    def test_get_workload_factory(self):
        """Factory returns callable configuring workload with designated intensity."""
        func = get_workload_function(WorkloadType.CRYPTO_HASH, intensity=10)
        res = func(0)
        self.assertEqual(res, 10)


class TestModule5Metrics(unittest.TestCase):
    """Module 5 unit tests: Performance measurement and statistical aggregation."""

    def test_performance_monitor_timing(self):
        """PerformanceMonitor records elapsed wall-clock time and metadata."""
        cores = [0, 1, 2, 3]
        mapping = {0: [0], 1: [1]}
        monitor = PerformanceMonitor(
            "test_config", 2, cores, mapping, run_number=1, sample_interval_sec=0.02
        )
        with monitor:
            import time
            time.sleep(0.08)
        metrics = monitor.metrics
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.execution_time_sec, 0.05)
        self.assertEqual(metrics.thread_count, 2)
        self.assertEqual(metrics.core_count, 4)
        self.assertEqual(metrics.run_number, 1)

    def test_run_metrics_serialization(self):
        """RunMetrics serializes correctly to dictionary and flat CSV row format."""
        rm = RunMetrics(
            config_name="fixed",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [0], 1: [1]},
            run_number=2,
            logical_cpu_count=4,
            physical_core_count=2,
            execution_time_sec=0.1234,
            system_avg_cpu=25.0,
            active_cores_avg_cpu=50.0,
            timestamp_iso="2026-10-06 12:00:00",
        )
        d = rm.to_dict()
        self.assertEqual(d["config_name"], "fixed")
        self.assertEqual(d["run_number"], 2)
        self.assertEqual(d["thread_count"], 2)

        row = rm.to_csv_row()
        self.assertEqual(row["Configuration"], "FIXED")
        self.assertEqual(row["Run_Number"], 2)
        self.assertEqual(row["Execution_Time_s"], "0.1234")
        self.assertEqual(row["System_Avg_CPU_pct"], "25.00")
        self.assertEqual(row["Target_Cores_Avg_CPU_pct"], "50.00")

    def test_aggregate_runs_statistics(self):
        """aggregate_runs computes mean, min, max, std dev, and CPU utilization across multiple runs."""
        rm1 = RunMetrics(
            config_name="fixed",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [0], 1: [1]},
            run_number=1,
            execution_time_sec=1.0,
            system_avg_cpu=20.0,
            active_cores_avg_cpu=40.0,
        )
        rm2 = RunMetrics(
            config_name="fixed",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [0], 1: [1]},
            run_number=2,
            execution_time_sec=2.0,
            system_avg_cpu=30.0,
            active_cores_avg_cpu=60.0,
        )
        summary = aggregate_runs("fixed", [rm1, rm2], physical_core_count=2)
        self.assertEqual(summary.run_count, 2)
        self.assertEqual(summary.avg_time_sec, 1.5)
        self.assertEqual(summary.min_time_sec, 1.0)
        self.assertEqual(summary.max_time_sec, 2.0)
        self.assertAlmostEqual(summary.std_dev_time_sec, 0.707106, places=4)
        self.assertEqual(summary.avg_system_cpu, 25.0)
        self.assertEqual(summary.avg_target_cpu, 50.0)

    def test_single_run_std_dev_zero(self):
        """A single benchmark run should have a sample standard deviation of 0.0."""
        rm = RunMetrics(
            config_name="baseline",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [], 1: []},
            run_number=1,
            execution_time_sec=1.234,
            system_avg_cpu=15.0,
            active_cores_avg_cpu=15.0,
        )
        summary = aggregate_runs("baseline", [rm], physical_core_count=2)
        self.assertEqual(summary.run_count, 1)
        self.assertEqual(summary.avg_time_sec, 1.234)
        self.assertEqual(summary.std_dev_time_sec, 0.0)

    def test_aggregate_runs_empty_error(self):
        """aggregate_runs raises ValueError when provided an empty run list."""
        with self.assertRaises(ValueError):
            aggregate_runs("baseline", [])

    def test_speedup_calculation(self):
        """ConfigurationSummary.speedup_vs computes accurate speedup ratios relative to baseline."""
        baseline_summary = ConfigurationSummary(
            config_name="baseline",
            thread_count=4,
            logical_cpu_count=8,
            physical_core_count=4,
            run_count=3,
            avg_time_sec=2.0,
            min_time_sec=1.9,
            max_time_sec=2.1,
            std_dev_time_sec=0.1,
            avg_system_cpu=30.0,
            avg_target_cpu=30.0,
        )
        faster_summary = ConfigurationSummary(
            config_name="distributed",
            thread_count=4,
            logical_cpu_count=8,
            physical_core_count=4,
            run_count=3,
            avg_time_sec=1.0,
            min_time_sec=0.9,
            max_time_sec=1.1,
            std_dev_time_sec=0.1,
            avg_system_cpu=50.0,
            avg_target_cpu=80.0,
        )
        slower_summary = ConfigurationSummary(
            config_name="concentrated",
            thread_count=4,
            logical_cpu_count=8,
            physical_core_count=4,
            run_count=3,
            avg_time_sec=4.0,
            min_time_sec=3.8,
            max_time_sec=4.2,
            std_dev_time_sec=0.2,
            avg_system_cpu=25.0,
            avg_target_cpu=99.0,
        )
        zero_summary = ConfigurationSummary(
            config_name="invalid",
            thread_count=4,
            logical_cpu_count=8,
            physical_core_count=4,
            run_count=1,
            avg_time_sec=0.0,
            min_time_sec=0.0,
            max_time_sec=0.0,
            std_dev_time_sec=0.0,
            avg_system_cpu=0.0,
            avg_target_cpu=0.0,
        )

        # Baseline vs itself: 2.0 / 2.0 = 1.0x
        self.assertAlmostEqual(baseline_summary.speedup_vs(baseline_summary), 1.0)
        # Faster config (1.0s vs 2.0s baseline): 2.0 / 1.0 = 2.0x
        self.assertAlmostEqual(faster_summary.speedup_vs(baseline_summary), 2.0)
        # Slower config (4.0s vs 2.0s baseline): 2.0 / 4.0 = 0.5x
        self.assertAlmostEqual(slower_summary.speedup_vs(baseline_summary), 0.5)
        # Edge case: zero avg_time returns fallback 1.0
        self.assertEqual(zero_summary.speedup_vs(baseline_summary), 1.0)


class TestModule6Display(unittest.TestCase):
    """Module 6 unit tests: Results display, ASCII load bars, and file export."""

    def setUp(self):
        self.rm1 = RunMetrics(
            config_name="baseline",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [], 1: []},
            run_number=1,
            execution_time_sec=1.0,
            avg_per_core_cpu=[20.0, 20.0, 20.0, 20.0],
            system_avg_cpu=20.0,
            active_cores_avg_cpu=20.0,
            timestamp_iso="2026-10-06 12:00:00",
        )
        self.rm2 = RunMetrics(
            config_name="fixed",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [0], 1: [1]},
            run_number=1,
            execution_time_sec=0.5,
            avg_per_core_cpu=[80.0, 80.0, 2.0, 2.0],
            system_avg_cpu=41.0,
            active_cores_avg_cpu=80.0,
            timestamp_iso="2026-10-06 12:00:05",
        )
        self.sum1 = aggregate_runs("baseline", [self.rm1], physical_core_count=2)
        self.sum2 = aggregate_runs("fixed", [self.rm2], physical_core_count=2)

    def test_render_bar_ascii(self):
        """_render_bar produces clean ASCII bar formatted with percentage."""
        bar50 = _render_bar(50.0, width=10)
        self.assertIn("#####-----", bar50)
        self.assertIn("50.0%", bar50)

        bar0 = _render_bar(0.0, width=10)
        self.assertIn("----------", bar0)
        self.assertIn("0.0%", bar0)

        bar100 = _render_bar(100.0, width=10)
        self.assertIn("##########", bar100)
        self.assertIn("100.0%", bar100)

        # Clamping checks
        bar_neg = _render_bar(-10.0, width=10)
        self.assertIn("----------", bar_neg)

        bar_over = _render_bar(150.0, width=10)
        self.assertIn("##########", bar_over)

    def test_display_functions_no_exception(self):
        """Console display functions execute without runtime or encoding exceptions."""
        display_single_run_result(self.rm1)
        display_configuration_summary(self.sum1)
        display_comparison_summary([self.sum1, self.sum2])
        # Also test with empty summary list
        display_comparison_summary([])

    def test_csv_export(self):
        """export_results_to_csv writes structured benchmark runs to CSV with valid header."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            path = tf.name
        try:
            exported_path = export_results_to_csv([self.rm1, self.rm2], filepath=path)
            self.assertEqual(exported_path, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Configuration", content)
            self.assertIn("Run_Number", content)
            self.assertIn("Execution_Time_s", content)
            self.assertIn("BASELINE", content)
            self.assertIn("FIXED", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_csv_export_empty(self):
        """export_results_to_csv returns empty string when runs list is empty."""
        result = export_results_to_csv([])
        self.assertEqual(result, "")

    def test_txt_export(self):
        """export_results_to_txt writes formal academic benchmark report with hardware and summaries."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            path = tf.name
        try:
            cpu_info = {
                "physical_cores": 2,
                "logical_cores": 4,
                "core_ids": [0, 1, 2, 3],
                "physical_core_mapping": {0: [0, 1], 1: [2, 3]},
            }
            exported_path = export_results_to_txt(
                [self.sum1, self.sum2],
                [self.rm1, self.rm2],
                filepath=path,
                cpu_details=cpu_info,
            )
            self.assertEqual(exported_path, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("PROJECT 31: THREAD AFFINITY MANAGEMENT BENCHMARK REPORT", content)
            self.assertIn("HARDWARE TOPOLOGY SPECIFICATIONS", content)
            self.assertIn("CONFIGURATION SUMMARY", content)
            self.assertIn("Fastest observed configuration", content)
            self.assertIn("BASELINE", content)
            self.assertIn("FIXED", content)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_txt_export_empty(self):
        """export_results_to_txt returns empty string when summaries list is empty."""
        result = export_results_to_txt([], [])
        self.assertEqual(result, "")


class TestModule6ChartGeneration(unittest.TestCase):
    """Module 6 unit tests: Graphical visualization and PNG chart generation."""

    def _create_mock_summary(
        self,
        config_name: str,
        avg_time: float,
        std_dev: float = 0.02,
        core_cpu_vals: Optional[List[float]] = None,
        workload_name: str = "crypto_hash",
    ) -> ConfigurationSummary:
        """Helper to create realistic ConfigurationSummary for chart tests."""
        cpu_vals = core_cpu_vals if core_cpu_vals is not None else [40.0, 45.0, 30.0, 25.0]
        rep_run = RunMetrics(
            config_name=config_name,
            thread_count=4,
            core_count=len(cpu_vals),
            thread_to_cores={i: [i] for i in range(4)},
            run_number=1,
            execution_time_sec=avg_time,
            avg_per_core_cpu=cpu_vals,
            system_avg_cpu=sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0,
            active_cores_avg_cpu=sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0,
            workload_name=workload_name,
        )
        return ConfigurationSummary(
            config_name=config_name,
            thread_count=4,
            logical_cpu_count=len(cpu_vals),
            physical_core_count=2,
            run_count=3,
            avg_time_sec=avg_time,
            min_time_sec=avg_time * 0.95,
            max_time_sec=avg_time * 1.05,
            std_dev_time_sec=std_dev,
            avg_system_cpu=rep_run.system_avg_cpu,
            avg_target_cpu=rep_run.active_cores_avg_cpu,
            runs=[rep_run],
            thread_to_cores=rep_run.thread_to_cores,
            representative_run=rep_run,
            workload_name=workload_name,
        )

    def test_generate_charts_standard(self):
        """generate_benchmark_charts creates execution time, speedup, and CPU utilization PNGs."""
        summaries = [
            self._create_mock_summary("baseline", 1.0, 0.05, [25.0, 25.0, 25.0, 25.0]),
            self._create_mock_summary("fixed", 0.5, 0.02, [80.0, 80.0, 5.0, 5.0]),
            self._create_mock_summary("distributed", 0.45, 0.01, [85.0, 5.0, 85.0, 5.0]),
            self._create_mock_summary("concentrated", 1.8, 0.08, [99.0, 99.0, 2.0, 2.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = generate_benchmark_charts(
                summaries, output_dir=tmp_dir, workload_name="crypto_hash"
            )
            self.assertEqual(len(created), 3)

            basenames = [os.path.basename(f) for f in created]
            self.assertIn("execution_time_comparison.png", basenames)
            self.assertIn("speedup_comparison.png", basenames)
            self.assertIn("cpu_utilization_comparison.png", basenames)

            for f in created:
                self.assertTrue(os.path.exists(f))
                self.assertGreater(os.path.getsize(f), 1000)

    def test_generate_charts_empty_data(self):
        """generate_benchmark_charts returns empty list on empty input without creating files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = generate_benchmark_charts([], output_dir=tmp_dir)
            self.assertEqual(created, [])

    def test_generate_charts_no_baseline(self):
        """Speedup chart is omitted when BASELINE configuration is not present."""
        summaries = [
            self._create_mock_summary("fixed", 0.5, 0.02, [80.0, 80.0, 5.0, 5.0]),
            self._create_mock_summary("distributed", 0.45, 0.01, [85.0, 5.0, 85.0, 5.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = generate_benchmark_charts(
                summaries, output_dir=tmp_dir, workload_name="matrix_mult"
            )
            basenames = [os.path.basename(f) for f in created]
            self.assertIn("execution_time_comparison.png", basenames)
            self.assertNotIn("speedup_comparison.png", basenames)

    def test_generate_charts_no_cpu_data(self):
        """Per-core CPU chart is omitted when all recorded CPU samples are 0.0 or absent."""
        summaries = [
            self._create_mock_summary("baseline", 1.0, 0.05, [0.0, 0.0, 0.0, 0.0]),
            self._create_mock_summary("fixed", 0.5, 0.02, [0.0, 0.0, 0.0, 0.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = generate_benchmark_charts(
                summaries, output_dir=tmp_dir, workload_name="prime_sieve"
            )
            basenames = [os.path.basename(f) for f in created]
            self.assertIn("execution_time_comparison.png", basenames)
            self.assertIn("speedup_comparison.png", basenames)
            self.assertNotIn("cpu_utilization_comparison.png", basenames)

    def test_generate_charts_multi_workload(self):
        """generate_benchmark_charts prefixes and separates charts by workload."""
        summaries = [
            self._create_mock_summary("baseline", 1.0, workload_name="crypto_hash"),
            self._create_mock_summary("fixed", 0.5, workload_name="crypto_hash"),
            self._create_mock_summary("baseline", 2.0, workload_name="matrix_mult"),
            self._create_mock_summary("fixed", 1.2, workload_name="matrix_mult"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            created = generate_benchmark_charts(summaries, output_dir=tmp_dir)
            basenames = [os.path.basename(f) for f in created]
            self.assertIn("crypto_hash_execution_time_comparison.png", basenames)
            self.assertIn("crypto_hash_speedup_comparison.png", basenames)
            self.assertIn("matrix_mult_execution_time_comparison.png", basenames)
            self.assertIn("matrix_mult_speedup_comparison.png", basenames)

    def test_cli_plot_dir_argument(self):
        """main.py CLI parser correctly recognizes optional --plot-dir argument."""
        from unittest.mock import patch
        import sys
        from main import parse_arguments

        with patch.object(sys, "argv", ["main.py", "--plot-dir", "custom_charts"]):
            args = parse_arguments()
            self.assertEqual(args.plot_dir, "custom_charts")

        with patch.object(sys, "argv", ["main.py"]):
            args_default = parse_arguments()
            self.assertIsNone(args_default.plot_dir)


class TestModule5AdvancedStatistics(unittest.TestCase):
    """
    Unit tests for Step 3: Advanced Statistics and Analysis.
    Tests percentiles (P50, P90, P99), sample variance (s^2, df=N-1),
    Student's t confidence intervals (95%), edge cases, and summary CSV export.
    """

    def test_calculate_percentile_deterministic(self):
        """Percentiles computed via linear rank interpolation on deterministic sequence."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # P50 (median) of 1..10 is 5.5
        self.assertAlmostEqual(calculate_percentile(data, 50.0), 5.5, places=4)
        # P90: index 0.9 * 9 = 8.1 -> 9.0 + 0.1*(10 - 9) = 9.1
        self.assertAlmostEqual(calculate_percentile(data, 90.0), 9.1, places=4)
        # P99: index 0.99 * 9 = 8.91 -> 9.0 + 0.91*(1) = 9.91
        self.assertAlmostEqual(calculate_percentile(data, 99.0), 9.91, places=4)
        # Min and Max percentiles
        self.assertAlmostEqual(calculate_percentile(data, 0.0), 1.0, places=4)
        self.assertAlmostEqual(calculate_percentile(data, 100.0), 10.0, places=4)

    def test_calculate_percentile_edge_cases(self):
        """calculate_percentile handles empty list and single-element inputs without error."""
        self.assertEqual(calculate_percentile([], 50.0), 0.0)
        self.assertEqual(calculate_percentile([42.5], 50.0), 42.5)
        self.assertEqual(calculate_percentile([42.5], 99.0), 42.5)

    def test_sample_variance_bessel_correction(self):
        """
        Sample variance uses Bessel's correction (N - 1 degrees of freedom),
        yielding unbiased sample variance s^2, NOT population variance sigma^2.
        """
        # Deterministic dataset: [2, 4, 4, 4, 5, 5, 7, 9], N = 8
        # Mean = 40 / 8 = 5.0
        # Sum of squared deviations = 9 + 1 + 1 + 1 + 0 + 0 + 4 + 16 = 32
        # Sample variance = 32 / (8 - 1) = 32 / 7 = 4.571428...
        # Population variance would be 32 / 8 = 4.0 (incorrect for sample estimation)
        runs = [
            RunMetrics(
                config_name="fixed",
                thread_count=2,
                core_count=4,
                thread_to_cores={0: [0], 1: [1]},
                execution_time_sec=t,
                run_number=i + 1,
            )
            for i, t in enumerate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        ]
        summary = aggregate_runs("fixed", runs)
        expected_sample_variance = 32.0 / 7.0
        expected_sample_stdev = (32.0 / 7.0) ** 0.5
        self.assertAlmostEqual(summary.variance_time_sec, expected_sample_variance, places=4)
        self.assertAlmostEqual(summary.std_dev_time_sec, expected_sample_stdev, places=4)
        # Confirm it is strictly greater than population variance
        self.assertGreater(summary.variance_time_sec, 4.0)

    def test_confidence_interval_students_t(self):
        """Student's t-distribution confidence interval calculated with correct critical value."""
        # N = 4: df = 3, t_crit = 3.182
        # mean = 10.0, std_dev = 2.0 -> SE = 2.0 / sqrt(4) = 1.0
        # margin = 3.182 * 1.0 = 3.182
        # CI = [6.818, 13.182]
        lower, upper, margin, warning = calculate_confidence_interval_95(
            mean=10.0, std_dev=2.0, n=4
        )
        self.assertAlmostEqual(margin, 3.182, places=3)
        self.assertAlmostEqual(lower, 6.818, places=3)
        self.assertAlmostEqual(upper, 13.182, places=3)
        self.assertIn("Small sample size (N=4, df=3)", warning)

        # N = 10: df = 9, t_crit = 2.262
        lower10, upper10, margin10, warning10 = calculate_confidence_interval_95(
            mean=10.0, std_dev=2.0, n=10
        )
        expected_se = 2.0 / (10.0 ** 0.5)
        self.assertAlmostEqual(margin10, 2.262 * expected_se, places=3)
        self.assertEqual(warning10, "")  # No small sample warning when N >= 5

    def test_single_run_statistics_edge_case(self):
        """Single benchmark run explicitly documents unavailable/unreliable dispersion metrics."""
        rm = RunMetrics(
            config_name="baseline",
            thread_count=2,
            core_count=4,
            thread_to_cores={0: [], 1: []},
            execution_time_sec=1.2345,
            run_number=1,
        )
        summary = aggregate_runs("baseline", [rm])
        self.assertEqual(summary.run_count, 1)
        self.assertEqual(summary.variance_time_sec, 0.0)
        self.assertEqual(summary.std_dev_time_sec, 0.0)
        self.assertEqual(summary.p50_time_sec, 1.2345)
        self.assertEqual(summary.p90_time_sec, 1.2345)
        self.assertEqual(summary.p99_time_sec, 1.2345)
        self.assertEqual(summary.ci_95_lower, 1.2345)
        self.assertEqual(summary.ci_95_upper, 1.2345)
        self.assertEqual(summary.ci_margin, 0.0)
        self.assertIn("Sample size N=1 is insufficient", summary.ci_warning)

    def test_empty_runs_raises_exception(self):
        """aggregate_runs strictly raises ValueError on empty run list."""
        with self.assertRaises(ValueError):
            aggregate_runs("baseline", [])

    def test_export_summary_to_csv(self):
        """export_summary_to_csv writes structured statistical configuration summary."""
        rm1 = RunMetrics("baseline", 2, 4, {0: [], 1: []}, 1, execution_time_sec=1.0)
        rm2 = RunMetrics("baseline", 2, 4, {0: [], 1: []}, 2, execution_time_sec=1.2)
        sum1 = aggregate_runs("baseline", [rm1, rm2], workload_name="crypto_hash")

        rm3 = RunMetrics("fixed", 2, 4, {0: [0], 1: [1]}, 1, execution_time_sec=0.5)
        rm4 = RunMetrics("fixed", 2, 4, {0: [0], 1: [1]}, 2, execution_time_sec=0.6)
        sum2 = aggregate_runs("fixed", [rm3, rm4], workload_name="crypto_hash")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tf_path = tf.name

        try:
            exported = export_summary_to_csv([sum1, sum2], filepath=tf_path)
            self.assertEqual(exported, tf_path)
            self.assertTrue(os.path.exists(tf_path))

            with open(tf_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Configuration,Workload,Thread_Count,Run_Count,Mean_Time_s", content)
            self.assertIn("Median_P50_s,P90_s,P99_s", content)
            self.assertIn("Sample_Variance_s2,Sample_StdDev_s", content)
            self.assertIn("CI_95_Lower_s,CI_95_Upper_s", content)
            self.assertIn("Speedup_vs_Baseline", content)
            self.assertIn("BASELINE", content)
            self.assertIn("FIXED", content)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_export_summary_to_csv_empty(self):
        """export_summary_to_csv returns empty string when summaries list is empty."""
        result = export_summary_to_csv([])
        self.assertEqual(result, "")

    def test_txt_export_contains_advanced_statistics(self):
        """export_results_to_txt writes percentiles, variance, and confidence intervals."""
        rm1 = RunMetrics("baseline", 2, 4, {0: [], 1: []}, 1, execution_time_sec=1.0)
        rm2 = RunMetrics("baseline", 2, 4, {0: [], 1: []}, 2, execution_time_sec=1.1)
        sum1 = aggregate_runs("baseline", [rm1, rm2], workload_name="crypto_hash")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
            tf_path = tf.name

        try:
            exported = export_results_to_txt([sum1], [rm1, rm2], filepath=tf_path)
            self.assertEqual(exported, tf_path)
            with open(tf_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("--- ADVANCED STATISTICAL ANALYSIS & PERCENTILES ---", content)
            self.assertIn("Median P50", content)
            self.assertIn("Variance (s^2)", content)
            self.assertIn("95% CI of Mean", content)
            self.assertIn("Student's t-distribution", content)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_throughput_crypto_hash(self):
        """Throughput calculated only when consistent measurable operations are present."""
        rm1 = RunMetrics("fixed", 2, 4, {0: [0], 1: [1]}, 1, execution_time_sec=0.5)
        rm2 = RunMetrics("fixed", 2, 4, {0: [0], 1: [1]}, 2, execution_time_sec=0.5)
        # Total operations = 2 threads * 50000 = 100,000 hashes; avg_time = 0.5s -> 200,000 hashes/s
        summary = aggregate_runs(
            "fixed",
            [rm1, rm2],
            workload_name="crypto_hash",
            total_operations=100000,
            throughput_unit="hashes/s",
        )
        self.assertIsNotNone(summary.ops_per_sec)
        self.assertAlmostEqual(summary.ops_per_sec, 200000.0, places=1)
        self.assertEqual(summary.throughput_unit, "hashes/s")

        # Incompatible or absent operation count leaves ops_per_sec as None
        summary_no_ops = aggregate_runs("fixed", [rm1, rm2], workload_name="matrix_mult")
        self.assertIsNone(summary_no_ops.ops_per_sec)
        self.assertEqual(summary_no_ops.throughput_unit, "N/A")


if __name__ == "__main__":
    unittest.main()
