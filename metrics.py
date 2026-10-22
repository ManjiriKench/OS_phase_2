"""
Module 5: Performance Measurement
Project: Thread Affinity Management for Multi-Core Processors (Phase 1)

This module handles:
1. High-precision wall-clock timing using time.perf_counter().
2. Tracking active thread counts, logical processor counts, physical core counts,
   and thread-to-core assignment metadata.
3. Background sampling of per-core CPU utilization via psutil.cpu_percent(percpu=True).
4. Generating structured RunMetrics objects for individual runs.
5. Statistical aggregation across multiple repeated benchmark runs
   (mean, min, max, standard deviation, and average CPU utilization).
"""

from __future__ import annotations

import datetime
import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class RunMetrics:
    """
    Structured performance metrics for a single execution run.
    """
    config_name: str
    thread_count: int
    core_count: int
    thread_to_cores: Dict[int, List[int]]
    run_number: int = 1
    logical_cpu_count: int = 0
    physical_core_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time_sec: float = 0.0
    per_core_samples: List[List[float]] = field(default_factory=list)
    avg_per_core_cpu: List[float] = field(default_factory=list)
    peak_per_core_cpu: List[float] = field(default_factory=list)
    system_avg_cpu: float = 0.0
    active_cores_avg_cpu: float = 0.0
    timestamp_iso: str = ""
    thread_statuses: Dict[int, str] = field(default_factory=dict)
    workload_name: str = ""

    def to_dict(self) -> dict:
        """Serializes the metrics to a standard dictionary."""
        return {
            "config_name": self.config_name,
            "workload_name": self.workload_name,
            "run_number": self.run_number,
            "thread_count": self.thread_count,
            "logical_cpu_count": self.logical_cpu_count or self.core_count,
            "physical_core_count": self.physical_core_count,
            "thread_to_cores": {str(k): v for k, v in self.thread_to_cores.items()},
            "execution_time_sec": round(self.execution_time_sec, 4),
            "system_avg_cpu": round(self.system_avg_cpu, 2),
            "active_cores_avg_cpu": round(self.active_cores_avg_cpu, 2),
            "avg_per_core_cpu": [round(val, 2) for val in self.avg_per_core_cpu],
            "peak_per_core_cpu": [round(val, 2) for val in self.peak_per_core_cpu],
            "timestamp_iso": self.timestamp_iso,
        }

    def to_csv_row(self) -> dict:
        """Returns flattened dictionary suitable for CSV export."""
        assigned_str = ";".join(
            f"T{t}->{','.join(str(c) for c in cores) if cores else 'OS'}"
            for t, cores in sorted(self.thread_to_cores.items())
        )
        row = {
            "Timestamp": self.timestamp_iso,
            "Configuration": self.config_name.upper(),
            "Run_Number": self.run_number,
            "Thread_Count": self.thread_count,
            "Logical_Processors": self.logical_cpu_count or self.core_count,
            "Physical_Cores": self.physical_core_count,
            "Assignments": assigned_str,
            "Execution_Time_s": f"{self.execution_time_sec:.4f}",
            "System_Avg_CPU_pct": f"{self.system_avg_cpu:.2f}",
            "Target_Cores_Avg_CPU_pct": f"{self.active_cores_avg_cpu:.2f}",
        }
        if self.workload_name:
            row["Workload"] = self.workload_name.upper()
        return row


# Two-tailed Student's t critical values for 95% confidence (alpha = 0.05, df = N - 1)
T_CRITICAL_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def calculate_percentile(data: List[float], percentile: float) -> float:
    """
    Computes the p-th percentile (0 <= percentile <= 100) using linear interpolation
    between nearest rank indices (standard linear quantile formula).
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = (percentile / 100.0) * (n - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    weight = idx - lower
    return sorted_data[lower] + weight * (sorted_data[upper] - sorted_data[lower])


def calculate_confidence_interval_95(
    mean: float, std_dev: float, n: int
) -> Tuple[float, float, float, str]:
    """
    Calculates the two-tailed 95% confidence interval for the mean execution time
    using Student's t-distribution with df = n - 1.

    Assumptions:
    1. Independent and identically distributed (i.i.d.) observations under constant nominal settings.
    2. Underlying execution-time population is approximately normally distributed.

    Returns:
        (ci_lower, ci_upper, margin_of_error, warning_message)
    """
    if n < 1:
        return (0.0, 0.0, 0.0, "Empty dataset; cannot compute confidence interval.")
    if n == 1:
        return (
            mean,
            mean,
            0.0,
            "Sample size N=1 is insufficient for statistical confidence interval inference (degrees of freedom = 0).",
        )

    df = n - 1
    t_crit = T_CRITICAL_95.get(df, 1.960 if df > 30 else 2.042)
    std_error = std_dev / math.sqrt(n)
    margin = t_crit * std_error
    lower = max(0.0, mean - margin)
    upper = mean + margin

    if n < 5:
        warning = (
            f"Small sample size (N={n}, df={df}). High t-critical value ({t_crit:.3f}) "
            "yields exploratory confidence bounds."
        )
    else:
        warning = ""

    return (lower, upper, margin, warning)


@dataclass
class ConfigurationSummary:
    """
    Statistical summary aggregated across multiple repeated runs of a configuration.
    Includes mean, median (P50), P90, P99, sample variance (s^2, df=N-1),
    sample standard deviation (s, df=N-1), 95% Student's t confidence intervals,
    and speedup vs baseline.
    """
    config_name: str
    thread_count: int
    logical_cpu_count: int
    physical_core_count: int
    run_count: int
    avg_time_sec: float
    min_time_sec: float
    max_time_sec: float
    std_dev_time_sec: float
    avg_system_cpu: float
    avg_target_cpu: float
    runs: List[RunMetrics] = field(default_factory=list)
    thread_to_cores: Dict[int, List[int]] = field(default_factory=dict)
    representative_run: Optional[RunMetrics] = None
    workload_name: str = ""
    p50_time_sec: float = 0.0
    p90_time_sec: float = 0.0
    p99_time_sec: float = 0.0
    variance_time_sec: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    ci_margin: float = 0.0
    ci_warning: str = ""
    ops_per_sec: Optional[float] = None
    throughput_unit: str = "N/A"

    def speedup_vs(self, baseline_summary: ConfigurationSummary) -> float:
        """
        Calculates speedup ratio relative to baseline average execution time:
            speedup = baseline_avg_time / configuration_avg_time
        """
        if self.avg_time_sec > 0:
            return baseline_summary.avg_time_sec / self.avg_time_sec
        return 1.0

    def to_summary_csv_row(self, baseline_summary: Optional[ConfigurationSummary] = None) -> dict:
        """Flattens configuration aggregated statistics for CSV export."""
        speedup = self.speedup_vs(baseline_summary) if baseline_summary else 1.0
        return {
            "Configuration": self.config_name.upper(),
            "Workload": self.workload_name.upper() if self.workload_name else "N/A",
            "Thread_Count": self.thread_count,
            "Run_Count": self.run_count,
            "Mean_Time_s": f"{self.avg_time_sec:.4f}",
            "Median_P50_s": f"{self.p50_time_sec:.4f}",
            "P90_s": f"{self.p90_time_sec:.4f}",
            "P99_s": f"{self.p99_time_sec:.4f}",
            "Min_Time_s": f"{self.min_time_sec:.4f}",
            "Max_Time_s": f"{self.max_time_sec:.4f}",
            "Sample_Variance_s2": f"{self.variance_time_sec:.6f}",
            "Sample_StdDev_s": f"{self.std_dev_time_sec:.4f}",
            "CI_95_Lower_s": f"{self.ci_95_lower:.4f}",
            "CI_95_Upper_s": f"{self.ci_95_upper:.4f}",
            "CI_95_Margin_s": f"{self.ci_margin:.4f}",
            "Speedup_vs_Baseline": f"{speedup:.2f}",
            "System_Avg_CPU_pct": f"{self.avg_system_cpu:.2f}",
            "Target_Cores_Avg_CPU_pct": f"{self.avg_target_cpu:.2f}",
        }


def aggregate_runs(
    config_name: str,
    runs: List[RunMetrics],
    physical_core_count: int = 0,
    workload_name: Optional[str] = None,
    total_operations: Optional[int] = None,
    throughput_unit: Optional[str] = None,
) -> ConfigurationSummary:
    """
    Aggregates multiple RunMetrics into a single ConfigurationSummary.
    Calculates sample mean, median (P50), P90, P99 percentiles,
    sample variance (s^2, df=N-1), sample standard deviation (s, df=N-1),
    and 95% Student's t confidence interval for the mean.
    """
    if not runs:
        raise ValueError("Cannot aggregate empty list of runs.")

    times = [r.execution_time_sec for r in runs]
    sys_cpus = [r.system_avg_cpu for r in runs]
    target_cpus = [r.active_cores_avg_cpu for r in runs]

    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    avg_sys = statistics.mean(sys_cpus) if sys_cpus else 0.0
    avg_target = statistics.mean(target_cpus) if target_cpus else 0.0

    # Sample variance (s^2) and sample standard deviation (s) using Bessel's correction (N-1)
    if len(times) > 1:
        variance = statistics.variance(times)
        std_dev = statistics.stdev(times)
    else:
        variance = 0.0
        std_dev = 0.0

    p50 = calculate_percentile(times, 50.0)
    p90 = calculate_percentile(times, 90.0)
    p99 = calculate_percentile(times, 99.0)

    ci_lower, ci_upper, ci_margin, ci_warn = calculate_confidence_interval_95(
        avg_time, std_dev, len(times)
    )

    ops_rate = None
    unit_str = throughput_unit or "N/A"
    if total_operations is not None and avg_time > 0:
        ops_rate = total_operations / avg_time

    first = runs[0]
    inferred_workload = workload_name or first.workload_name
    return ConfigurationSummary(
        config_name=config_name,
        thread_count=first.thread_count,
        logical_cpu_count=first.logical_cpu_count or first.core_count,
        physical_core_count=physical_core_count or first.physical_core_count,
        run_count=len(runs),
        avg_time_sec=avg_time,
        min_time_sec=min_time,
        max_time_sec=max_time,
        std_dev_time_sec=std_dev,
        avg_system_cpu=avg_sys,
        avg_target_cpu=avg_target,
        runs=runs,
        thread_to_cores=first.thread_to_cores,
        representative_run=runs[-1],
        workload_name=inferred_workload,
        p50_time_sec=p50,
        p90_time_sec=p90,
        p99_time_sec=p99,
        variance_time_sec=variance,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        ci_margin=ci_margin,
        ci_warning=ci_warn,
        ops_per_sec=ops_rate,
        throughput_unit=unit_str,
    )


class CpuUtilizationSampler(threading.Thread):
    """
    Background sampling thread that collects periodic per-core CPU utilization
    metrics without disturbing worker threads.
    """

    def __init__(self, sample_interval_sec: float = 0.08):
        super().__init__(daemon=True, name="CPUMonitorThread")
        self.sample_interval_sec = sample_interval_sec
        self._stop_event = threading.Event()
        self.samples: List[List[float]] = []

    def run(self) -> None:
        if not PSUTIL_AVAILABLE:
            return

        psutil.cpu_percent(percpu=True)
        while not self._stop_event.is_set():
            time.sleep(self.sample_interval_sec)
            if self._stop_event.is_set():
                break
            try:
                per_cpu = psutil.cpu_percent(percpu=True)
                if per_cpu:
                    self.samples.append(per_cpu)
            except Exception:
                pass

    def stop(self) -> List[List[float]]:
        """Stops the sampling thread and returns collected samples."""
        self._stop_event.set()
        self.join(timeout=1.0)
        return self.samples


class PerformanceMonitor:
    """
    Context manager for measuring execution time and per-core CPU metrics.
    """

    def __init__(
        self,
        config_name: str,
        thread_count: int,
        available_core_ids: List[int],
        thread_to_cores: Dict[int, List[int]],
        run_number: int = 1,
        physical_core_count: int = 0,
        sample_interval_sec: float = 0.08,
        workload_name: str = "",
    ):
        self.config_name = config_name
        self.thread_count = thread_count
        self.available_core_ids = available_core_ids
        self.thread_to_cores = thread_to_cores
        self.run_number = run_number
        self.physical_core_count = physical_core_count
        self.sample_interval_sec = sample_interval_sec
        self.workload_name = workload_name
        self.metrics: Optional[RunMetrics] = None

        self._start_perf: float = 0.0
        self._end_perf: float = 0.0
        self._sampler: Optional[CpuUtilizationSampler] = None

    def start(self) -> None:
        """Starts timing and CPU utilization monitoring."""
        self.metrics = None
        self._sampler = CpuUtilizationSampler(self.sample_interval_sec)
        self._sampler.start()
        self._start_perf = time.perf_counter()

    def stop(self, thread_statuses: Optional[Dict[int, str]] = None) -> RunMetrics:
        """
        Stops monitoring, calculates metrics, and stores the completed RunMetrics.
        """
        self._end_perf = time.perf_counter()
        raw_samples = self._sampler.stop() if self._sampler else []

        duration = max(0.0001, self._end_perf - self._start_perf)
        num_cores = len(self.available_core_ids)

        avg_per_core = [0.0] * num_cores
        peak_per_core = [0.0] * num_cores

        if raw_samples:
            for core_idx in range(num_cores):
                core_vals = [s[core_idx] for s in raw_samples if core_idx < len(s)]
                if core_vals:
                    avg_per_core[core_idx] = sum(core_vals) / len(core_vals)
                    peak_per_core[core_idx] = max(core_vals)

        if any(avg_per_core):
            system_avg = sum(avg_per_core) / len(avg_per_core)
        else:
            system_avg = 0.0

        assigned_cores_set = set()
        for cores in self.thread_to_cores.values():
            assigned_cores_set.update(cores)

        if assigned_cores_set:
            active_vals = [avg_per_core[c] for c in assigned_cores_set if c < len(avg_per_core)]
            active_avg = sum(active_vals) / len(active_vals) if active_vals else system_avg
        else:
            active_avg = system_avg

        self.metrics = RunMetrics(
            config_name=self.config_name,
            thread_count=self.thread_count,
            core_count=num_cores,
            thread_to_cores=self.thread_to_cores,
            run_number=self.run_number,
            logical_cpu_count=num_cores,
            physical_core_count=self.physical_core_count,
            start_time=self._start_perf,
            end_time=self._end_perf,
            execution_time_sec=duration,
            per_core_samples=raw_samples,
            avg_per_core_cpu=avg_per_core,
            peak_per_core_cpu=peak_per_core,
            system_avg_cpu=system_avg,
            active_cores_avg_cpu=active_avg,
            timestamp_iso=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            thread_statuses=thread_statuses or {},
            workload_name=self.workload_name,
        )
        return self.metrics

    def __enter__(self) -> PerformanceMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.metrics is None:
            self.stop()


if __name__ == "__main__":
    print("Testing Module 5: Performance Measurement standalone...")
    dummy_mapping = {0: [0], 1: [1]}
    cores = [0, 1, 2, 3]

    monitor = PerformanceMonitor("test_config", 2, cores, dummy_mapping, run_number=1, physical_core_count=2)
    with monitor:
        time.sleep(0.1)
    res = monitor.metrics
    print(f"  [+] Single run: time={res.execution_time_sec:.4f}s, sys_cpu={res.system_avg_cpu:.1f}%")

    summary = aggregate_runs("test_config", [res, res], physical_core_count=2)
    print(f"  [+] Aggregated summary: avg_time={summary.avg_time_sec:.4f}s, runs={summary.run_count}")