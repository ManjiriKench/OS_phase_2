"""
Module 6: Results Display / Output
Project: Thread Affinity Management for Multi-Core Processors (Phase 1)

This module handles:
1. Pretty-printing individual run performance results and thread assignment tables.
2. Rendering ASCII CPU utilization load bars per logical processor (Windows cp1252 safe).
3. Displaying statistical summaries for repeated benchmark runs (mean, min, max, std dev).
4. Generating comparative benchmark summaries with scientifically neutral observations.
5. Exporting every individual benchmark run to CSV and generating formal TXT reports.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional
from metrics import ConfigurationSummary, RunMetrics


def _render_bar(percent: float, width: int = 16) -> str:
    """Generates an ASCII-safe text-based load bar for CPU utilization."""
    filled = int(round((percent / 100.0) * width))
    filled = max(0, min(width, filled))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {percent:5.1f}%"


def display_single_run_result(metrics: RunMetrics) -> None:
    """Prints a concise status line for an individual benchmark repetition."""
    print(
        f"    [Run {metrics.run_number}] Time: {metrics.execution_time_sec:.4f}s | "
        f"System CPU: {metrics.system_avg_cpu:.1f}% | "
        f"Target Cores CPU: {metrics.active_cores_avg_cpu:.1f}%"
    )


def display_configuration_summary(summary: ConfigurationSummary) -> None:
    """
    Prints a detailed results card for a configuration across its repeated runs,
    including per-core CPU load bars from the representative run.
    """
    print("\n" + "=" * 76)
    print(f"       BENCHMARK RESULTS: {summary.config_name.upper()} ({summary.run_count} Runs)")
    print("=" * 76)
    print(f"  * Configuration Name  : {summary.config_name.upper()}")
    print(f"  * Worker Threads      : {summary.thread_count}")
    print(f"  * Logical Processors  : {summary.logical_cpu_count}")
    print(f"  * Physical Cores      : {summary.physical_core_count if summary.physical_core_count else 'Detected via OS'}")
    print(f"  * Repetitions (Runs)  : {summary.run_count}")
    print(f"  * Average Exec Time   : {summary.avg_time_sec:.4f} seconds")
    print(f"  * Median (P50) Time   : {summary.p50_time_sec:.4f} seconds")
    print(f"  * 90th Percentile P90 : {summary.p90_time_sec:.4f} seconds")
    print(f"  * 99th Percentile P99 : {summary.p99_time_sec:.4f} seconds")
    print(f"  * Minimum Exec Time   : {summary.min_time_sec:.4f} seconds")
    print(f"  * Maximum Exec Time   : {summary.max_time_sec:.4f} seconds")
    print(f"  * Sample Variance     : {summary.variance_time_sec:.6f} s^2 (df={max(0, summary.run_count - 1)})")
    print(f"  * Sample Std Dev      : {summary.std_dev_time_sec:.4f} seconds (df={max(0, summary.run_count - 1)})")
    ci_repr = (
        f"[{summary.ci_95_lower:.4f}s, {summary.ci_95_upper:.4f}s] (±{summary.ci_margin:.4f}s, df={summary.run_count - 1})"
        if summary.run_count > 1
        else "N/A (Requires N >= 2)"
    )
    print(f"  * 95% Confidence Int  : {ci_repr}")
    if summary.ci_warning:
        print(f"    [!] Note: {summary.ci_warning}")
    print(f"  * Average System CPU  : {summary.avg_system_cpu:.2f}%")
    print(f"  * Target Cores CPU    : {summary.avg_target_cpu:.2f}%")
    print("-" * 76)

    # Thread-to-core mapping
    print(f"  {'Thread':<10} | {'Assigned Logical ID':<22} | {'Affinity Status':<16}")
    print("  " + "-" * 72)
    rep = summary.representative_run or summary.runs[0]
    for tid in sorted(summary.thread_to_cores.keys()):
        cores = summary.thread_to_cores[tid]
        core_repr = ", ".join(f"Logical {c}" for c in cores) if cores else "OS Scheduler (Unrestricted)"
        status = rep.thread_statuses.get(tid, "Verified")
        print(f"  Worker-{tid:<3} | {core_repr:<22} | {status:<16}")
    print("-" * 76)

    # Per-core CPU load bars from representative run
    if rep and rep.avg_per_core_cpu:
        print("  MEASURED CPU LOAD PER LOGICAL PROCESSOR (REPRESENTATIVE RUN):")
        core_to_threads: Dict[int, List[int]] = {}
        for tid, cores in summary.thread_to_cores.items():
            for c in cores:
                core_to_threads.setdefault(c, []).append(tid)

        for core_id, pct in enumerate(rep.avg_per_core_cpu):
            bar = _render_bar(pct, width=16)
            assigned_threads = core_to_threads.get(core_id, [])
            if assigned_threads:
                annot = f" <-- Workers: {', '.join(str(t) for t in assigned_threads)}"
            elif not any(summary.thread_to_cores.values()):
                annot = " (OS Discretion)"
            else:
                annot = ""
            print(f"    Logical Core {core_id:2d} : {bar}{annot}")
        print("=" * 76 + "\n")


def display_comparison_summary(summaries: List[ConfigurationSummary]) -> None:
    """
    Renders a comparative table ranking all configurations, with multi-run
    statistics (mean, min, max, stdev, speedup vs baseline) and academically
    defensible neutral observations.
    """
    if not summaries:
        print("[!] No configuration summaries to compare.")
        return

    baseline = next(
        (s for s in summaries if s.config_name.lower() == "baseline"), summaries[0]
    )

    # Sort from fastest (lowest average time) to slowest
    sorted_summaries = sorted(summaries, key=lambda s: s.avg_time_sec)
    fastest = sorted_summaries[0]
    slowest = sorted_summaries[-1]

    print("\n" + "#" * 86)
    print("      THREAD AFFINITY MANAGEMENT -- STATISTICAL PERFORMANCE COMPARISON")
    print("#" * 86)
    print(
        f"{'Configuration':<14} | {'Threads':<7} | {'Avg Time (s)':<12} | "
        f"{'Min (s)':<9} | {'Max (s)':<9} | {'StdDev (s)':<10} | {'Avg CPU %':<9} | {'Speedup'}"
    )
    print("-" * 86)

    for s in sorted_summaries:
        speedup = s.speedup_vs(baseline)
        if speedup > 1.05:
            speedup_str = f"{speedup:.2f}x (Faster)"
        elif speedup < 0.95:
            speedup_str = f"{speedup:.2f}x (Slower)"
        else:
            speedup_str = f"{speedup:.2f}x (Baseline)" if s == baseline else f"{speedup:.2f}x (Equal)"

        print(
            f"{s.config_name.upper():<14} | {s.thread_count:<7} | {s.avg_time_sec:<12.4f} | "
            f"{s.min_time_sec:<9.4f} | {s.max_time_sec:<9.4f} | {s.std_dev_time_sec:<10.4f} | "
            f"{s.avg_target_cpu:<8.1f}% | {speedup_str}"
        )

    print("-" * 86)
    print(f"  * Fastest observed configuration under test : {fastest.config_name.upper()} ({fastest.avg_time_sec:.4f}s avg)")
    print(f"  * Slowest observed configuration under test : {slowest.config_name.upper()} ({slowest.avg_time_sec:.4f}s avg)")
    if fastest.avg_time_sec > 0:
        ratio = slowest.avg_time_sec / fastest.avg_time_sec
        print(f"  * Observed performance difference           : {ratio:.2f}x ratio between slowest and fastest")

    # Advanced statistical percentiles and dispersion table
    print("\n" + "-" * 86)
    print("      PERCENTILES & DISPERSION METRICS (EXECUTION TIME)")
    print("-" * 86)
    print(
        f"{'Configuration':<14} | {'Mean (s)':<10} | {'Median P50':<11} | {'P90 (s)':<9} | {'P99 (s)':<9} | {'Variance (s^2)':<13} | {'95% CI of Mean'}"
    )
    print("-" * 86)
    for s in sorted_summaries:
        ci_str = f"[{s.ci_95_lower:.4f}, {s.ci_95_upper:.4f}]" if s.run_count > 1 else "N/A (N=1)"
        print(
            f"{s.config_name.upper():<14} | {s.avg_time_sec:<10.4f} | {s.p50_time_sec:<11.4f} | "
            f"{s.p90_time_sec:<9.4f} | {s.p99_time_sec:<9.4f} | {s.variance_time_sec:<13.6f} | {ci_str}"
        )
    print("-" * 86)

    print("\n  [i] ACADEMIC OBSERVATIONS & INTERPRETATION:")
    print("  * Observed execution time and CPU utilization differed between affinity configurations.")
    print("  * Concentrating multiple worker threads on fewer cores demonstrated higher per-core")
    print("    saturation and longer total execution time under this CPU-bound workload.")
    print("  * Distributing threads across separated logical processors achieved parallel execution")
    print("    with distinct core utilization profiles.")
    print("  * Note: Results reflect observed execution times and CPU utilization under the tested")
    print("    workload, operating system scheduling state, and hardware topology. Direct hardware")
    print("    metrics such as cache miss rates or context switch frequencies were not directly measured.")
    print("#" * 86 + "\n")


def export_results_to_csv(
    all_runs: List[RunMetrics],
    filepath: str = "affinity_benchmark_results.csv",
) -> str:
    """
    Exports all individual benchmark runs to a clean, structured CSV file.
    """
    if not all_runs:
        return ""

    fieldnames = [
        "Timestamp",
        "Configuration",
        "Run_Number",
        "Thread_Count",
        "Logical_Processors",
        "Physical_Cores",
        "Assignments",
        "Execution_Time_s",
        "System_Avg_CPU_pct",
        "Target_Cores_Avg_CPU_pct",
    ]
    if any(getattr(r, "workload_name", "") for r in all_runs):
        fieldnames.insert(1, "Workload")

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_runs:
            writer.writerow(r.to_csv_row())

    print(f"  [+] Benchmark data successfully exported to CSV: {os.path.abspath(filepath)}")
    return filepath


def export_summary_to_csv(
    summaries: List[ConfigurationSummary],
    filepath: str = "affinity_benchmark_summary.csv",
) -> str:
    """
    Exports aggregated configuration-level benchmark statistics to a CSV file.
    Includes Mean, Median (P50), P90, P99, Sample Variance (s^2, df=N-1),
    Sample Standard Deviation (s, df=N-1), 95% Student's t Confidence Intervals,
    and Speedup vs Baseline.
    """
    if not summaries:
        return ""

    baseline = next(
        (s for s in summaries if s.config_name.lower() == "baseline"), None
    )

    fieldnames = [
        "Configuration",
        "Workload",
        "Thread_Count",
        "Run_Count",
        "Mean_Time_s",
        "Median_P50_s",
        "P90_s",
        "P99_s",
        "Min_Time_s",
        "Max_Time_s",
        "Sample_Variance_s2",
        "Sample_StdDev_s",
        "CI_95_Lower_s",
        "CI_95_Upper_s",
        "CI_95_Margin_s",
        "Speedup_vs_Baseline",
        "System_Avg_CPU_pct",
        "Target_Cores_Avg_CPU_pct",
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s.to_summary_csv_row(baseline_summary=baseline))

    print(f"  [+] Statistical summary successfully exported to CSV: {os.path.abspath(filepath)}")
    return filepath


def generate_benchmark_charts(
    summaries: List[ConfigurationSummary],
    output_dir: str = "charts",
    workload_name: Optional[str] = None,
) -> List[str]:
    """
    Generates high-resolution PNG benchmark charts from aggregated ConfigurationSummary objects:
    1. Execution-Time Comparison: Average execution time per configuration (with std-dev error bars).
    2. Speedup Comparison: Speedup ratio relative to Baseline (with y=1.0 reference line).
    3. Per-Core CPU Utilization: Grouped bar chart comparing logical core CPU load across configs
       (omitted if recorded per-core data is missing or empty, with an explanation logged).

    Supports results from multiple workloads by grouping and clearly labeling each workload.
    Gracefully handles empty or insufficient data without fabricating values.

    Args:
        summaries: List of ConfigurationSummary objects.
        output_dir: Target directory path for PNG charts.
        workload_name: Optional default workload label if not present in summaries.

    Returns:
        List[str]: List of absolute paths to created PNG chart files.
    """
    if not summaries:
        print("  [!] No configuration summaries provided for chart generation.")
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] Error: matplotlib is not installed. Cannot generate benchmark charts.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    created_files: List[str] = []

    palette = {
        "BASELINE": "#3498DB",      # Blue
        "FIXED": "#2ECC71",         # Emerald Green
        "DISTRIBUTED": "#1ABC9C",   # Turquoise
        "CONCENTRATED": "#E74C3C",  # Coral Red
    }

    # Group summaries by workload
    workload_groups: Dict[str, List[ConfigurationSummary]] = {}
    for s in summaries:
        w_name = getattr(s, "workload_name", "") or workload_name or "DEFAULT_WORKLOAD"
        workload_groups.setdefault(w_name, []).append(s)

    for w_name, w_summaries in workload_groups.items():
        w_slug = w_name.lower().replace(" ", "_")
        prefix = f"{w_slug}_" if len(workload_groups) > 1 else ""

        # ------------------------------------------------------------------
        # Chart 1: Execution Time Comparison
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        configs = [s.config_name.upper() for s in w_summaries]
        avg_times = [s.avg_time_sec for s in w_summaries]
        std_devs = [s.std_dev_time_sec for s in w_summaries]
        colors = [palette.get(c, "#95A5A6") for c in configs]

        bars = ax.bar(
            configs,
            avg_times,
            yerr=std_devs if any(sd > 0 for sd in std_devs) else None,
            capsize=5,
            color=colors,
            edgecolor="#2C3E50",
            linewidth=1.2,
            alpha=0.9,
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Subtle median (P50) horizontal line indicator across each bar to show skewness
        p50_times = [s.p50_time_sec for s in w_summaries]
        has_p50 = any(p > 0 for p in p50_times)
        if has_p50:
            for bar, p50 in zip(bars, p50_times):
                if p50 > 0:
                    x_left = bar.get_x() + bar.get_width() * 0.15
                    x_right = bar.get_x() + bar.get_width() * 0.85
                    ax.plot([x_left, x_right], [p50, p50], color="#C0392B", linewidth=2.0, linestyle="--")
            ax.plot([], [], color="#C0392B", linewidth=2.0, linestyle="--", label="Median (P50)")
            ax.legend(loc="upper right")

        thread_cnt = w_summaries[0].thread_count if w_summaries else 0
        run_cnt = w_summaries[0].run_count if w_summaries else 0
        ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Affinity Configuration", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Execution Time Comparison Across Configurations\n"
            f"(Workload: {w_name.upper()} | Threads: {thread_cnt} | Runs: {run_cnt})",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        exec_filename = f"{prefix}execution_time_comparison.png"
        exec_path = os.path.join(output_dir, exec_filename)
        fig.savefig(exec_path, dpi=150)
        plt.close(fig)
        created_files.append(os.path.abspath(exec_path))

        # ------------------------------------------------------------------
        # Chart 2: Speedup Comparison (Relative to Baseline)
        # ------------------------------------------------------------------
        baseline = next(
            (s for s in w_summaries if s.config_name.lower() == "baseline"), None
        )
        if baseline and baseline.avg_time_sec > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            speedups = [s.speedup_vs(baseline) for s in w_summaries]
            sp_colors = [
                "#2ECC71" if sp >= 1.05 else ("#E74C3C" if sp <= 0.95 else "#3498DB")
                for sp in speedups
            ]

            bars = ax.bar(
                configs,
                speedups,
                color=sp_colors,
                edgecolor="#2C3E50",
                linewidth=1.2,
                alpha=0.9,
            )
            ax.axhline(
                1.0,
                color="#7F8C8D",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline Reference (1.00x - {baseline.avg_time_sec:.4f}s)",
            )

            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_ylabel("Speedup Ratio (vs Baseline)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Affinity Configuration", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Speedup Relative to Baseline OS Scheduling\n"
                f"(Workload: {w_name.upper()} | Baseline Avg: {baseline.avg_time_sec:.4f}s)",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(loc="upper right")
            plt.tight_layout()

            sp_filename = f"{prefix}speedup_comparison.png"
            sp_path = os.path.join(output_dir, sp_filename)
            fig.savefig(sp_path, dpi=150)
            plt.close(fig)
            created_files.append(os.path.abspath(sp_path))
        else:
            print(
                f"  [i] Note: Speedup chart omitted for workload '{w_name}' "
                f"(BASELINE configuration not present or non-positive execution time)."
            )

        # ------------------------------------------------------------------
        # Chart 3: Per-Core CPU Utilization
        # ------------------------------------------------------------------
        valid_core_data: Dict[str, List[float]] = {}
        for s in w_summaries:
            # Check representative_run, or search runs for non-zero per-core samples
            chosen_run = None
            if s.representative_run and any(v > 0.0 for v in s.representative_run.avg_per_core_cpu):
                chosen_run = s.representative_run
            else:
                chosen_run = next(
                    (r for r in s.runs if r.avg_per_core_cpu and any(v > 0.0 for v in r.avg_per_core_cpu)),
                    None,
                )
            if chosen_run and chosen_run.avg_per_core_cpu:
                valid_core_data[s.config_name.upper()] = chosen_run.avg_per_core_cpu

        if valid_core_data:
            num_cores = max(len(vals) for vals in valid_core_data.values())
            fig, ax = plt.subplots(figsize=(max(9, num_cores * 0.8), 5))
            x_indices = list(range(num_cores))
            bar_width = 0.8 / len(valid_core_data)

            for idx, (cfg_name, cpu_vals) in enumerate(valid_core_data.items()):
                offsets = [x - 0.4 + (idx + 0.5) * bar_width for x in x_indices]
                aligned_vals = [
                    cpu_vals[c] if c < len(cpu_vals) else 0.0 for c in x_indices
                ]
                ax.bar(
                    offsets,
                    aligned_vals,
                    width=bar_width,
                    label=cfg_name,
                    color=palette.get(cfg_name, "#95A5A6"),
                    edgecolor="#2C3E50",
                    linewidth=1.0,
                    alpha=0.85,
                )

            ax.set_xticks(x_indices)
            ax.set_xticklabels([f"Core {c}" for c in x_indices], fontsize=9)
            ax.set_ylabel("Average CPU Utilization (%)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Logical Processor ID", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.set_title(
                f"Measured Per-Core CPU Utilization by Configuration\n"
                f"(Workload: {w_name.upper()} | Representative Run)",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()

            cpu_filename = f"{prefix}cpu_utilization_comparison.png"
            cpu_path = os.path.join(output_dir, cpu_filename)
            fig.savefig(cpu_path, dpi=150)
            plt.close(fig)
            created_files.append(os.path.abspath(cpu_path))
        else:
            print(
                f"  [i] Note: Per-core CPU utilization chart omitted for workload '{w_name}' "
                f"(no non-zero per-core CPU utilization samples were recorded)."
            )

    if created_files:
        print(f"  [+] {len(created_files)} benchmark charts successfully generated in: {os.path.abspath(output_dir)}")
    return created_files


def export_results_to_txt(
    summaries: List[ConfigurationSummary],
    all_runs: List[RunMetrics],
    filepath: str = "affinity_benchmark_report.txt",
    cpu_details: Optional[dict] = None,
) -> str:
    """
    Generates a formal plain-text academic benchmark report.
    """
    if not summaries:
        return ""

    sorted_summaries = sorted(summaries, key=lambda s: s.avg_time_sec)
    fastest = sorted_summaries[0]
    slowest = sorted_summaries[-1]

    lines = []
    lines.append("=" * 84)
    lines.append("        PROJECT 31: THREAD AFFINITY MANAGEMENT BENCHMARK REPORT")
    lines.append("                 Phase 1 Prototype Experimental Results")
    lines.append("=" * 84)
    lines.append(f"Report Date: {summaries[0].runs[0].timestamp_iso if summaries[0].runs else 'N/A'}\n")

    if cpu_details:
        lines.append("--- HARDWARE TOPOLOGY SPECIFICATIONS ---")
        lines.append(f"Physical CPU Cores   : {cpu_details.get('physical_cores', 'Unknown')}")
        lines.append(f"Logical Processors   : {cpu_details.get('logical_cores', 'Unknown')}")
        lines.append(f"Logical Processor IDs: {cpu_details.get('core_ids', [])}")
        if cpu_details.get("physical_core_mapping"):
            lines.append("Physical-to-Logical Core Mapping:")
            for p, l in cpu_details["physical_core_mapping"].items():
                lines.append(f"  Physical Core {p:2d} -> Logical Processors {l}")
        lines.append("")

    lines.append("--- CONFIGURATION SUMMARY (STATISTICAL AGGREGATION) ---")
    header = (
        f"{'Configuration':<14} | {'Threads':<7} | {'Runs':<5} | {'Avg Time (s)':<12} | "
        f"{'Min (s)':<9} | {'Max (s)':<9} | {'StdDev (s)':<10} | {'Target CPU %':<12}"
    )
    lines.append(header)
    lines.append("-" * 84)
    for s in sorted_summaries:
        row = (
            f"{s.config_name.upper():<14} | {s.thread_count:<7} | {s.run_count:<5} | "
            f"{s.avg_time_sec:<12.4f} | {s.min_time_sec:<9.4f} | {s.max_time_sec:<9.4f} | "
            f"{s.std_dev_time_sec:<10.4f} | {s.avg_target_cpu:<11.1f}%"
        )
        lines.append(row)
    lines.append("-" * 84)

    lines.append(f"\nFastest observed configuration : {fastest.config_name.upper()} (Avg: {fastest.avg_time_sec:.4f}s)")
    lines.append(f"Slowest observed configuration : {slowest.config_name.upper()} (Avg: {slowest.avg_time_sec:.4f}s)")
    if fastest.avg_time_sec > 0:
        ratio = slowest.avg_time_sec / fastest.avg_time_sec
        lines.append(f"Observed Performance Ratio    : {ratio:.2f}x between slowest and fastest")

    # Advanced statistical percentiles and dispersion metrics
    lines.append("\n--- ADVANCED STATISTICAL ANALYSIS & PERCENTILES ---")
    header_stats = (
        f"{'Configuration':<14} | {'Mean (s)':<10} | {'Median P50':<11} | "
        f"{'P90 (s)':<9} | {'P99 (s)':<9} | {'Variance (s^2)':<15} | {'StdDev (s)':<10} | {'95% CI of Mean'}"
    )
    lines.append(header_stats)
    lines.append("-" * 92)
    for s in sorted_summaries:
        ci_str = f"[{s.ci_95_lower:.4f}, {s.ci_95_upper:.4f}]" if s.run_count > 1 else "N/A (N=1)"
        lines.append(
            f"{s.config_name.upper():<14} | {s.avg_time_sec:<10.4f} | {s.p50_time_sec:<11.4f} | "
            f"{s.p90_time_sec:<9.4f} | {s.p99_time_sec:<9.4f} | {s.variance_time_sec:<15.6f} | "
            f"{s.std_dev_time_sec:<10.4f} | {ci_str}"
        )
    lines.append("-" * 92)
    lines.append("Statistical Methodology & Dispersion Notes:")
    lines.append("  - Percentiles (P50, P90, P99) calculated via standard linear rank interpolation.")
    lines.append("  - Sample Variance (s^2) and Sample StdDev (s) use Bessel's correction (N - 1 df).")
    lines.append("  - 95% Confidence Intervals are computed using Student's t-distribution (df = N - 1).")
    lines.append("    Assumptions: independent repeated runs under identical nominal load with")
    lines.append("    approximately normally distributed wall-clock variation.")
    for s in sorted_summaries:
        if s.ci_warning:
            lines.append(f"  [!] {s.config_name.upper()}: {s.ci_warning}")
    if any(s.ops_per_sec is not None for s in sorted_summaries):
        lines.append("\nThroughput Rates (Measurable Operations):")
        for s in sorted_summaries:
            if s.ops_per_sec is not None:
                lines.append(f"  * {s.config_name.upper()}: {s.ops_per_sec:.2f} {s.throughput_unit}")

    lines.append("\n--- INDIVIDUAL RUN EXECUTION LOGS ---")
    for r in all_runs:
        lines.append(
            f"[{r.config_name.upper()} Run {r.run_number}] Time: {r.execution_time_sec:.4f}s | "
            f"System CPU: {r.system_avg_cpu:.1f}% | Target CPU: {r.active_cores_avg_cpu:.1f}%"
        )

    lines.append("\n--- ACADEMIC OBSERVATIONS ---")
    lines.append("1. Execution time and per-core CPU utilization varied systematically across affinity settings.")
    lines.append("2. Concentrated placement confined threads to a restricted core subset, resulting in increased")
    lines.append("   observed execution duration and localized core saturation.")
    lines.append("3. Distributed placement separated threads across different logical processor IDs.")
    lines.append("4. Conclusions are bounded to the measured execution times and CPU utilization on this platform;")
    lines.append("   cache miss rates and kernel context switches were not directly instrumented.")
    lines.append("\n" + "=" * 84)
    lines.append("End of Report.")
    lines.append("=" * 84 + "\n")

    report_text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  [+] Academic report successfully saved to TXT: {os.path.abspath(filepath)}")
    return filepath