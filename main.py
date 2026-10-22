"""
Main Execution Script (Phase 1 — Fully Integrated Modules 1 to 6)
Project: Thread Affinity Management for Multi-Core Processors

This script integrates all 6 project modules:
- Module 1: CPU Core Detection (cpu_detection.py)
- Module 2: Worker Thread Creation (thread_manager.py)
- Module 3: Workload Engine (workload.py)
- Module 4: Thread Affinity Management (affinity_manager.py)
- Module 5: Performance Measurement (metrics.py)
- Module 6: Results Display & Output (display.py)

Usage:
    python main.py [num_threads] [--affinity CONFIG] [--runs N] [--workload TYPE]
    [--intensity INT] [--csv FILE] [--report FILE]

Options:
    num_threads          Number of worker threads (default: 4)
    --affinity CONFIG    all (default), baseline, fixed, distributed, concentrated
    --runs N             Number of repeated benchmark runs per config (default: 3)
    --workload TYPE      crypto_hash (default), matrix_mult, prime_sieve
    --intensity INT      Workload intensity (iterations / matrix size)
    --csv FILE           CSV output path (default: affinity_benchmark_results.csv)
    --report FILE        Text report path (default: affinity_benchmark_report.txt)

Examples:
    python main.py                     # Runs 3 repeated benchmarks across all 4 configs
    python main.py 4 --runs 5          # Runs 5 repeated benchmarks per config
    python main.py 4 --affinity fixed  # Runs only the FIXED configuration
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import Dict, List, Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("project31")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    return logging.getLogger(name)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for thread count, configuration, runs, and workload."""
    parser = argparse.ArgumentParser(
        description="Project 31: Thread Affinity Management for Multi-Core Processors (Phase 1)"
    )
    parser.add_argument(
        "num_threads",
        nargs="?",
        type=int,
        default=4,
        help="Number of worker threads to spawn (default: 4)",
    )
    parser.add_argument(
        "--affinity",
        type=str,
        choices=["all", "baseline", "fixed", "distributed", "concentrated"],
        default="all",
        help="Affinity configuration to run (default: all - complete benchmark suite)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark repetitions per configuration (default: 3)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["crypto_hash", "matrix_mult", "prime_sieve"],
        default="crypto_hash",
        help="Deterministic CPU workload (default: crypto_hash)",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        default=None,
        help="Override workload intensity (iterations / size / limit)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="affinity_benchmark_results.csv",
        help="Output CSV file path (default: affinity_benchmark_results.csv)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="affinity_benchmark_report.txt",
        help="Output report text file path (default: affinity_benchmark_report.txt)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional directory to save benchmark visualization charts (e.g. --plot-dir charts/)",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Run explainable rule-based affinity strategy recommendation",
    )
    parser.add_argument(
        "--validate-strategy",
        action="store_true",
        help="Validate the recommended strategy across distinct workloads (e.g. crypto_hash vs matrix_mult)",
    )
    parser.add_argument(
        "--verify-affinity",
        action="store_true",
        help="Perform pre-flight validation of affinity mappings and bitmasks across all configurations",
    )
    return parser.parse_args()


from cpu_detection import detect_cpu_cores, display_cpu_info
from thread_manager import ThreadManager
from affinity_manager import (
    AffinityConfiguration,
    AffinityError,
    apply_thread_affinity,
    build_affinity_mask,
    display_affinity_mapping,
    generate_thread_affinity_mapping,
    is_windows,
    verify_thread_affinity,
    validate_affinity_mask,
    verify_distinct_mappings,
    validate_all_affinity_configurations,
)
from workload import get_workload_function, WorkloadType
from metrics import (
    PerformanceMonitor,
    RunMetrics,
    ConfigurationSummary,
    aggregate_runs,
)
from display import (
    display_single_run_result,
    display_configuration_summary,
    display_comparison_summary,
    export_results_to_csv,
    export_summary_to_csv,
    export_results_to_txt,
    generate_benchmark_charts,
)
from strategy import (
    recommend_affinity_strategy,
    display_strategy_recommendation,
    evaluate_validation_results,
    display_validation_report,
)


def create_affinity_worker(
    affinity_mapping: Dict[int, List[int]],
    available_core_ids: List[int],
    barrier: threading.Barrier,
    shared_results: Dict[int, dict],
    workload_func,
):
    """
    Constructs a worker callable that binds thread affinity, validates the mask,
    synchronizes at a barrier, and then runs the compute workload.
    """

    def worker(thread_index: int) -> None:
        thread_name = f"Worker-{thread_index}"
        cores = affinity_mapping.get(thread_index, [])

        # Step 1: Apply OS Thread Affinity (Windows SetThreadAffinityMask)
        try:
            if cores:
                set_result = apply_thread_affinity(
                    cores, available_core_ids, thread_name
                )
                verified = verify_thread_affinity(
                    cores,
                    available_core_ids,
                    thread_name,
                    set_return_value=set_result,
                )
                if not verified:
                    raise AffinityError(f"Affinity verification failed for {thread_name}")
                shared_results[thread_index] = {
                    "cores": cores,
                    "status": "Verified",
                }
            else:
                shared_results[thread_index] = {
                    "cores": [],
                    "status": "OS Scheduled",
                }
        except AffinityError as exc:
            print(f"  [!] {thread_name} affinity error: {exc}")
            shared_results[thread_index] = {
                "cores": cores,
                "status": "Failed",
                "error": str(exc),
            }
            return

        # Step 2: Synchronization Barrier — wait until all workers applied affinity
        try:
            barrier.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            return

        # Step 3: Abort if any worker experienced affinity failure
        failed = [r for r in shared_results.values() if r.get("status") == "Failed"]
        if failed:
            return

        # Step 4: Execute deterministic CPU Workload
        workload_func(thread_index)

    return worker


def run_single_iteration(
    config: AffinityConfiguration,
    num_threads: int,
    available_core_ids: List[int],
    affinity_mapping: Dict[int, List[int]],
    workload_func,
    run_number: int,
    physical_core_count: int,
    workload_name: str = "",
) -> RunMetrics:
    """Executes a single benchmark iteration of the given configuration."""
    manager = ThreadManager(num_threads=num_threads)
    shared_results: Dict[int, dict] = {}
    barrier = threading.Barrier(num_threads)

    worker_task = create_affinity_worker(
        affinity_mapping=affinity_mapping,
        available_core_ids=available_core_ids,
        barrier=barrier,
        shared_results=shared_results,
        workload_func=workload_func,
    )

    per_thread_args = [(i,) for i in range(num_threads)]
    manager.create_threads(
        target_func=worker_task,
        per_thread_args=per_thread_args,
    )

    for info in manager.thread_info:
        idx = info["index"]
        info["assigned_cores"] = affinity_mapping.get(idx, [])
        info["affinity_status"] = "Pending"

    monitor = PerformanceMonitor(
        config_name=config.value,
        thread_count=num_threads,
        available_core_ids=available_core_ids,
        thread_to_cores=affinity_mapping,
        run_number=run_number,
        physical_core_count=physical_core_count,
        workload_name=workload_name,
    )

    monitor.start()
    manager.start_threads()
    manager.join_threads()

    thread_statuses = {
        idx: shared_results.get(idx, {}).get("status", "Completed")
        for idx in range(num_threads)
    }
    metrics = monitor.stop(thread_statuses=thread_statuses)
    return metrics


def run_configuration_benchmark(
    config: AffinityConfiguration,
    num_threads: int,
    available_core_ids: List[int],
    workload_func,
    num_runs: int = 3,
    physical_core_count: int = 0,
    workload_name: str = "",
    intensity: Optional[int] = None,
) -> ConfigurationSummary:
    """
    Executes multiple repeated runs for a single configuration, displays progress,
    and returns the aggregated statistical summary.
    """
    print(f"\n>>> EXECUTING CONFIGURATION: {config.value.upper()} ({num_runs} Runs) <<<")

    # Generate thread-to-core mapping
    try:
        affinity_mapping = generate_thread_affinity_mapping(
            config, num_threads, available_core_ids
        )
    except ValueError as exc:
        print(f"[!] Configuration error for {config.value.upper()}: {exc}")
        raise

    display_affinity_mapping(config, affinity_mapping)

    runs: List[RunMetrics] = []
    for r in range(1, num_runs + 1):
        metrics = run_single_iteration(
            config=config,
            num_threads=num_threads,
            available_core_ids=available_core_ids,
            affinity_mapping=affinity_mapping,
            workload_func=workload_func,
            run_number=r,
            physical_core_count=physical_core_count,
            workload_name=workload_name,
        )
        runs.append(metrics)
        display_single_run_result(metrics)
        if r < num_runs:
            time.sleep(0.2)  # Cooldown between runs

    # Consistent throughput operations for crypto_hash (hashes = threads * iterations)
    total_ops = None
    unit_str = "N/A"
    if workload_name == "crypto_hash" and intensity is not None:
        total_ops = num_threads * intensity
        unit_str = "hashes/s"

    summary = aggregate_runs(
        config_name=config.value,
        runs=runs,
        physical_core_count=physical_core_count,
        workload_name=workload_name,
        total_operations=total_ops,
        throughput_unit=unit_str,
    )
    display_configuration_summary(summary)
    return summary


def _print_header() -> None:
    """Print the project header using the configured logger."""
    logger.info("\n" + "#" * 80)
    logger.info("      PROJECT 31: THREAD AFFINITY MANAGEMENT FOR MULTI-CORE PROCESSORS")
    logger.info("                 Phase 1 Working Prototype (Modules 1 - 6)")
    logger.info("#" * 80 + "\n")


def _print_platform_warning() -> None:
    """Print platform warning if not on Windows."""
    if not is_windows():
        logger.warning(
            "Non-Windows operating system detected. "
            "SetThreadAffinityMask requires Windows kernel32. "
            "Runs will execute under default OS scheduling."
        )


def main() -> None:
    args = parse_arguments()

    _print_header()

    # ==========================================
    # STEP 1: MODULE 1 - CPU Hardware Topology
    # ==========================================
    cpu_info = detect_cpu_cores()
    display_cpu_info(cpu_info)
    available_core_ids = cpu_info["core_ids"]
    physical_count = cpu_info.get("physical_cores", 0) or 0

    num_threads = args.num_threads
    if num_threads < 1:
        logger.error(f"Invalid thread count '{num_threads}'. Defaulting to 4.")
        num_threads = 4

    num_runs = args.runs
    if num_runs < 1:
        logger.error(f"Invalid runs '{num_runs}'. Defaulting to 3.")
        num_runs = 3

    # Validate platform
    _print_platform_warning()

    # ==========================================
    # STEP 2: MODULE 4 PRE-FLIGHT AFFINITY VALIDATION
    # ==========================================
    logger.info("Performing pre-flight affinity configuration and mask validation...")
    try:
        validated_mappings = validate_all_affinity_configurations(
            thread_count=num_threads,
            available_core_ids=available_core_ids,
        )
        logger.info(
            f"[+] Pre-flight affinity validation PASSED for {len(validated_mappings)} configurations. "
            "FIXED and DISTRIBUTED mappings verified distinct."
        )
    except Exception as exc:
        logger.error(f"[!] Pre-flight affinity validation FAILED: {exc}")
        if args.verify_affinity:
            sys.exit(1)

    if args.verify_affinity:
        print("\n" + "=" * 70)
        print("          PRE-FLIGHT AFFINITY VALIDATION REPORT")
        print("=" * 70)
        for cfg_name, mapping in validated_mappings.items():
            print(f"  Configuration: {cfg_name.upper()}")
            for t_idx, cores in mapping.items():
                mask = build_affinity_mask(cores)
                core_label = ", ".join(str(c) for c in cores) if cores else "OS Scheduler"
                print(f"    Thread {t_idx} -> Cores [{core_label}] (Mask: {mask:#x})")
        print("=" * 70 + "\n")
        print("[+] All affinity masks and distinct thread mappings verified successfully.\n")
        return

    # ==========================================
    # STEP 3: MODULE 3 - Workload Engine Setup
    # ==========================================
    workload_type = WorkloadType(args.workload)
    workload_func = get_workload_function(workload_type, intensity=args.intensity)
    logger.info(f"Workload Engine : {workload_type.value.upper()} (Deterministic CPU compute)")

    # ==========================================
    # STEP 4: STRATEGY RECOMMENDER (PERSON 3)
    # ==========================================
    recommendation = recommend_affinity_strategy(
        cpu_info=cpu_info,
        thread_count=num_threads,
        workload_type=workload_type.value,
    )
    display_strategy_recommendation(recommendation)

    if args.recommend and not args.validate_strategy:
        # Exit early if user explicitly requested the recommendation
        return

    # ==========================================
    # STEP 5: SELECT CONFIGURATIONS TO EXECUTE
    # ==========================================
    if args.affinity == "all":
        configs_to_run = [
            AffinityConfiguration.BASELINE,
            AffinityConfiguration.FIXED,
            AffinityConfiguration.DISTRIBUTED,
            AffinityConfiguration.CONCENTRATED,
        ]
        if num_threads > len(available_core_ids):
            logger.info(
                f"Skipping FIXED config (threads {num_threads} > cores {len(available_core_ids)})."
            )
            configs_to_run.remove(AffinityConfiguration.FIXED)
    else:
        configs_to_run = [AffinityConfiguration(args.affinity)]

    logger.info(f"Configurations  : {[c.value.upper() for c in configs_to_run]}")
    logger.info(f"Worker Threads  : {num_threads}")
    logger.info(f"Repetitions/Cfg : {num_runs} runs per configuration")
    logger.info(f"Logical IDs     : {len(available_core_ids)} available\n")

    # ==========================================
    # STEP 6: EXECUTION PIPELINE (MODULES 4, 5, 6)
    # ==========================================
    all_summaries: List[ConfigurationSummary] = []
    all_individual_runs: List[RunMetrics] = []

    for config in configs_to_run:
        try:
            summary = run_configuration_benchmark(
                config=config,
                num_threads=num_threads,
                available_core_ids=available_core_ids,
                workload_func=workload_func,
                num_runs=num_runs,
                physical_core_count=physical_count,
                workload_name=workload_type.value,
                intensity=args.intensity,
            )
            all_summaries.append(summary)
            all_individual_runs.extend(summary.runs)
            time.sleep(0.3)  # Cooldown between configs
        except Exception as exc:
            logger.error(f"Execution failed for {config.value}: {exc}", exc_info=True)

    # ==========================================
    # STEP 7: STRATEGY VALIDATION (CROSS-WORKLOAD)
    # ==========================================
    val_report = None
    if args.validate_strategy and len(all_summaries) > 1:
        # Determine secondary validation workload
        val_workload_type = (
            WorkloadType.MATRIX_MULT
            if workload_type == WorkloadType.CRYPTO_HASH
            else WorkloadType.CRYPTO_HASH
        )
        logger.info(f"\n>>> EXECUTING STRATEGY VALIDATION BENCHMARK ON: {val_workload_type.value.upper()} <<<")
        val_workload_func = get_workload_function(val_workload_type, intensity=None)
        val_summaries: List[ConfigurationSummary] = []

        for config in configs_to_run:
            try:
                v_summary = run_configuration_benchmark(
                    config=config,
                    num_threads=num_threads,
                    available_core_ids=available_core_ids,
                    workload_func=val_workload_func,
                    num_runs=max(2, num_runs - 1),
                    physical_core_count=physical_count,
                    workload_name=val_workload_type.value,
                    intensity=None,
                )
                val_summaries.append(v_summary)
                all_individual_runs.extend(v_summary.runs)
                time.sleep(0.3)
            except Exception as exc:
                logger.error(f"Validation run failed for {config.value}: {exc}", exc_info=True)

        if len(val_summaries) > 1:
            val_report = evaluate_validation_results(
                recommended_config=recommendation.recommended_configuration,
                calibration_workload=workload_type.value,
                validation_workload=val_workload_type.value,
                summaries=val_summaries,
            )
            display_validation_report(val_report)
    elif len(all_summaries) > 1:
        # Evaluate recommendation directly on current run
        val_report = evaluate_validation_results(
            recommended_config=recommendation.recommended_configuration,
            calibration_workload=workload_type.value,
            validation_workload=workload_type.value,
            summaries=all_summaries,
        )
        display_validation_report(val_report)

    # ==========================================
    # STEP 8: COMPARATIVE SUMMARY & EXPORT (MODULE 6)
    # ==========================================
    if len(all_summaries) > 1:
        display_comparison_summary(all_summaries)

    if all_individual_runs:
        export_results_to_csv(all_individual_runs, filepath=args.csv)
        summary_csv_path = (
            args.csv.replace(".csv", "_summary.csv")
            if args.csv.endswith(".csv")
            else f"{args.csv}_summary.csv"
        )
        export_summary_to_csv(all_summaries, filepath=summary_csv_path)
        export_results_to_txt(
            all_summaries,
            all_individual_runs,
            filepath=args.report,
            cpu_details=cpu_info,
        )

        # Append strategy recommendation and validation findings to report
        if val_report:
            try:
                with open(args.report, "a", encoding="utf-8") as rf:
                    rf.write("\n" + "=" * 84 + "\n")
                    rf.write("--- AFFINITY STRATEGY RECOMMENDATION & EMPIRICAL VALIDATION (PHASE 2) ---\n")
                    rf.write(f"Recommended Strategy        : {recommendation.recommended_configuration.value.upper()}\n")
                    rf.write(f"Primary Rule                : {recommendation.primary_rule}\n")
                    rf.write(f"Confidence                  : {recommendation.confidence}\n")
                    rf.write(f"Plain-Language Rationale    : {recommendation.plain_language_rationale}\n")
                    rf.write(f"Bottleneck Anti-Pattern     : {recommendation.expected_bottleneck_config.value.upper()}\n")
                    rf.write(f"Validation Outcome          : {val_report.outcome.value}\n")
                    rf.write(f"Validation Speedup vs Base  : {val_report.speedup_vs_baseline:.2f}x\n")
                    rf.write(f"Validation Verdict          : {val_report.verdict}\n")
                    rf.write("=" * 84 + "\n")
            except Exception as e:
                logger.warning(f"Could not append strategy findings to report: {e}")

    # Optional graphical chart generation
    if args.plot_dir and all_summaries:
        try:
            chart_files = generate_benchmark_charts(
                all_summaries,
                output_dir=args.plot_dir,
                workload_name=workload_type.value,
            )
            if chart_files:
                logger.info(
                    f"Generated {len(chart_files)} benchmark visualization charts in: {os.path.abspath(args.plot_dir)}"
                )
        except Exception as exc:
            logger.error(f"Failed to generate benchmark charts: {exc}", exc_info=True)

    logger.info("\n[+] Project 31 Phase 2 pipeline finished cleanly.\n")


if __name__ == "__main__":
    main()
