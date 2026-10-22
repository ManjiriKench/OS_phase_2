"""
Module: Strategy Recommender and Validation (Person 3 Deliverable)
Project: Thread Affinity Management for Multi-Core Processors (Phase 2)

This module implements:
1. An explainable, rule-based affinity recommender that recommends the optimal
   affinity configuration (BASELINE, FIXED, DISTRIBUTED, CONCENTRATED) based on:
   - CPU hardware topology (physical cores vs. logical processors).
   - Worker thread count and core subscription ratio.
   - Workload computational characteristics (cache/memory footprint, compute intensity).
   - Empirical profiling data (when calibration runs are available).
2. Transparent decision rules documented in plain language.
3. Strategy validation logic: evaluating recommendations against BASELINE and other
   configurations across distinct workloads and input sizes.
4. Clean comparison and reporting functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from affinity_manager import (
    AffinityConfiguration,
    generate_thread_affinity_mapping,
    build_affinity_mask,
    validate_affinity_mask,
    verify_distinct_mappings,
)


class RecommendationOutcome(Enum):
    """Outcome of validating an affinity recommendation against actual measurements."""
    OPTIMAL = "VALIDATED_OPTIMAL"          # Fastest measured configuration
    EFFECTIVE = "VALIDATED_EFFECTIVE"      # Faster than Baseline (speedup > 1.0)
    EQUIVALENT = "VALIDATED_EQUIVALENT"    # Within +/- 2% of Baseline
    INFERIOR = "NOT_OPTIMAL"               # Slower than Baseline


@dataclass
class DecisionRule:
    """A transparent, explainable rule used by the strategy recommender."""
    rule_id: str
    name: str
    description: str
    is_triggered: bool
    explanation: str


@dataclass
class StrategyRecommendation:
    """Structured result produced by the affinity strategy recommender."""
    recommended_configuration: AffinityConfiguration
    confidence: str                        # "HIGH" or "MEDIUM"
    primary_rule: str
    plain_language_rationale: str
    active_rules: List[DecisionRule]
    expected_bottleneck_config: AffinityConfiguration
    bottleneck_rationale: str
    workload_type: str
    thread_count: int
    physical_cores: int
    logical_cores: int
    empirical_speedups: Optional[Dict[str, float]] = None
    empirical_mean_times: Optional[Dict[str, float]] = None


@dataclass
class StrategyValidationReport:
    """Report comparing strategy recommendations against actual benchmark measurements."""
    calibration_workload: str
    validation_workload: str
    recommended_configuration: str
    fastest_configuration: str
    baseline_avg_time: float
    recommended_avg_time: float
    speedup_vs_baseline: float
    speedup_vs_concentrated: Optional[float]
    outcome: RecommendationOutcome
    verdict: str
    configuration_times: Dict[str, float] = field(default_factory=dict)
    configuration_speedups: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Explainable Rule-Based Recommender
# ---------------------------------------------------------------------------

def recommend_affinity_strategy(
    cpu_info: dict,
    thread_count: int,
    workload_type: str = "crypto_hash",
    calibration_summaries: Optional[List] = None,
) -> StrategyRecommendation:
    """
    Recommends the best affinity configuration using transparent, explainable rules.

    Decision Rules in Plain Language:
    --------------------------------
    Rule 1 (Physical Core Capacity):
        If thread_count <= physical_cores, each thread can execute on its own
        physical silicon core without resource contention on SMT hyperthreads.
        Pinning threads avoids OS thread migration and preserves L1/L2 cache locality.

    Rule 2 (Workload Memory & Cache Footprint):
        - 'crypto_hash' (256KB buffer hashing) and 'matrix_mult' (square matrices)
          are memory/cache-sensitive. Spreading threads via DISTRIBUTED across distinct
          physical core regions maximizes cache isolation and memory bandwidth.
        - 'prime_sieve' is tight-loop arithmetic compute where dedicated 1-to-1 core
          pinning (FIXED or DISTRIBUTED) avoids time-slicing interruptions.

    Rule 3 (Oversubscription Management):
        If thread_count > physical_cores, threads must share execution units.
        DISTRIBUTED is recommended to disperse load evenly across all logical processors
        before stacking. If thread_count > logical_cores, BASELINE is recommended
        so the Windows scheduler can dynamically balance thread priority and sleep states.

    Rule 4 (Bottleneck Avoidance - Anti-Pattern):
        CONCENTRATED restricts all threads to 1 or 2 cores, creating artificial CPU
        saturation, run-queue lockup, and cache eviction. It is flagged as the
        expected worst-performing configuration.

    Rule 5 (Empirical Calibration Override):
        If historical or calibration measurements are provided, the recommender
        selects the configuration with the lowest observed mean execution time and
        validates the speedup ratio against BASELINE.

    Args:
        cpu_info: Dict returned by detect_cpu_cores() containing physical/logical core counts.
        thread_count: Number of worker threads to be executed.
        workload_type: Compute workload identifier ('crypto_hash', 'matrix_mult', 'prime_sieve').
        calibration_summaries: Optional list of ConfigurationSummary objects from a prior run.

    Returns:
        StrategyRecommendation with complete explanation and rule traces.
    """
    logical_cores = cpu_info.get("logical_cores", 1) or 1
    physical_cores = cpu_info.get("physical_cores")
    if physical_cores is None or physical_cores < 1:
        # Fallback estimation if physical cores could not be detected
        physical_cores = max(1, logical_cores // 2)

    available_core_ids = cpu_info.get("core_ids", list(range(logical_cores)))

    rules: List[DecisionRule] = []
    empirical_speedups: Optional[Dict[str, float]] = None
    empirical_times: Optional[Dict[str, float]] = None

    # Check for empirical data (Rule 5)
    if calibration_summaries and len(calibration_summaries) > 0:
        empirical_times = {}
        baseline_time = None
        for s in calibration_summaries:
            c_name = getattr(s, "config_name", str(s)).upper()
            avg_t = getattr(s, "avg_time_sec", None)
            if avg_t is None:
                avg_t = getattr(s, "avg_time", None)
            if avg_t is not None and avg_t > 0:
                empirical_times[c_name] = avg_t
                if c_name == "BASELINE":
                    baseline_time = avg_t

        if empirical_times and baseline_time and baseline_time > 0:
            empirical_speedups = {
                c: baseline_time / t for c, t in empirical_times.items()
            }
            # Pick configuration with lowest execution time
            best_config_str = min(empirical_times, key=empirical_times.get)
            try:
                best_config = AffinityConfiguration(best_config_str.lower())
            except ValueError:
                best_config = AffinityConfiguration.DISTRIBUTED

            rule5 = DecisionRule(
                rule_id="R5_EMPIRICAL_CALIBRATION",
                name="Empirical Measurement Calibration",
                description="Select configuration with lowest measured mean execution time.",
                is_triggered=True,
                explanation=(
                    f"Prior benchmark of '{workload_type}' measured {best_config_str} as fastest "
                    f"({empirical_times[best_config_str]:.4f}s avg, "
                    f"{empirical_speedups.get(best_config_str, 1.0):.2f}x vs BASELINE)."
                ),
            )
            rules.append(rule5)

            return StrategyRecommendation(
                recommended_configuration=best_config,
                confidence="HIGH",
                primary_rule="R5_EMPIRICAL_CALIBRATION",
                plain_language_rationale=(
                    f"Selected {best_config.value.upper()} based on empirical measurements showing "
                    f"it achieved the lowest average execution time ({empirical_times[best_config_str]:.4f}s) "
                    f"with {empirical_speedups.get(best_config_str, 1.0):.2f}x speedup over BASELINE."
                ),
                active_rules=rules,
                expected_bottleneck_config=AffinityConfiguration.CONCENTRATED,
                bottleneck_rationale=(
                    "CONCENTRATED is empirically and architecturally bottlenecked by confining "
                    f"{thread_count} threads to 1-2 cores."
                ),
                workload_type=workload_type,
                thread_count=thread_count,
                physical_cores=physical_cores,
                logical_cores=logical_cores,
                empirical_speedups=empirical_speedups,
                empirical_mean_times=empirical_times,
            )

    # Heuristic Rule-Based Decision (Rules 1 - 4)
    r1_triggered = (thread_count <= physical_cores)
    r1 = DecisionRule(
        rule_id="R1_PHYSICAL_CORE_CAPACITY",
        name="Physical Core Capacity Matching",
        description="Verify whether worker threads fit within dedicated physical execution cores.",
        is_triggered=r1_triggered,
        explanation=(
            f"Worker count ({thread_count}) <= physical cores ({physical_cores}). Each thread can "
            "receive a dedicated execution unit without hyperthread sharing."
            if r1_triggered else
            f"Worker count ({thread_count}) > physical cores ({physical_cores}). Threads must share physical cores."
        ),
    )
    rules.append(r1)

    r2_triggered = workload_type in ("crypto_hash", "matrix_mult")
    r2 = DecisionRule(
        rule_id="R2_CACHE_MEMORY_DISPERSION",
        name="Cache and Memory Bandwidth Isolation",
        description="Disperse threads across distinct core regions for memory-heavy or cache-sensitive tasks.",
        is_triggered=r2_triggered,
        explanation=(
            f"Workload '{workload_type}' benefits from strided physical placement (DISTRIBUTED) "
            "to maximize L1/L2 cache isolation and memory bus throughput."
            if r2_triggered else
            f"Workload '{workload_type}' is compute-bound trial division; pinned dedicated cores (FIXED/DISTRIBUTED) are optimal."
        ),
    )
    rules.append(r2)

    r3_triggered = (thread_count > physical_cores)
    r3 = DecisionRule(
        rule_id="R3_OVERSUBSCRIPTION_LOAD_BALANCE",
        name="Oversubscription Load Balancing",
        description="Manage CPU oversubscription when threads exceed physical core count.",
        is_triggered=r3_triggered,
        explanation=(
            f"Thread count ({thread_count}) exceeds physical cores ({physical_cores}). "
            "DISTRIBUTED evenly balances threads across all logical cores before co-locating threads."
            if r3_triggered else
            "System is not oversubscribed; 1-to-1 core allocation is achievable."
        ),
    )
    rules.append(r3)

    r4 = DecisionRule(
        rule_id="R4_BOTTLENECK_AVOIDANCE",
        name="Bottleneck Elimination (Anti-Pattern)",
        description="Detect and flag artificially restricted core configurations.",
        is_triggered=True,
        explanation=(
            f"CONCENTRATED forces {thread_count} worker threads onto a maximum of 2 cores, "
            "creating severe thread queueing, context-switch overhead, and cache thrashing."
        ),
    )
    rules.append(r4)

    # Determine recommended configuration
    if thread_count > logical_cores:
        # Severe oversubscription: let OS scheduler dynamically manage threads
        rec_config = AffinityConfiguration.BASELINE
        confidence = "MEDIUM"
        primary_rule = "R3_OVERSUBSCRIPTION_LOAD_BALANCE"
        rationale = (
            f"Thread count ({thread_count}) exceeds all {logical_cores} logical processors. "
            "BASELINE is recommended so the Windows kernel scheduler can dynamically time-slice "
            "threads across all available cores."
        )
    elif r1_triggered:
        # Thread count <= physical cores
        if workload_type in ("crypto_hash", "matrix_mult"):
            rec_config = AffinityConfiguration.DISTRIBUTED
            confidence = "HIGH"
            primary_rule = "R2_CACHE_MEMORY_DISPERSION"
            rationale = (
                f"For '{workload_type}' with {thread_count} threads on a {physical_cores}-core / "
                f"{logical_cores}-processor system, DISTRIBUTED is recommended. Strided core placement "
                "maximizes private cache retention, reduces memory bus contention, and avoids SMT pairing."
            )
        else:
            # prime_sieve
            rec_config = AffinityConfiguration.FIXED
            confidence = "HIGH"
            primary_rule = "R1_PHYSICAL_CORE_CAPACITY"
            rationale = (
                f"For arithmetic compute '{workload_type}' with {thread_count} threads <= {physical_cores} cores, "
                "FIXED is recommended to give each worker a dedicated sequential physical core without migration."
            )
    else:
        # thread_count > physical_cores but <= logical_cores
        rec_config = AffinityConfiguration.DISTRIBUTED
        confidence = "HIGH"
        primary_rule = "R3_OVERSUBSCRIPTION_LOAD_BALANCE"
        rationale = (
            f"With {thread_count} threads exceeding {physical_cores} physical cores, DISTRIBUTED "
            "spreads worker threads evenly across the full logical processor topology, ensuring "
            "each physical core has at least one thread before hyperthreads are shared."
        )

    return StrategyRecommendation(
        recommended_configuration=rec_config,
        confidence=confidence,
        primary_rule=primary_rule,
        plain_language_rationale=rationale,
        active_rules=rules,
        expected_bottleneck_config=AffinityConfiguration.CONCENTRATED,
        bottleneck_rationale=(
            f"CONCENTRATED restricts {thread_count} threads to 1-2 cores, creating an execution "
            "bottleneck with maximum observed queuing latency."
        ),
        workload_type=workload_type,
        thread_count=thread_count,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        empirical_speedups=None,
        empirical_mean_times=None,
    )


# ---------------------------------------------------------------------------
# Strategy Validation Engine
# ---------------------------------------------------------------------------

def evaluate_validation_results(
    recommended_config: AffinityConfiguration,
    calibration_workload: str,
    validation_workload: str,
    summaries: List,
) -> StrategyValidationReport:
    """
    Evaluates measured validation benchmark summaries against the recommended strategy.

    Compares:
    - Recommended configuration vs. BASELINE (speedup ratio)
    - Recommended configuration vs. FIXED, DISTRIBUTED, CONCENTRATED
    - Honest classification of outcome (OPTIMAL, EFFECTIVE, EQUIVALENT, INFERIOR)
    """
    times: Dict[str, float] = {}
    for s in summaries:
        name = getattr(s, "config_name", str(s)).upper()
        avg_t = getattr(s, "avg_time_sec", None)
        if avg_t is None:
            avg_t = getattr(s, "avg_time", None)
        if avg_t is not None:
            times[name] = avg_t

    if not times:
        raise ValueError("No configuration summaries with valid avg_time provided.")

    rec_name = recommended_config.value.upper()
    baseline_time = times.get("BASELINE", 0.0)
    rec_time = times.get(rec_name, 0.0)

    # Find fastest measured configuration
    fastest_name = min(times, key=times.get)
    fastest_time = times[fastest_name]

    speedup_vs_baseline = (baseline_time / rec_time) if (baseline_time > 0 and rec_time > 0) else 1.0

    conc_time = times.get("CONCENTRATED")
    speedup_vs_conc = (conc_time / rec_time) if (conc_time and rec_time > 0) else None

    # Calculate all speedups relative to BASELINE
    speedups = {}
    for cfg, t in times.items():
        speedups[cfg] = (baseline_time / t) if (baseline_time > 0 and t > 0) else 1.0

    # Determine validation outcome
    if rec_name == fastest_name:
        outcome = RecommendationOutcome.OPTIMAL
        verdict = (
            f"Recommendation confirmed: {rec_name} was the fastest configuration "
            f"({rec_time:.4f}s avg, {speedup_vs_baseline:.2f}x speedup vs BASELINE)."
        )
    elif rec_time < baseline_time * 0.98:
        outcome = RecommendationOutcome.EFFECTIVE
        diff_pct = ((rec_time - fastest_time) / fastest_time) * 100
        verdict = (
            f"Recommendation effective: {rec_name} outperformed BASELINE with {speedup_vs_baseline:.2f}x speedup "
            f"({rec_time:.4f}s avg), within {diff_pct:.1f}% of fastest configuration ({fastest_name})."
        )
    elif abs(rec_time - baseline_time) / baseline_time <= 0.02:
        outcome = RecommendationOutcome.EQUIVALENT
        verdict = (
            f"Recommendation equivalent: {rec_name} performed within +/-2% of BASELINE "
            f"({rec_time:.4f}s vs {baseline_time:.4f}s)."
        )
    else:
        outcome = RecommendationOutcome.INFERIOR
        slowdown = (rec_time / baseline_time) if baseline_time > 0 else 1.0
        verdict = (
            f"Recommendation did not outperform BASELINE on '{validation_workload}' "
            f"({rec_name} took {rec_time:.4f}s vs BASELINE {baseline_time:.4f}s, {slowdown:.2f}x relative time)."
        )

    return StrategyValidationReport(
        calibration_workload=calibration_workload,
        validation_workload=validation_workload,
        recommended_configuration=rec_name,
        fastest_configuration=fastest_name,
        baseline_avg_time=baseline_time,
        recommended_avg_time=rec_time,
        speedup_vs_baseline=speedup_vs_baseline,
        speedup_vs_concentrated=speedup_vs_conc,
        outcome=outcome,
        verdict=verdict,
        configuration_times=times,
        configuration_speedups=speedups,
    )


# ---------------------------------------------------------------------------
# Formatted Terminal Presentation
# ---------------------------------------------------------------------------

def display_strategy_recommendation(rec: StrategyRecommendation) -> None:
    """Print an academically formatted terminal box with the strategy recommendation."""
    print("\n" + "=" * 80)
    print("        AFFINITY STRATEGY RECOMMENDER (EXPLAINABLE RULE-BASED)")
    print("=" * 80)
    print(f"  * Recommended Configuration : {rec.recommended_configuration.value.upper()}")
    print(f"  * Recommendation Confidence : {rec.confidence}")
    print(f"  * Workload Under Analysis   : {rec.workload_type.upper()}")
    print(f"  * Worker Thread Count       : {rec.thread_count}")
    print(f"  * Detected Hardware Layout  : {rec.physical_cores} Physical Cores | {rec.logical_cores} Logical Processors")
    print(f"  * Primary Triggered Rule    : {rec.primary_rule}")
    print("-" * 80)
    print("  EXPLAINABLE RATIONALE:")
    print(f"  {rec.plain_language_rationale}")
    print("-" * 80)
    print("  ACTIVE DECISION RULES EVALUATED:")
    for rule in rec.active_rules:
        status_tag = "[TRIGGERED]" if rule.is_triggered else "[INACTIVE] "
        print(f"    {status_tag} {rule.rule_id} - {rule.name}")
        print(f"                 Reason: {rule.explanation}")
    print("-" * 80)
    print("  BOTTLENECK WARNING (ANTI-PATTERN):")
    print(f"  * Avoid Configuration      : {rec.expected_bottleneck_config.value.upper()}")
    print(f"  * Warning Explanation      : {rec.bottleneck_rationale}")

    if rec.empirical_speedups:
        print("-" * 80)
        print("  EMPIRICAL CALIBRATION EVIDENCE:")
        for cfg, spd in sorted(rec.empirical_speedups.items(), key=lambda x: x[1], reverse=True):
            avg_t = rec.empirical_mean_times.get(cfg, 0.0) if rec.empirical_mean_times else 0.0
            print(f"    - {cfg:12s} : {avg_t:.4f}s avg | {spd:5.2f}x vs BASELINE")

    print("=" * 80 + "\n")


def display_validation_report(val: StrategyValidationReport) -> None:
    """Print a clean summary table of validation results across workloads."""
    print("\n" + "#" * 80)
    print("        AFFINITY STRATEGY VALIDATION REPORT (CROSS-WORKLOAD VERIFICATION)")
    print("#" * 80)
    print(f"  * Calibration Workload       : {val.calibration_workload.upper()}")
    print(f"  * Validation Workload        : {val.validation_workload.upper()}")
    print(f"  * Recommended Configuration  : {val.recommended_configuration}")
    print(f"  * Measured Fastest Config    : {val.fastest_configuration}")
    print(f"  * Validation Outcome         : {val.outcome.value}")
    print(f"  * Speedup vs BASELINE        : {val.speedup_vs_baseline:.2f}x")
    if val.speedup_vs_concentrated:
        print(f"  * Speedup vs CONCENTRATED    : {val.speedup_vs_concentrated:.2f}x")
    print("-" * 80)
    print("  CROSS-CONFIGURATION MEASURED EXECUTION TIMES:")
    print("  Configuration  | Avg Time (s) | Speedup vs BASELINE | Status")
    print("  " + "-" * 65)
    for cfg, t in sorted(val.configuration_times.items(), key=lambda x: x[1]):
        spd = val.configuration_speedups.get(cfg, 1.0)
        marker = "<-- RECOMMENDED" if cfg == val.recommended_configuration else ""
        if cfg == val.fastest_configuration and cfg != val.recommended_configuration:
            marker += " [FASTEST]"
        print(f"  {cfg:14s} | {t:12.4f} | {spd:19.2f}x | {marker}")
    print("-" * 80)
    print(f"  VERDICT: {val.verdict}")
    print("#" * 80 + "\n")
