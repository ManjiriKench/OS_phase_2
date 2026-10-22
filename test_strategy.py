"""
Unit tests for strategy.py (Person 3 Deliverable)
Tests the explainable rule-based affinity recommender and validation engine.
"""

import unittest
from types import SimpleNamespace

from affinity_manager import AffinityConfiguration
from strategy import (
    DecisionRule,
    RecommendationOutcome,
    StrategyRecommendation,
    StrategyValidationReport,
    display_strategy_recommendation,
    display_validation_report,
    evaluate_validation_results,
    recommend_affinity_strategy,
)


class TestStrategyRecommender(unittest.TestCase):
    """Tests for recommend_affinity_strategy decision rules."""

    def setUp(self):
        # Synthetic standard 8-core / 16-thread CPU
        self.cpu_info_8c = {
            "physical_cores": 8,
            "logical_cores": 16,
            "core_ids": list(range(16)),
        }

    def test_recommend_under_physical_capacity_crypto_hash(self):
        """When thread_count <= physical_cores for crypto_hash, recommends DISTRIBUTED."""
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=4,
            workload_type="crypto_hash",
        )
        self.assertEqual(rec.recommended_configuration, AffinityConfiguration.DISTRIBUTED)
        self.assertEqual(rec.confidence, "HIGH")
        self.assertEqual(rec.expected_bottleneck_config, AffinityConfiguration.CONCENTRATED)
        self.assertIn("DISTRIBUTED is recommended", rec.plain_language_rationale)

    def test_recommend_under_physical_capacity_prime_sieve(self):
        """When thread_count <= physical_cores for prime_sieve, recommends FIXED."""
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=4,
            workload_type="prime_sieve",
        )
        self.assertEqual(rec.recommended_configuration, AffinityConfiguration.FIXED)
        self.assertEqual(rec.confidence, "HIGH")
        self.assertIn("FIXED is recommended", rec.plain_language_rationale)

    def test_recommend_moderate_oversubscription(self):
        """When physical < thread_count <= logical, recommends DISTRIBUTED to balance load."""
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=12,
            workload_type="crypto_hash",
        )
        self.assertEqual(rec.recommended_configuration, AffinityConfiguration.DISTRIBUTED)
        self.assertEqual(rec.confidence, "HIGH")
        self.assertEqual(rec.primary_rule, "R3_OVERSUBSCRIPTION_LOAD_BALANCE")

    def test_recommend_severe_oversubscription(self):
        """When thread_count > logical_cores, recommends BASELINE for OS dynamic balancing."""
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=32,
            workload_type="crypto_hash",
        )
        self.assertEqual(rec.recommended_configuration, AffinityConfiguration.BASELINE)
        self.assertEqual(rec.confidence, "MEDIUM")

    def test_recommend_bottleneck_identification(self):
        """Verify CONCENTRATED is flagged as anti-pattern bottleneck."""
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=4,
            workload_type="crypto_hash",
        )
        self.assertEqual(rec.expected_bottleneck_config, AffinityConfiguration.CONCENTRATED)
        self.assertIn("CONCENTRATED", rec.bottleneck_rationale)

    def test_empirical_calibration_override(self):
        """When calibration data is supplied, selects fastest observed configuration."""
        fake_summaries = [
            SimpleNamespace(config_name="baseline", avg_time=2.0),
            SimpleNamespace(config_name="fixed", avg_time=1.2),
            SimpleNamespace(config_name="distributed", avg_time=1.0),  # Fastest
            SimpleNamespace(config_name="concentrated", avg_time=3.5),
        ]
        rec = recommend_affinity_strategy(
            cpu_info=self.cpu_info_8c,
            thread_count=4,
            workload_type="crypto_hash",
            calibration_summaries=fake_summaries,
        )
        self.assertEqual(rec.recommended_configuration, AffinityConfiguration.DISTRIBUTED)
        self.assertEqual(rec.primary_rule, "R5_EMPIRICAL_CALIBRATION")
        self.assertIsNotNone(rec.empirical_speedups)
        self.assertAlmostEqual(rec.empirical_speedups["DISTRIBUTED"], 2.0 / 1.0)
        self.assertAlmostEqual(rec.empirical_speedups["CONCENTRATED"], 2.0 / 3.5)


class TestStrategyValidation(unittest.TestCase):
    """Tests for evaluate_validation_results."""

    def test_validation_optimal_outcome(self):
        """Recommended config is fastest -> VALIDATED_OPTIMAL."""
        summaries = [
            SimpleNamespace(config_name="baseline", avg_time=2.0),
            SimpleNamespace(config_name="fixed", avg_time=1.2),
            SimpleNamespace(config_name="distributed", avg_time=1.0),
            SimpleNamespace(config_name="concentrated", avg_time=3.5),
        ]
        val = evaluate_validation_results(
            recommended_config=AffinityConfiguration.DISTRIBUTED,
            calibration_workload="crypto_hash",
            validation_workload="matrix_mult",
            summaries=summaries,
        )
        self.assertEqual(val.outcome, RecommendationOutcome.OPTIMAL)
        self.assertEqual(val.fastest_configuration, "DISTRIBUTED")
        self.assertAlmostEqual(val.speedup_vs_baseline, 2.0)
        self.assertAlmostEqual(val.speedup_vs_concentrated, 3.5)
        self.assertIn("fastest configuration", val.verdict)

    def test_validation_effective_outcome(self):
        """Recommended config beats baseline but is slightly behind fastest -> VALIDATED_EFFECTIVE."""
        summaries = [
            SimpleNamespace(config_name="baseline", avg_time=2.0),
            SimpleNamespace(config_name="fixed", avg_time=1.0),        # Fastest
            SimpleNamespace(config_name="distributed", avg_time=1.05),  # Recommended (effective)
            SimpleNamespace(config_name="concentrated", avg_time=3.5),
        ]
        val = evaluate_validation_results(
            recommended_config=AffinityConfiguration.DISTRIBUTED,
            calibration_workload="crypto_hash",
            validation_workload="matrix_mult",
            summaries=summaries,
        )
        self.assertEqual(val.outcome, RecommendationOutcome.EFFECTIVE)
        self.assertEqual(val.fastest_configuration, "FIXED")
        self.assertIn("effective", val.verdict.lower())

    def test_validation_inferior_outcome(self):
        """Recommended config is slower than baseline -> NOT_OPTIMAL."""
        summaries = [
            SimpleNamespace(config_name="baseline", avg_time=1.5),
            SimpleNamespace(config_name="fixed", avg_time=1.6),
            SimpleNamespace(config_name="distributed", avg_time=2.0),  # Recommended is slower
            SimpleNamespace(config_name="concentrated", avg_time=3.0),
        ]
        val = evaluate_validation_results(
            recommended_config=AffinityConfiguration.DISTRIBUTED,
            calibration_workload="crypto_hash",
            validation_workload="matrix_mult",
            summaries=summaries,
        )
        self.assertEqual(val.outcome, RecommendationOutcome.INFERIOR)
        self.assertIn("did not outperform", val.verdict)

    def test_display_functions(self):
        """Ensure formatting and display functions execute without error."""
        rec = recommend_affinity_strategy(
            cpu_info={"physical_cores": 4, "logical_cores": 8},
            thread_count=4,
            workload_type="crypto_hash",
        )
        display_strategy_recommendation(rec)

        summaries = [
            SimpleNamespace(config_name="baseline", avg_time=2.0),
            SimpleNamespace(config_name="distributed", avg_time=1.0),
        ]
        val = evaluate_validation_results(
            recommended_config=AffinityConfiguration.DISTRIBUTED,
            calibration_workload="crypto_hash",
            validation_workload="matrix_mult",
            summaries=summaries,
        )
        display_validation_report(val)


if __name__ == "__main__":
    unittest.main()
