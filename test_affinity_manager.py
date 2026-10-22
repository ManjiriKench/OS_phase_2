"""
Tests for affinity_manager.py — policy layer (no OS dependency).

These tests validate the configuration-generation logic for all four
affinity configurations, plus mask construction. They do NOT require a
live Windows thread API and can run on any platform.

Run with:
    python -m pytest test_affinity_manager.py -v
    # or
    python -m unittest test_affinity_manager -v
"""

import unittest

from affinity_manager import (
    AffinityConfiguration,
    build_affinity_mask,
    generate_thread_affinity_mapping,
    validate_affinity_mask,
    verify_distinct_mappings,
    validate_all_affinity_configurations,
)


class TestBaselineConfiguration(unittest.TestCase):
    """BASELINE: no explicit affinity, all threads mapped to empty lists."""

    def test_baseline_maps_all_threads_to_empty(self):
        cores = [0, 1, 2, 3]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.BASELINE, 4, cores
        )
        self.assertEqual(len(mapping), 4)
        for idx in range(4):
            self.assertEqual(mapping[idx], [])

    def test_baseline_single_thread(self):
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.BASELINE, 1, [0, 1, 2]
        )
        self.assertEqual(mapping[0], [])

    def test_baseline_more_threads_than_cores(self):
        """BASELINE should allow more threads than cores (no pinning)."""
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.BASELINE, 10, [0, 1]
        )
        self.assertEqual(len(mapping), 10)
        for idx in range(10):
            self.assertEqual(mapping[idx], [])


class TestFixedConfiguration(unittest.TestCase):
    """FIXED: thread i pinned to core i. Rejects threads > cores."""

    def test_fixed_1to1_mapping(self):
        cores = [0, 1, 2, 3]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.FIXED, 4, cores
        )
        self.assertEqual(mapping[0], [0])
        self.assertEqual(mapping[1], [1])
        self.assertEqual(mapping[2], [2])
        self.assertEqual(mapping[3], [3])

    def test_fixed_uses_actual_core_ids(self):
        """FIXED should use the actual detected core IDs, not 0..N-1."""
        cores = [2, 5, 7, 9]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.FIXED, 3, cores
        )
        self.assertEqual(mapping[0], [2])
        self.assertEqual(mapping[1], [5])
        self.assertEqual(mapping[2], [7])

    def test_fixed_equal_threads_and_cores(self):
        cores = [0, 1]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.FIXED, 2, cores
        )
        self.assertEqual(mapping[0], [0])
        self.assertEqual(mapping[1], [1])

    def test_fixed_rejects_too_many_threads(self):
        """FIXED must raise ValueError when threads > cores."""
        cores = [0, 1, 2, 3]
        with self.assertRaises(ValueError) as ctx:
            generate_thread_affinity_mapping(
                AffinityConfiguration.FIXED, 5, cores
            )
        self.assertIn("FIXED", str(ctx.exception))

    def test_fixed_single_core(self):
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.FIXED, 1, [3]
        )
        self.assertEqual(mapping[0], [3])


class TestDistributedConfiguration(unittest.TestCase):
    """DISTRIBUTED: genuine wide-dispersion strategy across processor cores."""

    def test_distributed_strided_spread_12_cores(self):
        """On 12 cores with 4 threads, stride is 3 -> cores [0, 3, 6, 9]."""
        cores = list(range(12))
        dist_mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.DISTRIBUTED, 4, cores
        )
        self.assertEqual(dist_mapping[0], [0])
        self.assertEqual(dist_mapping[1], [3])
        self.assertEqual(dist_mapping[2], [6])
        self.assertEqual(dist_mapping[3], [9])

    def test_distributed_differs_from_fixed(self):
        """CRITICAL: FIXED and DISTRIBUTED must produce different mappings."""
        cores = list(range(12))
        fixed_mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.FIXED, 4, cores
        )
        dist_mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.DISTRIBUTED, 4, cores
        )
        self.assertNotEqual(
            fixed_mapping,
            dist_mapping,
            "FIXED and DISTRIBUTED mappings must NOT be identical",
        )
        # FIXED is consecutive 0, 1, 2, 3; DISTRIBUTED is strided 0, 3, 6, 9
        self.assertEqual([fixed_mapping[i][0] for i in range(4)], [0, 1, 2, 3])
        self.assertEqual([dist_mapping[i][0] for i in range(4)], [0, 3, 6, 9])

    def test_distributed_deterministic(self):
        """Same inputs produce the same mapping every time."""
        cores = [0, 1, 2, 3, 4, 5, 6, 7]
        mapping1 = generate_thread_affinity_mapping(
            AffinityConfiguration.DISTRIBUTED, 4, cores
        )
        mapping2 = generate_thread_affinity_mapping(
            AffinityConfiguration.DISTRIBUTED, 4, cores
        )
        self.assertEqual(mapping1, mapping2)

    def test_distributed_uses_actual_core_ids(self):
        """Mapping strictly uses IDs from available_core_ids."""
        cores = [4, 6, 8, 10]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.DISTRIBUTED, 2, cores
        )
        self.assertEqual(mapping[0], [4])
        self.assertEqual(mapping[1], [8])


class TestConcentratedConfiguration(unittest.TestCase):
    """CONCENTRATED: restrict all workers to a small subset of cores."""

    def test_concentrated_uses_two_cores(self):
        """With 2+ cores available, concentrated uses exactly 2."""
        cores = [0, 1, 2, 3]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.CONCENTRATED, 4, cores
        )
        # All threads should be on cores 0 or 1 only (first 2 of subset)
        for idx in range(4):
            self.assertIn(mapping[idx][0], [0, 1])

    def test_concentrated_alternating_pattern(self):
        """Threads alternate between the two subset cores."""
        cores = [0, 1, 2, 3]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.CONCENTRATED, 4, cores
        )
        self.assertEqual(mapping[0], [0])
        self.assertEqual(mapping[1], [1])
        self.assertEqual(mapping[2], [0])
        self.assertEqual(mapping[3], [1])

    def test_concentrated_single_core(self):
        """With only 1 core available, all threads go to that core."""
        cores = [7]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.CONCENTRATED, 3, cores
        )
        for idx in range(3):
            self.assertEqual(mapping[idx], [7])

    def test_concentrated_uses_actual_core_ids(self):
        """Subset must come from the detected core IDs."""
        cores = [5, 6, 7, 8]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.CONCENTRATED, 4, cores
        )
        # First two cores of the detected list
        for idx in range(4):
            self.assertIn(mapping[idx][0], [5, 6])

    def test_concentrated_many_threads_one_core(self):
        """Many threads on 1 core should all get that core."""
        cores = [2]
        mapping = generate_thread_affinity_mapping(
            AffinityConfiguration.CONCENTRATED, 6, cores
        )
        for idx in range(6):
            self.assertEqual(mapping[idx], [2])


class TestValidation(unittest.TestCase):
    """Input validation for generate_thread_affinity_mapping."""

    def test_invalid_thread_count_zero(self):
        for config in AffinityConfiguration:
            with self.assertRaises(ValueError):
                generate_thread_affinity_mapping(config, 0, [0, 1])

    def test_invalid_thread_count_negative(self):
        for config in AffinityConfiguration:
            with self.assertRaises(ValueError):
                generate_thread_affinity_mapping(config, -1, [0, 1])

    def test_empty_core_ids(self):
        for config in AffinityConfiguration:
            with self.assertRaises(ValueError):
                generate_thread_affinity_mapping(config, 4, [])

    def test_all_configs_produce_mapping_keys(self):
        """Every configuration must produce a mapping with thread_count keys."""
        cores = [0, 1, 2, 3]
        for config in AffinityConfiguration:
            mapping = generate_thread_affinity_mapping(config, 4, cores)
            self.assertEqual(
                set(mapping.keys()),
                {0, 1, 2, 3},
                f"Config {config} did not produce all thread indices",
            )


class TestBuildAffinityMask(unittest.TestCase):
    """Mask construction from core ID lists."""

    def test_single_core_masks(self):
        self.assertEqual(build_affinity_mask([0]), 1)
        self.assertEqual(build_affinity_mask([1]), 2)
        self.assertEqual(build_affinity_mask([2]), 4)
        self.assertEqual(build_affinity_mask([3]), 8)

    def test_multiple_core_masks(self):
        self.assertEqual(build_affinity_mask([0, 1]), 3)
        self.assertEqual(build_affinity_mask([0, 1, 2]), 7)
        self.assertEqual(build_affinity_mask([1, 2]), 6)
        self.assertEqual(build_affinity_mask([0, 3]), 9)

    def test_high_core_ids(self):
        self.assertEqual(build_affinity_mask([0, 7]), 1 + 128)
        self.assertEqual(build_affinity_mask([4, 5]), 16 + 32)

    def test_empty_core_list(self):
        self.assertEqual(build_affinity_mask([]), 0)


class TestAffinityMaskValidation(unittest.TestCase):
    """Validation of affinity bitmasks against assigned core IDs."""

    def test_valid_mask(self):
        self.assertTrue(validate_affinity_mask([0, 2], 5, [0, 1, 2, 3]))

    def test_mask_mismatch(self):
        with self.assertRaises(Exception):
            validate_affinity_mask([0, 1], 5)  # 5 is 0b101 (cores 0, 2), not cores 0, 1

    def test_negative_mask(self):
        with self.assertRaises(Exception):
            validate_affinity_mask([0], -1)

    def test_core_outside_available_cores(self):
        with self.assertRaises(Exception):
            validate_affinity_mask([5], 32, available_core_ids=[0, 1, 2, 3])


class TestDistinctMappingsVerification(unittest.TestCase):
    """Verification that FIXED and DISTRIBUTED use distinct mappings."""

    def test_distinct_mappings_standard_topology(self):
        """12 cores, 4 threads must produce distinct mappings."""
        cores = list(range(12))
        self.assertTrue(verify_distinct_mappings(4, cores))

    def test_distinct_mappings_8_cores(self):
        """8 cores, 4 threads must produce distinct mappings."""
        cores = list(range(8))
        self.assertTrue(verify_distinct_mappings(4, cores))

    def test_distinct_mappings_equal_threads_cores(self):
        """4 cores, 4 threads must produce distinct mappings."""
        cores = list(range(4))
        self.assertTrue(verify_distinct_mappings(4, cores))

    def test_validate_all_affinity_configurations(self):
        """Pre-flight check validates all four configurations."""
        cores = list(range(8))
        mappings = validate_all_affinity_configurations(4, cores)
        self.assertIn("baseline", mappings)
        self.assertIn("fixed", mappings)
        self.assertIn("distributed", mappings)
        self.assertIn("concentrated", mappings)
        self.assertNotEqual(mappings["fixed"], mappings["distributed"])


if __name__ == "__main__":
    unittest.main()

