"""
Module 4: Thread Affinity Management
Project: Thread Affinity Management for Multi-Core Processors (Phase 1)

This module handles:
1. Defining thread-affinity configurations (policy layer).
2. Generating deterministic thread-to-core mappings for each configuration.
3. Applying Windows thread affinity via SetThreadAffinityMask (ctypes).
4. Verifying applied affinity via GetThreadAffinityMask (ctypes).

Architecture:
- The policy layer (generate_thread_affinity_mapping, build_affinity_mask)
  is platform-independent and fully testable on any OS.
- The OS layer (apply_thread_affinity, verify_thread_affinity) uses
  Windows ctypes calls and is isolated to the target demonstration
  platform.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------

class AffinityConfiguration(Enum):
    """The four affinity configurations evaluated by the project."""

    BASELINE = "baseline"
    FIXED = "fixed"
    DISTRIBUTED = "distributed"
    CONCENTRATED = "concentrated"


class AffinityError(Exception):
    """Raised when thread affinity cannot be applied or verified."""
    pass


@dataclass
class AffinityVerificationReport:
    """Detailed record of thread affinity assignment and verification status."""
    thread_name: str
    thread_index: int
    assigned_cores: List[int]
    expected_mask: int
    actual_mask: Optional[int]
    verified: bool
    status: str
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Policy layer (platform-independent)
# ---------------------------------------------------------------------------

def generate_thread_affinity_mapping(
    configuration: AffinityConfiguration,
    thread_count: int,
    available_core_ids: List[int],
) -> Dict[int, List[int]]:
    """
    Produce a deterministic mapping of thread_index -> list of core IDs.

    Args:
        configuration: The affinity policy to apply.
        thread_count: Number of worker threads (must be >= 1).
        available_core_ids: Valid CPU core IDs from cpu_detection.

    Returns:
        Dictionary mapping each thread index to a list of allowed core IDs.
        BASELINE returns an empty list per thread (no explicit affinity).

    Raises:
        ValueError: On invalid thread_count, empty cores, or FIXED
                    exceeding available core count.
    """
    if thread_count < 1:
        raise ValueError("thread_count must be at least 1.")
    if not available_core_ids:
        raise ValueError("available_core_ids must be non-empty.")

    if configuration == AffinityConfiguration.BASELINE:
        # No explicit affinity: OS scheduler decides placement.
        return {i: [] for i in range(thread_count)}

    elif configuration == AffinityConfiguration.FIXED:
        # One thread per dedicated core. Reject if threads > cores.
        if thread_count > len(available_core_ids):
            raise ValueError(
                f"FIXED configuration requires thread_count <= core count "
                f"({thread_count} > {len(available_core_ids)}). "
                f"Use DISTRIBUTED or CONCENTRATED instead."
            )
        return {i: [available_core_ids[i]] for i in range(thread_count)}

    elif configuration == AffinityConfiguration.DISTRIBUTED:
        # Genuinely distributed placement strategy across available logical processors.
        # Unlike FIXED (which sequentially binds threads 0, 1, 2, ...),
        # DISTRIBUTED disperses threads across the CPU topology with a stride
        # so worker threads are separated across distinct processor regions.
        total_cores = len(available_core_ids)
        if thread_count < total_cores:
            stride = total_cores // thread_count
            if stride > 1:
                return {
                    i: [available_core_ids[(i * stride) % total_cores]]
                    for i in range(thread_count)
                }
            else:
                # When stride is 1 and thread_count > 1, spread across both ends
                # to guarantee distinct placement from sequential FIXED.
                indices = [
                    int(round(i * (total_cores - 1) / (thread_count - 1)))
                    for i in range(thread_count)
                ]
                return {
                    i: [available_core_ids[indices[i]]]
                    for i in range(thread_count)
                }
        elif thread_count == total_cores:
            if total_cores > 2:
                # Disperse by interleaving even and odd core indices
                even_cores = [available_core_ids[k] for k in range(0, total_cores, 2)]
                odd_cores = [available_core_ids[k] for k in range(1, total_cores, 2)]
                dispersed = even_cores + odd_cores
                return {i: [dispersed[i]] for i in range(thread_count)}
            elif total_cores == 2:
                # Reverse assignment so FIXED (0->c0, 1->c1) != DISTRIBUTED (0->c1, 1->c0)
                return {
                    0: [available_core_ids[1]],
                    1: [available_core_ids[0]],
                }
            else:
                return {0: [available_core_ids[0]]}
        else:
            # When thread_count > total_cores, round-robin with stride offset
            stride = max(1, total_cores // 2) if total_cores > 1 else 1
            return {
                i: [available_core_ids[(i * stride) % total_cores]]
                for i in range(thread_count)
            }

    elif configuration == AffinityConfiguration.CONCENTRATED:
        # Restrict to a small subset: 2 cores if available, else 1.
        subset_size = min(2, len(available_core_ids))
        subset = available_core_ids[:subset_size]
        return {
            i: [subset[i % subset_size]]
            for i in range(thread_count)
        }

    else:
        raise ValueError(f"Unknown affinity configuration: {configuration}")


def build_affinity_mask(core_ids: List[int]) -> int:
    """
    Convert a list of core IDs into a Windows processor affinity mask.

    Each core ID corresponds to one bit position in the mask:
        core 0 -> bit 0 (value 1)
        core 1 -> bit 1 (value 2)
        core N -> bit N (value 2^N)

    Multiple cores produce a combined mask (e.g. cores [0, 2] -> 0b101 = 5).
    """
    mask = 0
    for core_id in core_ids:
        mask |= (1 << core_id)
    return mask


def validate_affinity_mask(
    core_ids: List[int],
    mask: int,
    available_core_ids: Optional[List[int]] = None,
) -> bool:
    """
    Validate that an affinity mask accurately represents the assigned core IDs.

    Checks:
    1. The computed bitmask of core_ids matches mask exactly.
    2. Every core ID in core_ids is in available_core_ids (if provided).
    3. No unexpected bits are enabled in the mask.

    Args:
        core_ids: List of logical processor IDs assigned to the thread.
        mask: The integer bitmask to validate.
        available_core_ids: Optional list of valid logical processor IDs on the system.

    Returns:
        bool: True if the mask is valid.

    Raises:
        AffinityError: If any validation check fails.
    """
    if mask < 0:
        raise AffinityError(f"Affinity mask cannot be negative: {mask}")

    expected_mask = build_affinity_mask(core_ids)
    if mask != expected_mask:
        raise AffinityError(
            f"Affinity mask mismatch: given {mask:#x} ({bin(mask)}), "
            f"expected {expected_mask:#x} ({bin(expected_mask)}) for cores {core_ids}"
        )

    if available_core_ids is not None:
        valid_set = set(available_core_ids)
        for core_id in core_ids:
            if core_id not in valid_set:
                raise AffinityError(
                    f"Core ID {core_id} in mask {mask:#x} is not within valid logical processor IDs {available_core_ids}"
                )

    return True


def verify_distinct_mappings(
    thread_count: int,
    available_core_ids: List[int],
) -> bool:
    """
    Verify that FIXED and DISTRIBUTED produce distinct logical-processor mappings.

    Requirement:
    Ensures that FIXED and DISTRIBUTED use their intended, distinct logical-processor
    mappings whenever thread_count > 1 and len(available_core_ids) > 1.

    Args:
        thread_count: Number of worker threads.
        available_core_ids: List of available logical processor IDs.

    Returns:
        bool: True if mappings are distinct.

    Raises:
        AffinityError: If FIXED and DISTRIBUTED produce identical mappings.
    """
    if thread_count <= 1 or len(available_core_ids) <= 1:
        # On single thread or single core, distinct placement is mathematically impossible
        return True

    if thread_count > len(available_core_ids):
        # FIXED is not applicable when threads > cores (raises ValueError)
        return True

    fixed = generate_thread_affinity_mapping(
        AffinityConfiguration.FIXED, thread_count, available_core_ids
    )
    dist = generate_thread_affinity_mapping(
        AffinityConfiguration.DISTRIBUTED, thread_count, available_core_ids
    )

    if fixed == dist:
        raise AffinityError(
            f"FIXED and DISTRIBUTED produced identical thread mappings for "
            f"{thread_count} threads across cores {available_core_ids}: {fixed}"
        )

    return True


def validate_all_affinity_configurations(
    thread_count: int,
    available_core_ids: List[int],
) -> Dict[str, Dict[int, List[int]]]:
    """
    Pre-flight validation for all four affinity configurations.

    Validates:
    1. Mapping generation for BASELINE, FIXED, DISTRIBUTED, and CONCENTRATED.
    2. FIXED and DISTRIBUTED have distinct mappings.
    3. Every thread's core assignment produces a valid bitmask.
    4. All assigned core IDs exist in available_core_ids.

    Returns:
        Dict mapping configuration names to their verified thread mappings.
    """
    mappings: Dict[str, Dict[int, List[int]]] = {}

    for config in AffinityConfiguration:
        if config == AffinityConfiguration.FIXED and thread_count > len(available_core_ids):
            # FIXED is skipped if threads > cores
            continue

        mapping = generate_thread_affinity_mapping(
            config, thread_count, available_core_ids
        )

        for thread_idx, cores in mapping.items():
            mask = build_affinity_mask(cores)
            validate_affinity_mask(cores, mask, available_core_ids)

        mappings[config.value] = mapping

    if AffinityConfiguration.FIXED.value in mappings and AffinityConfiguration.DISTRIBUTED.value in mappings:
        verify_distinct_mappings(thread_count, available_core_ids)

    return mappings


def display_affinity_mapping(
    configuration: AffinityConfiguration,
    mapping: Dict[int, List[int]],
) -> None:
    """Print a clear thread-to-core mapping table for the selected config."""
    print("-" * 55)
    print(f"  AFFINITY MAPPING ({configuration.value.upper()})")
    print("-" * 55)
    for index in sorted(mapping.keys()):
        cores = mapping[index]
        if cores:
            core_str = ", ".join(f"Logical Processor {c}" for c in cores)
            print(f"  Thread {index} -> {core_str}")
        else:
            print(f"  Thread {index} -> OS Scheduler (Unrestricted)")
    print("-" * 55)


# ---------------------------------------------------------------------------
# Windows OS layer (ctypes)
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # GetCurrentThread returns a pseudo-handle (no CloseHandle needed).
    _kernel32.GetCurrentThread.restype = wintypes.HANDLE
    _kernel32.GetCurrentThread.argtypes = []

    # SetThreadAffinityMask(hThread, dwThreadAffinityMask) -> previous mask
    _kernel32.SetThreadAffinityMask.restype = wintypes.DWORD
    _kernel32.SetThreadAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.c_size_t,
    ]

    # GetThreadAffinityMask(hThread, pProcessAffinity, pThreadAffinity) -> BOOL
    # NOTE: not all Windows builds export this by name. Attempt the lookup
    # and degrade gracefully if unavailable.
    _GET_THREAD_AFFINITY_AVAILABLE = False
    try:
        _kernel32.GetThreadAffinityMask.restype = wintypes.DWORD
        _kernel32.GetThreadAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        _GET_THREAD_AFFINITY_AVAILABLE = True
    except AttributeError:
        pass


def is_windows() -> bool:
    """Return True if running on Windows (where the OS API is available)."""
    return _IS_WINDOWS


def apply_thread_affinity(
    core_ids: List[int],
    available_core_ids: List[int],
    thread_name: str = "worker",
) -> int:
    """
    Apply the given core IDs to the CURRENT calling thread using the
    Windows API SetThreadAffinityMask.

    This function MUST be called from inside the worker thread so that
    the mask is applied to the actual worker OS thread, not the
    controlling/main thread.

    Args:
        core_ids: List of core IDs to pin the thread to.
        available_core_ids: Valid core IDs from cpu_detection (for validation).
        thread_name: Human-readable name for logging.

    Returns:
        The previous affinity mask returned by SetThreadAffinityMask.
        A non-zero value indicates the API call succeeded.

    Raises:
        AffinityError: On non-Windows platform, invalid core IDs, or
                       SetThreadAffinityMask failure.
    """
    if not _IS_WINDOWS:
        raise AffinityError(
            "Thread affinity requires Windows (SetThreadAffinityMask). "
            "This implementation does not provide a Linux equivalent."
        )

    for core_id in core_ids:
        if core_id not in available_core_ids:
            raise AffinityError(
                f"Core ID {core_id} is not in available cores "
                f"{available_core_ids}."
            )

    mask = build_affinity_mask(core_ids)
    handle = _kernel32.GetCurrentThread()

    previous_mask = _kernel32.SetThreadAffinityMask(handle, mask)
    if previous_mask == 0:
        error_code = ctypes.get_last_error()
        raise AffinityError(
            f"SetThreadAffinityMask failed for {thread_name} "
            f"(cores {core_ids}, mask {mask:#x}, Windows error {error_code})."
        )

    print(
        f"  [{thread_name}] Affinity mask {mask:#x} applied "
        f"(cores {core_ids})"
    )
    return previous_mask


def verify_thread_affinity(
    expected_core_ids: List[int],
    available_core_ids: List[int],
    thread_name: str = "worker",
    set_return_value: int = 0,
) -> bool:
    """
    Verify that thread affinity was applied successfully.

    Primary verification:
        Uses GetThreadAffinityMask to read back the mask that was set for
        the current thread, then compares it with the expected mask
        constructed from expected_core_ids.

    Fallback verification (when GetThreadAffinityMask is unavailable):
        Checks that SetThreadAffinityMask returned a non-zero previous
        mask (set_return_value), which indicates the API call succeeded.

    Windows limitation:
        GetThreadAffinityMask returns the affinity mask that was *set* for
        the thread, not the core the thread is *currently executing on*.
        The OS scheduler may still migrate the thread within the allowed
        mask. True migration-tracking would require performance counters
        or NtQueryInformationThread, which is beyond this module's scope.

    Args:
        expected_core_ids: The core IDs that were intended for this thread.
        available_core_ids: Valid core IDs from cpu_detection (for validation).
        thread_name: Human-readable name for logging.
        set_return_value: The return value from SetThreadAffinityMask,
                          used as fallback verification evidence.

    Returns:
        True if verification succeeded (either read-back or fallback).

    Raises:
        AffinityError: On non-Windows platform or API failure.
    """
    if not _IS_WINDOWS:
        raise AffinityError(
            "Thread affinity verification requires Windows "
            "(GetThreadAffinityMask)."
        )

    expected_mask = build_affinity_mask(expected_core_ids)

    # --- Primary verification: read-back via GetThreadAffinityMask ---
    if _GET_THREAD_AFFINITY_AVAILABLE:
        handle = _kernel32.GetCurrentThread()
        process_affinity = ctypes.c_size_t(0)
        thread_affinity = ctypes.c_size_t(0)

        result = _kernel32.GetThreadAffinityMask(
            handle,
            ctypes.byref(process_affinity),
            ctypes.byref(thread_affinity),
        )

        if result == 0:
            error_code = ctypes.get_last_error()
            raise AffinityError(
                f"GetThreadAffinityMask failed for {thread_name} "
                f"(Windows error {error_code})."
            )

        actual_mask = thread_affinity.value
        if actual_mask == expected_mask:
            print(
                f"  [{thread_name}] Affinity verified (read-back) "
                f"(mask {actual_mask:#x} matches expected)"
            )
            return True
        else:
            print(
                f"  [{thread_name}] Affinity MISMATCH "
                f"(expected {expected_mask:#x}, actual {actual_mask:#x})"
            )
            return False

    # --- Fallback verification: SetThreadAffinityMask return value ---
    # SetThreadAffinityMask returns the *previous* affinity mask.
    # A non-zero return indicates the API call succeeded.
    if set_return_value != 0:
        print(
            f"  [{thread_name}] Affinity verified (API success) "
            f"(mask {expected_mask:#x}, SetThreadAffinityMask returned "
            f"{set_return_value:#x})"
        )
        return True
    else:
        print(
            f"  [{thread_name}] Affinity verification FAILED "
            f"(SetThreadAffinityMask returned 0 for mask {expected_mask:#x})"
        )
        return False


def verify_thread_affinity_detailed(
    expected_core_ids: List[int],
    available_core_ids: List[int],
    thread_name: str = "worker",
    thread_index: int = 0,
    set_return_value: int = 0,
) -> AffinityVerificationReport:
    """
    Detailed verification of thread affinity returning a structured report.

    Provides complete diagnostics:
    - Expected affinity bitmask
    - Actual read-back mask (or fallback verification status)
    - Verification outcome ("Verified", "Mismatch", "Failed", "OS Scheduled")
    - Informative error message on failure
    """
    if not expected_core_ids:
        return AffinityVerificationReport(
            thread_name=thread_name,
            thread_index=thread_index,
            assigned_cores=[],
            expected_mask=0,
            actual_mask=None,
            verified=True,
            status="OS Scheduled",
            error_message=None,
        )

    expected_mask = build_affinity_mask(expected_core_ids)
    if not _IS_WINDOWS:
        return AffinityVerificationReport(
            thread_name=thread_name,
            thread_index=thread_index,
            assigned_cores=expected_core_ids,
            expected_mask=expected_mask,
            actual_mask=None,
            verified=False,
            status="Failed",
            error_message="Non-Windows platform: SetThreadAffinityMask unavailable",
        )

    if _GET_THREAD_AFFINITY_AVAILABLE:
        try:
            handle = _kernel32.GetCurrentThread()
            process_affinity = ctypes.c_size_t(0)
            thread_affinity = ctypes.c_size_t(0)
            result = _kernel32.GetThreadAffinityMask(
                handle,
                ctypes.byref(process_affinity),
                ctypes.byref(thread_affinity),
            )
            if result == 0:
                err = ctypes.get_last_error()
                return AffinityVerificationReport(
                    thread_name=thread_name,
                    thread_index=thread_index,
                    assigned_cores=expected_core_ids,
                    expected_mask=expected_mask,
                    actual_mask=None,
                    verified=False,
                    status="Failed",
                    error_message=f"GetThreadAffinityMask error {err}",
                )
            actual = thread_affinity.value
            matches = (actual == expected_mask)
            return AffinityVerificationReport(
                thread_name=thread_name,
                thread_index=thread_index,
                assigned_cores=expected_core_ids,
                expected_mask=expected_mask,
                actual_mask=actual,
                verified=matches,
                status="Verified" if matches else "Mismatch",
                error_message=None if matches else f"Mask mismatch: expected {expected_mask:#x}, got {actual:#x}",
            )
        except Exception as exc:
            return AffinityVerificationReport(
                thread_name=thread_name,
                thread_index=thread_index,
                assigned_cores=expected_core_ids,
                expected_mask=expected_mask,
                actual_mask=None,
                verified=False,
                status="Failed",
                error_message=str(exc),
            )

    # Fallback via set_return_value
    success = (set_return_value != 0)
    return AffinityVerificationReport(
        thread_name=thread_name,
        thread_index=thread_index,
        assigned_cores=expected_core_ids,
        expected_mask=expected_mask,
        actual_mask=None,
        verified=success,
        status="Verified" if success else "Failed",
        error_message=None if success else "SetThreadAffinityMask returned 0",
    )
