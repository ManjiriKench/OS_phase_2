"""
Module 1: CPU Core Detection
Project: Thread Affinity Management for Multi-Core Processors (Phase 1)

This module handles:
1. Detecting physical CPU cores (actual silicon execution units) and
   logical processors (hardware threads exposed to the OS scheduler).
2. Generating 0-indexed logical processor IDs used by Windows affinity masks.
3. Discovering physical-to-logical core topology on Windows via
   GetLogicalProcessorInformation (ctypes) to distinguish SMT/hyperthreads.
4. Clearly presenting CPU hardware topology with precise academic terminology.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _query_windows_core_topology() -> Optional[Dict[int, List[int]]]:
    """
    Queries the Windows kernel for physical core to logical processor mappings.
    Returns:
        dict: {physical_core_index: [logical_processor_ids, ...]}
        or None if unavailable / non-Windows.
    """
    if sys.platform != "win32":
        return None

    try:
        class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("ProcessorMask", ctypes.c_size_t),
                ("Relationship", ctypes.c_int),
                ("Reserved", ctypes.c_ulonglong * 2),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        buf_size = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_size))

        elem_size = ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
        count = buf_size.value // elem_size
        buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * count)()
        
        if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(buf_size)):
            return None

        # Relationship 0 = RelationProcessorCore
        mapping: Dict[int, List[int]] = {}
        phys_index = 0
        for item in buf:
            if item.Relationship == 0:
                mask = item.ProcessorMask
                logical_ids = [bit for bit in range(64) if (mask & (1 << bit))]
                if logical_ids:
                    mapping[phys_index] = logical_ids
                    phys_index += 1
        return mapping if mapping else None
    except Exception:
        return None


def detect_cpu_cores() -> dict:
    """
    Detects CPU hardware specifications and logical processor layout.

    Returns:
        dict:
            - 'logical_cores': Total logical processors (execution contexts).
            - 'physical_cores': Number of physical hardware cores.
            - 'core_ids': List of valid 0-indexed logical processor IDs.
            - 'physical_core_mapping': Dict of {physical_core: [logical_ids]}.
    """
    if PSUTIL_AVAILABLE:
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
    else:
        logical_cores = os.cpu_count()
        physical_cores = None

    if logical_cores is None or logical_cores < 1:
        logical_cores = 1

    core_ids = list(range(logical_cores))
    phys_mapping = _query_windows_core_topology()

    if physical_cores is None and phys_mapping is not None:
        physical_cores = len(phys_mapping)

    return {
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "core_ids": core_ids,
        "physical_core_mapping": phys_mapping,
    }


def get_available_coreIds() -> List[int]:
    """Returns valid logical processor IDs."""
    return detect_cpu_cores()["core_ids"]


def display_cpu_info(cpu_info: Optional[dict] = None) -> None:
    """
    Prints a clear, academically precise summary of CPU hardware topology.
    Distinguishes Physical Cores from Logical Processors.
    """
    if cpu_info is None:
        cpu_info = detect_cpu_cores()

    logical = cpu_info["logical_cores"]
    physical = cpu_info.get("physical_cores")
    core_ids = cpu_info["core_ids"]
    mapping = cpu_info.get("physical_core_mapping")

    print("=" * 70)
    print("           MODULE 1: CPU HARDWARE TOPOLOGY DETECTION")
    print("=" * 70)
    print(f"  * Physical CPU Cores   : {physical if physical else 'Unknown'} (Actual hardware cores)")
    print(f"  * Logical Processors   : {logical} (OS scheduling units / hyperthreads)")
    print(f"  * Logical Processor IDs: {core_ids}")
    print("  * Note: Windows affinity bitmasks operate on Logical Processor IDs.")
    
    if mapping:
        print("\n  * Physical-to-Logical Topology:")
        for p_id, l_ids in mapping.items():
            core_type = "Hyperthreaded" if len(l_ids) > 1 else "Single-thread"
            print(f"    Physical Core {p_id:2d} -> Logical Processors {l_ids} ({core_type})")
    print("=" * 70)


if __name__ == "__main__":
    # Test Module 1 standalone
    info = detect_cpu_cores()
    display_cpu_info(info)
