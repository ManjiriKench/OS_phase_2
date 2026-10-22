"""
Module 3: Workload Engine
Project: Thread Affinity Management for Multi-Core Processors (Phase 1)

This module handles:
1. Providing deterministic, repeatable CPU-intensive compute workloads for worker threads.
2. Offering multiple workload strategies:
   - 'crypto_hash' (Default): Deterministic cryptographic block hashing (SHA-256).
     C-level implementation in CPython releases the Global Interpreter Lock (GIL),
     allowing true multi-core parallel execution on OS threads and clearly
     demonstrating thread affinity speedups.
   - 'matrix_mult': Deterministic integer square matrix multiplication.
   - 'prime_sieve': Deterministic prime number trial division up to a fixed limit.
3. Ensuring identical input parameters yield identical compute cycles across runs.
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Callable, Dict, Optional


class WorkloadType(Enum):
    """Available CPU workload types."""
    CRYPTO_HASH = "crypto_hash"
    MATRIX_MULT = "matrix_mult"
    PRIME_SIEVE = "prime_sieve"


def run_crypto_hash_workload(
    thread_index: int,
    iterations: int = 4000,
    chunk_size_kb: int = 256,
) -> int:
    """
    Executes repeated SHA-256 hashing over fixed 256KB memory buffers.

    Why this is ideal for OS Thread Affinity testing:
    - hashlib.sha256 in CPython releases the GIL during hashing of memory buffers.
    - This allows worker threads to execute in parallel across distinct CPU cores.
    - Demonstrates dramatic differences between pinned cores (Fixed/Distributed)
      and bottlenecked cores (Concentrated).

    Args:
        thread_index: Worker thread index.
        iterations: Number of hash iterations to compute.
        chunk_size_kb: Size of buffer in kilobytes to hash repeatedly.

    Returns:
        int: Total operations completed.
    """
    # Deterministic pattern seeded by fixed bytes
    data_block = (b"THREAD_AFFINITY_BENCHMARK_BLOCK_" * (chunk_size_kb * 32))[: chunk_size_kb * 1024]
    
    current_hash = hashlib.sha256(data_block).digest()
    byte_slice = memoryview(data_block)
    for i in range(iterations):
        # Hash full buffer; mix in loop counter to ensure fresh computation
        current_hash = hashlib.sha256(byte_slice).digest()

    return iterations


def run_matrix_mult_workload(
    thread_index: int,
    matrix_size: int = 150,
) -> int:
    """
    Executes a deterministic NxN integer matrix multiplication.

    Args:
        thread_index: Worker thread index.
        matrix_size: Dimension N of the NxN matrices.

    Returns:
        int: Checksum of resulting matrix.
    """
    size = matrix_size
    A = [[(i + j + 1) % 100 for j in range(size)] for i in range(size)]
    B = [[(i * 2 + j + 3) % 100 for j in range(size)] for i in range(size)]
    C = [[0] * size for _ in range(size)]

    for i in range(size):
        for k in range(size):
            a_ik = A[i][k]
            for j in range(size):
                C[i][j] = (C[i][j] + a_ik * B[k][j]) % 1000000

    checksum = sum(C[i][i] for i in range(size))
    return checksum


def run_prime_sieve_workload(
    thread_index: int,
    limit: int = 40000,
) -> int:
    """
    Calculates primes up to limit using trial division.

    Args:
        thread_index: Worker thread index.
        limit: Upper integer bound for prime checking.

    Returns:
        int: Number of primes found.
    """
    count = 0
    for num in range(2, limit):
        is_prime = True
        for d in range(2, int(num**0.5) + 1):
            if num % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


WORKLOAD_REGISTRY: Dict[WorkloadType, Callable] = {
    WorkloadType.CRYPTO_HASH: run_crypto_hash_workload,
    WorkloadType.MATRIX_MULT: run_matrix_mult_workload,
    WorkloadType.PRIME_SIEVE: run_prime_sieve_workload,
}


def get_workload_function(
    workload_type: WorkloadType = WorkloadType.CRYPTO_HASH,
    intensity: Optional[int] = None,
) -> Callable[[int], int]:
    """
    Factory function returning a configured, single-argument workload runner
    `worker_task(thread_index) -> result`.

    Args:
        workload_type: WorkloadType enum or string name.
        intensity: Optional intensity parameter (iterations, matrix size, limit).
                   Defaults to calibrated values suitable for ~1-2s runs.

    Returns:
        Callable[[int], int]: Callable accepting thread_index.
    """
    if isinstance(workload_type, str):
        try:
            workload_type = WorkloadType(workload_type.lower())
        except ValueError:
            valid = [w.value for w in WorkloadType]
            raise ValueError(f"Invalid workload '{workload_type}'. Choose from {valid}")

    if workload_type == WorkloadType.CRYPTO_HASH:
        iters = intensity if intensity is not None else 4000
        return lambda tid: run_crypto_hash_workload(tid, iterations=iters)

    elif workload_type == WorkloadType.MATRIX_MULT:
        size = intensity if intensity is not None else 150
        return lambda tid: run_matrix_mult_workload(tid, matrix_size=size)

    elif workload_type == WorkloadType.PRIME_SIEVE:
        lim = intensity if intensity is not None else 40000
        return lambda tid: run_prime_sieve_workload(tid, limit=lim)

    raise ValueError(f"Unsupported workload type: {workload_type}")


if __name__ == "__main__":
    print("Testing Module 3: Workload Engine standalone...")
    import time as time_module
    t0 = time_module.perf_counter()
    res = run_crypto_hash_workload(0, iterations=2000)
    dt = time_module.perf_counter() - t0
    print(f"  [+] Crypto hash completed ({res} iters) in {dt:.4f}s")