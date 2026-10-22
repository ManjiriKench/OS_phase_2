# Project 31: Thread Affinity Management for Multi-Core Processors

## Overview
This project implements a comprehensive thread affinity management system demonstrating how CPU core pinning affects performance on multi-core processors. It integrates all 6 project modules into a single Phase 1 prototype.

## Team Structure & Responsibilities

### Person 1 — Workload Engine
- **Module 3**: Expanded workload system with 3 deterministic CPU-intensive tasks
- **Workload types**: `crypto_hash` (SHA-256), `matrix_mult` (NxN integer multiplication), `prime_sieve` (trial division)
- **Responsibilities**: Implement workload selection, configurable intensity, deterministic repeatable computations
- **Deliverable**: `workload.py` with 3+ workload types integrated into Phase 1

### Person 2 — Metrics & Display
- **Module 5**: Performance measurement and statistical aggregation
- **Module 6**: Results display and output export
- **Responsibilities**: `RunMetrics`/`ConfigurationSummary` dataclasses, `PerformanceMonitor` context manager, CSV/TXT export, comparative summaries
- **Deliverable**: `metrics.py`, `display.py` with full export functionality

### Person 3 — Affinity Strategy, Validation, and Integration (Phase 2 Deliverable)
- **Module 4 & Strategy Engine**:
  - `affinity_manager.py`: Pre-flight mask validation (`validate_affinity_mask`), strict verification that FIXED and DISTRIBUTED mappings remain distinct (`verify_distinct_mappings`), and structured verification reporting (`verify_thread_affinity_detailed`).
  - `strategy.py`: Explainable rule-based affinity recommender (`recommend_affinity_strategy`) based on hardware topology, thread-to-core ratios, compute characteristics, and anti-pattern bottleneck detection (`CONCENTRATED`).
  - Empirical cross-workload validation engine (`evaluate_validation_results`) comparing recommendations against BASELINE and all configurations.
  - End-to-end integration into `main.py` with `--recommend`, `--validate-strategy`, and `--verify-affinity` CLI flags.
  - Comprehensive unit test suite in `test_strategy.py` and enhanced `test_affinity_manager.py`.

## Project Structure

```
project/
├── main.py                  # Entry point: runs benchmarks, recommendation & validation
├── cpu_detection.py         # Module 1: CPU hardware topology detection
├── thread_manager.py        # Module 2: Worker thread management & synchronization
├── workload.py              # Module 3: CPU-intensive workload engine (Person 1)
├── affinity_manager.py      # Module 4: Thread affinity & mask validation (Person 3)
├── metrics.py               # Module 5: Performance measurement & stats (Person 2)
├── display.py               # Module 6: Results display, charts & reports (Person 2)
├── strategy.py              # Phase 2: Explainable Strategy Recommender & Validator (Person 3)
├── test_affinity_manager.py # Unit tests for affinity manager & mask validation
├── test_strategy.py         # Unit tests for strategy recommender & validation
└── test_modules_5_and_6.py  # Unit tests for metrics, display & workload
```

## Getting Started

### Prerequisites
- Python 3.8+
- `psutil` package: `pip install psutil`

### Running the Benchmark & Strategy Engine

```bash
# Run default benchmark suite (all configs, 3 runs, recommendation & validation)
python main.py

# Pre-flight affinity validation (verifies bitmasks and distinct thread mappings)
python main.py --verify-affinity

# Explainable rule-based strategy recommendation (analyzes topology & workload)
python main.py --recommend
python main.py 4 --workload prime_sieve --recommend

# Cross-workload strategy validation (calibrates on workload A, validates on workload B)
python main.py 4 --validate-strategy --runs 3

# Run with 4 threads, 5 repetitions, matrix multiply workload
python main.py 4 --runs 5 --workload matrix_mult

# Run single configuration with custom intensity
python main.py 4 --affinity fixed --runs 3 --workload crypto_hash --intensity 5000

# Output to custom files
python main.py 4 --runs 2 --csv my_results.csv --report my_report.txt
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `num_threads` | 4 | Number of worker threads |
| `--affinity` | `all` | Configurations: `all`, `baseline`, `fixed`, `distributed`, `concentrated` |
| `--runs` | 3 | Benchmark repetitions per configuration |
| `--workload` | `crypto_hash` | `crypto_hash`, `matrix_mult`, `prime_sieve` |
| `--intensity` | auto | Override workload iterations/size/limit |
| `--csv` | `affinity_benchmark_results.csv` | CSV output path |
| `--report` | `affinity_benchmark_report.txt` | TXT report path |
| `--plot-dir` | None | Directory to export comparison PNG charts |
| `--recommend` | False | Run explainable rule-based strategy recommendation |
| `--validate-strategy` | False | Validate recommended strategy across distinct workloads |
| `--verify-affinity` | False | Perform pre-flight bitmask and distinct mapping validation |

## Configuration Details

### Baseline (No affinity)
- OS scheduler decides thread placement
- Best for reference/normalization

### Fixed (1:1 pinning)
- Thread i pinned to Core i
- Requires: `threads <= cores`
- Demonstrates dedicated core allocation

### Distributed (round-robin spread)
- Threads dispersed across all cores with stride
- Ensures FIXED != DISTRIBUTED mappings
- Maximizes parallel execution

### Concentrated (restricted core subset)
- All threads restricted to 1-2 cores
- Demonstrates core saturation effects
- Expected: longest execution times

## Testing

### Running Tests

```bash
# Affinity manager policy tests (platform-independent)
python -m unittest test_affinity_manager -v

# Module 5 & 6 tests (metrics, display, workload)
python -m unittest test_modules_5_and_6 -v
# (run from os-project_final/ directory)
```

### Test Coverage
- **25 tests** for affinity configurations and mask construction
- **11 tests** for workload functions, metrics, and display export
- All tests validated across multiple core counts and configurations

## Production Features

### Logging
- Structured logging with timestamps and log levels
- Module-specific loggers (`project31`, `affinity_manager`, etc.)
- Console output via `basicConfig`

### Error Handling
- `AffinityError` exceptions for OS API failures
- Input validation in all configuration functions
- Graceful degradation when `GetThreadAffinityMask` unavailable
- CLI argument validation

### Configuration Management
- `AffinityConfiguration` enum for type-safe config selection
- CLI-driven configuration with comprehensive help
- Platform detection (`is_windows()`) with appropriate warnings

### Output Export
- **CSV**: Structured benchmark data with per-run metrics
- **TXT**: Academic benchmark report with hardware specs, statistics, and observations
- Comparison summaries with speedup calculations vs baseline

## Key Implementations

### Affinity Configurations
- **Baseline**: Empty core mappings, OS discretion
- **Fixed**: `thread i -> core i` mapping, rejects `threads > cores`
- **Distributed**: Strided dispersion across all cores, interleave to differ from Fixed
- **Concentrated**: All threads on first 1-2 detected cores

### Windows API Integration
- `SetThreadAffinityMask` via ctypes for affinity application
- `GetThreadAffinityMask` for verification (with fallback)
- `GetLogicalProcessorInformation` for physical-to-logical core topology

### Workload Engine
- 3 deterministic CPU-intensive tasks
- GIL-release-friendly (crypto_hash uses `hashlib.sha256` which releases GIL)
- Configurable intensity per workload type
- Factory pattern via `get_workload_function()`

### Performance Metrics
- Wall-clock timing via `time.perf_counter()`
- Per-core CPU utilization sampling via `psutil.cpu_percent(percpu=True)`
- Statistical aggregation: mean, min, max, standard deviation
- Speedup calculation relative to baseline

## Academic Observations (from generated reports)

The system demonstrates expected behavior:
- **Concentrated** placement confines threads to restricted cores, increasing per-core saturation and execution time
- **Distributed** separation across logical processors achieves parallel execution with distinct core utilization profiles
- **Fixed** 1:1 pinning shows intermediate performance between concentrated and distributed
- Results are bounded to measured execution times and OS scheduling state; direct hardware metrics (cache misses, context switches) are not instrumented

## Development Notes

### Adding New Workloads
1. Add function to `workload.py` with signature `run_*_workload(thread_index, intensity) -> int`
2. Add `WorkloadType` enum value
3. Add case to `get_workload_function()` factory
4. Test under all 4 affinity configurations

### Adding New Configurations
1. Add value to `AffinityConfiguration` enum
2. Implement `generate_thread_affinity_mapping()` case
3. Update `main.py` configuration selection logic
4. Add test cases in `test_affinity_manager.py`

### Platform Considerations
- Full Windows API functionality requires Windows OS
- Non-Windows runs execute under default OS scheduling with appropriate warnings
- Mask construction and policy layer are platform-independent and testable on any OS

## Version & Maintenance

- **Phase**: 1 (Working Prototype, Modules 1-6)
- **Status**: Active development
- **Tests**: 36 total (25 + 11), all passing
- **Last updated**: Integration of pinkpixel-jpg/os-project_final repository

## License
This project is for educational purposes as part of a course assignment. Please refer to the original repository (`pinkpixel-jpg/os-project_final`) for licensing information.