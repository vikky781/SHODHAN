# Architecture

SHODHAN is a small C++20 library (`shodhan_core`) plus a command-line tool
(`shodhan`) and a test executable. This document describes the modules that
exist today and the rules the code follows.

## Module map

```
include/shodhan/   public headers
src/util/          status names, logger
src/model/         sparse matrix, LP model, model statistics
src/io/            MPS reader and writer
src/cli/           the `shodhan` command-line tool
tests/             unit tests, a header-only test harness, toy models
scripts/           helper scripts (dependency guard)
docs/              this file and the MPS conventions
data/              notes on where to download benchmarks (nothing committed)
```

Dependencies between modules point downwards only:

```
cli  ->  io  ->  model  ->  util
```

| Module | Main headers | What it does |
|--------|--------------|--------------|
| util   | `status.hpp`, `params.hpp`, `logger.hpp`, `constants.hpp`, `version.hpp` | `Status` enum, `Params` defaults, a leveled `Logger`, the infinity constant `kInf` |
| model  | `sparse_matrix.hpp`, `lp_model.hpp`, `model_stats.hpp` | CSC matrix (validation, triplet construction, CSR copy, SpMV), `LpModel`, descriptive statistics |
| io     | `mps.hpp` | MPS reader (fixed/free, optional `.gz`) and writer |
| cli    | (none public) | `shodhan info`, `shodhan solve` |

### Model conventions

- Everything is stored as `row_lower <= A x <= row_upper` and
  `col_lower <= x <= col_upper`. Infinity is stored as `+/-kInf` (1e30);
  `is_inf()` tests for it.
- `LpModel::quadratic` is reserved for the lower triangle of Q and stays
  empty until quadratic models are supported.
- Indices are 32-bit; a matrix may hold at most INT32_MAX nonzeros.

### Error handling

Reading never throws on bad input: `read_mps_*` returns a result with `ok`,
an `error` string (`file:line: message: 'offending text'`) and a list of
warnings. The solver entry point does not exist yet; `shodhan solve` reports
`Status::NotImplemented` and exits with code 2 rather than inventing a result.

## Portability

Code in `src/` and `include/` uses only the C++ standard library (no POSIX or
Win32 headers) and must build with GCC, Clang and MSVC. Warnings are enabled
at a high level (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, `/W4`) and
are errors in CI.

## Dependency policy

SHODHAN implements its numerical code itself.

Allowed:

- the C++ standard library, `std::thread`, OpenMP,
- optionally zlib, only to read `.gz` files (`SHODHAN_ENABLE_ZLIB`),
- later, the CUDA runtime.

Forbidden: HiGHS, SCIP, CLP/CBC/COIN-OR, GLPK, OR-Tools, SuiteSparse
(AMD/CHOLMOD/UMFPACK), Eigen, BLAS/LAPACK, cuDSS, cuSOLVER.

`scripts/check_deps.py` fails if a CMake file, an `#include` in `src/` or
`include/`, or the shared libraries linked by a built binary refer to any of
these. `--verbose` prints what was checked. CI runs it on Linux.

## Testing

`tests/test_harness.hpp` is a tiny header-only harness (`TEST_CASE`, `CHECK`,
`CHECK_EQ`, `CHECK_NEAR`, `REQUIRE`). `shodhan_tests [filter]` runs every
test whose name contains the filter. `tests/cli_check.py` drives the real
executable (exit codes and key output) and is registered with CTest when
Python 3 is available. The toy models in `tests/models/` and the strings in
`tests/mps_samples.hpp` are written for these tests; no benchmark instance is
reproduced.
