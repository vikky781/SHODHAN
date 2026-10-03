# Architecture

SHODHAN is a small C++20 library (`shodhan_core`) plus a command-line tool
(`shodhan`) and a test executable. This document describes the modules that
exist today and the rules the code follows.

## Module map

```
include/shodhan/   public headers
src/util/          status names, logger
src/model/         sparse matrix, LP model, model statistics, KKT checker
src/scaling/       row/column/objective scaling and unscaling
src/presolve/      presolve reductions and postsolve
src/io/            MPS reader and writer
src/linalg/        sparse LU of simplex bases, FTRAN/BTRAN, Forrest-Tomlin update
src/lp/            simplex engine (dual, primal cleanup) and the LP pipeline
bench/             LP set runner (Python, standard library only)
src/cli/           the `shodhan` command-line tool
tests/             unit tests, a header-only test harness, toy models
tests/support/     TEST-ONLY code: dense reference LP solver, dense LU oracle, random LP and basis generators
scripts/           helper scripts (dependency guard)
docs/              architecture, conventions, presolve, MPS conventions
data/              notes on where to download benchmarks (nothing committed)
```

Dependencies between modules point downwards only:

```
cli  ->  presolve  ->  scaling  ->  model  ->  util
cli  ->  io  ->  model  ->  util
cli  ->  lp  ->  presolve, scaling, linalg  ->  model  ->  util
```

| Module | Main headers | What it does |
|--------|--------------|--------------|
| util   | `status.hpp`, `params.hpp`, `logger.hpp`, `constants.hpp`, `version.hpp` | `Status` enum, `Params` defaults, a leveled `Logger`, the infinity constant `kInf` |
| model  | `sparse_matrix.hpp`, `lp_model.hpp`, `model_stats.hpp` | CSC matrix (validation, triplet construction, CSR copy, SpMV), `LpModel`, descriptive statistics |
| model  | `solution.hpp`, `kkt.hpp` | `Solution` (x, y, d, objective) and `check_kkt`, independent of any presolve code |
| scaling | `scaling.hpp` | geometric-mean + equilibration scaling with exact power-of-two factors; `unscale_solution` |
| presolve | `presolve.hpp` | nine reductions, each with a postsolve record; `postsolve` recovers x, y, d |
| io     | `mps.hpp` | MPS reader (fixed/free, optional `.gz`) and writer |
| linalg | `sparse_work.hpp`, `basis_factor.hpp` | `SparseWork` (dense array + index list), `BasisFactor`: LU of a basis of `[A | -I]`, `ftran`/`btran`, `repair`, Forrest-Tomlin `update` (see `docs/LU.md`) |
| lp     | `simplex_engine.hpp`, `lp_solver.hpp`, `rays.hpp` | `SimplexEngine` (dual simplex, primal cleanup, warm start), `LpSolver` pipeline, Farkas and ray checkers (docs/SIMPLEX.md) |
| cli    | (none public) | `shodhan info`, `shodhan presolve`, `shodhan solve`, `shodhan factor-bench` |

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
warnings. The solver entry point does not exist yet; `shodhan solve` runs presolve and
scaling, reports a status that presolve itself proves (infeasible, unbounded, solved), and
otherwise reports `Status::NotImplemented` and exits with code 2 rather than inventing a result.

## Test-only code

`tests/support/` holds a dense two-phase simplex, a dense LU with partial pivoting and seeded random LP and
basis generators. They exist so that presolve, scaling, postsolve and the basis factorization can be checked
against independent code before a real LP engine exists. They are compiled into the test executable only; `shodhan_core` and the CLI never see them, and
`scripts/check_deps.py` fails if `src/` or `include/` include them or if the CMake definition of the
library or the CLI mentions `tests/`.

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
