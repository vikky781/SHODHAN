# SHODHAN

A from-scratch LP/MILP/QP optimization solver core, written in C++20.

SHODHAN is being built step by step. At this stage it can read and write MPS
models, describe them, scale them, and presolve LPs and MIPs with a postsolve
that recovers primal and dual solutions; it cannot solve anything yet.

## Build

Requirements: a C++20 compiler (GCC, Clang or MSVC) and CMake 3.20 or newer.
Python 3 is optional and is used by one test and by the dependency guard.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

With a multi-config generator (Visual Studio, Xcode) add `--config Release` to
the build command and `-C Release` to `ctest`. If no build type is given,
Release is used.

### Options

| Option                | Default | Meaning                                              |
|-----------------------|---------|------------------------------------------------------|
| `SHODHAN_BUILD_TESTS` | ON      | Build the unit tests and register them with CTest    |
| `SHODHAN_ENABLE_ZLIB` | OFF     | Read `.gz` input; requires zlib (`find_package`)     |
| `SHODHAN_WERROR`      | OFF     | Treat compiler warnings as errors (used in CI)       |
| `SHODHAN_SANITIZE`    | OFF     | AddressSanitizer + UBSan (GCC/Clang only)            |

### Presets

`CMakePresets.json` defines `release` and `debug-sanitizers` (using presets
needs CMake 3.21; the project itself builds with 3.20):

```sh
cmake --preset debug-sanitizers
cmake --build --preset debug-sanitizers
ctest --preset debug-sanitizers
```

### Larger test runs

The randomized presolve tests use fixed seed ranges by default. To stress a wider
range (a failing seed is printed):

```sh
SHODHAN_SEED_FIRST=1 SHODHAN_SEED_COUNT=6000 build/shodhan_tests presolve
```

## Usage

```sh
shodhan info  model.mps    # summary: size, types, bounds, scaling indicator
shodhan presolve model.mps [--write-presolved out.mps] [--no-dual-needed] [--mip]
shodhan solve model.mps    # presolve + scaling, then reports NotImplemented
shodhan --help
shodhan --version
```

`presolve` prints the status, the sizes before and after, a counter per reduction, the
coefficient ratio before and after scaling, and the time taken.

`solve` runs presolve and scaling and prints their statistics. If presolve alone proves the
model infeasible or unbounded, or solves it completely, that is reported (exit code 0); otherwise
it reports `NotImplemented` (exit code 2) because there is no LP engine yet. Nothing is faked.

Exit codes: 0 ok, 1 usage, read or write error, 2 not implemented.

## Status

| Feature                                              | Status              |
|------------------------------------------------------|---------------------|
| Core types (status, params, logger)                  | implemented         |
| CSC sparse matrix, LP model container                | implemented         |
| MPS reader (fixed and free format)                   | implemented         |
| MPS reader: `.gz` input via optional zlib            | implemented, not exercised by the tests |
| MPS writer (round-trips the models in the tests)     | implemented         |
| KKT checker (primal/dual feasibility, gap)           | implemented         |
| Scaling (geometric + equilibration, exact powers of two) | implemented     |
| LP presolve: empty row/column, fixed column, singleton row, redundant row, forcing row, doubleton equation, dual fixing | implemented |
| MIP-safe presolve (integer rounding and bound tightening) | implemented    |
| Postsolve of primal values and duals (all reductions, including forcing rows) | implemented |
| `shodhan info`, `shodhan presolve`                   | implemented         |
| `shodhan solve`                                      | presolve + scaling, then `NotImplemented` |
| QPS files (QUADOBJ / QMATRIX)                        | not yet implemented |
| LP solver (simplex)                                  | not yet implemented |
| LP solver (interior point)                           | not yet implemented |
| MILP branch-and-bound                                | not yet implemented |
| Convex QP                                            | not yet implemented |
| GPU acceleration                                     | not yet implemented |

Testing note: LP presolve is checked against a small dense simplex that lives in
`tests/support/` and is a test oracle only (it is not part of the library). On extreme
coefficient ranges that oracle can report `NumericalError`; the tests count those seeds
explicitly and cap how many are allowed rather than ignoring them.

## Dependency policy

SHODHAN is written from scratch. The only things it may use are:

- the C++ standard library, `std::thread` and OpenMP,
- optionally zlib, for reading `.gz` files,
- later, the CUDA runtime.

No third-party solver or numerical library is allowed. In particular: HiGHS,
SCIP, CLP/CBC/COIN-OR, GLPK, OR-Tools, SuiteSparse (AMD/CHOLMOD/UMFPACK),
Eigen, BLAS/LAPACK, cuDSS and cuSOLVER are forbidden. `scripts/check_deps.py`
enforces this in CI, and also checks that test-only code (`tests/support/`) is not
linked into `shodhan_core`.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map,
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) for the sign and scaling conventions,
[docs/PRESOLVE.md](docs/PRESOLVE.md) for each reduction and its postsolve,
[docs/MPS_FORMAT.md](docs/MPS_FORMAT.md) for the MPS conventions SHODHAN
follows, and [data/README.md](data/README.md) for where to obtain benchmark
instances (they are never committed).

## License

Apache-2.0, see [LICENSE](LICENSE).
