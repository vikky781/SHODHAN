# SHODHAN

A from-scratch LP/MILP/QP optimization solver core, written in C++20.

SHODHAN is being built step by step. At this stage it can read and write MPS
models and describe them; it cannot solve anything yet.

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

## Usage

```sh
shodhan info  model.mps    # summary: size, types, bounds, scaling indicators
shodhan solve model.mps    # prints the summary, then reports NotImplemented
shodhan --help
shodhan --version
```

Exit codes: 0 ok, 1 usage or read error, 2 not implemented.

## Status

| Feature                                              | Status              |
|------------------------------------------------------|---------------------|
| Core types (status, params, logger)                  | implemented         |
| CSC sparse matrix, LP model container                | implemented         |
| MPS reader (fixed and free format)                   | implemented         |
| MPS reader: `.gz` input via optional zlib            | implemented, not exercised by the tests |
| MPS writer (round-trips the models in the tests)     | implemented         |
| `shodhan info`                                       | implemented         |
| `shodhan solve`                                      | stub: reports `NotImplemented` |
| QPS files (QUADOBJ / QMATRIX)                        | not yet implemented |
| Presolve                                             | not yet implemented |
| LP solver (simplex)                                  | not yet implemented |
| LP solver (interior point)                           | not yet implemented |
| MILP branch-and-bound                                | not yet implemented |
| Convex QP                                            | not yet implemented |
| GPU acceleration                                     | not yet implemented |

## Dependency policy

SHODHAN is written from scratch. The only things it may use are:

- the C++ standard library, `std::thread` and OpenMP,
- optionally zlib, for reading `.gz` files,
- later, the CUDA runtime.

No third-party solver or numerical library is allowed. In particular: HiGHS,
SCIP, CLP/CBC/COIN-OR, GLPK, OR-Tools, SuiteSparse (AMD/CHOLMOD/UMFPACK),
Eigen, BLAS/LAPACK, cuDSS and cuSOLVER are forbidden. `scripts/check_deps.py`
enforces this in CI.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map,
[docs/MPS_FORMAT.md](docs/MPS_FORMAT.md) for the MPS conventions SHODHAN
follows, and [data/README.md](data/README.md) for where to obtain benchmark
instances (they are never committed).

## License

Apache-2.0, see [LICENSE](LICENSE).
