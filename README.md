# SHODHAN

A from-scratch LP/MILP/QP optimization solver core, written in C++20.

SHODHAN is being built step by step. At this stage it can read and write MPS
models, describe them, scale them, and presolve LPs and MIPs with a postsolve
that recovers primal and dual solutions. It also has the sparse LU factorization
of simplex bases (with FTRAN/BTRAN and the Forrest-Tomlin update) that a simplex
solver sits on. LPs can now be solved end to end (presolve, scaling, dual simplex with a primal
cleanup, KKT check on the original model), and MILPs by branch and bound with reliability branching and primal
heuristics, a MIP presolve (propagation, probing, clique table) and a root cut loop (Gomory, MIR, cover, clique, implied
bound) (all tested on small generated instances only). Every LP result can be
written as a certificate and checked by KASAUTI, an independent verifier in standard-library Python
that works in exact arithmetic.

## Build

Requirements: a C++20 compiler (GCC, Clang or MSVC) and CMake 3.20 or newer.
Python 3 is optional and is used by the CLI test, the dependency guard and the KASAUTI tests
(`ctest` registers a visible skip when it is missing).

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

The MIP tests have their own counts: `SHODHAN_MIP_SEED_COUNT` (MIP presolve equivalence, default 1000),
`SHODHAN_CUT_SEEDS` (cut validity, models per family, default 90 over 8 families) and `SHODHAN_CUT_PIPE_SEEDS` (cuts in the whole
pipeline, per family, default 125). The sanitizer test preset (`ctest --preset debug-sanitizers`, used in CI) sets them to
150, 12 and 20, because the sanitizer build is several times slower; the release jobs use the defaults.

## Usage

```sh
shodhan info  model.mps    # summary: size, types, bounds, scaling indicator
shodhan presolve model.mps [--write-presolved out.mps] [--no-dual-needed] [--mip]
shodhan solve model.mps [--no-presolve] [--no-scaling] [--no-perturb] [--time-limit s]
                           [--iter-limit n] [--write-sol path] [--write-cert path] [--verbose]
                           [--mip-gap g] [--mip-abs-gap g] [--node-limit n] [--seed s]
                           [--branching reliability|pseudocost|mostfrac|first]
                           [--node-select bestbound|depth|bestestimate] [--heuristics on|off]
                           [--presolve on|off] [--probing on|off] [--cuts on|off] [--cut-rounds n]
shodhan dump-model model.mps   # canonical text of the parsed model (compares readers)
shodhan factor-bench model.mps [--threshold u] [--max-updates k]
shodhan --help
shodhan --version
```

`presolve` prints the status, the sizes before and after, a counter per reduction, the
coefficient ratio before and after scaling, and the time taken.

`solve` runs presolve and scaling, then the dual simplex with a primal cleanup, and prints the statistics and
the result. `--write-cert` also writes a certificate (see below). Models with integer columns are solved by branch
and bound ([docs/MIP.md](docs/MIP.md)): the output has the status, objective, best bound, gap, nodes, LP iterations, time
and which heuristics found solutions; every reported solution was verified against the original model. A quadratic
objective is reported as `NotImplemented` (exit code 2). Nothing is faked.

`factor-bench` builds a crash basis (structural columns in ascending nonzero-count order, logical
columns substituted for any rank deficiency), factorizes it, prints the sizes, fill and pivot counts, runs
50 random solves with sparse right-hand sides (counts per hypersparse/dense path and residuals) and a short
Forrest-Tomlin run. It is a developer diagnostic, not a benchmark, and says nothing about solver performance.

Exit codes: 0 ok (optimal for `solve`, for a MILP within the gap), 1 usage, read or write error, 2 not implemented, 3
`solve` ended infeasible, unbounded or numerically, 4 `solve` stopped at a time, iteration or node limit (for a MILP the
output says whether an incumbent exists). `solve` reports `Optimal` only after the KKT check on the
original model passed, and `Infeasible`/`Unbounded` only with a verified certificate.

## Certificates and the independent verifier

`shodhan solve model.mps --write-cert model.cert.json` writes a JSON certificate (format:
[docs/CERTIFICATES.md](docs/CERTIFICATES.md)) and prints the command that checks it. KASAUTI
(`verify/`, see [docs/KASAUTI.md](docs/KASAUTI.md)) has its own MPS parser and no dependency on the solver; it
verifies primal feasibility, a weak-duality dual bound, a Farkas proof or a feasible point plus recession
direction, in exact rational arithmetic:

```sh
shodhan solve tests/models/tiny_lp.mps --write-cert tiny.cert.json
cd verify && python -m kasauti ../tests/models/tiny_lp.mps ../tiny.cert.json
```

Excerpt of the output:

```
primal objective (model sense): 94
rigorous dual upper bound LB(y): 94 (valid for every feasible x by weak duality)
gap |primal objective - bound|: 0
the optimum lies in [94, 94]
VERDICT: PASS_OPTIMAL  (rigorous: exact arithmetic)
```

Multipliers are floating-point numbers, so many correct answers have no strict exact bound. KASAUTI then
reports `PASS_OPTIMAL_TOL` and says it is **tolerance-checked, not a proof**; the tolerance involved
(`--dual-zero-tol 1e-9`) was chosen after seeing a failing case. On the 426-certificate test corpus every
certificate passes in exact and in float mode, 120 of them rigorously and 306 tolerance-checked.

## Status

| Feature                                              | Status              |
|------------------------------------------------------|---------------------|
| Core types (status, params, logger)                  | implemented         |
| CSC sparse matrix, LP model container                | implemented         |
| MPS reader (fixed and free format)                   | implemented         |
| MPS reader: `.gz` input via optional zlib (`scripts/gunzip_mps.py` without it) | implemented; read in CI with zlib enabled (on a tiny model) |
| MPS writer (round-trips the models in the tests)     | implemented         |
| KKT checker (primal/dual feasibility, gap)           | implemented         |
| Scaling (geometric + equilibration, exact powers of two) | implemented     |
| LP presolve: empty row/column, fixed column, singleton row, redundant row, forcing row, doubleton equation, dual fixing | implemented |
| MIP-safe presolve (integer rounding and bound tightening) | implemented    |
| Postsolve of primal values and duals (all reductions, including forcing rows) | implemented |
| Sparse LU of simplex bases (singleton + Markowitz pivoting), FTRAN/BTRAN with a hypersparse path, basis repair | implemented |
| Forrest-Tomlin basis update                          | implemented         |
| `shodhan info`, `shodhan presolve`                   | implemented         |
| `shodhan factor-bench` (developer diagnostic for the LU) | implemented     |
| LP dual simplex (bound flipping, Harris, steepest edge, perturbation, phase 1, primal cleanup), Farkas and ray certificates | implemented; tested on generated models only |
| `shodhan solve`                                      | LPs: full pipeline; MILPs: branch and bound; quadratic objectives: `NotImplemented` |
| Certificates (`--write-cert`): own SHA-256, own JSON writer, optimal / infeasible / unbounded bodies | implemented |
| Weak-duality bound of the multipliers in the LP acceptance check, with a `rigorous` flag | implemented |
| KASAUTI verifier (`verify/`, standard-library Python, exact `Fraction` and float modes) | implemented; tested on generated models only |
| `shodhan dump-model` (differential test of the MPS readers) | implemented |
| Opt-in stress harness (`SHODHAN_BUILD_STRESS`, not part of ctest) | implemented |
| `bench/run_set.py` (LPs and MILPs; `run_lp_set.py` is a wrapper), `bench/gen_mip.py` | implemented; no Netlib or MIPLIB files were available, so nothing real was run |
| QPS files (QUADOBJ / QMATRIX)                        | not yet implemented |
| LP solver (interior point)                           | not yet implemented |
| MILP branch and bound: node tree, best-bound/depth/best-estimate selection with plunging, objective-integrality pruning | implemented; tested on small generated instances only |
| Branching: most fractional, first index, pseudocost, reliability (strong branching on engine copies) | implemented |
| Primal heuristics: trivial, simple rounding, fractional and coefficient diving, Feasibility Jump | implemented |
| MILP certificates (feasible point, exact integrality; the bound is claimed, not verified) | implemented |
| MIP presolve: bound propagation, coefficient tightening, probing, parallel rows, duplicate and dominated columns, clique table | implemented; tested on generated instances only |
| Root cut loop: Gomory mixed-integer, MIR with aggregation, lifted covers, clique, implied-bound cuts; `add_rows`/`remove_rows` on the engine | implemented; tested on generated instances only |
| Restarts, cuts at tree nodes, zero-half cuts, RENS/RINS, multithreading | not yet implemented |
| Convex QP                                            | not yet implemented |
| GPU acceleration                                     | not yet implemented |

Hypersensitive LPs: wide seed 450741 may end in an honest `NumericalError` on some platforms. The two earlier open findings
(seeds 450165 and 433) are fixed; details and evidence in [docs/KASAUTI.md](docs/KASAUTI.md).

Testing note: LP presolve is checked against a small dense simplex that lives in
`tests/support/` and is a test oracle only (it is not part of the library). On extreme
coefficient ranges that oracle can report `NumericalError`; the tests count those seeds
explicitly and cap how many are allowed rather than ignoring them.

## Dependency policy

SHODHAN is written from scratch. The only things it may use are:

- the C++ standard library, `std::thread` and OpenMP,
- optionally zlib, for reading `.gz` files,
- later, the CUDA runtime.

The verifier in `verify/` may use only the Python standard library; `scripts/check_verify_independence.py`
enforces that, and that it does not refer to the C++ sources.

No third-party solver or numerical library is allowed. In particular: HiGHS,
SCIP, CLP/CBC/COIN-OR, GLPK, OR-Tools, SuiteSparse (AMD/CHOLMOD/UMFPACK/KLU/CSparse),
Eigen, BLAS/LAPACK, cuDSS and cuSOLVER are forbidden. `scripts/check_deps.py`
enforces this in CI, and also checks that test-only code (`tests/support/`) is not
linked into `shodhan_core`.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map,
[docs/CONVENTIONS.md](docs/CONVENTIONS.md) for the sign and scaling conventions,
[docs/PRESOLVE.md](docs/PRESOLVE.md) for each reduction and its postsolve,
[docs/MIP.md](docs/MIP.md) for the branch and bound and [docs/CUTS.md](docs/CUTS.md) for the cutting planes,
[docs/LU.md](docs/LU.md) for the basis factorization and its update,
[docs/SIMPLEX.md](docs/SIMPLEX.md) for the simplex engine and the LP pipeline,
[docs/CERTIFICATES.md](docs/CERTIFICATES.md) for the certificate format and the checks,
[docs/KASAUTI.md](docs/KASAUTI.md) for the verifier, its tests and the adjudication of disputed LPs,
[docs/MPS_FORMAT.md](docs/MPS_FORMAT.md) for the MPS conventions SHODHAN
follows, and [data/README.md](data/README.md) for where to obtain benchmark
instances (they are never committed).

## License

Apache-2.0, see [LICENSE](LICENSE).
