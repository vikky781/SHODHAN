# KASAUTI: the independent certificate verifier

KASAUTI (`verify/kasauti/`) checks a solver's claim about an LP or a convex QP from the model file and a certificate, using
nothing but the Python standard library. It has no access to the solver: it contains its own MPS parser (written
from the conventions in [MPS_FORMAT.md](MPS_FORMAT.md), not translated from the C++ reader) and does its
arithmetic in exact rationals (`fractions.Fraction`) or, for very large models, in compensated floating point.
The certificate format and the mathematics of every check are in [CERTIFICATES.md](CERTIFICATES.md).

## Use

```sh
shodhan solve model.mps --write-cert model.cert.json
python -m kasauti model.mps model.cert.json            # run from verify/, or with PYTHONPATH=verify
```

The solve command prints the certificate path and the verification command. Options:

| Option | Meaning |
|--------|---------|
| `--mode auto\|exact\|float` | exact (Fractions) or float; `auto` is exact up to 200000 nonzeros and then float, with an explicit warning that the result is not an exact proof |
| `--primal-tol 1e-6` | relative bound violation accepted |
| `--gap-tol 1e-6` | relative gap between primal objective and dual bound |
| `--ray-tol 1e-9` | relative violation of the recession-cone conditions |
| `--farkas-zero-tol 1e-12` | retry an infeasibility proof dropping coefficients below this (0 = never) |
| `--dual-zero-tol 1e-9` | retry an optimality bound dropping multipliers below this (0 = never) |
| `--psd-cap 120` | largest number of columns for the exact positive-semidefiniteness test of Q; above it convexity is "not verified" and the verdict tolerance-level |
| `--sol FILE` | check a `.sol` file (feasibility and objective only) |
| `--report out.json` | machine-readable report |

Exit codes: `0` PASS, `1` FAIL, `2` INCONCLUSIVE (status `other`, unsupported or malformed input, usage error).

## Verdicts and what they prove

| Verdict | Meaning |
|---------|---------|
| `PASS_OPTIMAL` | exact mode and an exactly feasible point: the optimum lies in `[bound, objective]`. A proof. |
| `PASS_OPTIMAL_TOL` | the strict dual bound was `-infinity` because of tiny wrong-signed multipliers; a bound after dropping them agrees with the objective. **Tolerance-checked, not a proof.** |
| `PASS_INFEASIBLE`, `PASS_INFEASIBLE_TOL` | disjoint intervals, exactly / after dropping tiny coefficients |
| `PASS_UNBOUNDED`, `PASS_UNBOUNDED_TOL` | feasible point and improving recession direction, exactly / within tolerance |
| `PASS_FEASIBLE` | a MILP result: feasibility, exact integrality and the exact objective; any claimed bound is reported as "bound NOT verified"; optimality not certified |
| `FAIL` | a check failed, the hash does not match, or the certificate is malformed |

A verdict is called rigorous only if it has no `_TOL` suffix, was computed in exact mode, and (for optimal and
unbounded claims) the point is exactly feasible. The final line of the output says "rigorous: exact arithmetic"
or "tolerance-checked, not a proof". Corpus and test summaries always print both counts.

The default `--dual-zero-tol 1e-9` was chosen after seeing a failing case, not derived in advance: with the
first guess, `1e-12`, a correct answer whose smallest wrong-signed multiplier was `1.6e-12` of the largest one
failed. `1e-9` is 1000 times smaller than the solver's own dual tolerance.

Example (`tests/models/tiny_lp.mps`, a maximization):

```
primal objective (model sense): 94
rigorous dual upper bound LB(y): 94 (valid for every feasible x by weak duality)
gap |primal objective - bound|: 0
the optimum lies in [94, 94]
VERDICT: PASS_OPTIMAL  (rigorous: exact arithmetic)
```

## Independence

`scripts/check_verify_independence.py` fails if any file in `verify/` imports a module outside the standard
library (or the verifier's own packages) or mentions the C++ sources. The Python tests run the solver only as an
executable (`SHODHAN_EXE`) to produce inputs; no code is shared.

## What is tested

All of this runs under `ctest` (and the Python parts under `python -m unittest discover -s tests -t .` in `verify/`):

- **Unit tests**: the parser (every MPS feature, free and fixed format), every check, every verdict, malformed
  certificates.
- **Differential parser test**: 300 generated models covering every MPS feature are read by the C++ reader and
  by KASAUTI (`shodhan dump-model`); the numbers must agree as doubles (ranged-row bounds: a difference of up
  to `4 * 2^-52` times the larger of rhs and range is allowed, because C++ rounds `rhs + |range|` in double). Result: 300 identical, 0 mismatches. It found
  a real KASAUTI bug (a fixed-format number longer than the field was silently truncated).
- **Corpus integration test**: 426 seeded LPs (60 per family: degenerate, free variables, ranged rows, boxed,
  wide coefficients, infeasible, unbounded, plus 6 special models) and 150 seeded MILPs (15 per family, see
  [MIP.md](MIP.md); every fifth stopped by a node limit of 2) are written to MPS, read back, solved by the full
  pipeline, and each certificate is verified in exact and in float mode. Result: 576 certificates; the 546 that are
  expected to pass pass in both modes, the 30 that certify nothing (a MILP infeasible by branching, a MILP with an
  unbounded relaxation, a node-limited run without an incumbent) are INCONCLUSIVE in both modes, and the two modes
  agree on all 576. 237 passes are rigorous (for a MILP: feasibility and integrality exact; optimality is never
  claimed) and 309 are tolerance-checked. The attempts field shows 291 LP certificates from the first configuration
  and 135 from the second (exactly the 60 infeasible and 75 unbounded LPs, because presolve cannot certify those
  statuses in its reduced model) and 150 `branch-and-bound` entries.
- **Mutation tests**: valid certificates are corrupted (x, y, Farkas multipliers, ray, point, claimed objective,
  hash, row count, status), and an oracle written independently of the checker (plain `Fraction` code) classifies
  each mutation as harmful (violation or gap of at least `1e-4` relative, a destroyed proof), benign (at most
  `1e-8`) or gray. For MILP certificates the mutations break integrality (240), a row or bound (177 harmful), the
  claimed objective (80), the claimed bound (80) and claim certified optimality (80). Result: 2378 of 2378 harmful
  mutations rejected; 565 benign mutations accepted, none rejected; 112 gray-zone mutations skipped (they depend on
  the verifier's tolerances). One mutation that looked harmful was not: relabelling a feasible certificate as
  `optimal` passed once because that point really is LP-optimal with zero multipliers (a true claim); the status
  mutation now relabels to `unbounded`, which a feasible certificate can never satisfy.

## Adjudication of disputed LPs

The solver's pipeline and the dense test oracle disagreed on some wide-coefficient LPs. Each was solved
exactly (rational arithmetic on the exact value of every double in the model file, Bland's rule) and the exact
answer was verified by KASAUTI in strict mode (`--dual-zero-tol 0`), which gives an exact optimum or an exact
point and ray. Results (`tests/support/adjudicated_seeds.hpp`):

| Wide seed | Exact answer | Pipeline | Oracle | Verdict |
|-----------|--------------|----------|--------|---------|
| 453507 | optimal, -18266017920876808 | optimal, relative error 4e-16 | unbounded | oracle wrong |
| 450741 | optimal, -11.437507350760493 | optimal, relative error 4.8e-6 | relative error 9e-2 | within the conditioning limit (sum of the multipliers about 6e9); oracle further off |
| 450835 | optimal, -9.3281250009935022 | optimal, relative error 2.9e-10 | relative error 1.6e-4 | oracle wrong |
| 453961 | optimal, 20.123667009608404 | optimal, relative error 5.5e-9 | relative error 0.19 | oracle wrong |
| 451287 | unbounded | was NumericalError, fixed: unbounded | unbounded | engine bug fixed |
| 450165 | unbounded | was NumericalError, fixed: unbounded (point from a zero-cost feasibility solve, ray cleaned) | unbounded | engine bug fixed; KASAUTI `PASS_UNBOUNDED_TOL` |
| 433 | optimal, 7.4272773546400339 | was optimal with relative error 3.9e-6, fixed: error 5e-14 | correct | engine bug fixed; KASAUTI `PASS_OPTIMAL_TOL`, agreement 6.7e-6 before, 4.2e-13 after |
| 1999 | infeasible by far less than any tolerance | optimal within tolerance | optimal | tolerance-level agreement |

The pipeline's strict certificates for the optimal cases all `FAIL` in strict mode (the strict dual bound is
`-infinity`: the multipliers are floating-point numbers, 1e-17 to 1e-14 of their scale for the offenders). That
is expected and is why `PASS_OPTIMAL_TOL` exists; the exact solve is what decides who is right.

Fixed: the point of an unbounded certificate was the incrementally updated primal vector, which drifts on
ill-conditioned bases (seed 451287: rows violated by 1e-2 relative); it is now taken from a fresh factorization.

Fixed since: seed 450165 ended as `NumericalError` because the engine point came from a basis too ill-conditioned to
satisfy an equality row better than 2.3e-6 relative; it is now recomputed by a zero-cost feasibility solve at
`primal_tol 1e-10` (violation 0) and the ray is normalized and cleared of rounding noise. KASAUTI also measured ray row
violations against the magnitude of the row's nonzero terms under the ray, which is exactly 1 for a one-term row; it now
uses the normwise `||a_i||_1 ||r||_inf` like the columns. Seed 433 was accepted with a relative objective error of
3.9e-6 because polishing at 1e-13 took the dual-simplex branch, which shifts the costs of wrong-signed reduced costs
and hid a 2e-7 dual infeasibility; polishing now restores the costs before each check and alternates primal and dual
passes (up to four).

Evidence (generated with `shodhan_stress wide N N --emit-mps`, certificates by `shodhan solve --write-cert`):

| Seed | Before (commit d097fc0) | After |
|------|-------------------------|-------|
| 450165 | `NumericalError`, certificate status `other`, KASAUTI `INCONCLUSIVE` | `Unbounded`, ray rate -3.625, KASAUTI `PASS_UNBOUNDED_TOL` (strict and default mode) |
| 433 | objective 7.4273105236 (exact 7.4272773546400339), KASAUTI `PASS_OPTIMAL_TOL` with agreement 6.7e-6 | objective 7.42727735464, `PASS_OPTIMAL_TOL` with agreement 4.2e-13 |

In strict mode (`--dual-zero-tol 0`) seed 433 is `FAIL` before and after: a floating-point multiplier of 2.8e-8 sits on
a column without an upper bound, so no rigorous dual bound exists; that is the documented limit of strict mode, not a
defect of the answer. The unbounded certificate of 450165 is tolerance-checked (cone conditions hold at 1e-9), not
exact. Seed 450741 stays a hypersensitive case: the pipeline may return an honest `NumericalError` on some platforms.

## Quadratic objectives

The parser reads `QUADOBJ` and `QMATRIX` independently of the C++ reader (the differential test compares them on
generated QPS files). The optimal check uses `d = c + Q' x - A^T y` and the first-order bound of
[QP.md](QP.md), after an exact test that Q' is positive semidefinite. `FAIL` if it is not. Where the test cannot be
run (cap, float mode) the verdict is `PASS_OPTIMAL_TOL` with the note "convexity of Q is not verified". Unbounded QP
certificates are INCONCLUSIVE. Tests: `verify/tests/test_qp_checks.py`, `verify/tests/test_qp_mutations.py` (every
harmful corruption of a QP certificate must be rejected) and the QP part of the corpus.

## Pooling certificates

`python -m kasauti spec.pool cert.json` verifies a pooling certificate ([POOLING.md](POOLING.md), [CERTIFICATES.md](CERTIFICATES.md)).
The `.pool` file is read by `verify/kasauti/pool.py`, which has its own parser written from POOLING.md section 2 (a differential test compares
it with the C++ parser on 720 generated and damaged texts), and the relative residuals of the ORIGINAL nonlinear model (POOLING.md section 3)
are evaluated in exact rational arithmetic (`--mode float` for floats). `--pool-tol` (default 1e-6) is the largest accepted residual.

| Verdict | Meaning |
|---------|---------|
| `PASS_FEASIBLE` | every residual of the nonlinear model is exactly zero (exact mode) |
| `PASS_FEASIBLE_TOL` | every residual is within `--pool-tol`: the nonlinear constraints hold within the stated tolerance |
| `FAIL` | a residual above the tolerance, a wrong objective claim, a claimed bound below the objective, a claim of optimality, or a hash mismatch |
| `INCONCLUSIVE` | status `other`, an unparsable `.pool` file, or an unsupported status |

Neither verdict certifies optimality, and the McCormick bound claimed in the certificate is **not verified**: the output says so explicitly.
Tests: `verify/tests/test_pool.py` (hand-made certificates, a corpus of at least 100 certificates written by `shodhan pool` on seeded
SYNTHETIC instances verified in exact and float mode, mutation tests judged by an oracle written independently from the definitions,
and the parser differential test).

## Limits

- Float mode cannot prove that Q is positive semidefinite.

- Exact mode is used up to 200000 nonzeros; above that the float mode is not a proof.
- LP optimality, infeasibility and unboundedness are proved. For a MILP only feasibility, integrality and the objective
  value are verified; its optimality (the bound) is out of scope and is reported as not verified.
- Unsupported MPS input (sections the verifier does not implement, semi-continuous bounds, bounds that make
  the model ill-posed such as an infinite `FX`) gives `INCONCLUSIVE`, never a pass.
- The verifier checks the model file as bytes (SHA-256) and as parsed by its own reader; it cannot know that the
  file is the model you meant to solve.
