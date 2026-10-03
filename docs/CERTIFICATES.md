# Solution certificates

A certificate is a JSON file that lets a program with no access to any solver internals check a claim
about an LP: it is optimal, infeasible or unbounded (or, reserved for MILP, feasible). The checker
reads the ORIGINAL model file and the certificate and nothing else. `shodhan solve model.mps --write-cert
cert.json` writes one; KASAUTI (`verify/kasauti/`, see [KASAUTI.md](KASAUTI.md)) verifies it.

All names in a certificate are the MPS names of rows and columns, never indices, so the verifier does not
depend on any ordering.

## 1. Format

UTF-8 JSON, one object:

| Key | Meaning |
|-----|---------|
| `format` | the string `"shodhan-cert"` |
| `version` | integer, currently `1` |
| `solver` | `{name, version}` strings |
| `problem` | `{name, file_sha256, rows, cols, nnz, sense, n_integer}`: `file_sha256` is the SHA-256 (lower-case hex) of the exact bytes of the model file that was read; `sense` is `"min"` or `"max"`; `nnz` counts stored matrix entries |
| `status` | `optimal`, `infeasible`, `unbounded`, `feasible` or `other` |
| `claimed_objective` | the objective value the solver reports, in the model's own sense (omitted for `infeasible`, `unbounded`, `other`) |
| `tolerances` | `{primal_tol, dual_tol, kkt_tol}` the solver used |
| `attempts` | `{count, configuration}`: how many `LpSolver` attempts (fallbacks) were made and which configuration produced the result (`presolve+scaling`, `scaling`, `none`, ...) |
| certificate body | exactly one of the blocks below; absent when `status` is `other` |

Doubles are written with the shortest decimal that reads back to the identical double (`std::to_chars`), so a
reader that parses them as IEEE doubles recovers exactly what the solver had. Integers (counts) are JSON
integers. A value that is not finite is never written.

### Certificate bodies

- **`optimal`**: `x` (object, column name to value, nonzeros only) and `y` (object, row name to value, nonzeros
  only). `y` is in the convention of the internal minimization form: for a `max` model the verifier replaces
  the objective `c` by `-c` and checks as for a minimization problem (the same convention as
  [CONVENTIONS.md](CONVENTIONS.md)). Reduced costs are not in the file: the verifier derives `d = c - A^T y`.
- **`infeasible`**: `farkas`: `{y: {row name: multiplier}}`.
- **`unbounded`**: `point` (a primal feasible `x0`, nonzeros only) and `ray` (column name to value, nonzeros
  only). Both are required: a ray alone proves nothing without a feasible point.
- **`feasible`** (reserved for MILP): `x` only. The verifier checks feasibility and integrality and says that
  optimality is not certified.
- **`other`**: no body (the solver ended with `NumericalError`, a limit, ...). A certificate with status
  `other` certifies nothing and a verifier must report it as inconclusive.

## 2. Mathematics of the checks

Let the model be (after the `max` to `min` conversion `c := -c` when `sense` is `max`)

```
minimize  c^T x + offset     subject to   row_lo <= A x <= row_hi,   col_lo <= x <= col_hi
```

with infinite bounds written as `-inf`/`+inf`. The objective offset is the negative of the value in the
objective row of the RHS section (MPS convention, see MPS_FORMAT.md). All data are exact decimals in the file.

### Optimal: primal feasibility and a rigorous dual bound

1. **Primal feasibility.** From `x` compute the row activities `a_i x` exactly and the violations of the row and
   column bounds. The report gives the maximum absolute violation and the maximum relative one, where the
   relative violation of a bound is `violation / (1 + |bound|)`.
2. **Dual bound by weak duality.** Take any `y` and let `d = c - A^T y` (exact). For every feasible `x`,
   with `r = A x`:

   ```
   c^T x = y^T A x + d^T x = y^T r + d^T x
   ```

   Now `y_i r_i >= y_i * row_lo_i` if `y_i > 0` and `>= y_i * row_hi_i` if `y_i < 0`; likewise
   `d_j x_j >= d_j * col_lo_j` for `d_j > 0` and `>= d_j * col_hi_j` for `d_j < 0`. Therefore

   ```
   LB(y) = offset + sum_i (y_i > 0 ? y_i * row_lo_i : y_i * row_hi_i) + sum_j (d_j > 0 ? d_j * col_lo_j : d_j * col_hi_j)
   ```

   is a lower bound on the optimal value, **for any `y` whatsoever**: no sign condition and no tolerance is
   needed, only exact arithmetic. A term whose bound is infinite makes `LB = -inf` (the bound then proves
   nothing). The sign conventions of CONVENTIONS.md only decide which `y` gives a good bound.
3. **Primal objective.** `c^T x + offset` computed exactly is an upper bound on the optimum if `x` is feasible.
4. **Verdict.** The gap is `primal objective - LB`. `PASS_OPTIMAL` requires the primal violations to be within
   `primal_tol` (relative form above) and `gap <= gap_tol * (1 + |objective|)`. If `x` is exactly feasible the
   statement is rigorous: the optimum lies in `[LB, primal objective]`. If `x` is only feasible within
   tolerance, the report says so, and the claim is "optimal within the stated tolerances". The claimed objective
   in the file must match the exact one (in the model's sense) within tolerance.

### Infeasible: disjoint intervals

For multipliers `y` and `g = A^T y`, every feasible `x` gives `y^T A x` two descriptions:
`sum_i y_i r_i` with `r_i` in `[row_lo_i, row_hi_i]`, and `sum_j g_j x_j` with `x_j` in `[col_lo_j, col_hi_j]`.
So `y^T A x` lies in the intersection of

```
R = [ sum_i min(y_i row_lo_i, y_i row_hi_i) , sum_i max(y_i row_lo_i, y_i row_hi_i) ]
C = [ sum_j min(g_j col_lo_j, g_j col_hi_j) , sum_j max(g_j col_lo_j, g_j col_hi_j) ]
```

(an infinite endpoint where a nonzero coefficient meets an infinite bound). If `R` and `C` are disjoint the
model has no feasible point. The check is exact; endpoints are printed. Because the multipliers are doubles, a
coefficient `g_j` that is zero in theory can be a tiny nonzero in exact arithmetic and then meets an infinite
bound; the strict check fails in that case and the verifier reports it. The option `--farkas-zero-tol`
(default 0) lets the user drop coefficients below a relative tolerance, in which case the verdict is
`PASS_INFEASIBLE_TOL` and lists what was dropped (it is then not a rigorous proof).

### Unbounded: feasible point plus recession direction

1. `x0` is primal feasible (violations reported as above).
2. The ray `r` lies in the recession cone of the feasible region. Derivation: the feasible set is
   `{x : row_lo <= A x <= row_hi, col_lo <= x <= col_hi}`; `x0 + t r` stays feasible for all `t >= 0` iff
   `A r >= 0` in every row with a finite lower bound, `A r <= 0` in every row with a finite upper bound, and
   `r_j >= 0` for every column with a finite lower bound, `r_j <= 0` for every column with a finite upper bound.
   So an equality row needs `(A r)_i = 0`, a fixed or doubly bounded column needs `r_j = 0`, a free row or free
   column is unconstrained.
3. The objective strictly improves: `c^T r < 0` (after the `max` to `min` conversion).

With doubles in the file, `A r` is generally not exactly zero in an equality row. The check is exact in what it
computes; a row or column condition counts as satisfied if its violation is within `ray_tol` times the magnitude
of its terms (default `1e-9`), and the report states whether every condition held exactly (rigorous) or only
within tolerance.

### Feasible (reserved)

Feasibility within tolerance, integrality violation of every column listed as integer in the file, exact objective.
The verdict is `PASS_FEASIBLE` together with the line "optimality not certified".

### Solution files

A `.sol` file written by `shodhan solve --write-sol` (objective line, then `name value`) can be checked for
primal feasibility and objective only; it carries no optimality evidence.

## 3. What is and is not certified

Certified (for the model file as read): that the stated `x` is feasible within the reported violation; that the
optimum is not below `LB(y)`; that an infeasible model has no feasible point (rigorously when the intervals are
disjoint exactly); that an unbounded model has a feasible point and an improving recession direction.

Not certified:
- **Optimality of MILP solutions**: bounds from branch and bound are not part of version 1.
- **Presolve internals**: nothing about the reduced model; certificates are always stated in the original space.
- **Floating-point behaviour of the solver**: the solver's arithmetic is not examined; only its output is.
- **Tolerance-level claims**: where a check passes only within a tolerance, the verifier says so.
- **That the file is the model you intended**: only that it is the file whose SHA-256 is in the certificate.

## 4. Trust model

To trust a `PASS` you must trust (1) the verifier's MPS parser (written independently of the solver's reader
from the MPS conventions in this repository), (2) Python's `fractions.Fraction` arithmetic and standard library,
and (3) the input file itself. You do not have to trust the solver, its presolve, its scaling, its LU
factorization, or the C++ certificate writer: a wrong certificate is simply rejected.

## 5. Tolerance conventions

Defined by this project (`--primal-tol 1e-6`, `--gap-tol 1e-6`, `--ray-tol 1e-9`, relative forms as above).
They are not claimed to be identical to the conventions of any other solution checker.
