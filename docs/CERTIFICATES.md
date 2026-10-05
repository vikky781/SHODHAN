# Solution certificates

A certificate is a JSON file that lets a program with no access to any solver internals check a claim
about an LP: it is optimal, infeasible or unbounded (or, for a MILP result, a feasible point whose objective is claimed but not proved optimal). The checker
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
| `problem` | `{name, file_sha256, rows, cols, nnz, sense, n_integer}`: `file_sha256` is the SHA-256 (lower-case hex) of the exact bytes of the model file that was read; `sense` is `"min"` or `"max"`; `nnz` counts stored matrix entries; for a model with a quadratic objective also `quadratic` (true) and `q_nnz` (stored entries of the lower triangle of Q), absent otherwise ([QP.md](QP.md)) |
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
  [CONVENTIONS.md](CONVENTIONS.md)). Reduced costs are not in the file: the verifier derives `d = c + Q x - A^T y` (`Q = 0` for an LP).
  Also `dual_bound`, the solver's own claim about its multipliers: `{rigorous, available}` and, when a bound
  exists, `{value, dropped, gap_rel}`. It is a claim, not evidence (the verifier recomputes everything), but a
  claim of `rigorous: true` that exact arithmetic refutes makes the certificate FAIL.
- **`infeasible`**: `farkas`: `{y: {row name: multiplier}}`.
- **`unbounded`**: `point` (a primal feasible `x0`, nonzeros only) and `ray` (column name to value, nonzeros
  only). Both are required: a ray alone proves nothing without a feasible point.
- **`feasible`** (MILP results, see [MIP.md](MIP.md)): `x` (integer columns exactly integral, written as the
  integers the solver snapped them to), `claimed_objective`, optionally `claimed_best_bound`, `claimed_gap`
  (relative) and `claimed_gap_abs`, `nodes`, `mip_status` (the solver's own status, for example `NodeLimit`) and
  the explicit field `optimality_certified: false`. The verifier checks primal feasibility (relative `primal_tol`),
  that every integer column is **exactly** integral, and the exact objective against `claimed_objective`. The claimed
  bound is only reported ("bound NOT verified"); a bound on the wrong side of the objective, or any claim of
  `optimality_certified: true`, makes the certificate fail. A time- or node-limited run with an incumbent is written
  the same way.
- **`infeasible`** for a MILP: with `farkas` when the LP relaxation itself is infeasible (verified as above); otherwise
  with `certified: false` and no body (infeasibility was proved by branching or presolve): the verifier reports it as
  INCONCLUSIVE, exit code 2, never as a pass.
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
5. **A point that beats the bound.** If `x` is feasible only within tolerance, its objective can lie below
   `LB(y)`. That is consistent only up to what its own violations explain: `objective >= LB - sum_i |y_i| viol_i
   - sum_j |d_j| viol_j - gap_tol (1 + |objective|)`. A larger shortfall means the point and the multipliers are
   inconsistent and the certificate fails.
6. **Tolerance-level bound (`PASS_OPTIMAL_TOL`).** The multipliers are floating-point numbers, so a reduced cost
   that is zero in theory is a tiny nonzero in exact arithmetic; if it has the wrong sign for a column with an
   infinite bound, the strict `LB` is `-inf` although the answer is right. Only then does the verifier retry,
   treating a multiplier or reduced cost below `--dual-zero-tol` (default `1e-9`) of its scale as zero, and it
   reports how many it dropped and the largest change this can make to the primal objective. **This result is
   tolerance-checked, not a proof**: the retried bound is not a valid bound for every feasible point. The
   default `1e-9` was chosen after seeing a failing case (a correct answer with a multiplier of `1.6e-12` of the
   largest one failed at the first guess, `1e-12`); it is 1000 times smaller than the solver's own dual
   tolerance (`1e-6`), and it was not derived in advance.
7. **The solver's `rigorous` claim.** If the certificate says `dual_bound.rigorous = true`, the strict bound must
   exist in exact arithmetic; if it does not, the certificate fails (exact mode; float mode cannot judge it).
   The converse is not required: the solver's claim is deliberately conservative.

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
bound; the strict check fails in that case and the verifier reports it. When it fails, the verifier retries
once, dropping coefficients below `--farkas-zero-tol` (default `1e-12`, relative to the largest; `0` disables
the retry), in which case the verdict is `PASS_INFEASIBLE_TOL`, it lists what was dropped, and the result is
tolerance-checked, not a proof.

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

### Feasible (MILP results)

Feasibility within `primal_tol` (relative form above), integrality of every column listed as integer in the file
(**exactly** integral: the solver snaps them), and the exact objective compared with `claimed_objective`. A claimed best
bound and gap are printed as "bound NOT verified". The verdict is `PASS_FEASIBLE` together with the line "optimality not
certified"; it is called rigorous only if the point is exactly feasible and the arithmetic exact, and even then it says
nothing about optimality.

### Solution files

A `.sol` file written by `shodhan solve --write-sol` (objective line, then `name value`) can be checked for
primal feasibility and objective only; it carries no optimality evidence.

### Quadratic objectives

For `min offset + c^T x + (1/2) x^T Q x` the `optimal` body is the same (`x`, `y`, `dual_bound`). The verifier derives
`d = c + Q' x - A^T y` with exact `Q' x`, checks `x` for feasibility, and computes the weak-duality bound with the
first-order underestimate of the convex term at the certificate's own `x` (`Q'` is the minimization form, see
[QP.md](QP.md)). The bound is valid only if `Q'` is positive semidefinite, so the verifier tests that exactly
(symmetric elimination, up to `--psd-cap` columns, *default* 120):

`optimal` bodies of a QP also carry `convexity`, how the solver established convexity of Q: `float` (its floating-point
test found Q positive semidefinite with no pivot treated as zero), `tolerance` (it did, but only within a tolerance: the
sparse test factors Q + s I with s = 1e-9 max|q| and accepts an eigenvalue down to -s; the dense test sets a pivot
of absolute value at most that to zero). It is a claim, reported by the verifier next to its own exact result.

- not positive semidefinite: the certificate fails, whatever the solver claimed;
- positive semidefinite proven: the verdict can be rigorous;
- not verified (above the cap, or in float mode): the bound is reported as not rigorous and the verdict is
  tolerance-level (`PASS_OPTIMAL_TOL`). A certificate that claims `rigorous: true` in this situation is reported
  as such.

`infeasible` certificates do not involve `Q`. `unbounded` certificates are not supported for a quadratic objective
(the verifier answers INCONCLUSIVE). Multipliers from the interior-point method carry floating-point noise, so many
QP certificates pass only at tolerance level; the corpus counts are in the step report.

## 3. What is and is not certified

Certified (for the model file as read): that the stated `x` is feasible within the reported violation; that the
optimum is not below `LB(y)`; that an infeasible model has no feasible point (rigorously when the intervals are
disjoint exactly); that an unbounded model has a feasible point and an improving recession direction; for a quadratic objective, optimality within the stated gap, rigorously only when positive semidefiniteness of Q was verified exactly.

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

### Resolution floor of the multipliers

When the strict bound is -infinity because a reduced cost `d_j` has the wrong sign on an infinite bound, the retry drops it if
`|d_j| <= dual_zero_tol * scale_j`, with `scale_j = |c_j| + |(Qx)_j| + sum_i |a_ij y_i|`, **or** if `|d_j| <= 2^-52 * ||y||_inf *
||a_j||_1`. The second rule exists because `scale_j` degenerates when every term of the column is dust (c_j = 0 and the only
nonzero multiplier in the column is about 1e-17): `|d_j| / scale_j` is then 1 however small `d_j` is. A multiplier vector
of doubles from a factorization is accurate normwise, about `2^-52 ||y||_inf` per component, so `d_j` cannot be resolved
below `2^-52 ||y||_inf ||a_j||_1`. The constant is fixed in advance and was not tuned; the verdict stays tolerance-level.
It was added after three Netlib certificates (BNL2, FINNIS, PEROLD) failed the retry in exactly this way.

## 5. Tolerance conventions

Defined by this project (`--primal-tol 1e-6`, `--gap-tol 1e-6`, `--ray-tol 1e-9`, `--farkas-zero-tol 1e-12`,
`--dual-zero-tol 1e-9`, relative forms as above). They are not claimed to be identical to the conventions of any
other solution checker. Every verdict that depends on a tolerance carries the suffix `_TOL` or says
"tolerance-checked"; only a verdict without it, from exact mode, with an exactly feasible point, is a proof.
`--dual-zero-tol` was set after seeing a failing case, as explained under "Optimal" above.
