# Quadratic objectives

## 1. Model

    min  offset + c^T x + (1/2) x^T Q x      s.t.  row_lower <= A x <= row_upper,  col_lower <= x <= col_upper

`LpModel::quadratic` stores **the lower triangle** of the symmetric `Q`, in CSC form; an entry `(i, j)` with
`i != j` stands for `q_ij` and `q_ji`. For a maximization model the solver works with `c' = -c`, `Q' = -Q`
(*minimization form*). **Convexity means that `Q'` is positive semidefinite**, so a maximization model is convex when `Q` itself is
negative semidefinite.

## 2. QPS input and output

The MPS reader accepts the `QUADOBJ` section (each off-diagonal pair once, lower or upper triangle) and `QMATRIX`
(the full symmetric matrix; an unsymmetric one is rejected). The objective's `1/2` convention is the one of the
Maros-Meszaros files. `QCMATRIX` and `QSECTION` (quadratic constraints) are recognised and rejected with a message;
they are not supported. The writer emits `QUADOBJ`. Semantics and the round trip are tested in `tests/test_qp_model.cpp`
and `tests/test_mps_reader.cpp`; KASAUTI parses the same sections with its own reader.

## 3. Convexity

`check_convexity` (`include/shodhan/quadratic.hpp`) decides whether `Q'` is positive semidefinite: a dense
test up to 1200 columns, the sparse LDL^T beyond. It works in floating point with a tolerance, so its answer is
evidence, not a proof. A model reported not positive semidefinite gets the status `NonConvex` (exit 3) and is not
solved. `KASAUTI` repeats the test exactly (section 5).

## 4. KKT conditions and the dual bound

With `d = c + Q x - A^T y`, the optimality conditions are the LP ones with this `d`: primal feasibility, the sign rules
of `y` and `d`, and complementarity. The dual objective used for the gap is
`sum y*b_active + sum d*bound_active - (1/2) x^T Q x + offset`; its gap to the primal objective equals the
complementarity sum.

The *rigorous* lower bound of `compute_dual_bound` for a convex `Q` is the first-order underestimate at the given
point `x~`: `(1/2) x^T Q x >= x~^T Q x - (1/2) x~^T Q x~`, which makes the problem linear in `x` and reduces to the LP
bound with cost `c + Q x~`, evaluated in exact arithmetic over the box. It is valid only if `Q'` is positive
semidefinite.

## 5. Certificates and KASAUTI

An optimal certificate of a QP has the additive fields `quadratic` and `q_nnz` ([CERTIFICATES.md](CERTIFICATES.md)).
KASAUTI recomputes `d` with exact `Q x`, tests `Q'` for positive semidefiniteness by exact symmetric elimination (a
negative pivot, or a zero pivot with a nonzero entry in its row, proves it is not; finishing proves it is) up to
`--psd-cap` columns (*default* 120), and computes the bound of section 4.
- not positive semidefinite: `FAIL`;
- positive semidefinite proven, bound rigorous: a rigorous `PASS` is possible;
- above the cap, or in float mode: convexity is "not verified", the bound is not rigorous, and the verdict is
  tolerance-level (`PASS_OPTIMAL_TOL`).
Infeasible certificates (a Farkas ray of the constraints) do not involve `Q` and are checked as for an LP.
Unbounded certificates for a QP are **not supported** (`INCONCLUSIVE`).

## 6. Status of the solver for QPs

| Case | Result |
|------|--------|
| convex, continuous | interior-point method ([IPM.md](IPM.md)); `Optimal` only after the KKT check and the bound |
| indefinite `Q'` | `NonConvex`, exit 3 |
| integer columns with a quadratic objective | `NotImplemented` (MIQP), exit 2 |
| `--method simplex` on a QP | `NotImplemented`, exit 2 |
| infeasible | `Infeasible` with a Farkas certificate from the simplex |
| unbounded QP | reported without a certificate |
| quadratic constraints | rejected at read time |

Presolve for a QP is restricted to reductions that keep the problem exact: fixed columns (their terms are folded into
`c` and the offset), empty and singleton and redundant rows. The other LP reductions use dual arguments that ignore `Q`
and are switched off. Scaling uses `Q' = s C Q C` with the same column scaling `C` as the matrix.
