# Conventions

These conventions hold everywhere in SHODHAN: the model, the KKT checker, scaling,
presolve and postsolve, and (later) the solvers.

## Minimization form

Internally every model is a **minimization**. A model with `Sense::Maximize` is
converted by negating the costs and the objective offset (a flag remembers the
original sense) and signs are flipped back on output:

- `Solution::objective` is always `offset + c^T x` in the model's **own** sense (a
  maximization model reports the maximization value).
- `Solution::y` and `Solution::d` are always the multipliers of the
  **minimization form** (the one with costs `-c`).
- `PresolveResult::reduced` is always a minimization model.

## Primal problem

```
minimize    c^T x + offset
subject to  row_lo <= A x <= row_hi
            col_lo <=  x  <= col_hi
```

Infinite bounds are stored as `+/-kInf` (1e30); `is_inf(v)` is true for any value
with magnitude at least `kInf`. A "free row" has `row_lo = -kInf, row_hi = +kInf`.

## Duals and reduced costs

Row duals `y` and reduced costs `d = c - A^T y`.

KKT sign rules:

| Quantity | Situation | Requirement |
|----------|-----------|-------------|
| row `i`  | at its lower bound only | `y_i >= 0` |
| row `i`  | at its upper bound only | `y_i <= 0` |
| row `i`  | strictly between, or a free row | `y_i = 0` |
| row `i`  | equality row | `y_i` free |
| column `j` | at its lower bound only | `d_j >= 0` |
| column `j` | at its upper bound only | `d_j <= 0` |
| column `j` | strictly between | `d_j = 0` |
| column `j` | fixed | `d_j` free |

Equivalently (and this is how `check_kkt` evaluates them, without needing a
tolerance to decide which bound is "active"): a positive multiplier needs a finite
lower bound and must be complementary to it, a negative multiplier needs a finite
upper bound and must be complementary to it.

## Dual objective

```
dual objective = sum_i y_i * (active row bound)
               + sum_j d_j * (active column bound)
               + objective offset
```

where the active bound is the lower bound for a positive multiplier and the upper
bound for a negative one (a zero multiplier contributes nothing). At a KKT point it
equals the primal objective. Everything is in minimization form, so for a maximization
model the dual objective is the negated maximization value.

## Solution

```cpp
struct Solution { std::vector<double> x, y, d; double objective; };
```

`d` may be left empty; `check_kkt` then recomputes it as `c - A^T y`.

## KKT report scaling

`check_kkt` reports absolute violations and relative ones, divided by
`1 + (norm of the relevant data)`. The relevant data is the magnitude of the terms the
quantity is computed from: the finite bounds for primal infeasibility, per column
`|c_j| + sum_i |a_ij y_i|` for reduced costs, and the larger of `|p| + |d|` and the sums of
absolute objective terms for the duality gap. This is a backward-error scaling: a reduced
cost that is a tiny difference of huge terms can only be as accurate as those terms allow,
so demanding more is meaningless, but error that is large relative to the data is not
hidden.

## Scaling

With `A' = R A C`, `c' = s C c`, `x = C x'` (`R`, `C` diagonal, `s` scalar, all exact powers
of two):

```
row bounds' = R * row bounds     column bounds' = C^-1 * column bounds
objective'  = s * objective      offset' = s * offset
x = C x'     y = R y' / s        d = d' / (s C)     objective = objective' / s
```

Integer and binary columns always have `C_jj = 1`.

## Computational form: structural and logical variables

The simplex code and the basis factorization (`docs/LU.md`) work with the model written as

```
A x - r = 0
```

where `x` are the **structural** variables (bounds `col_lo`, `col_hi`, costs `c`) and `r` are the
**logical** variables, one per row, equal to the row activities (bounds `row_lo`, `row_hi`, cost 0).

- The logical column of row `i` is `-e_i` (the negative unit vector) and its variable index is `n + i`.
  Variable indices therefore run over `0..n-1` (structural, column `j` of `A`) and `n..n+m-1` (logical).
- A **basis** is a list of `m` variable indices; entry `p` is the variable at *basis position* `p`. The basis
  matrix `B` has `m` columns taken from `[A | -I]`, column `p` being the column of `basis[p]`.
- The reduced cost of logical `n + i` is `d = 0 - (-e_i)^T y = y_i`: it equals the row dual. This is the
  same sign rule as in the KKT table above (a row at its lower bound has `y_i >= 0`, which is a logical at its
  lower bound with a nonnegative reduced cost).
- `ftran` solves `B x = a` with `a` indexed by row and `x` by basis position; `btran` solves
  `B^T y = c` with `c` indexed by basis position and `y` by row.

## Quadratic objective

The objective of a model may carry a quadratic term:

    objective = offset + c^T x + (1/2) x^T Q x,      Q symmetric.

* **Storage.** `LpModel::quadratic` holds the LOWER TRIANGLE of Q (row index >= column index, diagonal included) as an
  `n_cols x n_cols` CSC matrix; the empty 0x0 matrix means no quadratic term. An entry q_ij with i > j stands for
  both q_ij and q_ji (the matrix is symmetric by construction, there is no way to store an asymmetric Q).
  `validate()` rejects entries above the diagonal, non-finite values and a wrong shape. `quad_full()` builds both triangles.
* **Contributions.** A diagonal entry q_jj adds (1/2) q_jj x_j^2 to the objective; an off-diagonal entry q_ij (i > j) adds
  q_ij x_i x_j once in total ((1/2)(q_ij x_i x_j + q_ji x_j x_i)).
* **Minimization form.** A maximization model has c AND Q negated: max c^T x + (1/2) x^T Q x is min (-c)^T x + (1/2) x^T (-Q) x.
  "Convex" always refers to the minimization form: for a maximization model -Q must be positive semidefinite.
* **KKT.** In minimization form the reduced costs are d = c + Q x - A^T y, with the same sign rules as for an LP
  (y_i > 0 on an active lower row bound, y_i < 0 on an active upper one; d_j > 0 on a lower column bound, d_j < 0 on an
  upper one). The primal objective is offset + c^T x + (1/2) x^T Q x and the dual objective is

      sum_i y_i (active row bound) + sum_j d_j (active column bound) - (1/2) x^T Q x + offset.

  Gap relation: with d = c + Q x - A^T y,  x^T d = c^T x + x^T Q x - y^T A x, hence

      primal - dual = sum_j d_j (x_j - active bound_j) + sum_i y_i (row activity_i - active bound_i),

  the complementarity sum, exactly as for an LP (Q only enters through d). For a convex QP any feasible (x, y) with the
  sign rules satisfied has dual objective <= optimum (weak duality, see docs/QP.md and docs/CERTIFICATES.md).
