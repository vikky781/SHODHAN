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
