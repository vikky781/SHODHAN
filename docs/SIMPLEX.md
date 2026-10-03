# Simplex engine

`SimplexEngine` (`include/shodhan/simplex_engine.hpp`, `src/lp/`) solves LPs with the bounded dual
simplex method and a primal simplex for cleanup. This document gives the algorithm as implemented, the
sign conventions, the formulas, and the defaults. Numbers marked *default* or *target* are starting points,
not guarantees. It builds on [CONVENTIONS.md](CONVENTIONS.md) (computational form) and [LU.md](LU.md) (basis
factorization).

## 1. Setting and signs

Computational form: `A x - r = 0`, variables `0..n-1` structural (`col_lo`, `col_hi`, cost `c_j`) and
`n..n+m-1` logical (bounds `row_lo`, `row_hi`, cost 0, column `-e_i`). Everything is a minimization: a
`Maximize` model has its costs negated inside the engine; `y` and `d` of the result follow the minimization
form, the reported objective is in the model's own sense.

With a basis `B` (m columns) and nonbasic `N`:

```
B x_B + N x_N = 0            x_B = -B^-1 N x_N          (one ftran of  -sum_j a_j x_j)
B^T y = c_B                  d_j = c_j - a_j^T y        (one btran; logical n+i: d = y_i)
```

Nonbasic variables sit at a bound (`AtLower`, `AtUpper`), at the common value of a fixed variable (`Fixed`),
or at 0 (`FreeAtZero`). Dual feasibility of a nonbasic variable: `d_j >= 0` at lower, `d_j <= 0` at upper,
`d_j = 0` when free, anything when fixed; a boxed variable may sit at either bound, so it is placed at the
bound its `d_j` favours. Tolerances: `primal_tol` (default 1e-6, absolute, on the scaled model) and `dual_tol`
(default 1e-6).

## 2. Dual simplex, phase 2

Start: a dual feasible basis (the slack basis when the costs allow it; see section 7 for the other cases).
One iteration:

1. **Leaving row.** For each basic variable with a bound violation `v_i > primal_tol`, the score is
   `v_i^2 / w_i`; the largest score wins, ties go to the smallest position. No violation: optimal.
2. **Pivot row.** `rho_r = B^-T e_r` (one btran). `alpha_j = rho_r^T a_j` for nonbasic `j`, computed row-wise
   from a CSR copy of `A` by looping over the nonzeros of `rho_r`; a logical `n+i` has `alpha = -rho_r[i]`.
3. **Ratio test.** Let `x_p` be the leaving variable, `delta = x_p - bound` (negative at the lower bound),
   `sigma = +1` if `x_p` leaves at its lower bound, `-1` at its upper bound. Moving `y` by
   `-sigma*theta*rho_r` (`theta >= 0`) changes `d_j -> d_j + sigma*theta*alpha_j` and gives the leaving variable
   `d_p = sigma*theta`. A variable at its lower bound blocks the step when `sigma*alpha_j < 0`, one at its upper
   bound when `sigma*alpha_j > 0`, a free one whenever `alpha_j != 0`; fixed variables never block. The textbook
   test picks the smallest `theta_j = |d_j| / |alpha_j|` (ties: larger `|alpha_j|`, then smaller index) among
   candidates with `|alpha_j| >= min_pivot_abs`. No candidate: the row proves primal infeasibility (section 8).
4. **Entering column and pivot check.** `alpha_q = B^-1 a_q` (one ftran, spike saved for the update). The pivot
   `alpha_q[r]` is compared with `alpha_r[q]` from the row; if they differ by more than a relative
   `pivot_agreement_tol` (default 1e-7), a refactorization is done and the iteration repeated; if the factor is
   fresh, the candidate is excluded and a trouble event is counted.
5. **Update.** `t = delta / alpha_q[r]`: `x_B -= t * alpha_q` (all positions except `r`), `x_q += t`, `x_p = bound`.
   `d_j += sigma*theta*alpha_j` for nonbasic `j`, `d_q = 0`, `d_p = sigma*theta`. The basis header and statuses
   change and `BasisFactor::update(r)` is called; `NeedRefactor` triggers a refactorization with recomputation
   of `x_B`, `y`, `d` from scratch.
6. Before declaring optimality, and before declaring infeasibility, the factorization is refreshed if any update
   was done since the last one.

A refactorization also happens when the update limit (`refactor_interval`, default 100) is reached or the LU
growth indicator (`FactorStats::growth`) exceeds `max_growth` (target 1e8).

## 3. Dual steepest edge

The weight of row `i` is `w_i = ||e_i^T B^-1||^2 = ||rho_i||^2`. The slack basis has all weights 1. For a start
basis chosen by the caller (`set_basis`) the weights are computed exactly (one btran per row); after a basis
repair they are reset to 1 and `weights_exact()` is false (an approximation).

Update after a pivot in row `r` with entering column `alpha = B^-1 a_q` and `tau = B^-1 rho_r` (one extra
ftran of `rho_r`), for `i != r`:

```
w_i' = max( w_i - 2 (alpha_i / alpha_r) tau_i + (alpha_i / alpha_r)^2 w_r , 1e-4 )
w_r' = max( w_r / alpha_r^2 , 1e-4 )
```

where `w_r` is recomputed exactly as `||rho_r||^2` at the pivot. Derivation: the rows of `B^-1` change as
`rho_i' = rho_i - (alpha_i/alpha_r) rho_r` and `rho_r' = rho_r / alpha_r`, and `rho_i . rho_r = (B^-1 rho_r)_i = tau_i`.
A test compares the updated weights with the exactly computed ones after four iterations.
