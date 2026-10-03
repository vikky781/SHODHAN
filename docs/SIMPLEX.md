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

## 4. Ratio test: bound flipping and Harris

The ratio test of section 2 is the textbook one. The default test combines the bound flipping ("long-step")
ratio test with Harris's two passes (Koberstein 2005, Maros 2003).

For the candidate set (variables that block the dual step) define `s_j = d_j` (at lower), `-d_j` (at upper), 0
(free), the exact breakpoint `t_j = max(s_j, 0) / |alpha_j|` and the relaxed one
`r_j = (max(s_j, 0) + tol) / |alpha_j|`, with `tol = dual_tol / 2` (Harris on) or 0 (off).

1. Candidates with `|alpha_j| < max(min_pivot_abs, min_pivot_rel * max_k |alpha_k|)` are dropped (defaults 1e-9 and
   1e-7, targets); if that removes all of them, the relative threshold is dropped.
2. Candidates are sorted by `t_j`. **Harris pass 1:** `theta_max` is the smallest `r_j` among the remaining
   ones, and the group `K` holds those with `t_j <= theta_max`.
3. **Bound flipping:** the slope of the dual objective along the step starts at `|delta|` (the infeasibility of
   the leaving variable). Passing a boxed candidate `j` lowers it by `|alpha_j| (hi_j - lo_j)`. If the slope
   stays above a margin `0.5 * primal_tol * (1 + |bound|)` after passing all of `K`, the variables of `K`
   are flipped to their other bound and the search continues with the remaining candidates; a variable that is
   not boxed stops the search (its slope decrease is infinite).
4. **Harris pass 2:** the entering variable is the member of the final group with the largest `|alpha_j|`
   (ties: smaller index), and `theta = t_q`.
5. If every candidate is passed and the slope is still above the margin, the row cannot be repaired: primal
   infeasibility (section 8).

The flips change `x_B` by `-B^-1 sum_j a_j * change_j`, done with one ftran of the combined column; the
infeasibility `delta` of the leaving variable is recomputed from the updated `x_p`, and the pivot then proceeds
with that `delta`. The entering column is computed and checked before the flips are applied, so a rejected pivot
leaves the state untouched.

After the step, `d_j += sigma*theta*alpha_j`. A nonbasic `d_j` that ends up wrong-signed by more than 1e-11
(possible by at most the Harris tolerance, or when `theta` had to be clamped at 0) is removed by **cost
shifting**: `c_j -= d_j`, `d_j = 0`. Shifted costs are part of the working costs and are removed with the
perturbation (section 6).

## 5. Tolerances

The primal tolerance is **relative to the violated bound**: a bound is violated when the violation exceeds
`primal_tol * (1 + |bound|)` (default `primal_tol` 1e-6), which is the measure `check_kkt` uses. An absolute
tolerance would mistake rounding noise for infeasibility on rows whose values are large; with absolute margins a
test LP with coefficients between 1e-4 and 1e4 was declared infeasible by a 3e-6 rounding difference on a row of
scale 1e4. The dual tolerance `dual_tol` (default 1e-6) is absolute, on the scaled model.

**Polishing.** After optimality, violations below the tolerances can still be visible after unscaling and
change the objective when the multipliers are large. The engine tightens both tolerances to `polish_tol`
(default 1e-13, target) and lets the dual simplex (remaining primal violations) or the primal simplex (remaining
dual violations) finish, with a cap of `100 + 2m` iterations so that noise cannot make it loop. The normal
tolerances are then restored and checked again.

## 6. Perturbation, stalling and cleanup

**Perturbation** (on by default, `perturb`). At the start of phase 2 the cost of every nonbasic structural
variable at a bound moves away from dual infeasibility: `+xi_j` at a lower bound, `-xi_j` at an upper bound, with
`xi_j = perturb_scale * (1 + |c_j|) * (1 + u_j)`, `u_j` uniform in `[0,1)` from a deterministic generator seeded
by `seed` and the perturbation count (`perturb_scale` default 5e-7, target). Free and fixed variables are not
perturbed. `d_j` changes by exactly the same amount.

**Stall detection.** The dual objective `c^T x` is tracked incrementally (including the effect of flips). If it
does not improve for `stall_iterations` iterations (default 500, target), a larger perturbation (10, 100, ...
times the base scale, at most five times) is applied.

**Removal and cleanup.** When the dual simplex ends optimal, the original costs are restored, the basis is
refactorized, and `x_B`, `y`, `d` are recomputed from scratch. If the point is primal and dual feasible within
the tolerances it is optimal. If there are dual infeasibilities (the basis is primal feasible: bounds do not
depend on the costs), the **primal simplex** removes them; primal infeasibilities with dual feasibility go back
to the dual simplex. The loop is bounded by `max_cleanup_rounds` (default 5, target) and then reports
`NumericalError`.

**Primal simplex** (a cleanup engine, not the main one). Dantzig pricing on the dual infeasibility; the entering
column `alpha = B^-1 a_q`; a Harris two-pass ratio test over the basic variables (relative tolerance
`0.5 * primal_tol`), choosing among ties the largest `|alpha|`; a bound flip when the entering variable's own
range is shorter than every ratio; Bland's rule (smallest index) after 100 degenerate steps in a row. The dual
update after a pivot in row `r` is `d_j -= (d_q / alpha_r) alpha_rj`, `d_p = -d_q / alpha_r`. If no basic
variable blocks and the entering variable is not boxed, the problem is unbounded and the direction is kept
(section 8).

## 7. Dual phase 1 and infeasible or unbounded problems

If some nonbasic reduced cost has the wrong sign for its status after the boxed variables were placed (a free
variable with a cost, a lower bounded one with a negative cost, ...), the basis is not dual feasible.

**Phase 1** (Koberstein's artificial bounds subproblem). Replace the bounds of every variable, including the
logicals, by `[-1000, 1000]` (free; `artificial_bound`, target), `[0, 1]` (lower bounded only), `[-1, 0]` (upper
bounded only) or `[0, 0]` (boxed or fixed), place each nonbasic variable at the bound its reduced cost favours
(every variable is boxed, so the basis is dual feasible) and solve `min c^T x` with the same dual simplex (without
Harris, perturbation or polishing). The subproblem is feasible (`x = 0`) and bounded, so it always ends optimal
(anything else is reported as a numerical failure). The real bounds are then restored and each nonbasic variable
gets its real status from the sign of `d_j`; the basis is dual feasible for the original problem exactly when no
variable remains wrong-signed, i.e. when the optimum of the subproblem is zero. Phase 1 iterations are counted
in `SimplexStats::phase1_iterations`.

**Dual infeasible: infeasible or unbounded.** If phase 1 does not reach a dual feasible basis the LP is infeasible
or unbounded. It is resolved:

1. Replace all costs by zero (every basis is dual feasible) and run phase 2. If the dual simplex proves primal
   infeasibility, the LP is `Infeasible`; the certificate does not depend on the costs.
2. Otherwise the basis found is primal feasible. Run the primal simplex with the true costs from it. If it ends
   optimal the dual infeasibility was a tolerance artifact and the usual cleanup and acceptance follow; if it finds
   a blocking-free direction the LP is `Unbounded` and the ray is kept.

## 8. Certificates

**Primal infeasibility (Farkas).** When no entering candidate remains in row `r` (including the case where every
candidate is a boxed variable that can be flipped and the infeasibility remains), the multipliers are
`y = rho_r`. For every feasible `(x, r)`, `y^T (A x - r) = 0`, i.e. the combination
`sum_j (A^T y)_j x_j - sum_i y_i r_i` is 0. `check_farkas(model, y, tol)` (`include/shodhan/rays.hpp`,
`src/model/rays.cpp`) computes the largest and smallest value of that combination over the box of column and row
bounds and accepts if the largest is below 0 or the smallest above 0 by more than `tol` times the magnitude of
the terms; an infinite bound reached by a nonzero coefficient defeats the proof (coefficients below 1e-12 of the
largest count as zero, because the basic variables of the row have coefficient 0 in theory and about 1e-17 in
floating point). Why it works for the simplex row: with `sigma = +1` (the leaving variable violates its lower
bound) the row `x_p + sum_j alpha_j x_j = 0` gives `x_p <= -sum_j min(alpha_j x_j)`, and the candidates are
exactly the variables for which this minimum is not attained at the current bound; when there is none, or all can
be flipped and the slope is still positive, even the best choice leaves `x_p` below its lower bound. The case
`sigma = -1` is symmetric.

**Unboundedness.** The primal simplex stops with a direction when the entering variable `q` (moving by
`dir * t`) is not boxed and no basic variable blocks: the structural part of the ray is `ray_q = dir` and
`ray_B = -dir * alpha` for the basic structurals. `check_unbounded_ray(model, ray, tol)` verifies that the column
bounds and row ranges allow motion along it forever (finite column bound in the direction of motion must have a
zero component; `A ray` must lie in the recession cone of the row ranges) and that the objective improves
(`c^T ray < 0` for minimization, `> 0` for maximization). Feasibility of the LP is not part of this check; it comes
from the primal feasible basis the primal simplex started from.

## 9. Numerical trouble and acceptance

- **Pivot disagreement** (row versus column value of the pivot): refactorize and repeat; if the factorization is
  fresh the candidate is excluded for this iteration and a trouble event is counted.
- **Drift:** after every refactorization `x_B` and `d` are recomputed from scratch and compared with the updated
  values; a primal drift above 1e-6 counts as a trouble event.
- **Rank deficiency of the basis:** `BasisFactor::repair` replaces the deficient columns by logicals; the replaced
  variables become nonbasic at a bound, the pricing weights are reset to 1 (`weights_exact()` becomes false) and the
  primal and dual values are recomputed.
- More than `max_trouble` (default 6, target) consecutive trouble events end the solve with `NumericalError`.
- **Acceptance** (`final_check`, default on): `x_B` and `y` are refined (up to two rounds of solving for the
  residual of `A x - r = 0` and `B^T y = c_B`), then `check_kkt` is run on the model with the current bounds at
  `final_tol` (default 1e-6). If it fails, the tolerances are tightened to 1e-9 once and the cleanup is redone; if
  it still fails the result is `NumericalError`, never `Optimal`. An `Infeasible` result must pass `check_farkas`
  (tolerance 1e-9) and an `Unbounded` one `check_unbounded_ray` (1e-7), otherwise it also becomes `NumericalError`.

## 10. The LP pipeline

`LpSolver` (`include/shodhan/lp_solver.hpp`) runs: presolve (optional) -> scaling (optional) -> `SimplexEngine` ->
unscale -> postsolve -> `check_kkt` on the ORIGINAL model with `kkt_tol` (default 1e-6). Maximization models are
negated inside the engine; `y` and `d` stay in minimization form and the objective is in the model's sense.

- `Optimal` is returned only if the KKT check on the original model passed.
- `Infeasible` carries Farkas multipliers (the engine's multipliers times the row scale factors) that pass
  `check_farkas` on the original model; `Unbounded` carries a ray (the engine's ray times the column scale
  factors) that passes `check_unbounded_ray`.
- A status reached by presolve alone (or by the engine on the reduced model) cannot be certified in the original
  space: it is confirmed by solving the scaled original without presolve.
- If an attempt cannot be certified, fallbacks follow: without presolve; with tolerances 100 times tighter; without
  scaling. Then `NumericalError`. `LpResult::attempts` and `message` say what happened.
- `TimeLimit` (`Params::time_limit`, whole pipeline) and `IterationLimit` are reported as such.

## 11. Defaults (all targets) and warm start

`primal_tol` 1e-6 (relative to the bound), `dual_tol` 1e-6, `refactor_interval` 100, `max_growth` 1e8,
`pivot_agreement_tol` 1e-7, `min_pivot_abs` 1e-9, `min_pivot_rel` 1e-7, `perturb_scale` 5e-7, `stall_iterations` 500,
`max_trouble` 6, `max_cleanup_rounds` 5, `artificial_bound` 1000, `polish_tol` 1e-13, `final_tol` 1e-6.

Warm start: after `solve()`, `change_col_bounds` / `change_row_bounds` move nonbasic variables to the new bound and mark the
primal values stale; the next `solve()` recomputes `x_B` with one ftran and resolves with the dual simplex from the current
basis (a test compares this with a cold solve after 300 random bound changes).

## 12. Known limitations

- Tested on generated models only; no Netlib instance was run.
- Wide coefficient ranges (1e-4..1e4) can make an LP hypersensitive (multipliers around 1e10): a point that is feasible to
  5e-10 can then have an objective that differs by several units from the exact vertex, so two solvers that both pass a
  relative KKT check can disagree. In a stress run of 28000 LPs this showed in 3 cases (the pipeline's point was the more
  precise one).
- In the same stress run, 3 unbounded wide-coefficient LPs ended as `NumericalError` because the primal simplex ray did not
  pass `check_unbounded_ray` (small entries that the pivot thresholds drop); this is not fixed.
- The primal simplex uses Dantzig pricing; it is a cleanup engine and may be slow on large problems.
- Bound flipping is not combined with a cost-shifting-free guarantee: shifted costs are removed by the cleanup, which can
  need primal iterations.

## 13. References

- A. Koberstein, "The dual simplex method, techniques for a fast and stable implementation", PhD thesis, Universitat Paderborn, 2005.
- I. Maros, "A generalized Dual Phase-2 simplex algorithm", European Journal of Operational Research 149, 2003.
- J. J. Forrest and D. Goldfarb, "Steepest-edge simplex algorithms for linear programming", Mathematical Programming 57, 1992.
- P. M. J. Harris, "Pivot selection methods of the Devex LP code", Mathematical Programming 5, 1973.
