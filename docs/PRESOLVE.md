# Presolve and postsolve

`presolve(model, options)` returns a smaller model plus a postsolve stack;
`postsolve(stack, reduced_solution)` maps a solution of the reduced model back to the
original model: primal `x`, row duals `y`, reduced costs `d` and the objective (in the
original sense). Conventions (minimization form, sign rules) are in
[CONVENTIONS.md](CONVENTIONS.md).

Presolve is deterministic: the same input gives byte-identical output. It uses no
unordered containers or pointer ordering.

## Result and statuses

| Status | Meaning |
|--------|---------|
| `Reduced` | `reduced` is a model to solve; postsolve maps its solution back |
| `SolvedByPresolve` | every row and column was eliminated; `postsolve(stack, {})` gives the solution |
| `Infeasible` | proven infeasible; `note` names the first conflict |
| `Unbounded` | an improving ray exists and presolve established feasibility (everything else was eliminated) |
| `InfeasibleOrUnbounded` | an improving ray exists but feasibility of the rest was not established |

For the last three the `reduced` model is empty.

## Working structure

A mutable model with row-wise and column-wise access (`WorkModel`). Removing a row or
column is lazy: its entries stay in the other view until the next compaction, which
happens once per pass in time linear in the nonzeros. Each row/column carries a dirty
flag; a pass visits only dirty items in index order, and reductions mark the neighbours
they affect. Passes repeat until nothing is dirty or `max_passes` is reached. Indices are
the original indices throughout; the reduced model is built once at the end, with row and
column index maps stored in the stack.

## Reductions

Each reduction is its own class (`src/presolve/reductions_*.cpp`) with its own postsolve
record (`src/presolve/postsolve_records.hpp`). Notation: the current model is
`M_k`, the model after the reduction is `M_{k+1}`; postsolve maps a KKT point of `M_{k+1}`
to a KKT point of `M_k`, keeping `d = c_k - A_k^T y` for every column alive in `M_k`.

### 1. Empty row
Removed if `0` lies in the row range, otherwise the model is infeasible. Postsolve: `y_i = 0`.

### 2. Empty column
Fixed at the bound its cost favours (`c > 0`: lower, `c < 0`: upper; `c = 0`: `0` clamped
into the bounds). If the favoured bound is infinite and `c != 0`, the column is an improving
ray: it is dropped and the final status becomes `Unbounded` (everything else eliminated) or
`InfeasibleOrUnbounded`. Postsolve: `x_j = value`, `d_j = c_j`.

### 3. Fixed column
If `lower == upper` (more precisely if `(upper - lower) * (1 + max(|c_j|, |a_ij|)) <= tol * (1 + |lower|)`,
so a nearly fixed column with large coefficients is not fixed), the column is substituted:
row bounds and the objective offset shift. Postsolve: `x_j = value`, `d_j = c_j - A_j^T y`.

### 4. Singleton row
`rl <= a x_j <= ru` becomes bounds on `x_j` and the row is removed. For an integer column in a
MIP the derived bounds are rounded (`ceil`/`floor`). Postsolve: let `d'` be the reduced cost of
`x_j` in `M_{k+1}`. If the bound derived from the row is at least as tight as the old one and
`d'` pushes against it (`d' > 0` for a lower bound, `d' < 0` for an upper bound), set
`y_i = d' / a` and `d_j = 0`; otherwise `y_i = 0`. The sign of `y_i` is the sign the active row
side requires (`a > 0`: lower bound comes from `rl`, `a < 0`: it comes from `ru`).

### 5. Redundant row
From the activity bounds of the row: a row side that the column bounds already imply is
dropped; a row with both sides implied (or a free row) is removed; a row whose activity
range cannot meet its bounds is infeasible. Postsolve: `y_i = 0`. A dropped side cannot be
active, so `y_i = 0` is complementary.

### 6. Forcing row
If the minimum activity equals the row upper bound (or the maximum activity equals the lower
bound) every column of the row is fixed at its activity-extreme bound and the row is removed.
Postsolve (derived from Andersen & Andersen, 1995): with `e_j = c_j - sum_{r != i} a_rj y_r`
and `d_j = e_j - a_ij y_i`, the sign rules on the forced columns are `y_i <= e_j / a_ij` for all
`j` (minimum-forcing, row at its upper side) or `y_i >= e_j / a_ij` (maximum-forcing). Taking
the extreme ratio, capped at 0 (resp. floored at 0) unless the row is an equality, satisfies all
of them at once; then `d_j = e_j - a_ij y_i` for every forced column. This is checked by hand-made
tests and by the property tests.

### 7. Doubleton equation aggregation
`a_j x_j + a_k x_k = b` substitutes `x_j = beta - alpha x_k` (`alpha = a_k / a_j`,
`beta = b / a_j`): costs, offset, other rows' coefficients and bounds, and the bounds of
`x_k` (implied by `x_j`'s bounds) are updated; the row and column `j` are removed.
Refused when
- the pivot `|a_j|` is below `doubleton_pivot_tolerance` (default `1e-3`) times the largest
  magnitude in column `j`;
- `|alpha|` is outside `[1/max_coefficient_growth, max_coefficient_growth]` (default `1e3`);
  a tiny or huge ratio makes the implied bounds `(beta - bound) / alpha` carry rounding error of
  order `eps / |alpha|`;
- a new coefficient would exceed `max_coefficient_growth` times the larger of the two it combines;
- a new coefficient is nonzero but below `min_cancellation_ratio` (default `1e-3`) of the terms that
  cancelled to form it (exact cancellation is accepted).

Postsolve: `x_j = beta - alpha x_k`. With `e_j`, `e_k` the reduced costs without the removed row,
`d'_k = e_k - alpha e_j`. Choosing `y_i = e_j / a_j` gives `d_j = 0`, `d_k = d'_k`. If `x_k` rests on
a bound that was derived from `x_j`'s bounds (strictly tighter than its own), that choice has the wrong
sign for `x_k`, so `y_i = e_k / a_k` is used instead: then `d_k = 0` and `d_j = -d'_k / alpha`, which
has the sign `x_j`'s active bound needs.

### 8. Dual fixing (dominated column)
If every row of column `j` has an infinite side in the direction of decreasing `x_j` and `c_j >= 0`,
`x_j` can be fixed at its lower bound (symmetrically for increasing and the upper bound). Then
`sum_i a_ij y_i <= 0` for every KKT point, so `d_j = c_j - A_j^T y >= c_j >= 0`, which is exactly what a
column at its lower bound needs: postsolve is the fixed-column recovery. If the needed bound is infinite
and the cost is nonzero, the column and its rows are dropped (the rows stay satisfiable along the ray)
and the final status is `Unbounded` or `InfeasibleOrUnbounded`. Valid for MIP.

### 9. Integer bounds and implied bounds
- MIP only: integer column bounds are rounded (`ceil` the lower, `floor` the upper); no integer in the range
  means infeasible. Integer columns are also tightened from row activities.
- LP: implied bounds are computed **for detection only** (a conflict with the column's own bounds proves
  infeasibility). They are never written into the model, because tightening LP bounds can create degeneracy.
No postsolve record is needed (primal values are untouched).

## MIP safety

When `is_mip` is set only reductions that preserve integer-feasible points are used:

| Reduction | MIP |
|-----------|-----|
| 1 empty row, 2 empty column, 3 fixed column, 5 redundant row | yes |
| 4 singleton row | yes, with rounding for integer columns |
| 6 forcing row | only if every integer column in it would be fixed at an integral bound |
| 7 doubleton | the eliminated column must be continuous, **or** both columns are integer, the pivot is `+-1` and `a_k`, `b` are integral (so `x_j = beta - alpha x_k` stays integral) |
| 8 dual fixing | yes (an integer column is fixed only at an integral bound) |
| 9 integer bounds | yes (this is where integer bounds are rounded and tightened) |

Duals are not reconstructed for a MIP (`need_duals` is ignored); postsolve returns only `x` and the objective.

## Tolerances

- `feasibility_tol` (default `1e-9`) is relative.
- A **conflict** is only declared infeasible if it exceeds `feasibility_tol` relative to the magnitude of the
  data behind the row (`row_mag`: bounds, `sum |a| |bound|`, shifts). Wide coefficient ranges let substitutions
  leave errors around `1e-12` of that magnitude, which must not be mistaken for infeasibility.
- A decision that **changes the problem** (forcing a row, dropping an implied row side) uses a strict tolerance of a
  few ulps of the compared value only: a looser margin would perturb the row by that amount and a large dual
  would turn that into a visible objective change.

## `need_duals`

With `need_duals = false` (or `is_mip`) postsolve returns `x` and the objective only (`y` and `d` empty). No reduction
is gated by this flag: the dual postsolve of every reduction above is implemented and tested, including forcing rows.

## What is tested

- One hand-made model per reduction (`tests/test_presolve_reductions.cpp`), asserting the reduced shape and the
  postsolved solution with `check_kkt`; dual values are asserted exactly where they can be derived by hand.
- Property tests (`tests/test_presolve_property.cpp`) against the dense test oracle in `tests/support/`: LP, LP through
  scaling, maximization, infeasible/unbounded constructions, small MIPs against brute-force enumeration, determinism.
  Seeds are printed on failure; `SHODHAN_SEED_FIRST` / `SHODHAN_SEED_COUNT` widen the range.
