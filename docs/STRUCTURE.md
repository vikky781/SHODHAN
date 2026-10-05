# Structure analysis and big-M tightening

Code: `include/shodhan/structure.hpp`, `src/structure/structure.cpp` (detection), `src/presolve/mip_reductions.cpp`
(tightening), `src/mip/cut_separators.cpp` (use by the MIR separator). Tests: `tests/test_structure.cpp` with the
generators of `tests/support/structure_families.*`. Everything here is synthetic test material; no plant data is
involved. Numbers marked *target* or *default* are starting points.

## 1. What is detected

`detect_structure(model)` makes one pass over the rows of a model (in the solver: the presolved MIP model).

| Kind | Row pattern | Output |
|------|-------------|--------|
| Variable upper bound (VUB) | exactly two columns, one finite side; after writing it as `ax x + ay y <= b` with `ax > 0`: `ay < 0, b = 0` is `x <= u y` with `u = -ay/ax`; `ay > 0, b = ay` is `x <= u (1 - y)` with `u = ay/ax`. `y` binary, `x` has lower bound 0 and is not binary. All four spellings (`<=` or `>=`, with the signs flipped) are found. | `VubRow{row, x, y, u, complemented}` |
| Flow balance | equality row, every coefficient `+1` or `-1`, both signs present, at least one column that is not binary | row index, `is_balance_row[row]` |
| Set row | all columns binary, all coefficients `+1`, right-hand side 1: `=` partitioning, `<=` packing, `>=` covering | `SetRow{row, kind}` |

Not detected, on purpose: a general variable bound `x <= a + b y` with `a != 0`, aggregated rows with three or more
columns (`x1 + x2 <= u y`), range and equality links, and a row over two binaries (an implication, which the clique
table of presolve handles). Binary means an integer column with bounds `[0, 1]`.

## 2. Big-M tightening with implied bounds

The MIP presolve already ran coefficient tightening (Savelsbergh 1994) for rows with one finite side, but only when every
column of the row had a finite bound in the model. It now also receives the *implied* bounds of the activity-based bound
propagation at its fixpoint (option `PresolveOptions::implied_bound_tightening`, default on; CLI `--structure on|off`):

* for a VUB row `x - M y <= 0`, `M` becomes the implied upper bound of `x`, derived from **all** rows, not only this one;
* for an indicator row `a^T x <= b + M (1 - y)`, the coefficient of `y` is tightened with the implied activity of the
  other columns.

**Validity.** An implied bound holds for every feasible point of the model. The coefficient rule only uses
`rest <= maxact` where `maxact` is computed from valid bounds, so replacing a coefficient removes no integer-feasible point; it
changes the LP relaxation only. Propagated bounds of continuous columns are rounded outwards by the same tolerance
as the other presolve passes (`btol`), so rounding noise can weaken a bound, never cut a feasible point off. No column is
changed, only a coefficient and a right-hand side, so no postsolve record is needed. The pre-existing derivation also
needs a binary column in the row and a finite activity bound; both are checked.

`PresolveStats::implied_bound_tightenings` counts rows whose tightening needed an implied bound that was strictly better
than the model's own bound.

## 3. Use by the cuts

The MIR separator (`docs/CUTS.md`) takes variable bounds for continuous columns from the implications that probing
found; probing is limited (`probing_column_limit`), so on large models VUB rows could be missed. Now:

* every detected VUB row is added as a variable bound of its column (`x <= u y`, or `x <= u - u y`);
* when the aggregation chooses a tight row to eliminate a continuous column and several candidates are equally tight, a
  flow-balance row is preferred over other rows, then the shorter row (previously: the shorter row only).

Option `MipOptions::cut_structure_aware` (default on, also switched by `--structure`).

## 4. What was measured (synthetic families, informational)

On 510 small generated MIPs (170 each of fixed-charge with capacity rows, single-machine sequencing, lot sizing with
setups; the big-M is deliberately loose; all optima verified against brute force) the optimum was unchanged with and
without the tightening. The tightening fired in 169 models (554 rows) and improved the root LP bound in 160 of them, all of
them lot-sizing models, where the implied bound comes from the chain of inventory balances. In the fixed-charge and sequencing
families the implied bound is a singleton row after dual fixing, which the older passes already turned into a column
bound, so the new code adds nothing there. Over the 484 models with a root gap, the tightening closed on average 0.27 of
it (all families together; this is not a claim about any other model class).

For the structure-aware MIR cuts, on 300 further MIPs the optimum was again equal to brute force in all runs; the root
bound with cuts was higher with the structure in 0 models, lower in 1, and equal in 257 of the 258 where the cut loop ran, and the
node counts were equal. On these tiny instances the change has no measurable effect; whether it matters on larger models
is tested in the suite run of `bench/refinery/` and reported there.

## 5. Tests

* `vub_shape_matcher_on_hand_made_rows`, `structure_detection_on_a_hand_made_model`;
* `planted_structure_is_found_exactly_and_decoys_are_left_alone`: models with planted VUB, balance and set rows in
  shuffled order plus decoy rows (wrong right-hand side, three columns, equality, range, lower bound instead of upper
  bound, coefficient 2, no minus sign, rhs 2, continuous column in a set row); the found lists must equal the planted
  lists exactly (`SHODHAN_STRUCT_SEEDS`, default 400);
* `implied_bound_tightening_keeps_the_optimum_and_never_weakens_the_root_bound` (`SHODHAN_BIGM_SEEDS`, default 510): the optimum
  of the reduced model equals the brute-force optimum of the original with the tightening off and on; the root LP bound with
  it is at least the bound without it and at most the optimum;
* `structure_aware_cuts_keep_the_optimum_and_the_effect_is_measured` (`SHODHAN_STRUCT_CUT_SEEDS`, default 300).
