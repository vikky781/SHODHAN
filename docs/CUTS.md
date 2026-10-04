# Cutting planes

Cuts are generated only at the root of the branch and bound (`src/mip/cut_*.cpp`, `include/shodhan/mip/cuts.hpp`).
They are global: they use only the root column bounds, so they stay valid at every node.

A cut is `sum val[k] x[idx[k]] <= rhs` in the presolved, unscaled, minimization model. **Valid** means every
integer-feasible point of that model (continuous part free) satisfies it. A cut that removes such a point is a bug.

## Engine support

`SimplexEngine::add_rows(rows)` appends rows (the engine then owns a copy of the model). The logical variable of each
new row enters the basis, so the basis stays dual feasible and the next `solve()` is a warm dual simplex start.
`remove_rows(rows)` removes rows whose logical variable is basic (slack rows) and refuses, changing nothing, if one is
tight. `tableau_row(pos, alpha)` returns a row of the simplex tableau for the Gomory separator. Tests
(`tests/test_engine_rows.cpp`): 300 random LPs, warm resolve after adding cuts equals a cold solve of the extended
model, removing slack rows keeps the optimum, adding then removing is the identity, a tight row cannot be removed.

## Separators

| Separator | Cuts | Needs |
|-----------|------|-------|
| Gomory mixed-integer | one per fractional basic integer column (at most `cut_gomory_rows` per round) | tableau row of the engine |
| MIR with aggregation (c-MIR) | from tight rows, up to `cut_mir_aggregation` rows combined | rows; variable bounds from implications |
| Knapsack cover, lifted | per row side, relaxed to a 0-1 knapsack | binary columns |
| Clique | stored cliques extended greedily by conflicting literals | clique table |
| Implied bound | `y <= B + (U-B) x_b` style inequalities from implications | implications |

* **Gomory** (Gomory 1960; Cornuejols, Li, Vandenbussche 2003). For a basic integer column with fractional value the
  row `x_B + sum a_j x_j = 0` over nonbasic variables is shifted to bounds (`t >= 0`); integer `t` get
  `min(f_j/f0, (1-f_j)/(1-f0))`, continuous `t` and logicals get `a/f0` or `-a/(1-f0)`; logicals are replaced by their
  rows. Derived in the scaled space and returned unscaled. Rows with a free nonbasic variable are skipped.
* **MIR** (Nemhauser, Wolsey 1990; Marchand, Wolsey 2001). Aggregation adds rows (multiplier >= 0, so the sum is valid)
  to eliminate continuous columns strictly between bounds. Continuous columns use their closest simple or variable
  bound; integer columns are shifted or complemented. For `delta` among the coefficients of integer columns strictly
  between bounds (and `delta/2, /4, /8` of the best), `f0 = frac(beta/delta)` must lie in `[0.05, 0.95]`; the best
  efficacy wins. Not done: improving the complementation by local search, and slack integrality.
* **Cover** (Balas 1975, Wolsey 1975; lifting after Padberg, exact by dynamic programming over the profit). Columns
  other than the binaries are relaxed to their minimum activity; fixed columns move to the right-hand side.
* **Clique** (Atamturk, Nemhauser, Savelsbergh 2000). Literals conflict when they share a stored clique.
* **Implied bound** (Savelsbergh 1994) from probing implications.
* **Zero-half** cuts (Caprara, Fischetti 1996) are a roadmap item only, not implemented.

When presolve is on, cliques and implications come from presolve (`PresolveResult::mip`). When it is off,
`find_mip_structure(model)` computes them by probing the model itself (`MipOptions::cut_structure`).

## Safety filters (`clean_cut`, `select_cuts`)

Applied to every raw candidate, in this order; each rejection is counted per separator.

1. Sort, merge duplicate columns, reject non-finite values.
2. **Tiny coefficients** (`|g| < 1e-9 max|g|`) are removed by relaxing the right-hand side with the column bound that
   makes the term smallest (valid); rejected when that bound is infinite.
3. **Dynamism** `max|g| / min|g| > cut_max_dynamism` (target 1e6): rejected.
4. **Density** above `cut_max_density` of the columns (only for more than 40 columns): rejected.
5. Scale to a largest coefficient of 1 and **relax the right-hand side** by `cut_rhs_relaxation` (relative, 1e-9) against
   rounding.
6. **Efficacy** `(g x* - rhs) / ||g||` must exceed `cut_min_efficacy` (1e-4).
7. Selection: most efficacious first, at most `cut_max_per_round` (100), dropping cuts whose cosine with an already
   chosen one exceeds `cut_max_parallelism` (0.95; opposite orientation is kept).

All numbers are targets, not tuned.

## Root loop (`run_root_cut_loop`)

Separate, clean, select, add, resolve; stop when the LP solution is integral, no cut survives, the bound gains less
than `cut_min_progress` (relative 1e-6) for `cut_stall_rounds` (3) rounds, or after `cut_rounds` (20). A cut slack for
`cut_age_limit` (5) rounds is removed; at the end all slack cuts are removed, so only binding cuts reach the tree. If a
resolve fails numerically the cuts are dropped and the search continues on a fresh engine (reported as
`abandoned`). If the LP with cuts is infeasible the model is reported infeasible **without a certificate** (the cuts
are not part of a Farkas proof of the original LP).

## Tests

`tests/test_cuts.cpp`, `tests/test_cuts_pipeline.cpp`, `tests/support/cut_harness.*`, `tests/support/cut_families.*`.

* **Validity** with presolve OFF: for every candidate cut of every round on 720 seeded models of eight families, every
  feasible integer assignment is enumerated; for continuous columns the maximum of the cut over the continuous
  polytope is computed with the dense reference LP. Any violation (tolerance 1e-9 relative for integer-only cuts, 1e-7
  with an LP) fails the build and prints family, seed, separator and the cut. Counts per separator are printed; below
  50 candidate cuts a separator reports `inconclusive for this separator`. The generators were tuned so that 72.8% of the
  models have a fractional root LP (the test requires at least 60%; set cover and set packing are the weakest families).
  Reference run: 524 models with cuts, 80768 feasible points, 4.66 million (cut, point) checks, 0 violations.
* The checker itself is tested with a deliberately invalid cut.
* Full pipeline, 1000 seeds, half with presolve on, against brute force; the root bound with cuts is valid and not
  weaker than without.
* Early termination: 200 runs with a node limit of 1 to 5, bound and incumbent valid.
* Determinism: 200 instances solved twice, identical results.
* Ablation (informational, generated instances only): printed by `cuts_ablation_on_generated_instances_is_informational`.
  On these tiny models the Gomory cuts do most of the work, and removing the clique or implied-bound separators changes
  nothing; that says nothing about real instances, none of which were available.
