# Mixed-integer programming: branch and bound

`shodhan solve model.mps` solves models with integer columns by branch and bound (`src/mip/`, `include/shodhan/mip/`).
Everything below describes what is implemented; values marked *default* or *target* are starting points, not
guarantees, and nothing here is a performance claim.

## 1. Pipeline

1. **Presolve** with the MIP-safe reductions of step 2 only ([PRESOLVE.md](PRESOLVE.md)): no dual information is
   reconstructed. The reduced model is always a minimization. If presolve proves the model infeasible the result is
   `Infeasible`; if it finds an improving ray the result is `InfeasibleOrUnbounded`; if it solves everything, the
   point goes through the incumbent manager like any other candidate.
2. **Scaling** ([CONVENTIONS.md](CONVENTIONS.md)); integer columns keep scale 1, so bounds and integrality mean the
   same in the scaled and the unscaled space.
3. **Heuristics that need no LP** (trivial, Feasibility Jump) run before the root LP.
4. **Root LP** with the dual simplex, then the search (section 2).
5. Every solution is mapped back to the original space and **verified against the original model** by the incumbent
   manager (section 6) before it is reported.

The search runs on the presolved, scaled model; plugins (heuristics, branching rules) see the presolved model in
unscaled values. Unbounded LP relaxations are reported as `InfeasibleOrUnbounded`: a MILP with an unbounded relaxation
can still be infeasible, and no certificate is produced, so `Unbounded` is never claimed.

## 2. The search

- **Nodes** live in an arena (`NodeTree`). A node stores its parent, the branching column, direction and value, its
  lower bound, an estimate, and the bound changes it adds; the bounds of a subproblem are rebuilt from the chain of
  parents (nothing is copied per node). Ids are assigned in creation order, so every tie-break by id is deterministic.
  Nodes are not freed during a search (memory grows with the number of nodes).
- **Node LPs** are solved by the warm-started dual simplex of [SIMPLEX.md](SIMPLEX.md): the engine's bounds are moved to
  the node's bounds and the solve starts from the parent's basis. The two children share one `BasisSnapshot` (one byte
  per variable) of the parent's optimum. A snapshot is stored only while fewer than `max_stored_bases` are alive
  (*default* 2000); a node without one continues from the engine's current basis, which is still dual feasible because
  only bounds changed.
- **Node selection** (`NodeSelector` plugins): best bound (*default*), depth first, best estimate; ties by node id.
- **Plunging**: after branching the search continues into one child immediately, without restoring a basis, and queues
  the other. The preferred child is the one in the nearer rounding direction (ties go up). With an incumbent, plunging
  stops when the child's bound exceeds `best_bound + plunge_gap_fraction * (incumbent - best_bound)` (*target* 0.25).
- **Pruning**: a node is pruned when its lower bound reaches the incumbent (within 1e-9 relative). **Objective
  integrality**: if every objective coefficient is a multiple of some g on integer columns and zero on continuous ones
  (g found by trying a few decimal scale factors and taking the gcd; conservative, 0 when in doubt), every bound is
  rounded up to the next value of `offset + k g` before it is compared, which is the same as pruning when
  `bound > incumbent - g + eps`.
- **Numerical trouble**: a node LP that ends with an iteration limit or a numerical error is retried once after a
  refactorization with relaxed ratio-test thresholds. If it still fails the node is **not** discarded silently: it is
  counted in `numerical_trouble_nodes`, its parent's bound is kept in the reported best bound, and the final status
  cannot be `Optimal`.
- **Stopping**: gap (`mip_gap` relative *default* 1e-4, `mip_abs_gap` *default* 1e-6), `node_limit`, `time_limit`.
- A sense of scale: a MILP whose root LP is integral is finished after one node; the test families below need between
  zero and a few hundred nodes.

## 3. Branching

All rules are `BranchingRule` plugins (`--branching`):

- `first`: the lowest-index fractional column; `mostfrac`: the most fractional one (lowest index on ties).
- `pseudocost`: the product score `max(down, eps) * max(up, eps)` with `down = pseudocost_down * f` and
  `up = pseudocost_up * (1 - f)`. Pseudocosts (objective gain per unit change) are kept per column and direction and
  are updated after every node LP; columns without data use the global average of that direction (1 if there is none),
  which makes the rule start out like `mostfrac`.
- `reliability` (*default*): Achterberg, Koch and Martin, "Branching rules revisited", Operations Research Letters 33
  (2005). Candidates are ranked by the pseudocost score. A candidate whose pseudocosts have fewer than
  `reliability_threshold` observations (*target* 4) in either direction is measured by **strong branching**: both
  children are solved with at most `strong_iteration_limit` dual simplex iterations (*target* 100) on a **copy of the
  engine**, so the search state cannot be disturbed (the search's engine is passed as `const&`, and a test compares its
  basis, primal and dual values, bounds and iteration count before and after, bit for bit). At most
  `strong_candidate_limit` candidates are evaluated per node (*target* 10) and the evaluation stops after
  `strong_lookahead` candidates without a better score (*target* 4). The measured gains also initialize the
  pseudocosts. A child that is infeasible or no better than the incumbent tightens the column's bound at the node (the
  node is then re-solved and the rule asked again); both children cut off prunes the node; a child solved to optimality
  gives a valid lower bound for that child. A child that hits the iteration limit contributes only an estimate.

## 4. Primal heuristics

`PrimalHeuristic` plugins; each has a frequency setting (`0` = only at the root), counts calls, submissions and
successes, and offers points to the incumbent manager only.

- **trivial**: all integer columns at their lower bounds, at their upper bounds, and at zero (clamped into the bounds).
- **rounding**: simple rounding with row locks: a fractional column is rounded in a direction with no locks, so no row
  can become violated; if some column is locked both ways it gives up.
- **diving-fractional / diving-coefficient**: tighten one column in a rounding direction (nearest integer for the
  fractional dive; fewest locks in the chosen direction for the coefficient dive) and resolve with the warm-started dual
  simplex until the LP solution is integral or the dive fails. A dive works on a copy of the engine, allows one
  backtrack (the other side of the last decision), uses at most `dive_iteration_floor + dive_iteration_fraction *
  (LP iterations so far)` iterations (*targets* 500 and 0.1) and stops after a depth limit or 12 steps without fewer
  fractional columns.
- **feasibility-jump**: Luteberget and Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic", Mathematical
  Programming Computation (2023), implemented from the paper's description. It minimizes the weighted constraint
  violation; each column's best value (its jump value) comes from the breakpoints of its rows, the score is the decrease
  of the weighted violation, moves are taken from a seeded random sample of positively scored columns, and the weights
  of violated rows grow at local minima. Differences from the paper: the jump values of all columns sharing a row with
  the moved column are recomputed from scratch instead of updated incrementally, and the objective is not optimized.
  Work limit `fj_work_limit` matrix entries (*target* 2,000,000). It runs before the root LP from the zero point
  (clamped into the bounds), and at nodes from the rounded LP solution when `feasibility_jump.frequency > 0`.

## 5. Statuses and the bound reported on early termination

| Status | Meaning |
|--------|---------|
| `Optimal` | the search finished with the gap within tolerance and **no node was discarded for numerical reasons**; `has_solution` is true |
| `Infeasible` | no integer feasible point: proved by branching or presolve (no certificate), or because the LP relaxation is infeasible (then a Farkas certificate is attached) |
| `InfeasibleOrUnbounded` | the relaxation is unbounded (or presolve found an improving ray): not certified either way |
| `TimeLimit`, `NodeLimit` | stopped; `has_solution` says whether an incumbent exists |
| `NumericalError` | the root LP failed, or a node was dropped (the search is incomplete); `has_solution` may be true |

`best_bound` is a lower bound for a minimization and an upper bound for a maximization, in the model's own sense. It is
the smallest bound of every node not yet decided (open nodes, the node being processed, and any dropped node), capped by
the incumbent. It is therefore valid after a time or node limit as well; a test checks 200 node-limited runs that really
stopped with open nodes. Gaps are `abs_gap = |objective - best_bound|` and `rel_gap = abs_gap / max(1, |objective|)`.

## 6. The incumbent manager

The only way a solution can enter the search (`IncumbentManager`): a candidate in the presolved space is (1) mapped back
to the original space by postsolve, (2) snapped (integer columns to the nearest integer, rejected if that leaves their
bounds), (3) completed: if the **original** model has continuous columns, the LP with the integer columns fixed is
re-solved so that the continuous values are consistent, (4) verified on the original model (rows and columns with
`primal_tol` in the relative form `violation / (1 + |bound|)`, integrality exactly after snapping) and (5) accepted only
if strictly better. Rejections are counted by reason (`not_better`, `row_violated`, `column_violated`, `not_integral`,
`lp_resolve_failed`, `wrong_size`). A test replaces a heuristic with one that returns garbage: every candidate is
rejected and counted and the results stay correct.

## 7. Certificates

A MILP result with an incumbent is written as status `feasible` ([CERTIFICATES.md](CERTIFICATES.md)): exactly integral
integer columns, the claimed objective, the claimed best bound and gap, and `optimality_certified: false`. KASAUTI
verifies feasibility, integrality and the exact objective and prints "bound NOT verified". Infeasibility is attached as a
Farkas certificate only when the LP relaxation itself is infeasible; otherwise the certificate says `certified: false`
and the verifier reports INCONCLUSIVE.

## 8. Plugin interfaces

`NodeSelector`, `BranchingRule`, `PrimalHeuristic` and `Separator` (interface only: cutting planes come with a later
step) are abstract classes with a documented contract in `include/shodhan/mip/plugins.hpp`, registered by name in
`plugin_registry()`. The incumbent manager is the single entry point for solutions.

## 9. What was tested (generated instances only)

Ten seeded families (multi-dimensional knapsack, set cover, assignment with a side constraint, fixed-charge flow with a
weak big-M, capacitated facility location, lot sizing, general integers 0..5, mixed, infeasible by parity, unbounded),
compared with a test-only brute-force enumerator (the dense reference LP solves the continuous part) and a test-only
dense branch and bound (`tests/support/mip_oracle.*`, never linked into the library):

- 1050 seeds: 1050 agree in status and objective (820 optimal, 125 infeasible, 105 unbounded; 291 of them needed more than
  one node); incumbents verified feasible on the original model.
- 720 further solves with heuristics off, simple branching rules and depth-first or best-estimate selection (half of them
  without presolve): 0 failed, 479 branched.
- every branching rule x node selector combination (4 x 3) on 300 MILPs: 3600 runs reach the oracle optimum.
- 200 node-limited runs on correlated 14-16 item knapsacks (node limit 1 to 5): all stopped with open nodes, the bound
  was valid in all 200 and the incumbent in all 134 that had one.
- heuristics on and off give the same optimum on 315 instances; determinism: 200 instances solved twice give identical
  node counts, iterations, objectives, bounds and incumbents.
- heuristic success on generated instances: simple rounding 318 of 318 where a locked direction exists and 0 of 24 where
  every column is locked both ways; trivial 240 of 240; fractional diving 342 of 386 and coefficient diving 386 of 386
  on four families; Feasibility Jump 575 of 575 (these instances are small and easy).
- 150 MILP certificates verified by KASAUTI in exact and float mode; mutation tests (broken integrality, a broken row,
  a changed objective, a wrong-side bound, a claimed optimality) are all detected.

## 10. Limitations

- No cutting planes (the `Separator` interface has no implementation), no MIP presolve beyond the step 2 reductions, no
  restarts, no reduced-cost fixing, no conflict analysis, single-threaded.
- Strong branching and diving copy the whole engine for every probe; this is simple and safe but wasteful on large models.
- The node arena is never compacted.
- Verified only on small generated instances; no MIPLIB or other real instance was run, so nothing is known about
  behaviour on hard or large problems.
- A node whose LP cannot be solved reliably makes the search incomplete rather than being resolved by a more careful
  method.
- Unbounded MILPs are reported as `InfeasibleOrUnbounded`, not `Unbounded`.

## 11. References

- T. Achterberg, T. Koch, A. Martin, "Branching rules revisited", Operations Research Letters 33 (2005) 42-54.
- B. Luteberget, G. Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic", Mathematical Programming Computation
  15 (2023).
