# The pooling driver

`shodhan pool` solves the standard **pooling problem** in the q-formulation. The problem is NONCONVEX (bilinear), so it
lives outside the LP/MILP/QP core (`include/shodhan/pooling.hpp`, `src/pooling/`, `src/cli/pool_command.cpp`). It builds
LPs with its own model code and solves them with `LpSolver`. It returns **local** solutions only, plus a McCormick upper
bound; it never claims global optimality unless the gap is within the tolerance, and it never returns a point that has not been
checked against the original nonlinear equations. All instances shipped with the repository are **synthetic** (invented
numbers); nothing here is taken from a publication.

## 1. The problem

Sources `s` (cost `c_s`, supply `A_s`, qualities `q_sk`), pools `p` (capacity `C_p`), terminals `t` (price `pi_t`, demand
`D_t`, maximum quality `Q_tk`), qualities `k = 1..K`. Flows `f_sp`, `f_pt`, `f_st >= 0` on the arcs of the network (source to
pool, pool to terminal, source directly to terminal); pool qualities `q_pk`.

    max   sum_t pi_t (sum_p f_pt + sum_s f_st)  -  sum_s c_s (sum_p f_sp + sum_t f_st)
    s.t.  sum_p f_sp + sum_t f_st <= A_s                                   (supply)
          sum_s f_sp <= C_p                                                (pool capacity)
          sum_p f_pt + sum_s f_st <= D_t                                   (terminal demand)
          sum_s f_sp = sum_t f_pt                                          (pool material balance)
          sum_s q_sk f_sp = q_pk * sum_t f_pt                              (pool quality balance, bilinear)
          sum_p q_pk f_pt + sum_s q_sk f_st <= Q_tk (sum_p f_pt + sum_s f_st)   (terminal quality, bilinear)

A pool quality balance is only meaningful for a pool with throughput; the pool quality of a pool that carries nothing is
free. `q_pk` always lies between the smallest and the largest quality of the sources that feed the pool `p`.
A specification `Q_tk = inf` means no specification for that quality.

## 2. The `.pool` file

Line based, plain text, no JSON. `#` starts a comment (anywhere on a line). Tokens are separated by blanks. Numbers are
decimal text (`3`, `2.5`, `1e-3`) or `inf` where a bound is allowed. Names contain no blank and none of `>`, `:`, `"`.

    name NAME                      optional; defaults to the file name
    synthetic yes|no               optional; yes makes the CLI print the SYNTHETIC notice
    qualities N [NAME1 ... NAMEN]  required before any source; N from 1 to 16; names default to Q1 ... QN
    source NAME COST SUPPLY q1 ... qN        supply may be inf; qualities must be finite
    pool NAME CAPACITY                       capacity may be inf
    terminal NAME PRICE DEMAND Q1 ... QN     demand and the maximum qualities may be inf
    arc FROM TO                              source->pool, pool->terminal or source->terminal

Errors are reported as `file:line: message`: unknown keyword, duplicate name or arc, a name with a forbidden character, an unknown name in
an arc, an arc between the wrong kinds, a wrong number of fields, a missing or repeated `qualities` line, a negative supply, capacity or
demand, an infinite cost, price or source quality, a pool without an incoming or without an outgoing arc, no arc that leaves a source, no
source or no terminal. The verifier (KASAUTI) has its own parser of this format and the two are compared by a differential test.
`tests/models/pool_synth_one_pool.pool` is a complete example.

## 3. What a returned point must satisfy (`check_pool_point`)

Every residual is **relative**: `residual / (1 + scale)` where `scale` is the sum of the absolute values of the terms behind
it. With `out_p = sum_t f_pt`, `in_p = sum_s f_sp`, `F_t = sum_p f_pt + sum_s f_st`:

| Residual | Definition |
|----------|------------|
| bounds | `max(0, -f)` over all flows, and for every pool quality the amount by which it leaves the range of the source qualities that feed its pool |
| supply, capacity, demand | `max(0, activity - limit) / (1 + abs(activity) + limit)` for each finite limit |
| material | `abs(in_p - out_p) / (1 + abs(in_p) + abs(out_p))` |
| quality balance | `abs(sum_s q_sk f_sp - q_pk out_p) / (1 + sum_s abs(q_sk f_sp) + abs(q_pk out_p))` |
| terminal quality | `max(0, sum_p q_pk f_pt + sum_s q_sk f_st - Q_tk F_t) / (1 + sum_p abs(q_pk f_pt) + sum_s abs(q_sk f_st) + abs(Q_tk) abs(F_t))` |

A point is accepted when the largest residual is at most the tolerance (`--tol`, *default* 1e-6). A point that fails is never
reported. The same definitions are implemented independently in KASAUTI (`verify/kasauti/pool.py`, which was written from
this section) and evaluated in exact rational arithmetic.

## 4. The methods

**Fixed-quality LP.** With every `q_pk` replaced by a constant, the terminal rows are linear:
`sum_p (q_pk - Q_tk) f_pt + sum_s (q_sk - Q_tk) f_st <= 0`. The pool quality balance is **not** enforced (it would be bilinear):
the LP may mix the sources of a pool as it likes, so its solution generally violates the balance for the assumed `q`.

**Distributive recursion** (`--method recursion`). Solve the fixed-quality LP; recompute the actual pool qualities from the source
mix, `q_implied = sum_s q_sk f_sp / sum_s f_sp`; set `q <- q + a (q_implied - q)` (the damping `a`, *default* 0.5; `--damping`);
repeat. It stops when the largest change of a pool quality is below `0.01 * tol * (largest quality)`: at that fixed point the
pool balances hold and the point is checked. A pool without throughput has no implied quality and keeps its value. The
iterates are stored; **revisiting an earlier iterate is reported as cycling** (status `Cycling`: with `--damping 1` some
instances oscillate between two qualities), and a run that does not stop within `--max-iter` (*default* 200) ends with
`IterationLimit`. Only a point that satisfies the nonlinear model within the tolerance is ever returned, and it is returned
labelled "not a converged solution". Reference: Haverly (1978), *to be verified by the author*.

**Sequential LP** (`--method slp`). Linearize the bilinear terms around the current point `(f0, q0)`:
`q T ~ q0 T + T0 q - q0 T0` with `T = sum_t f_pt`, and `q f_pt ~ q0 f_pt + f0 q - q0 f0`, add elastic variables for the
linearization residuals with a penalty weight `mu` (*default* `10 (max price + 1) / (largest quality)`, raised tenfold up to six
times when the residual does not vanish at a stationary point), put a trust region on the pool qualities (relative radius
0.25 of each range initially), solve the LP, accept or reject the step by the merit function
`objective - mu * (sum of absolute residuals of the nonlinear constraints)`, and adapt the radius (doubled after a good
step at the boundary, halved after a poor or rejected step). The point it returns has its pool qualities set to those implied
by the flows and is checked. Of all points met along the way, the one with the highest objective among those that pass a tolerance
1000 times tighter is kept (points that pass only the tolerance itself could owe part of their objective to its slack; this was
found in testing). Each LP is solved from scratch by `LpSolver`: **no warm start** is used between iterations. References:
Griffith and Stewart (1961), Baker and Lasdon (1985), *to be verified by the author*.

**Multi-start** (`--starts N`, `--seed s`). Start 0 sets every pool quality to the strictest specification of the terminals the
pool feeds (so the terminals can take material from the pool), start 1 to the midpoint of the ranges, the others are seeded random
(own splitmix64 generator, so results do not depend on the standard library). The best point that passes the check is kept; the
outcomes (objective value, number of starts) are printed.

**McCormick upper bound.** Each product `q_pk f_pt` is replaced by a variable `v` with the four McCormick inequalities from the
bounds `q in [qL, qU]` (range of the feeding sources) and `f in [0, fU]` (`fU = min(demand, pool capacity, supply that can reach the
pool)`, which must be finite): `v >= qL f`, `v >= qU f + fU q - qU fU`, `v <= qU f`, `v <= qL f + fU q - qL fU`. The pool quality
balance becomes `sum_s q_sk f_sp = sum_t v_pkt` and the terminal quality rows use `v`. The LP maximum is an upper bound of the
maximum of the pooling problem. It is computed in floating point by the simplex code: **KASAUTI does not verify it** and says so.
Piecewise McCormick (partitioned ranges with binaries) is not implemented.

## 5. Output, status and exit codes

`shodhan pool spec.pool` prints the local value, the McCormick bound and the relative gap, and states plainly whether the gap is within
the tolerance (global up to that tolerance and the accuracy of the bound) or **no global-optimality guarantee** (the usual case). Exit
codes: 0 converged and verified, 1 usage or read error, 3 no verified point (failure), 4 a verified point that is not a converged solution
(cycling, iteration limit).

## 6. Certificate (additive, see CERTIFICATES.md)

`--write-cert` writes status `feasible` with `flows` (arc `FROM>TO` to value), `q` (`POOL:QUALITY` to value), `claimed_objective`,
`claimed_upper_bound` (the McCormick bound, when computed), `converged`, `optimality_certified: false` and `nonconvex: true`; status `other`
without a body when no verified point exists. `python -m kasauti spec.pool cert.json` checks it (KASAUTI.md).

## 7. Limits

* Local solutions only: no spatial branch and bound, no global guarantee.
* The recursion is weak: it can end at a trivial fixed point (an idle pool whose assumed quality exceeds a specification keeps it for
  ever) and it cycles or stalls on some instances. The sequential LP is the more reliable method on the synthetic instances of the tests
  (numbers in the step report and in `tests/test_pooling.cpp` output); neither is claimed to be fast.
* The model has no minimum demands: the zero flow is always feasible, so a "no solution" outcome means that the methods failed, not that the
  instance is infeasible.
* Only maximum-quality specifications; no minimum qualities, no quality-dependent prices.
* No warm start between SLP iterations, no piecewise McCormick, no multiple periods.
