# Sparse LU of simplex bases

This document describes `BasisFactor` (`include/shodhan/basis_factor.hpp`, `src/linalg/`): a
sparse LU factorization of a simplex basis, the two triangular solves FTRAN and BTRAN, and the
Forrest-Tomlin update. It also separates what follows the published literature from what is a
decision made in this code base. All numbers marked *default* or *target* are starting points,
not guarantees.

## 1. Setting

The computational form is in [CONVENTIONS.md](CONVENTIONS.md): the model is `A x - r = 0`, the
logical column of row `i` is `-e_i` (variable `n + i`), and a basis is a list of `m` variable
indices. `B` is the `m x m` matrix whose column at *basis position* `p` is the column of
`basis[p]`.

Two index spaces matter and are never mixed:

| Space | Indexed by | Used for |
|-------|-----------|----------|
| row space | rows `0..m-1` | right-hand side of `B x = a`; solution `y` of `B^T y = c` |
| position space | basis positions `0..m-1` | solution `x` of `B x = a`; right-hand side `c` of `B^T y = c` |

`ftran` maps row space to position space and `btran` position space to row space, both in place
on a `SparseWork`.

`SparseWork` (`sparse_work.hpp`) is a dense value array, an index list of the entries that may be
nonzero, and a mark array. Invariant: every nonzero is listed, no entry is listed twice. `clear()`
costs O(count), not O(m); `drop_small(tol)` removes entries with `|v| <= tol` (absolute, default
1e-14); `reindex()` rebuilds the list after a kernel wrote into the dense array.

## 2. Factorization

### 2.1 Storage while eliminating

The active submatrix is stored column-wise with values (row index and value per entry) and
row-wise as a pattern only (column indices). Each side lives in a `ListArena`
(`include/shodhan/detail/list_arena.hpp`): all lists share two flat arrays, a list that outgrows
its capacity moves to the end of the arena, and the arena is compacted into a retained second
buffer when holes pile up. There is no vector of vectors in the factorization or in the solves.
Offsets are `std::size_t`; indices and lengths are `int32`. `factorize` throws
`std::overflow_error` if the dimension or the nonzero count does not fit in `int32`.

Values in columns (rather than rows) make the threshold test against the **column maximum**
cheap and give the multipliers of a pivot directly.

### 2.2 Pivot selection

Rows and columns of the active submatrix are kept in count buckets (doubly linked lists indexed
by the number of active entries). The search follows the Markowitz scheme with limited search,
as in Suhl and Suhl:

1. For `cnt = 1, 2, ...`: examine the columns with `cnt` active entries, then the rows with `cnt`
   entries. A column is examined by listing its entries that satisfy the threshold test
   `|a_ij| >= u * max_i |a_ij|` (`u` = `pivot_threshold`, default 0.1); a row by looking up each of
   its entries in its column and applying the same test.
2. The Markowitz count of a candidate is `(r_i - 1)(c_j - 1)`. The best candidate is the
   smallest count; ties go to the larger magnitude, then to the smaller row index, then to the
   smaller column index.
3. The search stops when a candidate with count 0 is found (a singleton), when
   `markowitz_search` candidate rows/columns have been examined and a candidate exists
   (default 4), or when the best count is at most `cnt * cnt` (nothing found later can be
   better).

Everything is deterministic: bucket order, tie-breaking and the order of every loop are fixed.

**Singleton triangularization.** Column singletons (one active entry in the column) and row
singletons have count 0 and are always found first, before any pivot of a larger count. Such
pivots produce no fill and no numerical search: a column singleton creates no multipliers and a
row singleton only a column of multipliers with no change to other rows. Because the search takes
them in the same loop, the effect is the one of a separate peeling pass over the matrix
(including the singletons that appear after Markowitz pivots), but there is no separate pass: it
is one mechanism. `FactorStats` counts singleton pivots (split into column and row singletons)
and Markowitz pivots separately.

### 2.3 Elimination step

For the pivot `(r, c)` with value `p`:

- the multipliers are `l_i = a_ic / p` for the other rows `i` of column `c`; they are stored as
  one L column eta (only nonempty etas are kept);
- column `c` leaves the active submatrix; row `r` becomes a row of U (its other entries, with
  values, are moved out of the active columns);
- for each remaining entry `a_rj` of row `r`, column `j` is updated by `a_ij -= l_i a_rj` through a
  scatter array; fill-in entries are appended to both the column and the row pattern;
- an updated entry whose magnitude is at most `1e-14` times the sum of the magnitudes of the two
  terms it came from is exact cancellation (rounding noise): it is removed, not stored.

### 2.4 Tolerances and rank deficiency

A column is **numerically empty** when its largest active entry is at most `abs_pivot_tol`
(default 1e-11, target) or at most `rel_pivot_tol` (default 1e-11) times the largest magnitude the
column had before elimination. Such a column gets no pivot: it is removed from the active
submatrix and recorded as deficient. A column that loses all its entries by cancellation is
deficient too. When the elimination ends, `deficient_positions()` lists the basis positions
without a pivot (ascending) and `unpivoted_rows()` the rows without one (ascending); both lists
have the same length, and `factorize` returns `RankDeficient` and leaves the object unusable for
solves until `repair()` succeeds.

Which of two duplicate columns is reported is a result of the pivot order and the tie-breaking
(deterministic, but arbitrary): the guarantee is that the remaining columns are independent and
span the same space.

### 2.5 Result storage

| Part | Storage |
|------|---------|
| L | etas in elimination order: pivot row, entries `(row, multiplier)` in flat arrays with `size_t` offsets; a row-wise copy (for each row, the etas that touch it) is built for the hypersparse btran |
| U | off-diagonal entries twice, in two `ListArena`s: by row `(position, value)` (used by btran and by the update) and by position `(row, value)` (used by ftran); diagonal in an array indexed by pivot row |
| permutations | `pos_of_row`, `row_of_pos` (the pivot pairing) and a triangular order given by *slots*: `row_of_slot[s]` (−1 for a dead slot) and `slot_of_row[r]` |
| R (updates) | row etas of the Forrest-Tomlin updates, flat arrays |

A pivot is identified by its row. After an update the pivot of the replaced column moves to a new
slot at the end of the order and its old slot is marked dead; slot arrays are sized in
`factorize` for `max_updates` extra slots, so updating never reallocates them.

## 3. Repair

`repair(A, basis_vars)` after a `RankDeficient` factorization pairs the deficient positions
(ascending) with the unpivoted rows (ascending), replaces the basis variable at each deficient
position by the logical variable `n + row`, refactorizes, and returns the
`(position, old_var, new_var)` triples. The new columns are unit vectors of exactly the rows the
elimination could not pivot on, so for an exactly singular basis the repaired basis is nonsingular
after this one round. For a *numerically* singular basis the refactorization may choose other pivots
and find new deficiencies; the round is then repeated, at most 8 times, and the triples returned are
the net substitutions relative to the basis passed in (ascending by position). `status()` reports the
result of the last refactorization (`RankDeficient` only if the rounds ran out). The procedure has no
random or address-dependent step: identical inputs give identical substitutions.

## 4. Solves

With `L^-1 = E_K ... E_1` (column etas), `R_1..R_k` the row etas of the updates and `U` upper
triangular in the slot order:

```
FTRAN   x = U^-1 R_k ... R_1 L^-1 a
BTRAN   y = L^-T R_1^T ... R_k^T U^-T c
```

Each triangular stage has two implementations that perform the same arithmetic in a different
loop order:

- **dense**: a loop over all etas or slots, skipping zero entries; the index list is rebuilt
  afterwards by one pass (`reindex`);
- **hypersparse**: the set of positions that can become nonzero is found first by a depth-first
  search over the graph of the triangular factor (edges from a pivot to the entries it updates),
  giving a topological order; only those pivots are processed. Work is proportional to the
  nonzeros touched. This is the symbolic-reach idea of Gilbert and Peierls.

A stage takes the hypersparse path when the number of listed nonzeros is below
`hyper_threshold * m` (default 0.10, target); `0` forces dense and a value above 1 forces
hypersparse. `FactorStats` counts stages per path (`hyper_solves`, `dense_solves`). Results are
cleaned with `drop_small(drop_tol)` (absolute, default 1e-14).

`ftran(rhs, save_spike = true)` also keeps the vector after `L` and the row etas (the *spike*) and
the final solution, for `update`. Solves with `save_spike = false` and `btran` do not disturb the
saved spike.

No buffer is allocated inside `ftran`, `btran` or `update` once the object is warm: the work vectors,
the DFS stacks and the heap are sized in `factorize`, and the vectors and arenas that grow with updates keep
their capacity across refactorizations. A test counts allocations (`tests/test_lu_alloc.cpp`).

## 5. Forrest-Tomlin update

Replacing the column at position `p` by `a_q`, with `s = R_k ... R_1 L^-1 a_q` the saved spike and
`r0` the pivot row of `p`:

1. **Pivot to the end.** Column `p` of U is replaced by `s`, and the pivot `(r0, p)` moves last in
   the triangular order. Row `r0` now has entries to the left of the diagonal (in the columns of the
   pivots that used to follow it).
2. **Row elimination.** These entries are eliminated against the rows of the later pivots, in
   triangular order (a heap over slots, because elimination creates entries in later columns): row
   `r0` -= `sum_j mu_j * row(r_j)`. The multipliers form the new row eta `R` (`w[r0] -= sum mu_j w[r_j]`).
3. **New diagonal, two ways.** From the spike, `d_new = s[r0] - sum_j mu_j s[r_j]`. From the old
   diagonal and the pivot element of the saved solution, `d_alt = d_old * x[p]`. In exact arithmetic
   they are equal, because the determinant ratio of the new and the old basis is `x[p]`.
4. **Stability check.** If `|d_new - d_alt| > update_mismatch_tol * max(|d_new|, |d_alt|)`
   (default 1e-8, target), or `|d_new|` or the pivot element `|x[p]|` is at most `update_pivot_tol`
   (default 1e-11, target), `update` returns `NeedRefactor`.
5. **Commit.** Otherwise the old column and the old row are removed from both copies of U, the
   spike entries are inserted into the column list and the row lists, the pivot gets a new slot,
   the diagonal is `d_new`, and the row eta is appended.

Limits checked before anything is modified: no saved spike, `update_count >= max_updates`
(default 100, target), no free slot, or `nnz(L) + nnz(U) + nnz(R) > max_growth * (nnz(L) + nnz(U))` of
the fresh factor (default 3, target).

**State after `NeedRefactor`.** The elimination of step 2 and the checks of steps 3-4 only read U; the
row eta is built in the appended part of its arrays and truncated on failure. So after
`NeedRefactor` the factorization is exactly as before and still valid for the old basis. The caller
is expected to update its basis header and call `factorize` with the new basis. The saved spike
stays valid until it is consumed by a successful update, replaced by another `ftran` with
`save_spike = true`, or discarded by `factorize`.

## 6. Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pivot_threshold` | 0.1 (target) | relative threshold `u` against the column maximum |
| `abs_pivot_tol` | 1e-11 (target) | absolute size below which a column is numerically empty |
| `rel_pivot_tol` | 1e-11 | same, relative to the column's original largest entry |
| `markowitz_search` | 4 (target) | candidate columns/rows examined after a pivot was found |
| `drop_tol` | 1e-14 | absolute drop tolerance for solve results and saved spikes |
| `max_updates` | 100 (target) | updates before a refactor is requested |
| `max_growth` | 3 (target) | growth of `nnz(L)+nnz(U)+nnz(R)` over the fresh factor |
| `update_mismatch_tol` | 1e-8 (target) | tolerance of the two diagonal computations |
| `update_pivot_tol` | 1e-11 (target) | smallest acceptable new diagonal / pivot element |
| `hyper_threshold` | 0.10 (target) | density below which a stage runs hypersparse |

### Growth indicators

`FactorStats` carries `max_abs_basis` (largest magnitude in `B`), `max_abs_u` (largest magnitude of a pivot or
off-diagonal of `U`, including entries added by updates), `growth = max_abs_u / max_abs_basis`, and
`min_pivot_ratio` (smallest `|pivot|` divided by the largest original magnitude of its column; after an update
the new diagonal is divided by `max_abs_basis` instead, which can only make the ratio smaller). They are
indicators for the caller, which can refactorize or distrust the basis when growth is large or the ratio tiny.
They are not condition numbers: a triangular basis can be badly conditioned with growth 1.

## 7. Failure modes and limits

- **Borderline rank.** For a basis that is numerically within rounding of singular, "the rank" is
  not well defined: the tests only require the factorization's rank to lie between the dense
  oracle's rank at relative tolerances 1e-8 and 1e-14 (on the equilibrated matrix). An accepted
  pivot is not a certificate of good conditioning.
- **Rank detection is noise-limited, not rank revealing.** In a stress run over 15000 seeded bases of
  random LPs and structured families, `rel_pivot_tol` 1e-11 missed one of two dependent columns in 2 bases whose
  coefficients span about eight decades (unscaled data); 1e-9 instead flagged one healthy column in 1 basis, and 1e-8
  in 8. No single value removes both kinds of error, which is why the default stays at 1e-11 and the model should
  be scaled first.
- **Accepted but badly conditioned bases.** A basis can pass all pivot tests and still have a huge
  condition number (a triangular-like basis with large off-diagonals, for example: the crash basis of a randomly
  generated LP with coefficients over three decades gave FTRAN results with entries around 1e29 while the relative
  residual stayed at rounding level).
  LU with threshold pivoting is backward stable, so the residual `||B x - a|| / (||B|| ||x|| + ||a||)` stays
  small, but the digits of `x` are only as good as `kappa * eps` allows. The factorization does not estimate
  `kappa`; the simplex code that uses it must watch growth (for example the size of FTRAN results) itself.
  `shodhan factor-bench` prints the largest result entry as a hint.
- **The drop tolerance is absolute.** Dropping solution entries of size 1e-14 is harmless for a scaled
  model, but with coefficients up to 1e6 a dropped entry can contribute up to about 1e-8 to the residual
  (seen in a test: worst residual 1.9e-9 with the default, 1.8e-16 with `drop_tol = 0`). Scale the model
  (step 2) or lower `drop_tol` for unscaled data.
- **Threshold on columns only.** Multipliers are bounded by `1/u`; entries of U are not bounded.
- **Hypersparse is not always a gain.** It only pays when the right-hand side and the result are sparse. The
  switch is a fixed density test per stage; there is no prediction of the result density. The test
  `lu_hypersparse_quick_timing_report` prints a timing of both paths on a generated nearly triangular basis for the
  record; no performance claim is made here, and the result for other matrices may be different.
- **Update cost grows with the triangular structure.** The row elimination of an update is not bounded by a
  small constant: a dense row of U makes it expensive; `max_growth` and `max_updates` bound the damage.
- **Allocation.** The first factorizations and updates grow buffers; the no-allocation property holds after
  that warm-up, not from the first call.
- **Not thread-safe.** One `BasisFactor` per thread.
- **Range.** `m` and the nonzero count must fit in `int32`.

## 8. What follows the literature and what is a decision here

Follows the literature:

- the Markowitz criterion with threshold partial pivoting, count buckets and a limited search, and the
  LP-specific preference for singletons (Markowitz 1957; Suhl and Suhl 1990);
- storing L as column etas and keeping a row-wise and a column-wise copy of U for the two solves
  (Suhl and Suhl 1990, 1993);
- the Forrest-Tomlin update: replace the column by the partially transformed spike, move the pivot to the
  end, eliminate its row into a row eta, and use the new diagonal as the pivot (Forrest and Tomlin 1972;
  the bookkeeping with two copies of U resembles Suhl and Suhl 1993);
- symbolic reach by depth-first search for sparse triangular solves (Gilbert and Peierls 1988) and the use of
  it for the hypersparse simplex solves with a density switch (Hall and McKinnon 2005).

Decisions made in this code (the derivation is in sections 2 to 5, nothing was copied from another solver):

- one elimination loop in which singletons are simply the first candidates, instead of a separate pre-pass;
- values stored by column and patterns by row during elimination, so that the column-maximum threshold test is
  cheap; the threshold is against the column maximum only;
- the exact-cancellation rule (`1e-14` times the terms) and the relative tolerance `rel_pivot_tol` against the
  original column magnitude;
- pivots identified by their row, with a slot array and dead slots to represent the triangular order after an
  update, and a heap over slots for the row elimination;
- the second diagonal `d_old * x[p]` taken from a saved copy of the solution of the entering column;
- `update` leaves the factorization untouched on `NeedRefactor` (the row eta is built in a scratch region and
  truncated);
- `repair` pairing deficient positions and unpivoted rows by ascending order;
- the values of the defaults other than the ones the specification gave, and the density switch applied per stage
  from the current number of listed nonzeros.

## 9. References

- H. M. Markowitz, "The elimination form of the inverse and its application to linear programming",
  *Management Science* 3(3), 1957.
- U. H. Suhl and L. M. Suhl, "Computing sparse LU factorizations for large-scale linear programming bases",
  *ORSA Journal on Computing* 2(4), 1990.
- U. H. Suhl and L. M. Suhl, "A fast LU update for linear programming", *Annals of Operations Research* 43, 1993.
- J. J. H. Forrest and J. A. Tomlin, "Updated triangular factors of the basis to maintain sparsity in the
  product form simplex method", *Mathematical Programming* 2, 1972.
- J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time proportional to arithmetic operations",
  *SIAM Journal on Scientific and Statistical Computing* 9(5), 1988.
- J. A. J. Hall and K. I. M. McKinnon, "Hyper-sparsity in the revised simplex method and how to exploit it",
  *Computational Optimization and Applications* 32, 2005.
