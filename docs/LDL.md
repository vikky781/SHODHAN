# Sparse LDL^T for quasi-definite systems

`SparseLdl` (`include/shodhan/sparse_ldl.hpp`, `src/linalg/sparse_ldl.cpp`, `src/linalg/amd.cpp`) factors the symmetric
matrices of the interior-point method ([IPM.md](IPM.md)). It is written from the literature, with no third-party code.
Values marked *default* or *target* are starting points, not guarantees.

## 1. Pipeline

1. **Ordering.** Approximate minimum degree on the graph of `K + K^T` (`amd.cpp`): quotient graph with element
   absorption, supervariable detection, mass elimination and approximate external degree (Amestoy, Davis, Duff 1996).
   There is no special handling of dense rows, so one dense row of `A` makes the ordering and the fill poor.
2. **Symbolic analysis.** Elimination tree and column counts after Liu; the pattern of `L` is allocated once.
3. **Numeric factorization.** Up-looking `L D L^T`, row by row, without pivoting: the matrix is expected to be
   quasi-definite (a positive block and a negative block, each shifted by a regularization), for which every
   permutation has a factorization (Vanderbei 1995).
4. **Solve.** Forward, diagonal and backward substitution, followed by iterative refinement against the
   *unregularized* matrix.

## 2. Regularization and pivots

- Static regularization by sign block: `+rho` on the positive block, `-delta` on the negative block (*default* `rho = delta = 1e-9`, `LdlParams`).
- A pivot whose sign is wrong or whose absolute value is below `pivot_tol` (*default* `1e-13`) is replaced by
  `sign * dynamic_delta` (*default* `1e-8`). The threshold is **absolute**, not relative: a relative threshold replaced healthy small pivots and made the
  interior-point method fail in a measurable share of the generated problems (see IPM.md, section 6).
- The number of replacements is reported (`LdlStats::dynamic_regularizations`) so that a caller can tell a clean factorization from a
  repaired one.

## 3. Refinement

The factorization is of the regularized matrix, so a solve is a preconditioned iteration against the true matrix:
a bounded number of steps (`solve` takes `max_refinement`, default 3; the interior-point method asks for more),
stopping on a componentwise backward error and early when the error stops decreasing. The achieved error is
returned in `LdlStats::residual` and the steps taken in `refinement_steps`, not hidden.

## 4. Allocation

Everything is allocated in `analyze` and `factor`. `solve` allocates nothing; `tests/test_lu_alloc.cpp` counts
allocations around repeated solves.

## 5. Tests

`tests/test_sparse_ldl.cpp`: AMD returns a permutation and reports its work; the symbolic analysis matches a dense
symbolic elimination; the factorization matches a dense oracle (`tests/support/dense_ldl.*`, an oracle only) on random
quasi-definite matrices; dynamic regularization is counted and refinement recovers accuracy; refactorization reuses
the analysis; and the fill of AMD against the natural order is printed as a table (informational, except that
the arrow-head case must not fill in). Exact counts of a run are in the step report.

## 6. Known limits

- No dense-row handling in AMD; no supernodes, so the numeric phase is a scalar code (speed is *not* claimed).
- Quasi-definiteness is assumed, not checked: an indefinite input is repaired by pivot replacement, and the
  refinement result then tells whether that was enough.
- Single-threaded.
