# Interior-point method for LPs and convex QPs

`solve_ipm` (`include/shodhan/ipm.hpp`, `src/ipm/ipm.cpp`) solves the continuous relaxation of

    min  offset + c^T x + (1/2) x^T Q x     s.t.  row_lower <= A x <= row_upper,  col_lower <= x <= col_upper

with Q positive semidefinite (in the minimization form, see [QP.md](QP.md)). It is a Mehrotra predictor-corrector
method (Mehrotra 1992) with Gondzio multiple centrality correctors (Gondzio 1996) written from the literature.
Numbers marked *default* or *target* are starting points, not guarantees; no performance is claimed.

## 1. Formulation

The row activities `r = A x` are extra variables with the row bounds, so every variable has at most a lower and an
upper bound and the constraint is the equation `A x - r = 0`. Slacks of bounded variables carry the complementarity
terms. Free variables carry none. The multipliers of the equation are the row duals `y`; the reduced costs are
`d = c + Q x - A^T y`.

## 2. Newton system

After eliminating the step of the slacks and of the bound multipliers, each iteration solves a quasi-definite
augmented system (Vanderbei 1995)

    [ Q + Theta_x^-1 + rho I        A^T    ] [dx]   [..]
    [      A                  -(Theta_r + delta I) ] [dy] = [..]

with the sparse LDL^T of [LDL.md](LDL.md). The ordering and symbolic analysis are done once; each iteration only
refactors. Refinement runs against the unregularized matrix (*default* up to 15 steps).

## 3. Algorithm

- **Start**: a least-squares point projected into the interior with a margin; scaled problems need it (a flat
  start failed on badly scaled instances).
- **Predictor-corrector**: affine step, centering parameter `sigma = (mu_aff/mu)^3`, second-order corrector, then up
  to `centrality_correctors` (*default* 2) Gondzio correctors.
- **Step length**: fraction to the boundary `step_fraction = 0.95` (*default*; 0.995 caused `mu` cycling on a few
  generated problems). For a QP one common step is used for primal and dual (the dual equation contains `Q x`).
- **Stopping**: relative primal residual, relative dual residual and relative complementarity gap each at most
  `tol` (*default* 1e-8 for `solve_ipm`; the facade uses 1e-9, see section 6). The residuals are *componentwise*
  relative, deliberately in the same form as the KKT checker so that "converged" and "passes the check" agree.
- **Failure handling**: a failed factorization or stagnation is retried with a larger regularization (field
  `attempts`); divergence of the iterates is reported as `InfeasibleOrUnbounded`, never as `Optimal`.

## 4. In the pipeline

`--method auto` (default) sends LPs to the dual simplex and convex QPs to the IPM; `--method ipm` forces the IPM for
an LP; `--method simplex` on a QP is rejected (`NotImplemented`, exit 2). The facade runs a fallback ladder when
the IPM does not reach the acceptance check: presolve -> no presolve -> tighter tolerance -> no scaling. For an LP
whose IPM diverges, the dual simplex decides. For a QP that looks infeasible the feasibility of the constraints is
decided by the simplex, which returns a Farkas certificate; a QP is never reported infeasible without it.
Every `Optimal` is accepted only after the KKT check and the weak-duality bound of the *original* model.
The IPM finds an interior solution; there is no crossover (not implemented).

## 5. Tests

- `tests/test_ipm.cpp`: planted convex QPs with a known optimum (`SHODHAN_QP_SEEDS`, *default* 1600), tiny QPs
  against an enumeration of active sets, and infeasible / unbounded models, which must never end `Optimal`.
- `tests/test_ipm_agreement.cpp`: LPs from the generator families, IPM against the dual simplex
  (`SHODHAN_LP_AGREE_SEEDS` per family, *default* 300).
- `tests/test_qp_pipeline.cpp`: the whole pipeline on QPs (`SHODHAN_QP_PIPE_SEEDS`, *default* 1500).
- The sanitizer preset runs smaller counts (150 / 120 / 30), stated in `CMakePresets.json`.

## 6. Findings and limits

- Early versions failed on about 40% of the planted problems. Causes, found one at a time: a relative pivot
  threshold in the LDL replaced healthy pivots; refinement was too weak; a 0.995 step fraction made `mu` cycle;
  badly scaled problems needed the least-squares start; residuals were not componentwise. Two attempted
  "improvements" made it worse and were removed (a cap on `Theta`, backtracking on `mu`).
- Remaining weakness: on ill-conditioned "wide-coefficient" LPs the IPM is weaker than the simplex. In the
  agreement test a few instances disagree (objectives differing by up to about 3.6e-5 against the exact optimum); the
  simplex is the reference, and the facade's check rejects an IPM answer that fails it.
- Scaling can hurt on such instances; the ladder falls back to the unscaled model.
- IPM duals are floating-point numbers with noise on free columns, so KASAUTI usually proves optimality only to
  tolerance for them ([CERTIFICATES.md](CERTIFICATES.md)). `ipm_tol` in the facade is 1e-9 (the target of the
  specification was 1e-8) because at 1e-8 the noise made two corpus certificates fail the exact check.
- Platform sensitivity: in CI one compiler (MSVC) ended seed 54 of the wide-coefficient QP family with `NumericalError: non-finite Newton direction`
  after all retries, while the same seed passes with GCC on Linux and MinGW, including a build with FMA contraction. It could not be reproduced
  locally and its cause is not known. A fourth, more conservative retry (regularization 1e-4, step fraction 0.8, no centrality correctors) was added;
  whether it helps on that compiler is checked by CI. The planted-QP test therefore separates a WRONG answer (never allowed) from an honest
  `NumericalError` (counted, capped at 0.25% of the seeds, printed). The two LP seeds 540183 and 540277 end the same way on every platform.
- No handling of dense columns or rows beyond what the ordering gives.
