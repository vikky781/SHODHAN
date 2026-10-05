# Synthetic refinery and process case studies

`bench/refinery/` generates MPS models for refinery and process-industry planning, scheduling and logistics. **Every model is
SYNTHETIC: its structure follows textbook formulations; it is not plant or MRPL data.** The line
`SYNTHETIC: structure follows textbook formulations; not plant or MRPL data` is the first line of every `.mps` file (an MPS comment), the
first line of every `.meta.txt` sidecar, and is printed by `run_suite.py` and `ablation.py`. No instance data from a paper is used or typed
from memory; model structures are described generically, and no citation is given because none is claimed to be exact: **citations
are "to be verified by the author"**. To plug in real data, write an MPS file in the same variable and row naming, or replace the
generator's data tables; nothing else changes.

## Reproducibility

`gen_core.Rng` is a generator of our own (splitmix64 seeding, xorshift64\*), so the same arguments give byte-identical files on every platform
and Python version. All coefficients, bounds and witness values are dyadic rationals (multiples of 1/16 or integers), written as their
**exact** decimal expansion, so a reader that parses the text exactly sees the same numbers, and a witness point that is feasible by
construction is feasible in exact arithmetic (`check_witness.py` verifies that with KASAUTI's parser). Each model has a sidecar `NAME.meta.txt`:
the family, the size knobs, the seed, row, column, binary and nonzero counts, the witness point and its objective value.

Command lines: `python bench/refinery/generate.py --family R3 --size small --seed 1 [--loose 50] [--ties 0.3] [--spread 1] --out DIR`
or `--suite DIR [--sizes tiny,small,medium,large]` (R2 also has `--size huge`).

## Knobs

| Knob | Meaning |
|------|---------|
| size | `tiny`, `small`, `medium`, `large` (R2: `huge`, more than 100000 columns) |
| seed | fixes everything |
| loose | at least 1: factor on the big-M coefficients of the links (R3, R4, R5); the suite also writes `_weak` variants with 50 |
| ties | share of cost coefficients replaced by one common value: degeneracy (many ties) |
| spread | decades of coefficient range: costs are multiplied by a power of two up to 2^(3 spread) |

The physical limits (pumping rate, CDU maximum feed, blender capacity, capacities of facilities) are rows of their own, so `loose` only
weakens the big-M **links**: the feasible set and the optimum of a model are the same for every `loose` (the C++ test `refinery_tiny_models_...`
checks that on the tiny models; an earlier draft folded the rate into the big-M and the check caught it).

## The families

Names: `R<family>_<size>_<seed>[_weak]`.

**R1 crude selection and blending (LP).** Crudes `c` (price `p_c`, availability `a_c`), streams `s`, products `p`, qualities `k`.
Variables: purchases `B_c in [0, a_c]`, stream-to-product flows `F_sp >= 0`, stream sales `W_s >= 0`, product sales `S_p in [0, D_p]`.
Rows: CDU capacity `sum_c B_c <= Cap`; stream balance (fixed yields `y_cs`, sum over `s` at most 1) `sum_c y_cs B_c = sum_p F_sp + W_s`;
product `sum_s F_sp = S_p`; linear blending with a fixed quality index per stream `sum_s (q_sk - Qmax_pk) F_sp <= 0`. Objective (minimized):
`sum_c p_c B_c - sum_p pi_p S_p - sum_s r_s W_s` (a negated margin). Structure: crude assay and blending LP of the textbook kind.

**R2 multi-period planning (LP).** Periods `t`, crudes `c`, products `p`. Variables: purchases `B_ct <= a_ct`, crude inventory `V_ct <= tank_c`,
processed crude `R_ct`, product inventory `I_pt >= 0`, backlog `K_pt >= 0`. Rows: crude balance `V_ct = V_c,t-1 + B_ct - R_ct`; unit capacity
`sum_c R_ct <= cap_t`; product position `I_pt - K_pt = I_p,t-1 - K_p,t-1 + sum_c y_cp R_ct - d_pt` (fixed yields). Objective: purchase plus
holding plus backlog penalty. The witness buys and processes nothing and backlogs the cumulative demand. Scales with `T x (3 C + 2 P)`
columns: `huge` is 52 periods, 520 crudes, 210 products (more than 100000 columns).

**R3 crude-oil scheduling (MILP, time-discretized).** Vessels `v` (arrival slot `a_v`, parcel `Q_v`), tanks `k` (capacity `cap_k`, initial
inventory), one CDU, slots `t`. `X_vkt` unloaded volume, `Z_vkt` binary "v unloads into k in t" (link `X <= min(Q_v, cap_k) loose Z`);
`sum_k Z_vkt <= 1`; one berth `sum_vk Z_vkt <= 1`; pumping rate `sum_vk X_vkt <= rate`; unfulfilled `U_v = Q_v - sum X` penalized. Tank balance
`N_kt = N_k,t-1 + sum_v X_vkt - FD_kt`, `0 <= N <= cap`; feed `FD_kt <= min(cap_k, Fmax) loose W_kt` with `W_kt` binary "k feeds the CDU in t", at
most one feeding tank per slot, CDU maximum `sum_k FD_kt <= Fmax`; settling `Z_vkt + W_kt <= 1`; CDU minimum `sum_k FD_kt + E_t >= lo` with a penalized
shortfall `E_t`; changeover `G_kt >= W_kt - W_k,t-1` at a cost. Weak variants: `loose = 50`. Witness: nothing unloaded, nothing fed, shortfall paid.

**R4 blend and changeover (MILP).** One blender, grades `g`, slots `t`. `S_gt` binary "set up for g in t" (at most one per slot), capacity
`sum_g Q_gt <= cap`, link `Q_gt <= cap loose S_gt` (fixed charge), changeover `H_gt >= S_gt - S_g,t-1` at a sequence-independent cost, position
`I_gt - K_gt = I_g,t-1 - K_g,t-1 + Q_gt - d_gt`. Witness: no production.

**R5 capacitated facility location with transportation, multi-commodity (MILP).** Facilities `f` (open `Y_f`, fixed cost, capacity `C_f`),
customers `j`, commodities `k` with demand `d_jk`. `X_fjk >= 0` at unit cost; `sum_f X_fjk = d_jk`; `sum_jk X_fjk <= C_f Y_f`; link
`X_fjk <= min(d_jk, C_f) loose Y_f`. Witness: all facilities open, customers served greedily.

**R6 unit commitment: not implemented** (optional in the specification).

## Tests

* `bench/refinery/tests` (Python, run by ctest as `refinery_unittests`): the RNG, exact number formatting, determinism (two runs, and two processes
  with different hash seeds, give identical bytes), the SYNTHETIC label, exact feasibility of every witness with KASAUTI's parser for the
  suite sizes tiny, small and medium, detection of a broken witness, at most 14 binaries for the tiny R3 and R4 models, more than 100000 columns for
  R2 huge, and that tight and weak variants differ only in the big-M coefficients.
* `tests/test_refinery_models.cpp` (ctest `refinery_cpp`, files written by `refinery_generate`): every file parses and validates in the C++ reader, a
  read-write-read round trip is stable, the tiny R3, R4 and R5 models solved by SHODHAN equal the brute-force oracle, and each tight/weak pair has the same optimum.
* `run_suite.py` solves every model, writes certificates and verifies them with KASAUTI; a model is WRONG when KASAUTI says FAIL or when an
  Optimal objective is worse than the witness. `ablation.py` runs {tightening off/on} x {cuts off/on}. `crosscheck_highs.py` is an optional
  HiGHS reference check (skipped when `highspy` is absent). Results of the runs are in the step report; this document makes no performance claim.
