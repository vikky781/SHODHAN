# LP stress harness

An opt-in program that runs the full LP pipeline on many seeded LPs and compares it with the dense
test oracle. It is not part of the default build or of `ctest`.

```sh
cmake -S . -B build/stress -DCMAKE_BUILD_TYPE=Release -DSHODHAN_BUILD_STRESS=ON
cmake --build build/stress --target shodhan_stress

build/stress/shodhan_stress <family> <first_seed> <last_seed> [--emit-mps DIR] [--quiet]
```

Families: `degenerate`, `free`, `ranged`, `boxed`, `wide`, `infeasible`, `unbounded`, or `all`.
The LP of a family for a seed is the same one the property test (`tests/test_lp_property.cpp`) uses,
so a seed that fails here can be reproduced exactly. Examples:

```sh
build/stress/shodhan_stress wide 450000 454000      # 4000 wide-coefficient LPs
build/stress/shodhan_stress all 1 300               # every family
build/stress/shodhan_stress wide 453507 453507 --emit-mps /tmp/lps   # write the model of one seed as MPS
```

Output: one `MISMATCH <family> seed <n> ...` line per disagreement (status, objective, KKT on the
original model, or an invalid certificate), `HYPERSENSITIVE` lines for objectives that differ while
both points pass the KKT check and sum|y| * 1e-9 exceeds the comparison tolerance, and a summary per
family. The exit status is 1 if any mismatch was found.

`--emit-mps` writes the model instead of solving it, so an instance can be turned into a certificate
with `shodhan solve --write-cert` and checked with KASAUTI (`docs/KASAUTI.md`).
