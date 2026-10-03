# Benchmark data

Benchmark instances are **not** stored in this repository. Download them
yourself into `data/raw/`, which is git-ignored. Never commit benchmark files;
their licences and terms of use belong to the sites that host them.

| Collection        | Contents                                  | Where to get it                      |
|-------------------|-------------------------------------------|--------------------------------------|
| Netlib LP         | classic LP test problems (MPS)            | https://www.netlib.org/lp/data/      |
| MIPLIB 2017       | mixed-integer instances (MPS, often .gz)  | https://miplib.zib.de/               |
| Maros-Meszaros    | convex QP test set (QPS)                  | the data page of I. Maros (Imperial College London); search for "Maros Meszaros QP test set" |
| QPLIB             | quadratic programming library             | https://qplib.zib.de/                |

Notes:

- Many Netlib files are distributed in a compressed form that needs the
  `emps` expander published alongside them; decompress them to plain MPS
  before use.
- MIPLIB instances are `.mps.gz`. Reading them directly needs a build with
  `-DSHODHAN_ENABLE_ZLIB=ON`; otherwise decompress them first.
- QPS files (Maros-Meszaros, QPLIB) contain quadratic sections that SHODHAN
  does not read yet; they are rejected with a clear error.
- These locations are given from memory and were not re-checked when this file
  was written; verify them against the hosting sites.

Example:

```sh
mkdir -p data/raw/netlib
# download an instance (for example afiro.mps) into data/raw/netlib/
shodhan info data/raw/netlib/afiro.mps
```

## Netlib LPs and the LP runner

The original Netlib LP files (https://www.netlib.org/lp/data/) are distributed in a compressed form
that needs a separate expander, the `emps` tool published next to them, before they are plain MPS files
that SHODHAN can read. SHODHAN does not include the expander: compile `emps` yourself, expand each file
into `data/raw/netlib/`, then run

```sh
python bench/run_lp_set.py data/raw/netlib --exe build/release/shodhan --timeout 300 --csv netlib.csv
```

To compare objectives, give a reference CSV with columns `name,objective` taken from a source you trust
(the repository contains no reference values); the runner reports how many instances were solved, failed
and mismatched at a relative tolerance of 1e-6.
