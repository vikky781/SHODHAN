"""Core of the synthetic refinery / process case-study generators (Python standard library only).

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

* `Rng` is a deterministic generator of our own (splitmix64 seeding, xorshift64*), so that the files are
  byte-identical on every platform and Python version; no library random module is used.
* All data and all witness values are dyadic rationals (`fractions.Fraction` with a power-of-two denominator), so
  every number is written exactly and a witness point is feasible in exact arithmetic when it is feasible by construction.
* `Model` collects rows and columns and writes the MPS file and the `.meta.txt` sidecar.
"""

from fractions import Fraction

MASK = (1 << 64) - 1

SYNTHETIC_LINE = "SYNTHETIC: structure follows textbook formulations; not plant or MRPL data"


class Rng:
    def __init__(self, seed):
        z = (int(seed) + 0x9E3779B97F4A7C15) & MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        self.s = z or 0x2545F4914F6CDD1D

    def next(self):
        x = self.s
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & MASK

    def unit(self):
        return (self.next() >> 11) / 9007199254740992.0

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi]."""
        return lo + self.next() % (hi - lo + 1)

    def chance(self, p):
        return self.unit() < p

    def choice(self, seq):
        return seq[self.next() % len(seq)]

    def shuffle(self, seq):
        for i in range(len(seq) - 1, 0, -1):
            j = self.next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def dyadic(self, lo, hi, den=16):
        """A multiple of 1/den in [lo, hi] (lo, hi integers or multiples of 1/den)."""
        a, b = int(Fraction(lo) * den), int(Fraction(hi) * den)
        return Fraction(self.randint(a, b), den)


def fmt(v):
    """The EXACT decimal expansion of a dyadic Fraction or int (n / 2**m = n * 5**m / 10**m has m decimal digits), so that a
    reader that parses the text exactly recovers the same rational number. (The shortest round-trip repr of the double is
    not good enough: it is a different rational number whenever the exact expansion is longer.)"""
    v = Fraction(v)
    d = v.denominator
    if d & (d - 1):
        raise ValueError("not a dyadic rational: %s" % v)
    if d == 1:
        return str(v.numerator)
    m = d.bit_length() - 1
    n = v.numerator * 5 ** m
    sign = "-" if n < 0 else ""
    digits = str(abs(n)).rjust(m + 1, "0")
    return sign + digits[:-m] + "." + digits[-m:].rstrip("0")


INF = None  # bound meaning "infinite"


class Model:
    def __init__(self, name, family, sense="min"):
        self.name = name
        self.family = family
        self.sense = sense
        self.cols = []        # [name, lo, hi, cost, is_int]
        self.col_index = {}
        self.rows = []        # [name, type ('L','G','E'), rhs]
        self.row_index = {}
        self.entries = []     # per column: list of (row index, coef)
        self.meta = []
        self.witness = {}
        self.offset = Fraction(0)

    def col(self, name, lo=0, hi=INF, cost=0, integer=False):
        if name in self.col_index:
            raise ValueError("duplicate column " + name)
        self.col_index[name] = len(self.cols)
        self.cols.append([name, None if lo is None else Fraction(lo), None if hi is None else Fraction(hi), Fraction(cost), integer])
        self.entries.append([])
        return name

    def binary(self, name, cost=0):
        return self.col(name, 0, 1, cost, True)

    def row(self, name, rtype, rhs, terms):
        """terms: iterable of (column name, coefficient); zero coefficients are dropped, repeated columns are added."""
        if name in self.row_index:
            raise ValueError("duplicate row " + name)
        ri = len(self.rows)
        self.row_index[name] = ri
        self.rows.append([name, rtype, Fraction(rhs)])
        acc = {}
        order = []
        for c, v in terms:
            v = Fraction(v)
            if c not in acc:
                acc[c] = Fraction(0)
                order.append(c)
            acc[c] += v
        for c in order:
            if acc[c] != 0:
                self.entries[self.col_index[c]].append((ri, acc[c]))
        return name

    def n_binaries(self):
        return sum(1 for c in self.cols if c[4] and c[1] == 0 and c[2] == 1)

    def nnz(self):
        return sum(len(e) for e in self.entries)

    def mps_text(self):
        out = ["* " + SYNTHETIC_LINE, "NAME          %s" % self.name, "ROWS", " N  OBJ"]
        for rname, rtype, _ in self.rows:
            out.append(" %s  %s" % (rtype, rname))
        out.append("COLUMNS")
        in_int, marker = False, 0
        for (cname, lo, hi, cost, is_int), ent in zip(self.cols, self.entries):
            if is_int and not in_int:
                out.append("    MARK%04d  'MARKER'                 'INTORG'" % marker)
                marker += 1
                in_int = True
            if not is_int and in_int:
                out.append("    MARK%04d  'MARKER'                 'INTEND'" % marker)
                marker += 1
                in_int = False
            items = []
            if cost != 0:
                items.append(("OBJ", cost))
            for ri, v in ent:
                items.append((self.rows[ri][0], v))
            if not items:
                items.append(("OBJ", 0))  # keep the column declared
            for rname, v in items:
                out.append("    %s  %s  %s" % (cname, rname, fmt(v)))
        if in_int:
            out.append("    MARK%04d  'MARKER'                 'INTEND'" % marker)
        out.append("RHS")
        if self.offset != 0:
            out.append("    RHS  OBJ  %s" % fmt(-self.offset))  # MPS: the objective constant is the negated RHS entry
        for rname, rtype, rhs in self.rows:
            if rhs != 0:
                out.append("    RHS  %s  %s" % (rname, fmt(rhs)))
        bounds = []
        for cname, lo, hi, cost, is_int in self.cols:
            if is_int and lo == 0 and hi == 1:
                bounds.append(" BV BND  %s" % cname)
                continue
            if lo is None and hi is None:
                bounds.append(" FR BND  %s" % cname)
                continue
            if lo is not None and hi is not None and lo == hi:
                bounds.append(" FX BND  %s  %s" % (cname, fmt(lo)))
                continue
            if lo is None:
                bounds.append(" MI BND  %s" % cname)
            elif lo != 0:
                bounds.append(" LO BND  %s  %s" % (cname, fmt(lo)))
            if hi is not None:
                bounds.append(" UP BND  %s  %s" % (cname, fmt(hi)))
        if bounds:
            out.append("BOUNDS")
            out.extend(bounds)
        out.append("ENDATA")
        return "\n".join(out) + "\n"

    def meta_text(self, knobs, seed):
        lines = [SYNTHETIC_LINE, "family: %s" % self.family, "name: %s" % self.name, "seed: %d" % seed]
        for k in sorted(knobs):
            lines.append("knob %s: %s" % (k, knobs[k]))
        lines.append("rows: %d" % len(self.rows))
        lines.append("cols: %d" % len(self.cols))
        lines.append("binaries: %d" % self.n_binaries())
        lines.append("nnz: %d" % self.nnz())
        lines.append("sense: %s" % self.sense)
        lines.append("witness_objective: %s" % self.witness_objective())
        lines.append("witness_count: %d" % len(self.witness))
        for cname, _, _, _, _ in self.cols:
            if cname in self.witness:
                lines.append("witness %s %s" % (cname, fmt(self.witness[cname])))
        return "\n".join(lines) + "\n"

    def witness_objective(self):
        total = self.offset
        for cname, lo, hi, cost, is_int in self.cols:
            total += cost * self.witness.get(cname, Fraction(0))
        return fmt(total)

    def write(self, directory, basename, knobs, seed):
        import os
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, basename + ".mps"), "w", newline="\n", encoding="ascii") as f:
            f.write(self.mps_text())
        with open(os.path.join(directory, basename + ".meta.txt"), "w", newline="\n", encoding="ascii") as f:
            f.write(self.meta_text(knobs, seed))
