"""Seeded SYNTHETIC pooling instances for the verifier tests (invented numbers, written as .pool text). Python standard
library only; its own random generator; all numbers are multiples of 1/8 and written exactly."""

from fractions import Fraction

MASK = (1 << 64) - 1


class Rng:
    def __init__(self, seed):
        self.s = (seed * 0x9E3779B97F4A7C15 + 0x1234567) & MASK or 1

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK
        return self.s >> 33

    def randint(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def chance(self, p):
        return self.next() % 1000 < p * 1000

    def eighth(self, lo, hi):
        return Fraction(self.randint(int(lo * 8), int(hi * 8)), 8)


def num(v):
    if v is None:
        return "inf"
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    m = v.denominator.bit_length() - 1
    n = v.numerator * 5 ** m
    digits = str(abs(n)).rjust(m + 1, "0")
    return ("-" if n < 0 else "") + digits[:-m] + "." + digits[-m:].rstrip("0")


def generate(seed):
    """Returns (text, description). Roughly 60% one pool and one quality, 40% two pools and two qualities."""
    r = Rng(seed)
    two = seed % 5 in (1, 3)
    K = 2 if two else 1
    nS = r.randint(3, 4) if two else r.randint(2, 4)
    nP = 2 if two else 1
    nT = 2 if two else r.randint(1, 2)
    lines = ["# SYNTHETIC verifier-test instance %d: invented numbers" % seed, "name pool_test_%d" % seed, "synthetic yes",
             "qualities %d %s" % (K, " ".join("Q%d" % (k + 1) for k in range(K)))]
    qmin, qmax = [10 ** 9] * K, [-10 ** 9] * K
    for s in range(nS):
        q = [r.eighth(Fraction(1, 2), 6) for _ in range(K)]
        for k in range(K):
            qmin[k], qmax[k] = min(qmin[k], q[k]), max(qmax[k], q[k])
        cost = max(2, 22 - int(sum(q) * Fraction(14, 10) / K * 2) + r.randint(-2, 2))
        supply = None if r.chance(0.45) else r.eighth(25, 90)
        lines.append("source S%d %s %s %s" % (s + 1, num(cost), num(supply), " ".join(num(x) for x in q)))
    for p in range(nP):
        lines.append("pool P%d %s" % (p + 1, num(r.eighth(40, 110))))
    for t in range(nT):
        spec = [r.eighth(qmin[k] + (qmax[k] - qmin[k]) / 6, qmax[k] - (qmax[k] - qmin[k]) / 6) for k in range(K)]
        price = 30 - int(sum(spec)) + r.randint(0, 3)
        lines.append("terminal T%d %s %s %s" % (t + 1, num(price), num(r.eighth(25, 90)), " ".join(num(x) for x in spec)))
    for s in range(nS):
        for p in range(nP):
            if p == 0 or s == 0 or r.chance(0.7):
                lines.append("arc S%d P%d" % (s + 1, p + 1))
    for p in range(nP):
        for t in range(nT):
            lines.append("arc P%d T%d" % (p + 1, t + 1))
    for s in range(nS):
        for t in range(nT):
            if r.chance(0.3):
                lines.append("arc S%d T%d" % (s + 1, t + 1))
    return "\n".join(lines) + "\n", "%d pool(s), %d quality(ies), %d sources" % (nP, K, nS)
