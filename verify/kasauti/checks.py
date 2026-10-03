"""Certificate checks for KASAUTI (see docs/CERTIFICATES.md for the mathematics).

All arithmetic is exact (``fractions.Fraction``) in exact mode and plain floats with compensated
summation (``math.fsum``) in float mode. Infinite bounds are ``None`` (``lo is None``: -inf,
``hi is None``: +inf) and are handled explicitly, never through floating-point infinities.
"""

import math
from fractions import Fraction

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


class Options:
    def __init__(self, primal_tol=1e-6, gap_tol=1e-6, ray_tol=1e-9, int_tol=1e-6, farkas_zero_tol=1e-12):
        self.primal_tol = primal_tol
        self.gap_tol = gap_tol
        self.ray_tol = ray_tol
        self.int_tol = int_tol
        self.farkas_zero_tol = farkas_zero_tol


class Report:
    """Verdict plus the checks that led to it. ``lines`` is the human text, ``data`` the machine form."""

    def __init__(self):
        self.verdict = INCONCLUSIVE
        self.detail = ""          # PASS_OPTIMAL, PASS_INFEASIBLE_TOL, FAIL, ...
        self.lines = []
        self.data = {"checks": []}
        self.rigorous = False

    def say(self, text):
        self.lines.append(text)

    def check(self, name, ok, **numbers):
        self.data["checks"].append({"name": name, "ok": ok, **{k: _jsonable(v) for k, v in numbers.items()}})
        return ok


def _jsonable(v):
    if isinstance(v, Fraction):
        return {"decimal": "%.17g" % float(v), "exact": "%d/%d" % (v.numerator, v.denominator) if v.denominator != 1 else str(v.numerator)}
    if isinstance(v, float):
        return "inf" if v == math.inf else "-inf" if v == -math.inf else v
    return v


def fmt(v):
    """Human text of a number: the decimal value and, in exact mode, the exact fraction when it is short."""
    if v is None:
        return "inf"
    if isinstance(v, Fraction):
        dec = "%.17g" % float(v)
        if v.denominator == 1:
            return str(v.numerator) if abs(v.numerator) < 10 ** 18 else dec
        text = "%d/%d" % (v.numerator, v.denominator)
        return dec if len(text) > 40 else "%s (= %s)" % (dec, text)
    return "%.17g" % v


class Arith:
    def __init__(self, exact):
        self.exact = exact
        self.zero = Fraction(0) if exact else 0.0

    def num(self, v):
        if isinstance(v, bool):
            raise ValueError("boolean where a number was expected")
        if self.exact:
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("non-finite number")
            return Fraction(v)
        return float(v)

    def total(self, terms):
        return sum(terms, Fraction(0)) if self.exact else math.fsum(terms)

    def absval(self, v):
        return -v if v < 0 else v


class CertError(Exception):
    """The certificate is malformed (reported as INCONCLUSIVE or FAIL by the caller)."""


def _vector(arith, mapping, index, kind):
    """name -> value mapping to a dense list by index; unknown names are an error."""
    out = [arith.zero] * len(index)
    if not isinstance(mapping, dict):
        raise CertError("%s must be an object" % kind)
    for name, v in mapping.items():
        if name not in index:
            raise CertError("unknown %s name %r" % (kind, name))
        out[index[name]] = arith.num(v)
    return out


def activities(model, ar, x):
    terms = [[] for _ in range(model.n_rows)]
    for j, col in enumerate(model.col_entries):
        xj = x[j]
        if xj != 0:
            for i, a in col:
                terms[i].append(ar.num(a) * xj)
    return [ar.total(t) for t in terms]


def primal_violation(model, ar, x):
    """Largest absolute and relative violation of the row and column bounds by x."""
    act = activities(model, ar, x)
    worst_abs = worst_rel = ar.zero
    where = None
    def note(v, bound, what):
        nonlocal worst_abs, worst_rel, where
        if v > 0:
            rel = v / (1 + ar.absval(ar.num(bound)))
            if v > worst_abs:
                worst_abs = v
            if rel > worst_rel:
                worst_rel = rel
                where = what
    for i in range(model.n_rows):
        lo, hi = model.row_lo[i], model.row_hi[i]
        if lo is not None and act[i] < lo:
            note(ar.num(lo) - act[i], lo, "row %s (below its lower bound)" % model.row_names[i])
        if hi is not None and act[i] > hi:
            note(act[i] - ar.num(hi), hi, "row %s (above its upper bound)" % model.row_names[i])
    for j in range(model.n_cols):
        lo, hi = model.col_lo[j], model.col_hi[j]
        if lo is not None and x[j] < lo:
            note(ar.num(lo) - x[j], lo, "column %s (below its lower bound)" % model.col_names[j])
        if hi is not None and x[j] > hi:
            note(x[j] - ar.num(hi), hi, "column %s (above its upper bound)" % model.col_names[j])
    return worst_abs, worst_rel, where, act


def _sense_sign(model):
    return -1 if model.sense == "max" else 1


def internal_costs(model, ar):
    """Costs and offset of the minimization form (negated for a max model)."""
    s = _sense_sign(model)
    return [s * ar.num(c) for c in model.col_cost], s * ar.num(model.offset)


# --------------------------------------------------------------------------------------------
def check_optimal(model, cert, ar, opt, rep):
    x = _vector(ar, cert.get("x", {}), model.col_index, "column")
    y = _vector(ar, cert.get("y", {}), model.row_index, "row")
    c, offset = internal_costs(model, ar)
    sgn = _sense_sign(model)

    abs_v, rel_v, where, _ = primal_violation(model, ar, x)
    primal_ok = rel_v <= opt.primal_tol
    rep.say("primal feasibility: max violation %s absolute, %s relative%s" % (fmt(abs_v), fmt(rel_v), (" at " + where) if where else ""))
    rep.check("primal_feasibility", primal_ok, max_abs=abs_v, max_rel=rel_v, tolerance=opt.primal_tol)

    # d = c - A^T y, exactly.
    d = []
    for j, col in enumerate(model.col_entries):
        d.append(c[j] - ar.total([ar.num(a) * y[i] for i, a in col if y[i] != 0]))
    lb_terms = []
    infinite_reason = None
    for i in range(model.n_rows):
        if y[i] > 0:
            if model.row_lo[i] is None:
                infinite_reason = "row %s has y > 0 but no lower bound" % model.row_names[i]
            else:
                lb_terms.append(y[i] * ar.num(model.row_lo[i]))
        elif y[i] < 0:
            if model.row_hi[i] is None:
                infinite_reason = "row %s has y < 0 but no upper bound" % model.row_names[i]
            else:
                lb_terms.append(y[i] * ar.num(model.row_hi[i]))
    for j in range(model.n_cols):
        if d[j] > 0:
            if model.col_lo[j] is None:
                infinite_reason = "column %s has d > 0 but no lower bound" % model.col_names[j]
            else:
                lb_terms.append(d[j] * ar.num(model.col_lo[j]))
        elif d[j] < 0:
            if model.col_hi[j] is None:
                infinite_reason = "column %s has d < 0 but no upper bound" % model.col_names[j]
            else:
                lb_terms.append(d[j] * ar.num(model.col_hi[j]))
    primal_obj = offset + ar.total([c[j] * x[j] for j in range(model.n_cols) if x[j] != 0])
    obj_model = sgn * primal_obj
    rep.say("primal objective (model sense): %s" % fmt(obj_model))
    claimed = cert.get("claimed_objective")
    claim_ok = True
    if claimed is not None:
        cl = ar.num(claimed)
        diff = ar.absval(cl - obj_model)
        claim_ok = diff <= opt.gap_tol * (1 + ar.absval(obj_model))
        rep.say("claimed objective %s versus exact %s: difference %s" % (fmt(cl), fmt(obj_model), fmt(diff)))
        rep.check("claimed_objective", claim_ok, claimed=cl, exact=obj_model, difference=diff)
    if infinite_reason is not None:
        rep.say("dual bound: -infinity (%s): the dual side proves nothing" % infinite_reason)
        rep.check("dual_bound", False, reason=infinite_reason)
        gap_ok = False
        lb_model = None
    else:
        lb = offset + ar.total(lb_terms)
        gap = primal_obj - lb
        lb_model = sgn * lb
        word = "lower" if sgn == 1 else "upper"
        rep.say("rigorous dual %s bound LB(y): %s (valid for every feasible x by weak duality)" % (word, fmt(lb_model)))
        rep.say("gap |primal objective - bound|: %s" % fmt(ar.absval(gap)))
        scale = 1 + ar.absval(obj_model)
        gap_ok = gap <= opt.gap_tol * scale
        if gap < -opt.gap_tol * scale:
            rep.say("note: the primal point beats the dual bound by %s: it is infeasible by tolerance, not exactly feasible" % fmt(-gap))
        rep.check("dual_gap", gap_ok, lower_bound=lb_model, primal_objective=obj_model, gap=gap, tolerance=opt.gap_tol)
    ok = primal_ok and gap_ok and claim_ok
    rep.data.update({"primal_objective": _jsonable(obj_model), "dual_bound": _jsonable(lb_model), "max_primal_violation": _jsonable(abs_v)})
    if ok:
        rep.detail = "PASS_OPTIMAL"
        rep.rigorous = abs_v == 0 and ar.exact
        rep.say("the optimum lies in [%s, %s]%s" % (fmt(min(obj_model, lb_model)), fmt(max(obj_model, lb_model)),
                                                      "" if rep.rigorous else " (within the stated tolerances: the point is not exactly feasible or the mode is float)"))
    else:
        rep.detail = "FAIL"
    return ok


# --------------------------------------------------------------------------------------------
def _scaled_interval(ar, coef, lo, hi):
    """(min, max) of coef * v for lo <= v <= hi; each endpoint is a number or None for -inf/+inf."""
    if coef > 0:
        mn = None if lo is None else coef * ar.num(lo)
        mx = None if hi is None else coef * ar.num(hi)
    else:
        mn = None if hi is None else coef * ar.num(hi)
        mx = None if lo is None else coef * ar.num(lo)
    return mn, mx


def _sum_interval(ar, intervals):
    mn_terms, mx_terms = [], []
    mn_inf = mx_inf = False
    for mn, mx in intervals:
        if mn is None:
            mn_inf = True
        else:
            mn_terms.append(mn)
        if mx is None:
            mx_inf = True
        else:
            mx_terms.append(mx)
    return (None if mn_inf else ar.total(mn_terms)), (None if mx_inf else ar.total(mx_terms))


def _disjoint(r, c):
    """Are the intervals r = (min, max) and c disjoint? None endpoints are infinite."""
    if r[1] is not None and c[0] is not None and r[1] < c[0]:
        return True, c[0] - r[1]
    if c[1] is not None and r[0] is not None and c[1] < r[0]:
        return True, r[0] - c[1]
    return False, None


def check_infeasible(model, cert, ar, opt, rep):
    body = cert.get("farkas")
    if not isinstance(body, dict) or "y" not in body:
        raise CertError("the infeasible certificate needs farkas.y")
    y = _vector(ar, body["y"], model.row_index, "row")
    g = []
    for col in model.col_entries:
        g.append(ar.total([ar.num(a) * y[i] for i, a in col if y[i] != 0]))
    row_iv = [(_scaled_interval(ar, y[i], model.row_lo[i], model.row_hi[i])) for i in range(model.n_rows) if y[i] != 0]
    R = _sum_interval(ar, row_iv)

    def column_interval(drop):
        ivs = []
        for j in range(model.n_cols):
            if g[j] != 0 and j not in drop:
                ivs.append(_scaled_interval(ar, g[j], model.col_lo[j], model.col_hi[j]))
        return _sum_interval(ar, ivs)

    def show(iv):
        return "[%s, %s]" % ("-inf" if iv[0] is None else fmt(iv[0]), "+inf" if iv[1] is None else fmt(iv[1]))

    C = column_interval(set())
    rep.say("y^T r ranges over %s (rows), y^T A x ranges over %s (columns)" % (show(R), show(C)))
    ok, sep = _disjoint(R, C)
    if ok:
        rep.check("farkas_intervals_disjoint", True, row_interval=show(R), column_interval=show(C), separation=sep)
        rep.say("the intervals are disjoint (separation %s): the model is infeasible" % fmt(sep))
        rep.detail = "PASS_INFEASIBLE"
        rep.rigorous = ar.exact
        return True
    # Strict check failed. Multipliers are doubles, so a coefficient that is zero in theory can be a tiny
    # nonzero that meets an infinite bound. Optionally drop coefficients below a relative tolerance.
    gmax = max([ar.absval(v) for v in g] + [ar.zero])
    if opt.farkas_zero_tol > 0 and gmax > 0:
        drop = {j for j in range(model.n_cols) if g[j] != 0 and ar.absval(g[j]) <= opt.farkas_zero_tol * gmax}
        if drop:
            C2 = column_interval(drop)
            ok2, sep2 = _disjoint(R, C2)
            rep.say("after dropping %d coefficient(s) of A^T y below %g times the largest: columns range over %s" % (len(drop), opt.farkas_zero_tol, show(C2)))
            if ok2:
                rep.check("farkas_intervals_disjoint", True, row_interval=show(R), column_interval=show(C2), separation=sep2, dropped=len(drop))
                rep.say("the intervals are disjoint (separation %s) only after dropping those coefficients: NOT a rigorous proof" % fmt(sep2))
                rep.detail = "PASS_INFEASIBLE_TOL"
                rep.rigorous = False
                return True
    rep.check("farkas_intervals_disjoint", False, row_interval=show(R), column_interval=show(C))
    rep.say("the intervals overlap or an infinite bound defeats the proof: the certificate does not prove infeasibility")
    rep.detail = "FAIL"
    return False


# --------------------------------------------------------------------------------------------
def check_unbounded(model, cert, ar, opt, rep):
    x0 = _vector(ar, cert.get("point", {}), model.col_index, "column")
    r = _vector(ar, cert.get("ray", {}), model.col_index, "column")
    if "point" not in cert or "ray" not in cert:
        raise CertError("an unbounded certificate needs both point and ray")
    c, _ = internal_costs(model, ar)

    abs_v, rel_v, where, _ = primal_violation(model, ar, x0)
    point_ok = rel_v <= opt.primal_tol
    rep.say("point feasibility: max violation %s absolute, %s relative%s" % (fmt(abs_v), fmt(rel_v), (" at " + where) if where else ""))
    rep.check("point_feasible", point_ok, max_abs=abs_v, max_rel=rel_v, tolerance=opt.primal_tol)

    rmax = max([ar.absval(v) for v in r] + [ar.zero])
    if rmax == 0:
        rep.say("the ray is zero")
        rep.check("ray_nonzero", False)
        rep.detail = "FAIL"
        return False
    exact_ok = True
    tol_ok = True
    worst = ar.zero
    worst_where = None
    for j in range(model.n_cols):
        v = r[j]
        viol = ar.zero
        if v > 0 and model.col_hi[j] is not None:
            viol = v
        elif v < 0 and model.col_lo[j] is not None:
            viol = -v
        if viol > 0:
            exact_ok = False
            if viol > opt.ray_tol * rmax:
                tol_ok = False
            if viol / rmax > worst:
                worst, worst_where = viol / rmax, "column %s" % model.col_names[j]
    ar_terms = [[] for _ in range(model.n_rows)]
    for j, col in enumerate(model.col_entries):
        if r[j] != 0:
            for i, a in col:
                ar_terms[i].append(ar.num(a) * r[j])
    for i in range(model.n_rows):
        act = ar.total(ar_terms[i])
        viol = ar.zero
        if act < 0 and model.row_lo[i] is not None:
            viol = -act
        elif act > 0 and model.row_hi[i] is not None:
            viol = act
        if viol > 0:
            exact_ok = False
            scale = ar.total([ar.absval(t) for t in ar_terms[i]])
            if viol > opt.ray_tol * scale:
                tol_ok = False
            if scale > 0 and viol / scale > worst:
                worst, worst_where = viol / scale, "row %s" % model.row_names[i]
    rate = ar.total([c[j] * r[j] for j in range(model.n_cols) if r[j] != 0])
    improves = rate < 0
    rep.say("ray: recession-cone conditions %s (largest relative violation %s%s)" % (
        "hold exactly" if exact_ok else ("hold within %g" % opt.ray_tol if tol_ok else "FAIL"), fmt(worst), (" at " + worst_where) if worst_where else ""))
    rep.say("objective rate c^T r = %s (must be < 0 in the minimization form)" % fmt(rate))
    rep.check("ray_recession_cone", tol_ok, exact=exact_ok, largest_relative_violation=worst, tolerance=opt.ray_tol)
    rep.check("ray_improves_objective", improves, rate=rate)
    ok = point_ok and tol_ok and improves
    if ok:
        rep.detail = "PASS_UNBOUNDED" if exact_ok and abs_v == 0 else "PASS_UNBOUNDED_TOL"
        rep.rigorous = exact_ok and abs_v == 0 and ar.exact
        if not rep.rigorous:
            rep.say("not every condition holds exactly: the claim is within the stated tolerances")
    else:
        rep.detail = "FAIL"
    return ok


# --------------------------------------------------------------------------------------------
def check_feasible(model, cert, ar, opt, rep):
    x = _vector(ar, cert.get("x", {}), model.col_index, "column")
    abs_v, rel_v, where, _ = primal_violation(model, ar, x)
    feas_ok = rel_v <= opt.primal_tol
    rep.say("feasibility: max violation %s absolute, %s relative%s" % (fmt(abs_v), fmt(rel_v), (" at " + where) if where else ""))
    rep.check("primal_feasibility", feas_ok, max_abs=abs_v, max_rel=rel_v, tolerance=opt.primal_tol)
    worst_int = ar.zero
    for j in range(model.n_cols):
        if model.col_integer[j]:
            nearest = round(x[j]) if not ar.exact else Fraction(round(x[j]))
            dev = ar.absval(x[j] - nearest)
            worst_int = max(worst_int, dev)
    int_ok = worst_int <= opt.int_tol
    rep.say("integrality: largest distance from an integer %s over %d integer column(s)" % (fmt(worst_int), model.n_integer))
    rep.check("integrality", int_ok, max_deviation=worst_int, tolerance=opt.int_tol)
    c, offset = internal_costs(model, ar)
    obj = _sense_sign(model) * (offset + ar.total([c[j] * x[j] for j in range(model.n_cols) if x[j] != 0]))
    rep.say("exact objective (model sense): %s" % fmt(obj))
    ok = feas_ok and int_ok
    rep.data["primal_objective"] = _jsonable(obj)
    rep.detail = "PASS_FEASIBLE" if ok else "FAIL"
    if ok:
        rep.say("optimality not certified")
    return ok


def check_solution_file(model, values, claimed_objective, ar, opt, rep):
    """A .sol file: primal feasibility and objective only."""
    x = [ar.zero] * model.n_cols
    for name, v in values.items():
        if name not in model.col_index:
            raise CertError("unknown column name %r in the solution file" % name)
        x[model.col_index[name]] = ar.num(v)
    abs_v, rel_v, where, _ = primal_violation(model, ar, x)
    feas_ok = rel_v <= opt.primal_tol
    rep.say("feasibility: max violation %s absolute, %s relative%s" % (fmt(abs_v), fmt(rel_v), (" at " + where) if where else ""))
    rep.check("primal_feasibility", feas_ok, max_abs=abs_v, max_rel=rel_v, tolerance=opt.primal_tol)
    c, offset = internal_costs(model, ar)
    obj = _sense_sign(model) * (offset + ar.total([c[j] * x[j] for j in range(model.n_cols) if x[j] != 0]))
    rep.say("exact objective (model sense): %s" % fmt(obj))
    ok = feas_ok
    if claimed_objective is not None:
        cl = ar.num(claimed_objective)
        diff = ar.absval(cl - obj)
        match = diff <= opt.gap_tol * (1 + ar.absval(obj))
        rep.say("objective in the file %s, difference from the exact value %s" % (fmt(cl), fmt(diff)))
        rep.check("objective_matches", match, claimed=cl, exact=obj, difference=diff)
        ok = ok and match
    rep.detail = "PASS_SOLUTION_FEASIBLE" if ok else "FAIL"
    if ok:
        rep.say("a solution file certifies feasibility and the objective only, not optimality")
    return ok
