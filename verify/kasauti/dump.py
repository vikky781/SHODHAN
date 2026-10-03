"""Canonical text dump of a parsed model, and a comparison with the C++ ``shodhan dump-model`` output.

Exact values are converted to the nearest double before printing, so the dump of the exact parse can be
compared number by number with the doubles the C++ reader produced.
"""

import math


def _num(v, neg_inf):
    if v is None:
        return "-inf" if neg_inf else "inf"
    return repr(float(v))


def dump_model(model):
    lines = [
        "name\t%s" % model.name,
        "sense\t%s" % model.sense,
        "offset\t%s" % repr(float(model.offset)),
        "rows\t%d" % model.n_rows,
        "cols\t%d" % model.n_cols,
    ]
    for i in range(model.n_rows):
        lines.append("row\t%s\t%s\t%s" % (model.row_names[i], _num(model.row_lo[i], True), _num(model.row_hi[i], False)))
    for j in range(model.n_cols):
        lines.append("col\t%s\t%s\t%s\t%s\t%s" % (model.col_names[j], repr(float(model.col_cost[j])),
                                                  _num(model.col_lo[j], True), _num(model.col_hi[j], False),
                                                  "I" if model.col_integer[j] else "C"))
    for j in range(model.n_cols):
        for i, v in sorted(model.col_entries[j]):
            lines.append("entry\t%s\t%s\t%s" % (model.row_names[i], model.col_names[j], repr(float(v))))
    return "\n".join(lines) + "\n"


def _value(text):
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    return float(text)


_NUMERIC_FIELDS = {"offset": (1,), "row": (2, 3), "col": (2, 3, 4), "entry": (3,)}


def compare_dumps(a, b, row_scale=None):
    """Differences between two dumps (strings); numbers are compared as doubles, bit for bit.

    The bounds of a ranged row (rhs +- |range|) are the one exception: the C++ reader adds two
    already-rounded doubles, KASAUTI adds the exact decimals, so they may differ by the rounding of the
    inputs. For those rows, row_scale[i] = max(|rhs|, |range|) allows a difference of 4 * 2^-52 * scale."""
    row_no = -1
    la, lb = a.rstrip("\n").split("\n"), b.rstrip("\n").split("\n")
    diffs = []
    if len(la) != len(lb):
        diffs.append("different number of lines: %d versus %d" % (len(la), len(lb)))
    for k, (x, y) in enumerate(zip(la, lb)):
        fx, fy = x.split("\t"), y.split("\t")
        if len(fx) != len(fy) or fx[0] != fy[0]:
            diffs.append("line %d: %r versus %r" % (k + 1, x, y))
            continue
        numeric = _NUMERIC_FIELDS.get(fx[0], ())
        tol = 0.0
        if fx[0] == "row":
            row_no += 1
            if row_scale is not None and row_no < len(row_scale):
                tol = 4 * 2.0 ** -52 * row_scale[row_no]
        for idx, (p, q) in enumerate(zip(fx, fy)):
            if idx in numeric:
                try:
                    vp, vq = _value(p), _value(q)
                    same = (vp == vq and math.copysign(1.0, vp) == math.copysign(1.0, vq)) or (vp == 0 and vq == 0) or (
                        tol > 0 and fx[0] == "row" and math.isfinite(vp) and math.isfinite(vq) and abs(vp - vq) <= tol)
                except ValueError:
                    same = False
                if not same:
                    diffs.append("line %d field %d: %s versus %s  (%r / %r)" % (k + 1, idx, p, q, x, y))
            elif p != q:
                diffs.append("line %d field %d: %r versus %r" % (k + 1, idx, p, q))
    return diffs
