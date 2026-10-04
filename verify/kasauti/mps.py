"""MPS parser for KASAUTI.

Written from the conventions in docs/MPS_FORMAT.md and docs/CONVENTIONS.md, not from the C++ reader.
Numbers are parsed from their decimal text as exact fractions (``exact=True``) or as floats.

Infinite bounds are represented by ``None``: ``lo is None`` means -inf, ``hi is None`` means +inf.
A quadratic OBJECTIVE is read from QUADOBJ or QMATRIX (docs/QP.md); quadratic constraints (QCMATRIX, QSECTION), semi-continuous
bounds, SOS and indicators raise
``Unsupported``; malformed input raises ``MpsError``. Nothing is guessed.
"""

import gzip
import math
import re
from fractions import Fraction

INF_LIMIT = Fraction(10) ** 30
MAX_EXPONENT = 10000

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?$")
_UNSUPPORTED_SECTIONS = {"QSECTION", "QCMATRIX", "SOS", "INDICATORS", "OBJNAME"}
_QUAD_SECTIONS = ("QUADOBJ", "QMATRIX")
_VALUE_BOUNDS = {"UP", "LO", "FX", "LI", "UI", "SC"}
_NO_VALUE_BOUNDS = {"FR", "MI", "PL", "BV"}


class MpsError(Exception):
    """Malformed input."""


class Unsupported(MpsError):
    """A feature this verifier does not support."""


def parse_number(text, exact=True):
    """Parses decimal text. Returns a Fraction (exact) or float, or +-math.inf for Inf/Infinity."""
    t = text.strip()
    body = t.lstrip("+-").lower()
    if body in ("inf", "infinity"):
        return -math.inf if t.startswith("-") else math.inf
    if not _NUMBER.match(t):
        raise MpsError("not a number: %r" % text)
    t = t.replace("D", "e").replace("d", "e")
    if "e" in t.lower():
        exp = int(re.split("[eE]", t)[1])
        if abs(exp) > MAX_EXPONENT:
            raise Unsupported("exponent too large: %r" % text)
    if exact:
        v = Fraction(t)
        return v
    return float(t)


def is_infinite(v):
    if isinstance(v, float):
        return math.isinf(v) or abs(v) >= 1e30
    return abs(v) >= INF_LIMIT


class Model:
    """An LP as read from an MPS file."""

    def __init__(self, exact):
        self.exact = exact
        self.name = ""
        self.sense = "min"
        self.offset = 0
        self.row_names = []
        self.row_types = []          # 'L', 'G', 'E'
        self.row_lo = []             # None = -inf
        self.row_hi = []             # None = +inf
        self.row_scale = []          # ranged rows: max(|rhs|, |range|) as a float, else 0
        self.col_names = []
        self.col_cost = []
        self.col_lo = []
        self.col_hi = []
        self.col_integer = []
        self.col_entries = []        # per column: list of (row index, coefficient)
        self.row_index = {}
        self.col_index = {}
        self.warnings = []
        self.quad = {}               # lower triangle of Q: (i, j) with i >= j (column indices) -> value
        self.quad_kind = None        # "QUADOBJ" or "QMATRIX"

    @property
    def n_rows(self):
        return len(self.row_names)

    @property
    def n_cols(self):
        return len(self.col_names)

    @property
    def n_quad(self):
        return len(self.quad)

    @property
    def nnz(self):
        return sum(len(e) for e in self.col_entries)

    @property
    def n_integer(self):
        return sum(1 for f in self.col_integer if f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def decode(data):
    """File bytes to text (gzip is detected by its magic bytes). Undecodable bytes survive as lone
    surrogates (the same mapping the certificate writer uses for names that are not UTF-8)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="surrogateescape")


def _split_lines(text):
    return re.split(r"\r\n|\n|\r", text)


class _Line:
    __slots__ = ("no", "raw")

    def __init__(self, no, raw):
        self.no = no
        self.raw = raw


def _fixed_fields(raw):
    """The six fixed-format fields of a data line, or None if the line does not follow the layout."""
    line = raw.rstrip("\r\n")
    if len(line) > 0 and line[0] != " ":
        return None
    if len(line.rstrip()) > 61:  # text beyond column 61 would be silently lost
        return None
    padded = line.ljust(61)
    gaps = (padded[3], padded[12:14], padded[22:24], padded[36:39], padded[47:49])
    if any(g.strip() for g in gaps):
        return None
    f = [padded[1:3], padded[4:12], padded[14:22], padded[24:36], padded[39:47], padded[49:61]]
    return [x.strip() for x in f]


def _is_number_text(t):
    t = t.strip()
    return bool(_NUMBER.match(t)) or t.lstrip("+-").lower() in ("inf", "infinity")


def _layout_ok(section, line):
    """Does a data line follow the fixed-format layout (numeric fields where numbers belong)?"""
    if section == "ROWS":
        raw = line.raw.rstrip()
        return len(raw) >= 5 and raw[0] == " " and raw[1:3].strip() in ("N", "L", "G", "E") and raw[3] == " " and raw[4:].strip() != ""
    f = _fixed_fields(line.raw)
    if f is None:
        return False
    if section in ("COLUMNS", "RHS", "RANGES") or section in _QUAD_SECTIONS:
        if not f[1] and (section == "COLUMNS" or section in _QUAD_SECTIONS):
            return False
        if not f[2] or not f[3] or not _is_number_text(f[3]):
            return False
        if f[4] or f[5]:
            return bool(f[4]) and bool(f[5]) and _is_number_text(f[5])
        return True
    if section == "BOUNDS":
        if f[0] not in _VALUE_BOUNDS | _NO_VALUE_BOUNDS or not f[2]:
            return False
        if f[0] in _VALUE_BOUNDS:
            return bool(f[3]) and _is_number_text(f[3])
        return f[3] == "" or _is_number_text(f[3])
    return False


def _has_space(names):
    return any(" " in n for n in names)


def parse_mps(data, exact=True):
    """Parses MPS file bytes into a Model."""
    text = decode(data)
    lines = []
    for no, raw in enumerate(_split_lines(text), start=1):
        if raw.strip() == "" or raw.startswith("*"):
            continue
        lines.append(_Line(no, raw))

    # ---- split into sections --------------------------------------------------------------
    sections = []  # (keyword, header line, [data lines])
    saw_endata = False
    for line in lines:
        raw = line.raw
        first_col_text = raw[0] not in " \t"
        tokens = raw.split()
        is_header = first_col_text and (len(tokens) == 1 or tokens[0].upper() in ("NAME", "OBJSENSE", "QCMATRIX", "QSECTION"))
        if is_header:
            kw = tokens[0].upper()
            if kw in _UNSUPPORTED_SECTIONS:
                raise Unsupported("line %d: quadratic constraints not supported / section %s is not supported" % (line.no, kw))
            if kw not in ("NAME", "OBJSENSE", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA") + _QUAD_SECTIONS:
                raise MpsError("line %d: unknown section %r" % (line.no, tokens[0]))
            sections.append((kw, line, []))
            if kw == "ENDATA":
                saw_endata = True
                break
        else:
            if not sections:
                raise MpsError("line %d: data before the first section" % line.no)
            sections[-1][2].append(line)
    if not saw_endata:
        raise MpsError("missing ENDATA")

    # ---- fixed or free format ------------------------------------------------------------
    layout_lines = []
    for kw, _, data_lines in sections:
        if kw in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS") + _QUAD_SECTIONS:
            for ln in data_lines:
                if kw == "COLUMNS" and "MARKER" in ln.raw.split():
                    continue
                layout_lines.append((kw, ln))
    fixed = False
    if layout_lines and all(_layout_ok(kw, ln) for kw, ln in layout_lines):
        names = []
        for kw, ln in layout_lines:
            if kw == "ROWS":
                names.append(ln.raw.rstrip()[4:].strip())
            else:
                f = _fixed_fields(ln.raw)
                names.extend(x for x in (f[1], f[2], f[4]) if x)
        fixed = _has_space(names)

    model = Model(exact)
    num = lambda s: parse_number(s, exact)

    def records(kw, data_lines):
        """Yields (line number, fields) with the fields of each data line as the section expects."""
        for ln in data_lines:
            toks = ln.raw.split()
            if kw == "COLUMNS" and "MARKER" in toks:
                yield ln.no, ("MARKER", toks)
            elif fixed:
                if kw == "ROWS":
                    raw = ln.raw.rstrip()
                    yield ln.no, (raw[1:3].strip(), raw[4:].strip())
                else:
                    f = _fixed_fields(ln.raw)
                    yield ln.no, tuple(f)
            else:
                yield ln.no, tuple(toks)

    rows_by_type = {}
    objective_row = None
    ignored_rows = set()
    first_sets = {"RHS": None, "RANGES": None, "BOUNDS": None}
    seen_set_flag = {"RHS": False, "RANGES": False, "BOUNDS": False}
    rhs = {}
    ranges = {}
    in_marker = False
    last_col = None
    seen_cols_closed = set()

    for kw, header, data_lines in sections:
        if kw == "NAME":
            parts = header.raw.split(None, 1)
            model.name = parts[1].strip() if len(parts) > 1 else ""
        elif kw == "OBJSENSE":
            sense_tokens = header.raw.split()[1:]
            for ln in data_lines:
                sense_tokens.extend(ln.raw.split())
            if sense_tokens:
                s = sense_tokens[0].upper()
                if s in ("MAX", "MAXIMIZE"):
                    model.sense = "max"
                elif s in ("MIN", "MINIMIZE"):
                    model.sense = "min"
                else:
                    raise MpsError("line %d: bad OBJSENSE %r" % (header.no, sense_tokens[0]))
        elif kw == "ROWS":
            for no, rec in records(kw, data_lines):
                if fixed:
                    typ, name = rec
                else:
                    if len(rec) != 2:
                        raise MpsError("line %d: a ROWS line needs a type and a name" % no)
                    typ, name = rec
                typ = typ.upper()
                if typ not in ("N", "L", "G", "E"):
                    raise MpsError("line %d: unknown row type %r" % (no, typ))
                if name in model.row_index or name in ignored_rows or name == objective_row:
                    raise MpsError("line %d: duplicate row name %r" % (no, name))
                if typ == "N":
                    if objective_row is None:
                        objective_row = name
                    else:
                        ignored_rows.add(name)
                        model.warnings.append("extra N row %r ignored" % name)
                else:
                    model.row_index[name] = len(model.row_names)
                    model.row_names.append(name)
                    model.row_types.append(typ)
        elif kw == "COLUMNS":
            if objective_row is None:
                raise MpsError("COLUMNS before ROWS defined an objective (N) row")
            for no, rec in records(kw, data_lines):
                if rec[0] == "MARKER":
                    toks = rec[1]
                    tag = toks[-1].strip("'").upper()
                    if tag == "INTORG":
                        in_marker = True
                    elif tag == "INTEND":
                        in_marker = False
                    else:
                        raise MpsError("line %d: bad MARKER line" % no)
                    continue
                if fixed:
                    colname = rec[1]
                    pairs = [(rec[2], rec[3])]
                    if rec[4]:
                        pairs.append((rec[4], rec[5]))
                else:
                    if len(rec) not in (3, 5):
                        raise MpsError("line %d: a COLUMNS line has 3 or 5 fields" % no)
                    colname = rec[0]
                    pairs = [(rec[1], rec[2])]
                    if len(rec) == 5:
                        pairs.append((rec[3], rec[4]))
                if colname != last_col:
                    if colname in model.col_index:
                        raise MpsError("line %d: column %r appears again after other columns" % (no, colname))
                    if last_col is not None:
                        seen_cols_closed.add(last_col)
                    model.col_index[colname] = len(model.col_names)
                    model.col_names.append(colname)
                    model.col_cost.append(0 if exact else 0.0)
                    model.col_lo.append(0 if exact else 0.0)
                    model.col_hi.append(None)
                    model.col_integer.append(in_marker)
                    model.col_entries.append([])
                    last_col = colname
                j = model.col_index[colname]
                for rname, vtext in pairs:
                    v = num(vtext)
                    if isinstance(v, float) and math.isinf(v):
                        raise MpsError("line %d: infinite matrix coefficient" % no)
                    if rname == objective_row:
                        model.col_cost[j] = v
                    elif rname in ignored_rows:
                        continue
                    elif rname in model.row_index:
                        i = model.row_index[rname]
                        if any(e[0] == i for e in model.col_entries[j]):
                            raise MpsError("line %d: duplicate entry for column %r row %r" % (no, colname, rname))
                        model.col_entries[j].append((i, v))
                    else:
                        raise MpsError("line %d: unknown row %r" % (no, rname))
            if in_marker:
                model.warnings.append("integer MARKER block was never closed")
        elif kw in ("RHS", "RANGES"):
            target = rhs if kw == "RHS" else ranges
            for no, rec in records(kw, data_lines):
                if fixed:
                    setname, pairs = rec[1], [(rec[2], rec[3])] + ([(rec[4], rec[5])] if rec[4] else [])
                    use = True
                    setname = setname or None
                else:
                    n = len(rec)
                    if n in (3, 5):
                        setname = rec[0]
                        pairs = [(rec[1], rec[2])] + ([(rec[3], rec[4])] if n == 5 else [])
                    elif n in (2, 4):
                        setname = None
                        pairs = [(rec[0], rec[1])] + ([(rec[2], rec[3])] if n == 4 else [])
                    else:
                        raise MpsError("line %d: bad %s line" % (no, kw))
                if not seen_set_flag[kw]:
                    seen_set_flag[kw] = True
                    first_sets[kw] = setname
                elif setname != first_sets[kw]:
                    model.warnings.append("%s set %r ignored" % (kw, setname))
                    continue
                for rname, vtext in pairs:
                    v = num(vtext)
                    if rname in ignored_rows:
                        continue
                    if kw == "RHS" and rname == objective_row:
                        if isinstance(v, float) and math.isinf(v):
                            raise MpsError("line %d: infinite objective offset" % no)
                        model.offset = -v
                        continue
                    if rname not in model.row_index:
                        raise MpsError("line %d: unknown row %r" % (no, rname))
                    target[model.row_index[rname]] = v
        elif kw == "BOUNDS":
            for no, rec in records(kw, data_lines):
                if fixed:
                    typ, setname, cname, vtext = rec[0].upper(), rec[1] or None, rec[2], rec[3]
                else:
                    typ = rec[0].upper()
                    toks = list(rec[1:])
                    if typ in _VALUE_BOUNDS:
                        if len(toks) == 3:
                            setname, cname, vtext = toks
                        elif len(toks) == 2:
                            setname, cname, vtext = None, toks[0], toks[1]
                        else:
                            raise MpsError("line %d: bad BOUNDS line" % no)
                    elif typ in _NO_VALUE_BOUNDS:
                        if len(toks) == 3:
                            setname, cname, vtext = toks
                        elif len(toks) == 2 and toks[0] in model.col_index and (toks[1] in model.col_index) is False and _is_number_text(toks[1]):
                            setname, cname, vtext = None, toks[0], toks[1]
                        elif len(toks) == 2:
                            setname, cname, vtext = toks[0], toks[1], ""
                        elif len(toks) == 1:
                            setname, cname, vtext = None, toks[0], ""
                        else:
                            raise MpsError("line %d: bad BOUNDS line" % no)
                    else:
                        raise MpsError("line %d: unknown bound type %r" % (no, typ))
                if typ == "SC":
                    raise Unsupported("line %d: semi-continuous (SC) bounds are not supported" % no)
                if typ not in _VALUE_BOUNDS | _NO_VALUE_BOUNDS:
                    raise MpsError("line %d: unknown bound type %r" % (no, typ))
                if not seen_set_flag["BOUNDS"]:
                    seen_set_flag["BOUNDS"] = True
                    first_sets["BOUNDS"] = setname
                elif setname != first_sets["BOUNDS"]:
                    model.warnings.append("BOUNDS set %r ignored" % setname)
                    continue
                if cname not in model.col_index:
                    raise MpsError("line %d: bound on unknown column %r" % (no, cname))
                j = model.col_index[cname]
                v = None
                if typ in _VALUE_BOUNDS:
                    v = num(vtext)
                zero = 0 if exact else 0.0
                if typ in ("UP", "UI"):
                    if is_infinite(v):
                        if v < 0:
                            raise Unsupported("line %d: upper bound -infinity" % no)
                        model.col_hi[j] = None
                    else:
                        model.col_hi[j] = v
                        if v < 0 and model.col_lo[j] == zero:
                            model.col_lo[j] = None
                            model.warnings.append("negative UP bound on column %r with lower 0: lower set to -inf" % cname)
                    if typ == "UI":
                        model.col_integer[j] = True
                elif typ in ("LO", "LI"):
                    if is_infinite(v):
                        if v > 0:
                            raise Unsupported("line %d: lower bound +infinity" % no)
                        model.col_lo[j] = None
                    else:
                        model.col_lo[j] = v
                    if typ == "LI":
                        model.col_integer[j] = True
                elif typ == "FX":
                    if is_infinite(v):
                        raise Unsupported("line %d: infinite FX bound" % no)
                    model.col_lo[j] = v
                    model.col_hi[j] = v
                elif typ == "FR":
                    model.col_lo[j] = None
                    model.col_hi[j] = None
                elif typ == "MI":
                    model.col_lo[j] = None
                elif typ == "PL":
                    model.col_hi[j] = None
                elif typ == "BV":
                    model.col_lo[j] = zero
                    model.col_hi[j] = 1 if exact else 1.0
                    model.col_integer[j] = True

    # ---- the quadratic objective (docs/QP.md) -------------------------------------------------
    quad_sections = [sec for sec in sections if sec[0] in _QUAD_SECTIONS]
    if len(quad_sections) > 1:
        raise MpsError("QUADOBJ and QMATRIX (or a repeated section) cannot both be given")
    for kw, header, data_lines in quad_sections:
        model.quad_kind = kw
        listed = {}
        for no, rec in records(kw, data_lines):
            if fixed:
                first = rec[1]
                pairs = [(rec[2], rec[3])] + ([(rec[4], rec[5])] if rec[4] else [])
            else:
                if len(rec) not in (3, 5):
                    raise MpsError("line %d: a %s line has 3 or 5 fields" % (no, kw))
                first = rec[0]
                pairs = [(rec[1], rec[2])] + ([(rec[3], rec[4])] if len(rec) == 5 else [])
            for second, vtext in pairs:
                if first not in model.col_index or second not in model.col_index:
                    raise MpsError("line %d: unknown column in %s" % (no, kw))
                a, b = model.col_index[first], model.col_index[second]
                v = num(vtext)
                if isinstance(v, float) and math.isinf(v):
                    raise MpsError("line %d: infinite quadratic coefficient" % no)
                key = (max(a, b), min(a, b)) if kw == "QUADOBJ" else (a, b)
                if key in listed:
                    raise MpsError("line %d: duplicate %s entry" % (no, kw))
                listed[key] = v
        for (a, b), v in listed.items():
            if kw == "QMATRIX":
                if (b, a) not in listed:
                    raise MpsError("QMATRIX is not symmetric: entry (%s, %s) has no mirror entry" % (model.col_names[a], model.col_names[b]))
                w = listed[(b, a)]
                if a > b and abs(float(v) - float(w)) > 1e-12 * max(1.0, abs(float(v)), abs(float(w))):
                    raise MpsError("QMATRIX is not symmetric: entries (%s, %s) differ" % (model.col_names[a], model.col_names[b]))
                if a < b:
                    continue
            if v != 0:
                model.quad[(a, b)] = v

    # ---- row bounds from the right-hand sides and ranges ----------------------------------
    zero = 0 if exact else 0.0
    for i, typ in enumerate(model.row_types):
        b = rhs.get(i, zero)
        r = ranges.get(i)
        model.row_scale.append(float(max(abs(b), abs(r))) if (r is not None and not is_infinite(b) and not is_infinite(r) and r != 0) else 0.0)
        lo = hi = None
        if is_infinite(b):
            if typ == "L" and b > 0 or typ == "G" and b < 0:
                lo, hi = None, None  # a free row
            elif typ == "E":
                raise Unsupported("equality row with an infinite right-hand side")
            else:
                raise Unsupported("row %r has an infeasible infinite right-hand side" % model.row_names[i])
            model.row_lo.append(lo)
            model.row_hi.append(hi)
            continue
        if typ == "L":
            lo, hi = None, b
            if r is not None and not is_infinite(r):
                lo = b - abs(r)
        elif typ == "G":
            lo, hi = b, None
            if r is not None and not is_infinite(r):
                hi = b + abs(r)
        else:
            lo, hi = b, b
            if r is not None and not is_infinite(r):
                if r > 0:
                    hi = b + r
                elif r < 0:
                    lo = b - abs(r)
        model.row_lo.append(lo)
        model.row_hi.append(hi)

    # Bounds of magnitude >= 1e30 are infinite.
    for j in range(model.n_cols):
        if model.col_lo[j] is not None and is_infinite(model.col_lo[j]):
            model.col_lo[j] = None
        if model.col_hi[j] is not None and is_infinite(model.col_hi[j]):
            model.col_hi[j] = None
    return model


def parse_file(path, exact=True):
    return parse_mps(read_bytes(path), exact)
