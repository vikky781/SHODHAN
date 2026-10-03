"""Random MPS text generator that exercises every documented feature (free and fixed format)."""

import random

ALL_NUMBERS = ["0", "1", "-1", "2", "-3", "0.5", "-0.25", "0.1", "-0.3", "12.75", "1000", "-4.5", "3.14159265358979", "1e-3", "-2.5e2",
           "1.0D+2", "6d-1", "7E+1", "100000.5", "0.0001", ".5", "5.", "123456.789", "1e-8", "-7.77e-5"]
SHORT = [n for n in ALL_NUMBERS if len(n) <= 12]  # fixed format has 12 characters per number


def fixed_line(f1="", f2="", f3="", f4="", f5="", f6=""):
    return " " + f1.ljust(2) + " " + f2.ljust(8) + "  " + f3.ljust(8) + "  " + f4.rjust(12) + "   " + f5.ljust(8) + "  " + f6.rjust(12)


def generate(seed):
    """One random model as MPS text; returns (text, description)."""
    rnd = random.Random(seed)
    fixed = rnd.random() < 0.4
    NUMBERS = SHORT if fixed else ALL_NUMBERS
    NONZERO = [n for n in NUMBERS if float(n.replace("D", "e").replace("d", "e")) != 0]
    n_rows = rnd.randint(1, 7)
    n_cols = rnd.randint(1, 8)
    spaced = fixed
    rname = ["ROW %d" % (i + 1) if spaced and (i == 0 or rnd.random() < 0.3) else "R%d" % (i + 1) for i in range(n_rows)]
    cname = ["COL %d" % (j + 1) if spaced and (j == 0 or rnd.random() < 0.3) else "C%d" % (j + 1) for j in range(n_cols)]
    if fixed:
        cname = [c[:8] for c in cname]
        rname = [r[:8] for r in rname]
    types = [rnd.choice("LGE") for _ in range(n_rows)]
    obj = "COST"
    extra_n = rnd.random() < 0.2
    lines = []
    desc = []
    if rnd.random() < 0.8:
        lines.append("NAME          " + rnd.choice(["T%d" % seed, "MODEL"]))
    else:
        desc.append("no NAME")
    sense = rnd.choice(["none", "none", "section", "inline", "min_section", "maximize"])
    if sense == "section":
        lines += ["OBJSENSE", "    MAX"]
    elif sense == "inline":
        lines.append("OBJSENSE MAX")
    elif sense == "min_section":
        lines += ["OBJSENSE", "    MINIMIZE"]
    elif sense == "maximize":
        lines += ["OBJSENSE", "    MAXIMIZE"]
    desc.append("sense " + sense)
    if rnd.random() < 0.3:
        lines.append("* a comment line")

    def row_line(t, name):
        return " %s  %s" % (t, name)

    lines.append("ROWS")
    lines.append(row_line("N", obj))
    if extra_n:
        lines.append(row_line("N", "OTHEROBJ"))
    for t, n in zip(types, rname):
        lines.append(row_line(t, n))

    def data(f2, f3, v3, f5=None, v5=None, f1=""):
        if fixed:
            return fixed_line(f1, f2, f3, v3, f5 or "", v5 or "")
        pad = " " * rnd.randint(1, 3)
        text = "    %s%s%s%s%s" % (f2, pad, f3, pad, v3) if f2 != "" else "    %s%s%s" % (f3, pad, v3)
        if f5 is not None:
            text += pad + f5 + pad + v5
        return text

    lines.append("COLUMNS")
    in_marker = False
    int_cols = set()
    for j, name in enumerate(cname):
        want_int = rnd.random() < 0.25
        if want_int and not in_marker:
            lines.append("    MARKER                 'MARKER'                 'INTORG'")
            in_marker = True
        elif not want_int and in_marker:
            lines.append("    MARKER                 'MARKER'                 'INTEND'")
            in_marker = False
        if want_int:
            int_cols.add(j)
        entries = []
        if rnd.random() < 0.8:
            entries.append((obj, rnd.choice(NUMBERS)))
        for i in range(n_rows):
            if rnd.random() < 0.45:
                entries.append((rname[i], rnd.choice(NONZERO if rnd.random() < 0.9 else NUMBERS)))
        if extra_n and rnd.random() < 0.5:
            entries.append(("OTHEROBJ", rnd.choice(NONZERO)))
        if not entries:
            entries.append((obj, "0"))
        for k in range(0, len(entries), 2):
            chunk = entries[k:k + 2]
            if len(chunk) == 2:
                lines.append(data(name, chunk[0][0], chunk[0][1], chunk[1][0], chunk[1][1]))
            else:
                lines.append(data(name, chunk[0][0], chunk[0][1]))
    if in_marker and rnd.random() < 0.7:
        lines.append("    MARKER                 'MARKER'                 'INTEND'")

    omit_rhs_set = (not fixed) and rnd.random() < 0.25

    def section_entries(rows_pool):
        out = []
        for name in rows_pool:
            if rnd.random() < 0.6:
                out.append((name, rnd.choice(NUMBERS)))
        return out

    def emit_pairs(setname, entries, omit):
        res = []
        for k in range(0, len(entries), 2):
            chunk = entries[k:k + 2]
            if fixed:
                res.append(fixed_line("", setname, chunk[0][0], chunk[0][1], chunk[1][0] if len(chunk) == 2 else "", chunk[1][1] if len(chunk) == 2 else ""))
            else:
                toks = ([] if omit else [setname]) + [chunk[0][0], chunk[0][1]] + ([chunk[1][0], chunk[1][1]] if len(chunk) == 2 else [])
                res.append("    " + "  ".join(toks))
        return res

    rhs_entries = section_entries(rname)
    if rnd.random() < 0.5:
        rhs_entries.append((obj, rnd.choice(NUMBERS)))
    if rnd.random() < 0.15:
        k = rnd.randrange(n_rows)  # an infinite right-hand side only where it makes a free row
        if types[k] == "L":
            rhs_entries.append((rname[k], "1e30"))
        elif types[k] == "G":
            rhs_entries.append((rname[k], "-1e30"))
    ranges_first = rnd.random() < 0.3
    rng_entries = section_entries(rname)
    if rhs_entries:
        rhs_lines = ["RHS"] + emit_pairs("RHS", rhs_entries, omit_rhs_set)
        # a second RHS set that must be ignored
        if not fixed and not omit_rhs_set and rnd.random() < 0.2:
            rhs_lines.append("    OTHERSET  %s  99" % rname[0])
    else:
        rhs_lines = []
    rng_lines = (["RANGES"] + emit_pairs("RNG", rng_entries, omit_rhs_set)) if rng_entries else []
    if ranges_first:
        lines += rng_lines + rhs_lines
    else:
        lines += rhs_lines + rng_lines

    bounds = []
    for j, name in enumerate(cname):
        if rnd.random() < 0.55:
            bt = rnd.choice(["UP", "LO", "FX", "FR", "MI", "PL", "BV", "LI", "UI", "UP", "LO", "UP"])
            val = rnd.choice(NUMBERS + (["1e30"] if bt == "UP" else ["-1e30"]) if bt in ("UP", "LO") else NUMBERS)
            if bt in ("FX", "LI", "UI"):
                val = rnd.choice(NUMBERS)
            bounds.append((bt, name, val))
            if rnd.random() < 0.3:  # a second entry on the same column: later entries win
                bt2 = rnd.choice(["UP", "LO", "PL", "MI"])
                bounds.append((bt2, name, rnd.choice(NUMBERS)))
    if bounds:
        lines.append("BOUNDS")
        omit_b = (not fixed) and rnd.random() < 0.25
        for bt, name, val in bounds:
            needs_value = bt in ("UP", "LO", "FX", "LI", "UI")
            if fixed:
                lines.append(fixed_line(bt, "BND", name, val if needs_value else ""))
            elif omit_b:
                lines.append("  %s %s%s" % (bt, name, (" " + val) if needs_value else ""))
            else:
                lines.append("  %s BND %s%s" % (bt, name, (" " + val) if needs_value else ""))
    lines.append("ENDATA")
    return "\n".join(lines) + "\n", ", ".join(desc + (["fixed"] if fixed else ["free"]))
