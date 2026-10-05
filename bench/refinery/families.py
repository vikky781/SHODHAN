"""The synthetic refinery / process case-study families (docs/REFINERY.md). Python standard library only.

SYNTHETIC: structure follows textbook formulations; not plant or MRPL data.

Every generator returns (Model, knobs) and fills `model.witness` with a point that is feasible by construction.
All data are dyadic rationals (multiples of 1/16 or integers), so the witness is feasible in exact arithmetic.

Knobs common to all families:
  size    tiny | small | medium | large   (R2 also: huge, more than 1e5 columns)
  seed    integer; the generator RNG is our own (gen_core.Rng)
  loose   >= 1: factor by which the big-M coefficients exceed their tight value (R3, R4, R5)
  ties    0..1: share of cost coefficients that are replaced by a common value (degeneracy: many ties)
  spread  >= 0: decades of coefficient range (costs and quantities are multiplied by 2**randint(0, spread*3))
"""

from fractions import Fraction

from gen_core import Model, Rng

F = Fraction

SIZES = ("tiny", "small", "medium", "large")


def _scale(rng, spread):
    return F(2) ** rng.randint(0, 3 * spread) if spread else F(1)


def _cost(rng, lo, hi, ties, tie_value, spread):
    if ties and rng.chance(ties):
        return F(tie_value)
    return rng.dyadic(lo, hi) * _scale(rng, spread)


# ---------------------------------------------------------------------------------------------------------------
# R1: crude selection and blending LP
# ---------------------------------------------------------------------------------------------------------------
R1_SIZES = {"tiny": (3, 4, 3, 2), "small": (6, 5, 4, 2), "medium": (12, 5, 6, 3), "large": (40, 6, 8, 4)}


def r1_blend(size, seed, loose=1, ties=0, spread=0):
    """Crude purchases b_c in [0, avail_c] feed a CDU of capacity Cap with fixed yields y_cs of intermediate streams s.
    Streams go to products p (f_sp) or are sold as stream s at a low price (w_s). Linear blending: for every product p
    and quality k, sum_s (q_sk - Qmax_pk) f_sp <= 0 (a fixed quality index per stream, blended linearly on volume).
    Minimize  crude cost - product revenue - stream revenue  (a negated margin)."""
    nC, nS, nP, nQ = R1_SIZES[size]
    rng = Rng(seed * 1000003 + 1)
    m = Model("R1_%s_%d" % (size, seed), "R1 crude selection and blending LP")
    price_c = [_cost(rng, 40, 80, ties, 60, spread) for _ in range(nC)]
    avail = [rng.dyadic(20, 100) for _ in range(nC)]
    # fixed yields: every crude gives each stream a share (multiples of 1/16) that sum to at most 1
    yld = []
    for _ in range(nC):
        raw = [rng.randint(0, 6) for _ in range(nS)]
        while sum(raw) > 16:
            raw[rng.randint(0, nS - 1)] = 0
        yld.append([F(r, 16) for r in raw])
    qual = [[rng.dyadic(0, 8) for _ in range(nQ)] for _ in range(nS)]
    price_p = [_cost(rng, 70, 130, ties, 100, spread) for _ in range(nP)]
    price_s = [rng.dyadic(20, 50) for _ in range(nS)]
    demand = [rng.dyadic(10, 60) for _ in range(nP)]
    cap = sum(avail, F(0)) * F(3, 4)
    for c in range(nC):
        m.col("B%d" % c, 0, avail[c], price_c[c])
    for s in range(nS):
        m.col("W%d" % s, 0, None, -price_s[s])
        for p in range(nP):
            m.col("F%d_%d" % (s, p), 0, None, 0)
    for p in range(nP):
        m.col("S%d" % p, 0, demand[p], -price_p[p])
    m.row("CAP", "L", cap, [("B%d" % c, 1) for c in range(nC)])
    for s in range(nS):
        terms = [("B%d" % c, yld[c][s]) for c in range(nC)]
        terms += [("F%d_%d" % (s, p), -1) for p in range(nP)] + [("W%d" % s, -1)]
        m.row("BAL%d" % s, "E", 0, terms)
    for p in range(nP):
        m.row("PRD%d" % p, "E", 0, [("F%d_%d" % (s, p), 1) for s in range(nS)] + [("S%d" % p, -1)])
    # witness: buy a dyadic share of each availability, send each stream to one product (or sell it), then the quality limits
    # are set to the largest quality among the streams that the witness sends to the product
    share = [F(rng.randint(0, 8), 8) for _ in range(nC)]
    bw = [avail[c] * share[c] for c in range(nC)]
    if sum(bw, F(0)) > cap:
        bw = [v * F(1, 2) for v in bw]
    stream = [sum((yld[c][s] * bw[c] for c in range(nC)), F(0)) for s in range(nS)]
    target = [rng.randint(-1, nP - 1) for _ in range(nS)]  # -1: sold as a stream
    flow = [[F(0)] * nP for _ in range(nS)]
    sales = [F(0)] * nP
    wsell = [F(0)] * nS
    for s in range(nS):
        if target[s] < 0:
            wsell[s] = stream[s]
        else:
            p = target[s]
            take = min(stream[s], demand[p] - sales[p])
            flow[s][p] = take
            sales[p] += take
            wsell[s] = stream[s] - take
    used = [[s for s in range(nS) if flow[s][p] > 0] for p in range(nP)]
    qmax = [[max([qual[s][k] for s in used[p]] + [F(0)]) + rng.dyadic(0, 2) for k in range(nQ)] for p in range(nP)]
    for p in range(nP):
        for k in range(nQ):
            m.row("SPC%d_%d" % (p, k), "L", 0, [("F%d_%d" % (s, p), qual[s][k] - qmax[p][k]) for s in range(nS)])
    for c in range(nC):
        m.witness["B%d" % c] = bw[c]
    for s in range(nS):
        m.witness["W%d" % s] = wsell[s]
        for p in range(nP):
            m.witness["F%d_%d" % (s, p)] = flow[s][p]
    for p in range(nP):
        m.witness["S%d" % p] = sales[p]
    return m, dict(size=size, crudes=nC, streams=nS, products=nP, qualities=nQ, ties=ties, spread=spread)


# ---------------------------------------------------------------------------------------------------------------
# R2: multi-period planning LP
# ---------------------------------------------------------------------------------------------------------------
R2_SIZES = {"tiny": (3, 2, 2), "small": (6, 4, 3), "medium": (12, 10, 8), "large": (26, 40, 20), "huge": (52, 520, 210)}


def r2_planning(size, seed, loose=1, ties=0, spread=0):
    """Periods t, crudes c, products p. b_ct purchases (<= avail), v_ct crude inventory (<= tank), r_ct processed crude.
    Crude balance  v_ct = v_c,t-1 + b_ct - r_ct;  unit capacity  sum_c r_ct <= cap_t;  fixed yields y_cp of products;
    product position  I_pt - K_pt = I_p,t-1 - K_p,t-1 + sum_c y_cp r_ct - d_pt  with inventory I >= 0 and backlog K >= 0.
    Minimize purchase + holding + backlog penalty (revenue is constant: demand is met or backlogged)."""
    T, nC, nP = R2_SIZES[size]
    rng = Rng(seed * 1000003 + 2)
    m = Model("R2_%s_%d" % (size, seed), "R2 multi-period planning LP")
    cost_c = [_cost(rng, 40, 80, ties, 60, spread) for _ in range(nC)]
    hold_c = [rng.dyadic(0, 2) for _ in range(nC)]
    hold_p = [rng.dyadic(1, 3) for _ in range(nP)]
    pen_p = [rng.dyadic(20, 60) for _ in range(nP)]
    yld = []
    for _ in range(nC):
        raw = [rng.randint(0, 3) for _ in range(nP)]
        while sum(raw) > 16:
            raw[rng.randint(0, nP - 1)] = 0
        yld.append([F(r, 16) for r in raw])
    avail = [[rng.dyadic(5, 40) for _ in range(nC)] for _ in range(T)]
    tank = [rng.dyadic(20, 80) for _ in range(nC)]
    cap = [rng.dyadic(nC * 4, nC * 20) for _ in range(T)]
    d = [[rng.dyadic(0, 12) for _ in range(nP)] for _ in range(T)]
    for t in range(T):
        for c in range(nC):
            m.col("B%d_%d" % (c, t), 0, avail[t][c], cost_c[c])
            m.col("V%d_%d" % (c, t), 0, tank[c], hold_c[c])
            m.col("R%d_%d" % (c, t), 0, None, 0)
        for p in range(nP):
            m.col("I%d_%d" % (p, t), 0, None, hold_p[p])
            m.col("K%d_%d" % (p, t), 0, None, pen_p[p])
    for t in range(T):
        for c in range(nC):
            terms = [("V%d_%d" % (c, t), 1), ("B%d_%d" % (c, t), -1), ("R%d_%d" % (c, t), 1)]
            if t > 0:
                terms.append(("V%d_%d" % (c, t - 1), -1))
            m.row("CB%d_%d" % (c, t), "E", 0, terms)
        m.row("UC%d" % t, "L", cap[t], [("R%d_%d" % (c, t), 1) for c in range(nC)])
        for p in range(nP):
            terms = [("I%d_%d" % (p, t), 1), ("K%d_%d" % (p, t), -1)]
            terms += [("R%d_%d" % (c, t), -yld[c][p]) for c in range(nC) if yld[c][p] != 0]
            if t > 0:
                terms += [("I%d_%d" % (p, t - 1), -1), ("K%d_%d" % (p, t - 1), 1)]
            m.row("PB%d_%d" % (p, t), "E", -d[t][p], terms)
    # witness: buy and process nothing, carry the cumulative demand as backlog
    cum = [F(0)] * nP
    for t in range(T):
        for c in range(nC):
            for nm in ("B", "V", "R"):
                m.witness["%s%d_%d" % (nm, c, t)] = F(0)
        for p in range(nP):
            cum[p] += d[t][p]
            m.witness["I%d_%d" % (p, t)] = F(0)
            m.witness["K%d_%d" % (p, t)] = cum[p]
    return m, dict(size=size, periods=T, crudes=nC, products=nP, ties=ties, spread=spread)


# ---------------------------------------------------------------------------------------------------------------
# R3: crude-oil scheduling MILP, time-discretized
# ---------------------------------------------------------------------------------------------------------------
R3_SIZES = {"tiny": (1, 2, 3), "small": (2, 3, 6), "medium": (4, 4, 12), "large": (6, 5, 24)}


def r3_scheduling(size, seed, loose=1, ties=0, spread=0):
    """Vessels v (arrival slot a_v, parcel Q_v), tanks k (capacity cap_k, initial inventory), one CDU, slots t.
    x_vkt >= 0 unloaded volume, z_vkt binary 'v unloads into k in slot t' (link x <= M z with M = min(Q_v, cap_k) * loose),
    sum_k z_vkt <= 1 per vessel, one berth: sum_vk z_vkt <= 1; pumping rate: sum_vk x_vkt <= rate; unfulfilled volume
    u_v = Q_v - sum x costs a penalty. Tank balance  inv_kt = inv_k,t-1 + sum_v x_vkt - f_kt, 0 <= inv <= cap; feed
    f_kt <= min(cap_k, Fmax) * loose * w_kt, w binary 'k feeds the CDU in slot t', at most one feeding tank per slot, CDU
    maximum feed sum_k f_kt <= Fmax; settling: z_vkt + w_kt <= 1; CDU minimum feed  sum_k f_kt + e_t >= lo, e_t >= 0 a
    shortfall with a penalty; changeover  g_kt >= w_kt - w_k,t-1 with a cost. The physical limits (rate, Fmax) are rows
    of their own, so `loose` only weakens the big-M links and every loose value has the same feasible set."""
    nV, nK, T = R3_SIZES[size]
    rng = Rng(seed * 1000003 + 3)
    m = Model("R3_%s_%d%s" % (size, seed, "" if loose == 1 else "_weak"), "R3 crude-oil scheduling MILP")
    rate = rng.dyadic(8, 16)
    fmax = rng.dyadic(6, 12)
    lo_feed = rng.dyadic(2, 5)
    arrive = [rng.randint(0, max(0, T // 2)) for _ in range(nV)]
    parcel = [rng.dyadic(10, 40) for _ in range(nV)]
    capk = [rng.dyadic(30, 70) for _ in range(nK)]
    inv0 = [capk[k] * F(rng.randint(0, 4), 8) for k in range(nK)]
    pen_u = rng.dyadic(30, 60) * _scale(rng, spread)
    pen_e = rng.dyadic(20, 40)
    chg = rng.dyadic(1, 4)
    hold = rng.dyadic(0, 1)
    L = F(loose)
    for v in range(nV):
        m.col("U%d" % v, 0, parcel[v], pen_u)
    for t in range(T):
        m.col("E%d" % t, 0, None, pen_e)
    for k in range(nK):
        for t in range(T):
            m.col("N%d_%d" % (k, t), 0, capk[k], hold)
            m.col("FD%d_%d" % (k, t), 0, None, 0)
            m.binary("W%d_%d" % (k, t))
            m.col("G%d_%d" % (k, t), 0, None, chg)
    for v in range(nV):
        for k in range(nK):
            for t in range(arrive[v], T):
                m.col("X%d_%d_%d" % (v, k, t), 0, None, 0)
                m.binary("Z%d_%d_%d" % (v, k, t))
    for v in range(nV):
        m.row("VES%d" % v, "E", parcel[v], [("U%d" % v, 1)] + [("X%d_%d_%d" % (v, k, t), 1) for k in range(nK) for t in range(arrive[v], T)])
        for t in range(arrive[v], T):
            m.row("ONE%d_%d" % (v, t), "L", 1, [("Z%d_%d_%d" % (v, k, t), 1) for k in range(nK)])
            for k in range(nK):
                big = min(parcel[v], capk[k]) * L
                m.row("LNK%d_%d_%d" % (v, k, t), "L", 0, [("X%d_%d_%d" % (v, k, t), 1), ("Z%d_%d_%d" % (v, k, t), -big)])
    for t in range(T):
        zs = [("Z%d_%d_%d" % (v, k, t), 1) for v in range(nV) for k in range(nK) if t >= arrive[v]]
        if zs:
            m.row("BERTH%d" % t, "L", 1, zs)
            m.row("RATE%d" % t, "L", rate, [("X%d_%d_%d" % (v, k, t), 1) for v in range(nV) for k in range(nK) if t >= arrive[v]])
        m.row("CDUMAX%d" % t, "L", fmax, [("FD%d_%d" % (k, t), 1) for k in range(nK)])
        m.row("ONEFEED%d" % t, "L", 1, [("W%d_%d" % (k, t), 1) for k in range(nK)])
        m.row("CDU%d" % t, "G", lo_feed, [("FD%d_%d" % (k, t), 1) for k in range(nK)] + [("E%d" % t, 1)])
        for k in range(nK):
            terms = [("N%d_%d" % (k, t), 1), ("FD%d_%d" % (k, t), 1)] + [("X%d_%d_%d" % (v, k, t), -1) for v in range(nV) if t >= arrive[v]]
            if t > 0:
                terms.append(("N%d_%d" % (k, t - 1), -1))
            m.row("INV%d_%d" % (k, t), "E", inv0[k] if t == 0 else 0, terms)
            m.row("FLK%d_%d" % (k, t), "L", 0, [("FD%d_%d" % (k, t), 1), ("W%d_%d" % (k, t), -min(capk[k], fmax) * L)])
            for v in range(nV):
                if t >= arrive[v]:
                    m.row("SET%d_%d_%d" % (v, k, t), "L", 1, [("Z%d_%d_%d" % (v, k, t), 1), ("W%d_%d" % (k, t), 1)])
            gt = [("G%d_%d" % (k, t), 1), ("W%d_%d" % (k, t), -1)]
            if t > 0:
                gt.append(("W%d_%d" % (k, t - 1), 1))
            m.row("CHG%d_%d" % (k, t), "G", 0, gt)
    # witness: nothing is unloaded, nothing is fed, the CDU feed shortfall is paid
    for v in range(nV):
        m.witness["U%d" % v] = parcel[v]
    for t in range(T):
        m.witness["E%d" % t] = lo_feed
    for k in range(nK):
        for t in range(T):
            m.witness["N%d_%d" % (k, t)] = inv0[k]
            m.witness["FD%d_%d" % (k, t)] = F(0)
            m.witness["W%d_%d" % (k, t)] = F(0)
            m.witness["G%d_%d" % (k, t)] = F(0)
    for v in range(nV):
        for k in range(nK):
            for t in range(arrive[v], T):
                m.witness["X%d_%d_%d" % (v, k, t)] = F(0)
                m.witness["Z%d_%d_%d" % (v, k, t)] = F(0)
    return m, dict(size=size, vessels=nV, tanks=nK, slots=T, loose=loose, ties=ties, spread=spread)


# ---------------------------------------------------------------------------------------------------------------
# R4: blend / changeover MILP
# ---------------------------------------------------------------------------------------------------------------
R4_SIZES = {"tiny": (3, 4), "small": (4, 8), "medium": (8, 14), "large": (15, 24)}


def r4_blend_changeover(size, seed, loose=1, ties=0, spread=0):
    """One blender, grades g, slots t. s_gt binary 'the blender is set up for grade g in slot t' (at most one grade per
    slot), blender capacity sum_g q_gt <= cap, production link q_gt <= cap * loose * s_gt (fixed charge; the capacity row
    keeps every loose value equivalent), changeover  h_gt >= s_gt - s_g,t-1  at a sequence-independent cost, inventory I_gt and backlog K_gt: I_gt - K_gt = I_g,t-1 - K_g,t-1 + q_gt - d_gt."""
    G, T = R4_SIZES[size]
    rng = Rng(seed * 1000003 + 4)
    m = Model("R4_%s_%d%s" % (size, seed, "" if loose == 1 else "_weak"), "R4 blend and changeover MILP")
    cap = rng.dyadic(20, 40)
    L = F(loose)
    setup = [_cost(rng, 20, 60, ties, 40, spread) for _ in range(G)]
    prod_cost = [rng.dyadic(1, 4) for _ in range(G)]
    hold = [rng.dyadic(1, 2) for _ in range(G)]
    pen = [rng.dyadic(15, 40) for _ in range(G)]
    d = [[rng.dyadic(0, 12) if rng.chance(0.4) else F(0) for _ in range(G)] for _ in range(T)]
    i0 = [rng.dyadic(0, 15) for _ in range(G)]
    for t in range(T):
        for g in range(G):
            m.binary("S%d_%d" % (g, t))
            m.col("Q%d_%d" % (g, t), 0, None, prod_cost[g])
            m.col("H%d_%d" % (g, t), 0, None, setup[g])
            m.col("I%d_%d" % (g, t), 0, None, hold[g])
            m.col("K%d_%d" % (g, t), 0, None, pen[g])
    for t in range(T):
        m.row("ONE%d" % t, "L", 1, [("S%d_%d" % (g, t), 1) for g in range(G)])
        m.row("CAPB%d" % t, "L", cap, [("Q%d_%d" % (g, t), 1) for g in range(G)])
        for g in range(G):
            m.row("LNK%d_%d" % (g, t), "L", 0, [("Q%d_%d" % (g, t), 1), ("S%d_%d" % (g, t), -cap * L)])
            ch = [("H%d_%d" % (g, t), 1), ("S%d_%d" % (g, t), -1)]
            if t > 0:
                ch.append(("S%d_%d" % (g, t - 1), 1))
            m.row("CHG%d_%d" % (g, t), "G", 0, ch)
            terms = [("I%d_%d" % (g, t), 1), ("K%d_%d" % (g, t), -1), ("Q%d_%d" % (g, t), -1)]
            rhs = -d[t][g]
            if t > 0:
                terms += [("I%d_%d" % (g, t - 1), -1), ("K%d_%d" % (g, t - 1), 1)]
            else:
                rhs += i0[g]
            m.row("BAL%d_%d" % (g, t), "E", rhs, terms)
    net = [i0[g] for g in range(G)]
    for t in range(T):
        for g in range(G):
            net[g] -= d[t][g]
            m.witness["S%d_%d" % (g, t)] = F(0)
            m.witness["Q%d_%d" % (g, t)] = F(0)
            m.witness["H%d_%d" % (g, t)] = F(0)
            m.witness["I%d_%d" % (g, t)] = max(net[g], F(0))
            m.witness["K%d_%d" % (g, t)] = max(-net[g], F(0))
    return m, dict(size=size, grades=G, slots=T, loose=loose, ties=ties, spread=spread)


# ---------------------------------------------------------------------------------------------------------------
# R5: capacitated facility location with transportation, multi-commodity
# ---------------------------------------------------------------------------------------------------------------
R5_SIZES = {"tiny": (2, 3, 2), "small": (4, 8, 2), "medium": (8, 30, 3), "large": (20, 120, 3)}


def r5_facility(size, seed, loose=1, ties=0, spread=0):
    """Facilities f (open y_f binary, fixed cost, capacity C_f), customers j, commodities k with demand d_jk.
    Flows x_fjk >= 0 at unit cost c_fjk; demand  sum_f x_fjk = d_jk; capacity  sum_jk x_fjk <= C_f y_f;
    link  x_fjk <= min(d_jk, C_f) * loose * y_f (loose = 1 is the strong form)."""
    nF, nJ, nK = R5_SIZES[size]
    rng = Rng(seed * 1000003 + 5)
    m = Model("R5_%s_%d%s" % (size, seed, "" if loose == 1 else "_weak"), "R5 facility location and transportation")
    L = F(loose)
    d = [[rng.dyadic(1, 12) for _ in range(nK)] for _ in range(nJ)]
    total = sum((sum(row, F(0)) for row in d), F(0))
    base = F(-((-(total * 3)) // (2 * nF))) + 20  # an integer, so that every capacity stays dyadic
    capf = [base * F(rng.randint(8, 16), 16) for _ in range(nF)]
    # make sure the capacities cover the demand with room to spare
    while sum(capf, F(0)) < total * F(5, 4):
        capf = [c * F(5, 4) for c in capf]
    fixed = [_cost(rng, 100, 300, ties, 200, spread) for _ in range(nF)]
    for f in range(nF):
        m.binary("Y%d" % f, fixed[f])
    for f in range(nF):
        for j in range(nJ):
            for k in range(nK):
                m.col("X%d_%d_%d" % (f, j, k), 0, None, _cost(rng, 1, 9, ties, 5, spread))
    for j in range(nJ):
        for k in range(nK):
            m.row("DEM%d_%d" % (j, k), "E", d[j][k], [("X%d_%d_%d" % (f, j, k), 1) for f in range(nF)])
    for f in range(nF):
        m.row("CAP%d" % f, "L", 0, [("X%d_%d_%d" % (f, j, k), 1) for j in range(nJ) for k in range(nK)] + [("Y%d" % f, -capf[f])])
        for j in range(nJ):
            for k in range(nK):
                big = min(d[j][k], capf[f]) * L
                m.row("LNK%d_%d_%d" % (f, j, k), "L", 0, [("X%d_%d_%d" % (f, j, k), 1), ("Y%d" % f, -big)])
    # witness: all facilities open, customers served greedily from the first facility with spare capacity
    left = list(capf)
    flow = {}
    for j in range(nJ):
        for k in range(nK):
            rest = d[j][k]
            for f in range(nF):
                take = min(rest, left[f])
                if take > 0:
                    flow[(f, j, k)] = take
                    left[f] -= take
                    rest -= take
            assert rest == 0
    for f in range(nF):
        m.witness["Y%d" % f] = F(1)
        for j in range(nJ):
            for k in range(nK):
                m.witness["X%d_%d_%d" % (f, j, k)] = flow.get((f, j, k), F(0))
    return m, dict(size=size, facilities=nF, customers=nJ, commodities=nK, loose=loose, ties=ties, spread=spread)


FAMILIES = {
    "R1": r1_blend,
    "R2": r2_planning,
    "R3": r3_scheduling,
    "R4": r4_blend_changeover,
    "R5": r5_facility,
}
