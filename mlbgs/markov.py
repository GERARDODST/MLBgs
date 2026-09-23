"""DIAMANTE-24: modelo por turno al bate para el Pro-Lab.

1. Probabilidad de cada resultado del turno (K, BB, 1B, 2B, 3B, HR, OUT) bateador vs pitcher con el
   método de razón de momios multinomial (log5 generalizado, sección 5.7.2) sobre tasas regresadas a la
   media con los puntos de estabilización del framework (5.7.4), splits por mano, parque y viento.
2. Cadena de Markov de las 24 situaciones base-out (5.7.3): matriz de transición Q, matriz fundamental
   N = (I − Q)⁻¹ y matriz de expectativa de carreras RE24 = N·r; también P(anotar ≥1).
3. Cadena por lineup (24 × 9 = 216 estados): carreras esperadas de una entrada según quién la abre y
   matriz de rotación del lineup T (quién abre la siguiente entrada).
4. Simulación Monte Carlo del partido completo (5.7.9): abridor con límite de bateadores, bullpen real
   por rol y disponibilidad, penalización por veces en el orden, corredor fantasma en extra innings.

Todo en Python estándar.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from . import mathlib as M

EVENTS = ("K", "BB", "1B", "2B", "3B", "HR", "E", "OUT")
# puntos de estabilización (PA/BF) por evento — tabla 5.7.4 del framework
K_BAT = {"K": 60, "BB": 120, "HR": 170, "2B": 160, "3B": 160, "1B": 1100}
K_PIT = {"K": 70, "BB": 170, "HR": 1150, "2B": 2800, "3B": 2800, "1B": 2800}
TTO = {1: 0.96, 2: 1.00, 3: 1.07, 4: 1.10}  # penalización por veces en el orden (Tango/Lichtman)
STATES = [(b, o) for o in range(3) for b in range(8)]  # b = bits (1B=1, 2B=2, 3B=4)
BASE_NAMES = ["---", "1--", "-2-", "12-", "--3", "1-3", "-23", "123"]


# ============================================================ tasas por turno

def event_counts(st: dict, pitcher: bool = False) -> tuple[dict, float]:
    st = st or {}
    n = st.get("battersFaced" if pitcher else "plateAppearances", 0) or 0
    h = st.get("hits", 0)
    d, t, hr = st.get("doubles", 0), st.get("triples", 0), st.get("homeRuns", 0)
    c = {"K": st.get("strikeOuts", 0), "BB": st.get("baseOnBalls", 0) + st.get("hitByPitch", 0),
         "HR": hr, "3B": t, "2B": d, "1B": max(0, h - d - t - hr)}
    return c, float(n)


def league_rates(lg_tot: dict) -> dict:
    c, n = event_counts(lg_tot, pitcher=True)
    r = {e: c[e] / n for e in c}
    r["OUT"] = 1 - sum(r.values())
    return {e: r[e] for e in ("K", "BB", "1B", "2B", "3B", "HR", "OUT")}


def shrunk_rates(counts: dict, n: float, prior: dict, k: dict) -> dict:
    r = {e: M.shrink(counts[e] / n if n else prior[e], n, k[e], prior[e]) for e in k}
    r["OUT"] = max(0.30, 1 - sum(r.values()))
    s = sum(r.values())
    return {e: v / s for e, v in r.items()}


BASE_EVENTS = ("K", "BB", "1B", "2B", "3B", "HR", "OUT")


def combine(b: dict, p: dict, l: dict) -> dict:
    """Log5 multinomial: P(e) ∝ b_e · p_e / l_e."""
    w = {e: b[e] * p[e] / l[e] for e in BASE_EVENTS}
    s = sum(w.values())
    return {e: w[e] / s for e in BASE_EVENTS}


def scale(r: dict, mult: dict) -> dict:
    w = {e: r[e] * mult.get(e, 1.0) for e in BASE_EVENTS}
    s = sum(w.values())
    return {e: w[e] / s for e in BASE_EVENTS}


def with_residual(r: dict, e_rate: float) -> dict:
    """Agrega el evento residual E (errores, robos, wild pitches) restándolo de los outs."""
    out = dict(r)
    out["E"] = e_rate
    out["OUT"] = max(0.0, r["OUT"] - e_rate)
    return out


def calibrate_residual(lg_rates: dict, target: float, gb_share: float) -> float:
    """Tasa de E tal que la cadena reproduzca las carreras reales por media entrada de la liga (bisección)."""
    lo, hi = 0.0, 0.08
    for _ in range(40):
        mid = (lo + hi) / 2
        if re24(with_residual(lg_rates, mid), gb_share)["RE"][0] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ============================================================ transiciones base-out

def transitions(state, ev, gb_share=0.45):
    """Lista de (probabilidad, nuevas bases, outs añadidos, carreras) para un evento en un estado."""
    b, o = state
    r1, r2, r3 = b & 1, (b >> 1) & 1, (b >> 2) & 1
    two = o == 2
    out = []
    if ev == "K":
        return [(1.0, b, 1, 0)]
    if ev == "BB":
        runs = 1 if (r1 and r2 and r3) else 0
        n1, n2, n3 = 1, r2 or r1, r3 or (r1 and r2)
        return [(1.0, n1 | (n2 << 1) | (n3 << 2), 0, runs)]
    if ev == "HR":
        return [(1.0, 0, 0, 1 + r1 + r2 + r3)]
    if ev == "3B":
        return [(1.0, 4, 0, r1 + r2 + r3)]
    if ev == "2B":
        p1 = 0.60 if two else 0.40
        base_runs = r2 + r3
        if r1:
            return [(p1, 2, 0, base_runs + 1), (1 - p1, 2 | 4, 0, base_runs)]
        return [(1.0, 2, 0, base_runs)]
    if ev in ("1B", "E"):
        p2 = 0.85 if two else 0.60   # corredor de 2ª anota con sencillo
        p13 = 0.28                   # corredor de 1ª llega a 3ª
        runs3 = r3
        opts = []
        for s2, p_s2 in (((True, p2), (False, 1 - p2)) if r2 else ((False, 1.0),)):
            third_open = not (r2 and not s2)
            for to3, p_to3 in (((True, p13), (False, 1 - p13)) if (r1 and third_open) else ((False, 1.0),)):
                nb = 1
                if r1:
                    nb |= 4 if to3 else 2
                if r2 and not s2:
                    nb |= 4
                opts.append((p_s2 * p_to3, nb, 0, runs3 + (1 if (r2 and s2) else 0)))
        return opts
    # OUT en juego: rodado (posible doble play) o elevado (posible elevado de sacrificio)
    if two:
        return [(1.0, b, 1, 0)]
    gb, fb = gb_share, 1 - gb_share
    P_DP, P_3_GB, P_SF, P_2TO3_FB = 0.40, 0.50, 0.55, 0.25
    # --- rodado
    if r1:
        # doble play: fuera el corredor de 1ª y el bateador; con 0 outs anota el de 3ª y el de 2ª avanza
        if o == 0:
            out.append((gb * P_DP, (4 if r2 else 0), 2, r3))
        else:
            out.append((gb * P_DP, 0, 2, 0))  # termina la entrada: la carrera no cuenta
        # jugada de selección: fuera el corredor de 1ª, bateador en 1ª
        for sc, ps in (((True, P_3_GB), (False, 1 - P_3_GB)) if r3 else ((False, 1.0),)):
            nb = 1
            if r3 and not sc:
                nb |= 4
                if r2:
                    nb |= 2
            elif r2:
                nb |= 4
            out.append((gb * (1 - P_DP) * ps, nb, 1, 1 if sc else 0))
    else:
        for sc, ps in (((True, P_3_GB), (False, 1 - P_3_GB)) if r3 else ((False, 1.0),)):
            nb = 0
            if r3 and not sc:
                nb |= 4
                if r2:
                    nb |= 2
            elif r2:
                nb |= 4
            out.append((gb * ps, nb, 1, 1 if sc else 0))
    # --- elevado
    for sf, ps in (((True, P_SF), (False, 1 - P_SF)) if r3 else ((False, 1.0),)):
        third_free = not (r3 and not sf)
        for adv, pa in (((True, P_2TO3_FB), (False, 1 - P_2TO3_FB)) if (r2 and third_free) else ((False, 1.0),)):
            nb = b & 1
            if r3 and not sf:
                nb |= 4
            if r2:
                nb |= 4 if adv else 2
            out.append((fb * ps * pa, nb, 1, 1 if sf else 0))
    return out


def step_matrix(probs: dict, gb_share: float = 0.45):
    """Q (24×24, transiciones entre estados transitorios), absorción y carreras esperadas r por estado."""
    idx = {s: i for i, s in enumerate(STATES)}
    n = len(STATES)
    Q = [[0.0] * n for _ in range(n)]
    absorb = [0.0] * n
    r = [0.0] * n
    q0 = [[0.0] * n for _ in range(n)]   # transiciones sin carrera (para P(anotar))
    score_now = [0.0] * n
    for s in STATES:
        i = idx[s]
        for ev in EVENTS:
            pe = probs.get(ev, 0.0)
            for p, nb, dout, runs in transitions(s, ev, gb_share):
                w = pe * p
                no = s[1] + dout
                r[i] += w * runs
                if runs:
                    score_now[i] += w
                if no >= 3:
                    absorb[i] += w
                else:
                    j = idx[(nb, no)]
                    Q[i][j] += w
                    if not runs:
                        q0[i][j] += w
    return Q, absorb, r, q0, score_now


def solve(A, bvec):
    """Eliminación gaussiana con pivoteo parcial (A x = b)."""
    n = len(A)
    M_ = [row[:] + [bvec[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r_: abs(M_[r_][c]))
        M_[c], M_[piv] = M_[piv], M_[c]
        pv = M_[c][c]
        for r_ in range(c + 1, n):
            f = M_[r_][c] / pv
            if f:
                for k in range(c, n + 1):
                    M_[r_][k] -= f * M_[c][k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (M_[c][n] - sum(M_[c][k] * x[k] for k in range(c + 1, n))) / M_[c][c]
    return x


def inverse(A):
    n = len(A)
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def re24(probs: dict, gb_share: float = 0.45) -> dict:
    Q, absorb, r, q0, score_now = step_matrix(probs, gb_share)
    n = len(STATES)
    I_Q = [[(1.0 if i == j else 0.0) - Q[i][j] for j in range(n)] for i in range(n)]
    N = inverse(I_Q)
    RE = [sum(N[i][k] * r[k] for k in range(n)) for i in range(n)]
    I_Q0 = [[(1.0 if i == j else 0.0) - q0[i][j] for j in range(n)] for i in range(n)]
    P1 = solve(I_Q0, score_now)
    return {"Q": Q, "N": N, "RE": RE, "P1": P1, "r": r, "absorb": absorb,
            "table": [[RE[STATES.index((b, o))] for o in range(3)] for b in range(8)],
            "p1table": [[P1[STATES.index((b, o))] for o in range(3)] for b in range(8)],
            "expectedPA": sum(N[0])}


# ============================================================ cadena por lineup (216 estados)

def lineup_chain(pa_probs: list[dict], gb_share: float = 0.45, iters: int = 80) -> dict:
    """pa_probs[k] = probabilidades del bateador k (0..8) contra el pitcher.
    Devuelve R[k] = carreras esperadas de una entrada que abre el bateador k y T[k][j] = P(la siguiente la abre j)."""
    trans = {}
    for s in STATES:
        for k in range(9):
            lst = []
            for ev in EVENTS:
                pe = pa_probs[k].get(ev, 0.0)
                for p, nb, dout, runs in transitions(s, ev, gb_share):
                    lst.append((pe * p, nb, s[1] + dout, runs))
            trans[(s, k)] = lst
    V = {(s, k): 0.0 for s in STATES for k in range(9)}
    for _ in range(iters):
        V = {(s, k): sum(w * (runs + (V[((nb, no), (k + 1) % 9)] if no < 3 else 0.0))
                         for w, nb, no, runs in trans[(s, k)]) for s in STATES for k in range(9)}
    # próxima entrada: la abre el bateador siguiente al que hizo el 3er out
    T = [[0.0] * 9 for _ in range(9)]
    for lead in range(9):
        dist = defaultdict(float)
        dist[((0, 0), lead)] = 1.0
        for _ in range(60):
            nd = defaultdict(float)
            for (s, k), pm in dist.items():
                if pm < 1e-12:
                    continue
                for w, nb, no, runs in trans[(s, k)]:
                    if no >= 3:
                        T[lead][(k + 1) % 9] += pm * w
                    else:
                        nd[((nb, no), (k + 1) % 9)] += pm * w
            dist = nd
    R = [V[((0, 0), k)] for k in range(9)]
    return {"R": R, "T": T}


def innings_projection(R, T, innings: int = 9):
    pi = [1.0] + [0.0] * 8
    rows = []
    for i in range(innings):
        rows.append({"inning": i + 1, "runs": sum(pi[k] * R[k] for k in range(9)), "leadoff": pi[:]})
        pi = [sum(pi[k] * T[k][j] for k in range(9)) for j in range(9)]
    return rows


# ============================================================ simulación Monte Carlo del partido

class Team:
    def __init__(self, name, lineup, starter, bullpen, starter_bf):
        self.name = name
        self.lineup = lineup        # [{id, name, rates_vs: fn(pitcher, tto) -> cum}]
        self.starter = starter      # dict
        self.bullpen = bullpen      # list of dicts con role, pUse
        self.starter_bf = starter_bf  # (media, sd)


def cumulative(p: dict):
    acc = 0.0
    out = []
    for e in EVENTS:
        acc += p[e]
        out.append(acc)
    out[-1] = 1.0
    return out


def simulate(game, n=50000, seed=24, checkpoints=50):
    """game: {"away": TeamSpec, "home": TeamSpec, "prob": fn(batter, pitcher, tto)->cum, "gb": {pitcherId: gb}}"""
    rng = random.Random(seed)
    rand = rng.random
    prob = game["prob"]
    cache = {}

    def cum_for(b, p, tto):
        key = (b["id"], p["id"], min(tto, 4))
        c = cache.get(key)
        if c is None:
            c = prob(b, p, min(tto, 4))
            cache[key] = c
        return c

    stats = {
        "homeWin": 0, "runs": {"away": defaultdict(int), "home": defaultdict(int)}, "total": defaultdict(int),
        "margin": defaultdict(int), "f5": {"away": 0, "home": 0, "tie": 0}, "f5total": defaultdict(int),
        "f3": {"away": 0, "home": 0, "tie": 0}, "f3total": defaultdict(int), "nrfi": 0,
        "first": {"away": 0, "home": 0, "none": 0}, "extras": 0, "byInning": {"away": [0.0] * 9, "home": [0.0] * 9},
        "starterK": {"away": defaultdict(int), "home": defaultdict(int)},
        "starterOuts": {"away": defaultdict(int), "home": defaultdict(int)},
        "starterER": {"away": defaultdict(int), "home": defaultdict(int)},
        "relUse": {"away": defaultdict(int), "home": defaultdict(int)},
        "convergence": [],
    }
    step = max(1, n // checkpoints)

    for it in range(n):
        score = {"away": 0, "home": 0}
        inn_runs = {"away": [], "home": []}
        batter = {"away": 0, "home": 0}
        faced = {"away": defaultdict(int), "home": defaultdict(int)}   # veces que el abridor rival vio a cada bateador
        pit = {}
        bf = {}
        used = {"away": set(), "home": set()}
        sK = {"away": 0, "home": 0}
        sOuts = {"away": 0, "home": 0}
        sRuns = {"away": 0, "home": 0}
        rel_outs = {"away": 0, "home": 0}
        rel_cap = {"away": 3, "home": 3}
        limit = {}
        for side in ("away", "home"):
            t = game[side]
            mu, sd = t.starter_bf
            limit[side] = max(6, min(mu + 8, int(round(rng.gauss(mu, sd)))))
            pit[side] = t.starter
            bf[side] = 0
        first_scorer = None
        inning = 0
        f5_done = f3_done = False
        while True:
            inning += 1
            for half in ("away", "home"):   # half = equipo que batea
                field = "home" if half == "away" else "away"
                if inning >= 9 and half == "home" and score["home"] > score["away"]:
                    break
                t_field = game[field]
                # cambio de pitcher al iniciar la entrada: el relevista sale al completar su carga típica de outs
                cur = pit[field]
                if cur is not t_field.starter and rel_outs[field] >= rel_cap[field]:
                    pit[field] = choose_reliever(t_field, used[field], inning, score, field, rng)
                    rel_outs[field], rel_cap[field] = 0, stint_cap(pit[field], rng)
                outs = 0
                bases = 2 if inning >= 10 else 0   # corredor fantasma en 2ª
                runs_half = 0
                while outs < 3:
                    p_now = pit[field]
                    if (p_now is t_field.starter and (bf[field] >= limit[field] or sRuns[field] >= 7)) or \
                            (p_now is not t_field.starter and rel_outs[field] >= rel_cap[field] + 2):
                        pit[field] = choose_reliever(t_field, used[field], inning, score, field, rng)
                        rel_outs[field], rel_cap[field] = 0, stint_cap(pit[field], rng)
                        p_now = pit[field]
                    b = game[half].lineup[batter[half]]
                    tto = 1
                    if p_now is t_field.starter:
                        faced[half][batter[half]] += 1
                        tto = faced[half][batter[half]]
                        bf[field] += 1
                    c = cum_for(b, p_now, tto)
                    u = rand()
                    ev = 0
                    while u > c[ev]:
                        ev += 1
                    e = EVENTS[ev]
                    opts = transitions((bases, outs), e, p_now.get("gb", 0.45))
                    if len(opts) == 1:
                        _, nb, dout, runs = opts[0]
                    else:
                        v = rand()
                        acc = 0.0
                        for pp, nb, dout, runs in opts:
                            acc += pp
                            if v <= acc:
                                break
                    if e == "K" and p_now is t_field.starter:
                        sK[field] += 1
                    if p_now is t_field.starter:
                        sOuts[field] += min(dout, 3 - outs)
                        sRuns[field] += runs
                    else:
                        rel_outs[field] += min(dout, 3 - outs)
                    outs += dout
                    bases = nb if outs < 3 else 0
                    if runs:
                        runs_half += runs
                        score[half] += runs
                        if first_scorer is None:
                            first_scorer = half
                    batter[half] = (batter[half] + 1) % 9
                    if half == "home" and inning >= 9 and score["home"] > score["away"]:
                        break   # walk-off
                inn_runs[half].append(runs_half)
            if inning == 3 and not f3_done:
                f3_done = True
                a3, h3 = sum(inn_runs["away"][:3]), sum(inn_runs["home"][:3])
                stats["f3"]["home" if h3 > a3 else "away" if a3 > h3 else "tie"] += 1
                stats["f3total"][a3 + h3] += 1
            if inning == 5 and not f5_done:
                f5_done = True
                a5, h5 = sum(inn_runs["away"][:5]), sum(inn_runs["home"][:5])
                stats["f5"]["home" if h5 > a5 else "away" if a5 > h5 else "tie"] += 1
                stats["f5total"][a5 + h5] += 1
            if inning >= 9 and score["away"] != score["home"]:
                break
            if inning >= 20:
                break
        a, h = score["away"], score["home"]
        if inning > 9:
            stats["extras"] += 1
        stats["homeWin"] += h > a
        stats["runs"]["away"][a] += 1
        stats["runs"]["home"][h] += 1
        stats["total"][a + h] += 1
        stats["margin"][h - a] += 1
        if inn_runs["away"][0] == 0 and inn_runs["home"][0] == 0:
            stats["nrfi"] += 1
        stats["first"][first_scorer or "none"] += 1
        for side in ("away", "home"):
            for i in range(min(9, len(inn_runs[side]))):
                stats["byInning"][side][i] += inn_runs[side][i]
            fld = side
            stats["starterK"][fld][sK[fld]] += 1
            stats["starterOuts"][fld][sOuts[fld]] += 1
            stats["starterER"][fld][sRuns[fld]] += 1
            for rid in used[fld]:
                stats["relUse"][fld][rid] += 1
        if (it + 1) % step == 0:
            k = it + 1
            p = stats["homeWin"] / k
            stats["convergence"].append({"n": k, "p": p, "se": math.sqrt(p * (1 - p) / k)})
    stats["n"] = n
    return stats


def stint_cap(r: dict, rng) -> int:
    """Outs que suele cubrir el relevista por aparición (IP/G de la temporada), con variación."""
    mean_outs = max(2.0, min(9.0, 3 * (r.get("ipPerG") or 1.0)))
    return max(1, int(round(rng.gauss(mean_outs, 1.2))))


def choose_reliever(team: Team, used: set, inning: int, score: dict, field: str, rng) -> dict:
    """Selección de relevista por rol, situación y disponibilidad (cada brazo lanza una vez)."""
    lead = score[field] - score["home" if field == "away" else "away"]
    avail = [r for r in team.bullpen if r["id"] not in used and r.get("avail", 1.0) > 0]

    def pick(cands):
        cands = [c for c in cands if rng.random() < c.get("avail", 1.0)]
        return cands[0] if cands else None

    choice = None
    save_sit = 1 <= lead <= 3 or (lead == 0 and field == "home")
    if inning <= 5:
        # entradas tempranas (opener o abridor que salió pronto): brazos largos para cubrir volumen
        choice = pick(sorted([r for r in avail if r["role"] in ("Largo", "Intermedio")],
                             key=lambda r: -(r.get("ipPerG") or 0)))
    elif inning >= 9 and save_sit:
        choice = pick([r for r in avail if r["role"] == "Cerrador"] + [r for r in avail if r["role"] == "Setup"])
    if choice is None and inning == 8 and abs(lead) <= 3:
        choice = pick([r for r in avail if r["role"] == "Setup"] + [r for r in avail if r["role"] == "Cerrador" and lead <= 0 and False])
    if choice is None and 6 <= inning <= 8 and abs(lead) <= 2:
        choice = pick(sorted([r for r in avail if r["role"] in ("Setup", "Intermedio")], key=lambda r: r["fip"]))
    if choice is None:
        pool = [r for r in avail if r["role"] in ("Intermedio", "Largo", "Sin uso")] or avail
        if pool:
            weights = [max(0.02, r.get("pUse", 0.2)) * r.get("avail", 1.0) for r in pool]
            x = rng.random() * sum(weights)
            acc = 0.0
            for r, w in zip(pool, weights):
                acc += w
                if x <= acc:
                    choice = r
                    break
    if choice is None:
        choice = team.bullpen_rest
    else:
        used.add(choice["id"])
    return choice


def dist_view(d: dict, lo=0, hi=None):
    n = sum(d.values()) or 1
    hi = hi if hi is not None else (max(d) if d else 0)
    return [d.get(k, 0) / n for k in range(lo, hi + 1)]
