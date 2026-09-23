"""KRONOS: el partido como proceso estocástico, lanzamiento por lanzamiento.

Tercer modelo del Pro-Lab. Cada nivel del juego es un proceso estocástico distinto y cada uno se
estima con datos (Statcast congelado antes del partido) y se contrasta con la literatura:

1. Cadena de Markov de la cuenta (12 estados bolas-strikes + absorbentes K, BB, HBP y bola en juego).
   Probabilidades por lanzamiento de cada bateador y cada pitcher, regresadas a la liga por cuenta y
   combinadas con log5 multinomial por cuenta. Matriz fundamental N = (I − Q)⁻¹ → P(K), P(BB), lanzamientos
   esperados por turno (Powers & Yurko 2025; Albert, "Count Effects").
2. Cadena base-out de 24 estados (Lindsey 1963; Bukiet, Harold & Palacios 1997) + residuo de corrido
   (robos, wild pitches) calibrado a la liga 2026.
3. Deriva continua del pitcher con los bateadores enfrentados (Brill, Deshpande & Wyner 2023: la
   penalización por veces en el orden es continua, no un salto).
4. Salida del abridor como proceso de conteo con umbral aleatorio de lanzamientos que se acelera con las
   carreras permitidas (modelo de tiempo de falla acelerado estimado con las aperturas de la liga).
5. Fragilidad gamma del partido (estado oculto persistente): los datos 2026 no muestran "momentum" entre
   entradas (correlación lag-1 ≈ 0), pero sí un efecto común a todo el juego → mezcla persistente, no Hawkes.
6. Monte Carlo lanzamiento a lanzamiento con reglas oficiales 2026 (corredor en 2ª en extra innings 7.01(b),
   el local no batea la 9ª si va ganando).
7. Contraste: movimiento browniano de Stern (1994) con varianza dependiente del tiempo y volatilidad
   implícita de Polson & Stern (2015); distribución cuasigeométrica de Glass & Lowry (2008).
"""
from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict

from . import markov as MK

COUNTS = [(b, s) for s in range(3) for b in range(4)]          # 12 cuentas, orden: 0-0,1-0,2-0,3-0,0-1,...
CKEY = [f"{b}-{s}" for b, s in COUNTS]
CATS = ("B", "CS", "SS", "F", "FB", "HBP", "X")
BIP = ("OUT", "1B", "2B", "3B", "HR", "E")
EV_MAP = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR", "field_error": "E"}
K_COUNT = 25          # peso (lanzamientos) del prior en cada cuenta
K_FACTOR = 40         # peso del factor global del jugador por categoría
K_BIP_BAT = {"OUT": 150, "1B": 150, "2B": 150, "3B": 150, "HR": 200, "E": 400}
K_BIP_PIT = {"OUT": 500, "1B": 500, "2B": 500, "3B": 500, "HR": 400, "E": 800}


# ============================================================ tablas de la cuenta

def norm(d):
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def league_tables(pitch_league):
    """Tabla de la liga (P(categoría | cuenta)), bola en juego, TTO y largo del turno a partir de los días congelados."""
    T = {c: defaultdict(float) for c in CKEY}
    bip = defaultdict(float)
    tto = {k: defaultdict(float) for k in ("1", "2", "3")}
    palen = defaultdict(float)
    clock = defaultdict(float)
    n = 0
    for d in pitch_league.values():
        if "error" in d:
            continue
        n += d["n"]
        for c, dd in d["trans"].items():
            if c in T:
                for k, v in dd.items():
                    if k in CATS:
                        T[c][k] += v
        for e, v in d["inplay"].items():
            bip[EV_MAP.get(e, "OUT")] += v
        for k, dd in (d.get("tto") or {}).items():
            for kk, v in dd.items():
                tto[k][kk] += v
        for k, v in d["paLen"].items():
            palen[int(k)] += v
        for k, v in d["clock"].items():
            clock[k] += v
    return {"count": {c: norm({k: T[c].get(k, 0.0) + 1e-6 for k in CATS}) for c in CKEY}, "countN": {c: sum(T[c].values()) for c in CKEY},
            "bip": norm(dict(bip)), "tto": {k: dict(v) for k, v in tto.items()}, "paLen": dict(palen), "pitches": n,
            "clock": dict(clock)}


def player_table(pd_, L):
    """P(categoría | cuenta) de un jugador: factor global por categoría + regresión a la liga en cada cuenta."""
    trans = pd_.get("trans") or {}
    obs = defaultdict(float)
    exp = defaultdict(float)
    for c in CKEY:
        row = trans.get(c) or {}
        n = sum(v for k, v in row.items() if k in CATS)
        for k in CATS:
            obs[k] += row.get(k, 0)
            exp[k] += n * L["count"][c].get(k, 0)
    fac = {k: (obs[k] + K_FACTOR * _share(exp, k)) / (exp[k] + K_FACTOR * _share(exp, k)) if exp[k] > 0 else 1.0 for k in CATS}
    out = {}
    for c in CKEY:
        prior = norm({k: L["count"][c].get(k, 0) * fac[k] for k in CATS})
        row = trans.get(c) or {}
        n = sum(v for k, v in row.items() if k in CATS)
        out[c] = {k: (row.get(k, 0) + K_COUNT * prior[k]) / (n + K_COUNT) for k in CATS}
    return out, fac


def _share(exp, k):
    tot = sum(exp.values())
    return exp[k] / tot if tot else 1 / len(CATS)


def bip_table(pd_, L, kmap):
    raw = defaultdict(float)
    for e, v in (pd_.get("inplay") or {}).items():
        raw[EV_MAP.get(e, "OUT")] += v
    n = sum(raw.values())
    out = {}
    for k in BIP:
        kk = kmap[k]
        out[k] = (raw[k] + kk * L["bip"].get(k, 0)) / (n + kk)
    return norm(out), n


def combine(bat, pit, lg):
    """log5 multinomial: P(e) ∝ bateador_e · pitcher_e / liga_e (por cuenta o por bola en juego)."""
    return norm({k: bat[k] * pit.get(k, 0.0) / max(lg.get(k, 0.0), 1e-9) for k in bat})


def combine_counts(bat, pit, L):
    return {c: combine(bat[c], pit[c], L["count"][c]) for c in CKEY}


# ============================================================ cadena absorbente de la cuenta

def next_state(b, s, cat):
    """Siguiente cuenta o estado absorbente tras un lanzamiento."""
    if cat == "B":
        return ("BB",) if b == 3 else (b + 1, s)
    if cat in ("CS", "SS", "FB"):
        return ("K",) if s == 2 else (b, s + 1)
    if cat == "F":
        return (b, s) if s == 2 else (b, s + 1)
    if cat == "HBP":
        return ("HBP",)
    return ("X", b, s)


def absorb(P):
    """Q (12×12), R (12×15: K, BB, HBP, X en cada cuenta), N = (I − Q)⁻¹, B = N·R y lanzamientos esperados."""
    idx = {c: i for i, c in enumerate(COUNTS)}
    absn = ["K", "BB", "HBP"] + [f"X{b}-{s}" for b, s in COUNTS]
    aidx = {a: i for i, a in enumerate(absn)}
    Q = [[0.0] * 12 for _ in range(12)]
    R = [[0.0] * len(absn) for _ in range(12)]
    for (b, s), i in idx.items():
        row = P[f"{b}-{s}"]
        for cat in CATS:
            p = row.get(cat, 0.0)
            ns = next_state(b, s, cat)
            if len(ns) == 2:
                Q[i][idx[ns]] += p
            elif ns[0] == "X":
                R[i][aidx[f"X{b}-{s}"]] += p
            else:
                R[i][aidx[ns[0]]] += p
    IQ = [[(1.0 if i == j else 0.0) - Q[i][j] for j in range(12)] for i in range(12)]
    N = MK.inverse(IQ)
    B = [[sum(N[i][k] * R[k][j] for k in range(12)) for j in range(len(absn))] for i in range(12)]
    pitches = [sum(N[i]) for i in range(12)]
    b0 = B[0]
    return {"Q": Q, "N": N, "B": B, "abs": absn, "pitches": pitches,
            "K": b0[0], "BB": b0[1], "HBP": b0[2], "X": sum(b0[3:]), "Xby": {absn[j][1:]: b0[j] for j in range(3, len(absn))},
            "visits": N[0]}


def pa_length_dist(P, kmax=15):
    """Distribución exacta del número de lanzamientos del turno (propagando la cadena)."""
    dist = [0.0] * (kmax + 1)
    cur = {(0, 0): 1.0}
    for n in range(1, kmax + 1):
        nxt = defaultdict(float)
        for (b, s), p in cur.items():
            row = P[f"{b}-{s}"]
            for cat in CATS:
                q = row.get(cat, 0.0)
                if q <= 0:
                    continue
                ns = next_state(b, s, cat)
                if len(ns) == 2:
                    nxt[ns] += p * q
                else:
                    dist[n] += p * q
        cur = nxt
    dist[kmax] += sum(cur.values())
    return dist


# ============================================================ simulación lanzamiento a lanzamiento

class Pitcher:
    def __init__(self, pid, name, role, P, bip, gb, stint=None, avail=1.0, pUse=0.5, hand=None):
        self.id, self.name, self.role, self.P, self.bip, self.gb = pid, name, role, P, bip, gb
        self.stint, self.avail, self.pUse, self.hand = stint, avail, pUse, hand


def cum(d, keys):
    acc, out = 0.0, []
    for k in keys:
        acc += d.get(k, 0.0)
        out.append((acc, k))
    return out


def draw(cm, u):
    for acc, k in cm:
        if u <= acc:
            return k
    return cm[-1][1]


class Matchups:
    """Caché de probabilidades bateador × pitcher (por cuenta y bola en juego) con deriva por bateadores enfrentados."""

    def __init__(self, L, park, drift):
        self.L, self.park, self.drift = L, park, drift
        self.cache = {}

    def get(self, bat, pit, tbucket):
        key = (bat["id"], pit.id, tbucket)
        v = self.cache.get(key)
        if v is None:
            P = combine_counts(bat["P"], pit.P, self.L)
            bp = combine(bat["bip"], pit.bip, self.L["bip"])
            bp = norm({k: bp[k] * self.park.get(k, 1.0) for k in BIP})
            # deriva continua (Brill et al. 2023): más bolas y más contacto dañino conforme avanza el pitcher
            t = tbucket
            fB, fS, fH = 1 + self.drift["B"] * t, max(0.5, 1 - self.drift["SS"] * t), 1 + self.drift["H"] * t
            P = {c: norm({k: row[k] * (fB if k == "B" else fS if k == "SS" else 1.0) for k in CATS}) for c, row in P.items()}
            hits = {k: bp[k] * fH for k in ("1B", "2B", "3B", "HR")}
            bp = dict(bp, **hits)
            bp["OUT"] = max(0.05, 1 - sum(bp[k] for k in ("1B", "2B", "3B", "HR", "E")))
            bp = norm(bp)
            v = ({c: cum(P[c], CATS) for c in CKEY}, bp)
            self.cache[key] = v
        return v


def simulate(game, n=20000, seed=7, checkpoints=40):
    """game: {side: {"lineup": [bat], "starter": Pitcher, "pen": [Pitcher], "leash": (mu, sd)}, ...} + parámetros."""
    rng = random.Random(seed)
    L, M = game["L"], game["matchups"]
    beta_runs, q_adv, v_frail = game["leashBeta"], game["qAdv"], game["frailty"]
    sides = ("away", "home")
    runs_d = {s: defaultdict(int) for s in sides}
    tot_d, margin_d, f5_d, f5tot_d = defaultdict(int), defaultdict(int), {"away": 0, "home": 0, "tie": 0}, defaultdict(int)
    k_d = {s: defaultdict(int) for s in sides}
    pc_d = {s: defaultdict(int) for s in sides}
    outs_d = {s: defaultdict(int) for s in sides}
    rel_use = {s: defaultdict(int) for s in sides}
    wins = 0
    nrfi = 0
    extras = 0
    first_score = {"away": 0, "home": 0}
    pa_len_hist = defaultdict(int)
    conv = []
    shape = 1 / v_frail if v_frail > 1e-6 else None

    for it in range(n):
        frail = {s: (rng.gammavariate(shape, v_frail) if shape else 1.0) for s in sides}
        st = {}
        for s in sides:
            T = game[s]
            mu, sd = T["leash"]
            st[s] = {"pitcher": T["starter"], "starter": True, "bf": 0, "pitches": 0, "sp_pitches": 0, "sp_runs": 0,
                     "sp_k": 0, "sp_outs": 0, "leash": rng.gauss(mu, sd), "used": {T["starter"].id}, "stint_p": 0,
                     "stint_cap": None, "order": 0, "k_sp": 0}
        score = {"away": 0, "home": 0}
        inn_runs = []
        first = None
        inning = 1
        while True:
            line = [0, 0]
            for hi, bat_side in enumerate(sides):
                fld = "home" if bat_side == "away" else "away"
                if inning >= 9 and bat_side == "home" and score["home"] > score["away"]:
                    line[1] = None
                    break
                F = st[fld]
                # cambio de pitcher entre entradas si el relevista cumplió su tramo
                if not F["starter"] and F["stint_cap"] is not None and F["stint_p"] >= F["stint_cap"]:
                    _bring(game[fld], F, inning, score, fld, rng, rel_use[fld])
                bases, outs, runs = (2 if inning >= 10 else 0), 0, 0
                B = game[bat_side]
                while outs < 3:
                    pit = F["pitcher"]
                    bat = B["lineup"][st[bat_side]["order"] % 9]
                    st[bat_side]["order"] += 1
                    tb = min(4, F["bf"] // 5) * 5 / 9 if F["starter"] else 0
                    Pc, bp = M.get(bat, pit, round(tb, 3))
                    b = s_ = 0
                    np_ = 0
                    res = None
                    while res is None:
                        cat = draw(Pc[f"{b}-{s_}"], rng.random())
                        np_ += 1
                        # residuo de corrido: robo / wild pitch con corredores en base
                        if bases and cat == "B" and rng.random() < q_adv:
                            r3 = (bases >> 2) & 1
                            runs += r3
                            bases = ((bases << 1) & 7)
                        ns = next_state(b, s_, cat)
                        if len(ns) == 2:
                            b, s_ = ns
                        elif ns[0] == "X":
                            ev = draw_bip(bp, frail[bat_side], rng)
                            res = ev
                        else:
                            res = {"K": "K", "BB": "BB", "HBP": "BB"}[ns[0]]
                    F["pitches"] += np_
                    F["stint_p"] += np_
                    F["bf"] += 1
                    pa_len_hist[min(np_, 12)] += 1
                    if res == "K" and F["starter"]:
                        F["k_sp"] += 1
                    tr = MK.transitions((bases, outs), res, pit.gb)
                    u = rng.random()
                    acc = 0.0
                    for p, nb, do, r in tr:
                        acc += p
                        if u <= acc:
                            break
                    # carrera de oro: el local gana al anotar en la última entrada
                    before = runs
                    bases, outs, runs = nb, outs + do, runs + r
                    if F["starter"]:
                        F["sp_runs"] += runs - before
                        F["sp_outs"] += do
                    if inning >= 9 and bat_side == "home" and score["home"] + runs > score["away"]:
                        break
                    # ¿sale el pitcher? (proceso de conteo con umbral acelerado por carreras)
                    if outs < 3:
                        if F["starter"]:
                            if F["pitches"] >= F["leash"] - beta_runs * F["sp_runs"]:
                                _pull_starter(game[fld], F, inning, score, fld, rng, rel_use[fld])
                        elif F["stint_cap"] is not None and F["stint_p"] >= F["stint_cap"] + 8:
                            _bring(game[fld], F, inning, score, fld, rng, rel_use[fld])
                if first is None and runs:
                    first = bat_side
                score[bat_side] += runs
                line[hi] = runs
                if F["starter"] and (F["pitches"] >= F["leash"] - beta_runs * F["sp_runs"] - 6):
                    _pull_starter(game[fld], F, inning + 1, score, fld, rng, rel_use[fld])
            inn_runs.append(line)
            if inning == 1 and line[0] == 0 and (line[1] or 0) == 0:
                nrfi += 1
            if inning >= 9 and score["away"] != score["home"]:
                break
            inning += 1
            if inning > 20:
                break
        a, h = score["away"], score["home"]
        extras += inning > 9
        runs_d["away"][a] += 1
        runs_d["home"][h] += 1
        tot_d[a + h] += 1
        margin_d[h - a] += 1
        wins += h > a
        f5a = sum(x[0] for x in inn_runs[:5])
        f5h = sum((x[1] or 0) for x in inn_runs[:5])
        f5_d["home" if f5h > f5a else "away" if f5a > f5h else "tie"] += 1
        f5tot_d[f5a + f5h] += 1
        if first:
            first_score[first] += 1
        # ponches y lanzamientos del abridor de cada lado (lado = equipo del pitcher)
        for s in sides:
            k_d[s][st[s]["k_sp"]] += 1
            pc_d[s][min(st[s]["sp_pitches"] or st[s]["pitches"], 140) // 5 * 5] += 1
            outs_d[s][st[s]["sp_outs"]] += 1
        if (it + 1) % max(1, n // checkpoints) == 0:
            p = wins / (it + 1)
            conv.append({"n": it + 1, "p": p, "se": math.sqrt(max(p * (1 - p), 1e-9) / (it + 1))})

    def dist(d, hi):
        return [d.get(k, 0) / n for k in range(hi + 1)]

    return {"n": n, "pHome": wins / n, "se": math.sqrt(wins / n * (1 - wins / n) / n),
            "runs": {s: dist(runs_d[s], 20) for s in sides}, "total": dist(tot_d, 30),
            "margin": {k: v / n for k, v in sorted(margin_d.items())},
            "f5": {k: v / n for k, v in f5_d.items()}, "f5total": dist(f5tot_d, 20), "nrfi": nrfi / n,
            "first": {k: v / n for k, v in first_score.items()}, "extras": extras / n,
            "k": {s: dist(k_d[s], 15) for s in sides}, "spPitches": {s: {k: v / n for k, v in sorted(pc_d[s].items())} for s in sides},
            "spOuts": {s: dist(outs_d[s], 27) for s in sides},
            "relievers": {s: {k: v / n for k, v in sorted(rel_use[s].items(), key=lambda x: -x[1])} for s in sides},
            "paLen": {k: v / sum(pa_len_hist.values()) for k, v in sorted(pa_len_hist.items())}, "convergence": conv,
            "meanRuns": {s: sum(k * v for k, v in runs_d[s].items()) / n for s in sides}}


def draw_bip(bp, g, rng):
    """Bola en juego con la fragilidad del partido multiplicando la probabilidad de hit."""
    if g != 1.0:
        hit = {k: bp[k] * g for k in ("1B", "2B", "3B", "HR")}
        e = bp["E"]
        out = max(0.05, 1 - sum(hit.values()) - e)
        u = rng.random() * (out + e + sum(hit.values()))
        acc = out
        if u <= acc:
            return "OUT"
        for k in ("1B", "2B", "3B", "HR"):
            acc += hit[k]
            if u <= acc:
                return k
        return "E"
    u = rng.random()
    acc = 0.0
    for k in BIP:
        acc += bp[k]
        if u <= acc:
            return k
    return "OUT"


def _pull_starter(T, F, inning, score, side, rng, use):
    F["sp_pitches"] = F["pitches"]
    F["starter"] = False
    _bring(T, F, inning, score, side, rng, use)


def _bring(T, F, inning, score, side, rng, use):
    """Elige relevista por rol, situación y disponibilidad (cerrador en salvamento, setup en la 8ª, largos temprano)."""
    lead = score[side] - score["home" if side == "away" else "away"]
    pool = [p for p in T["pen"] if p.id not in F["used"] and rng.random() < p.avail]
    if not pool:
        pool = [p for p in T["pen"] if p.id not in F["used"]] or T["pen"]
    def w(p):
        base = max(p.pUse, 0.02)
        if p.role == "Cerrador":
            return base * (6 if inning >= 9 and 0 < lead <= 3 else 3 if inning >= 9 and lead == 0 else 0.15)
        if p.role == "Setup":
            return base * (4 if inning in (7, 8) and -1 <= lead <= 3 else 1.2 if inning >= 9 else 0.5)
        if p.role == "Largo":
            return base * (4 if inning <= 5 else 0.6)
        return base * (1.5 if 5 <= inning <= 7 else 1.0)
    ws = [w(p) for p in pool]
    tot = sum(ws)
    u = rng.random() * tot
    acc = 0.0
    pick = pool[-1]
    for p, x in zip(pool, ws):
        acc += x
        if u <= acc:
            pick = p
            break
    F["pitcher"] = pick
    F["used"].add(pick.id)
    F["stint_p"] = 0
    F["bf"] = 0
    mu, sd = pick.stint or (18, 6)
    F["stint_cap"] = max(8, rng.gauss(mu, sd))
    use[pick.name] += 1


# ============================================================ calibración y análisis de la liga

def league_matchup(L):
    return {"id": "LG", "P": L["count"], "bip": L["bip"]}


def calibrate_adv(L, gb, target, drift, n_half=12000, seed=3):
    """Busca la probabilidad de avance extra por lanzamiento (robos, WP, PB) que reproduce las carreras por media entrada."""
    lgp = Pitcher("LGP", "Liga", "SP", L["count"], L["bip"], gb)
    lo, hi = 0.0, 0.08
    res = None
    for _ in range(9):
        q = (lo + hi) / 2
        m = half_inning_mean(L, lgp, q, n_half, seed)
        res = (q, m)
        if m < target:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2, res[1]


def half_inning_mean(L, lgp, q, n_half, seed):
    rng = random.Random(seed)
    P = {c: cum(L["count"][c], CATS) for c in CKEY}
    tot = 0
    dist = defaultdict(int)
    for _ in range(n_half):
        bases, outs, runs = 0, 0, 0
        while outs < 3:
            b = s = 0
            res = None
            while res is None:
                cat = draw(P[f"{b}-{s}"], rng.random())
                if bases and cat == "B" and rng.random() < q:
                    runs += (bases >> 2) & 1
                    bases = (bases << 1) & 7
                ns = next_state(b, s, cat)
                if len(ns) == 2:
                    b, s = ns
                elif ns[0] == "X":
                    res = draw_bip(L["bip"], 1.0, rng)
                else:
                    res = "K" if ns[0] == "K" else "BB"
            tr = MK.transitions((bases, outs), res, lgp.gb)
            u = rng.random()
            acc = 0.0
            for p, nb, do, r in tr:
                acc += p
                if u <= acc:
                    break
            bases, outs, runs = nb, outs + do, runs + r
        tot += runs
        dist[min(runs, 8)] += 1
    half_inning_mean.last = {k: v / n_half for k, v in sorted(dist.items())}
    return tot / n_half


def tto_drift(L):
    """Deriva por bateador enfrentado a partir de TTO1 → TTO3 de la liga (efecto continuo, Brill et al. 2023)."""
    t1, t3 = L["tto"].get("1") or {}, L["tto"].get("3") or {}
    def rate(t, k):
        n = sum(v for kk, v in t.items() if kk in CATS)
        return t.get(k, 0) / n if n else 0
    def hit(t):
        x = t.get("X", 0)
        return (t.get("X_H", 0) + t.get("X_HR", 0)) / x if x else 0
    span = 2.0   # de la 1ª a la 3ª vez: ~18 bateadores = 2 unidades de 9
    dB = (rate(t3, "B") / rate(t1, "B") - 1) / span if rate(t1, "B") else 0
    dS = (1 - rate(t3, "SS") / rate(t1, "SS")) / span if rate(t1, "SS") else 0
    dH = (hit(t3) / hit(t1) - 1) / span if hit(t1) else 0
    return {"B": dB, "SS": dS, "H": dH, "raw": {"t1": {"B": rate(t1, "B"), "SS": rate(t1, "SS"), "hit": hit(t1)},
                                                    "t3": {"B": rate(t3, "B"), "SS": rate(t3, "SS"), "hit": hit(t3)}}}


def inning_dependence(results):
    """Prueba de 'momentum': correlación de carreras de un equipo entre entradas del mismo juego, con y sin calidad del equipo."""
    seqs = []
    off, dfn = defaultdict(list), defaultdict(list)
    for g in results:
        inn = g.get("inn") or []
        if len(inn) < 9:
            continue
        a = [x[0] for x in inn[:8]]
        h = [x[1] for x in inn[:8]]
        if None in a or None in h:
            continue
        seqs.append((g["away"], g["home"], a))
        seqs.append((g["home"], g["away"], h))
        for y in a:
            off[g["away"]].append(y)
            dfn[g["home"]].append(y)
        for y in h:
            off[g["home"]].append(y)
            dfn[g["away"]].append(y)
    lg = statistics.fmean([y for v in off.values() for y in v])
    om = {t: statistics.fmean(v) for t, v in off.items()}
    dm = {t: statistics.fmean(v) for t, v in dfn.items()}

    def corr(qs, lag):
        xs = [q[i] for q in qs for i in range(len(q) - lag)]
        ys = [q[i + lag] for q in qs for i in range(len(q) - lag)]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return num / den
    raw = [q for _, _, q in seqs]
    res = [[y - (om[o] + dm[d] - lg) for y in q] for o, d, q in seqs]
    rc = [corr(raw, l) for l in range(1, 7)]
    rr = [corr(res, l) for l in range(1, 7)]
    vi = statistics.pvariance([y for q in res for y in q])
    cov = statistics.fmean(rr[1:]) * vi
    tot = [sum(q) for q in raw]
    per = sum(statistics.pvariance([q[i] for q in raw]) for i in range(8))
    return {"seqs": len(seqs), "lagRaw": rc, "lagResid": rr, "varInning": vi, "cov": cov, "frailtyRuns": cov / (lg * lg),
            "var8": statistics.pvariance(tot), "var8Indep": per, "mean8": statistics.fmean(tot), "lgInning": lg}


def hmm_fit(results, K=2, iters=40, seed=5):
    """HMM de K estados sobre las carreras por entrada (0,1,2,3+) de cada equipo-partido: ¿persisten los estados?"""
    rng = random.Random(seed)
    seqs = []
    for g in results:
        inn = g.get("inn") or []
        if len(inn) < 9:
            continue
        for k in (0, 1):
            q = [x[k] for x in inn[:8]]
            if None not in q:
                seqs.append([min(y, 3) for y in q])
    M_ = 4
    pi = [1 / K] * K
    A = [[0.8 if i == j else 0.2 / (K - 1) for j in range(K)] for i in range(K)]
    base = [0.72, 0.15, 0.07, 0.06]
    E = [[max(1e-3, base[m] * (1 + (0.4 if i == 0 else -0.3) * (1 if m else -0.5)) + rng.random() * 1e-3) for m in range(M_)] for i in range(K)]
    E = [[v / sum(r) for v in r] for r in E]
    ll = None
    for _ in range(iters):
        pi_n = [0.0] * K
        A_n = [[0.0] * K for _ in range(K)]
        E_n = [[0.0] * M_ for _ in range(K)]
        ll = 0.0
        for q in seqs:
            T = len(q)
            al = [[0.0] * K for _ in range(T)]
            c = [0.0] * T
            for i in range(K):
                al[0][i] = pi[i] * E[i][q[0]]
            c[0] = sum(al[0])
            al[0] = [v / c[0] for v in al[0]]
            for t in range(1, T):
                for j in range(K):
                    al[t][j] = sum(al[t - 1][i] * A[i][j] for i in range(K)) * E[j][q[t]]
                c[t] = sum(al[t])
                al[t] = [v / c[t] for v in al[t]]
            be = [[1.0] * K for _ in range(T)]
            for t in range(T - 2, -1, -1):
                for i in range(K):
                    be[t][i] = sum(A[i][j] * E[j][q[t + 1]] * be[t + 1][j] for j in range(K)) / c[t + 1]
            ll += sum(math.log(x) for x in c)
            for t in range(T):
                gam = [al[t][i] * be[t][i] for i in range(K)]
                sg = sum(gam)
                for i in range(K):
                    E_n[i][q[t]] += gam[i] / sg
                    if t == 0:
                        pi_n[i] += gam[i] / sg
                if t < T - 1:
                    for i in range(K):
                        for j in range(K):
                            A_n[i][j] += al[t][i] * A[i][j] * E[j][q[t + 1]] * be[t + 1][j] / c[t + 1]
        pi = [v / sum(pi_n) for v in pi_n]
        A = [[v / sum(r) for v in r] for r in A_n]
        E = [[max(v, 1e-9) / sum(r) for v in r] for r in E_n]
    # comparación: independiente (1 estado)
    cnt = [0] * M_
    for q in seqs:
        for y in q:
            cnt[y] += 1
    p1 = [v / sum(cnt) for v in cnt]
    ll1 = sum(cnt[m] * math.log(p1[m]) for m in range(M_))
    nobs = sum(len(q) for q in seqs)
    kpar = K - 1 + K * (K - 1) + K * (M_ - 1)
    order = sorted(range(K), key=lambda i: E[i][0], reverse=True)
    return {"K": K, "pi": [pi[i] for i in order], "A": [[A[i][j] for j in order] for i in order],
            "E": [E[i] for i in order], "means": [sum(m * E[i][m] for m in range(M_)) for i in order],
            "ll": ll, "llIndep": ll1, "bic": -2 * ll + kpar * math.log(nobs), "bicIndep": -2 * ll1 + (M_ - 1) * math.log(nobs),
            "nobs": nobs, "seqs": len(seqs)}


def brownian(results):
    """Stern (1994) con varianza dependiente del tiempo; calibración contra la frecuencia real de victorias."""
    paths = []
    for g in results:
        inn = g.get("inn") or []
        if len(inn) < 9:
            continue
        d, p = 0, []
        for i in range(9):
            a, h = inn[i]
            d += (h or 0) - (a or 0)
            p.append(d)
        paths.append((p, g["hr"] - g["ar"]))
    var_t = [statistics.pvariance([p[t] for p, _ in paths]) for t in range(9)]
    mean_t = [statistics.fmean([p[t] for p, _ in paths]) for t in range(9)]
    mu = mean_t[7] * 9 / 8          # deriva hasta la 8ª completa extrapolada (la 9ª está truncada)
    s2 = var_t[7] * 9 / 8
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def stern(l, k, tv=True):
        rem_var = (s2 - var_t[k - 1]) if tv else s2 * (1 - k / 9)
        rem_mu = mu * (1 - k / 9)
        return Phi((l + rem_mu) / math.sqrt(max(rem_var, 1e-6)))
    calib = []
    for k in (3, 5, 7, 8):
        for l in range(-3, 4):
            rows = [f for p, f in paths if p[k - 1] == l]
            if len(rows) < 40:
                continue
            calib.append({"inning": k, "lead": l, "n": len(rows), "real": sum(1 for f in rows if f > 0) / len(rows),
                          "stern": stern(l, k, tv=False), "sternTv": stern(l, k, tv=True)})
    return {"games": len(paths), "mu": mu, "sigma": math.sqrt(s2), "varT": var_t, "meanT": mean_t, "calib": calib}


def quasigeometric(results):
    """Glass & Lowry (2008): P(0) = a y P(n) = (1−a)(1−d)d^(n−1); d constante entre equipos."""
    cnt = defaultdict(int)
    for g in results:
        for x in (g.get("inn") or [])[:9]:
            for v in x:
                if v is not None:
                    cnt[v] += 1
    n = sum(cnt.values())
    p0 = cnt[0] / n
    pos = sum(k * v for k, v in cnt.items() if k > 0)
    npos = n - cnt[0]
    mean_pos = pos / npos
    d = 1 - 1 / mean_pos
    fit = [{"runs": k, "real": cnt.get(k, 0) / n, "qg": p0 if k == 0 else (1 - p0) * (1 - d) * d ** (k - 1)} for k in range(7)]
    return {"a": p0, "d": d, "fit": fit, "n": n, "mean": sum(k * v for k, v in cnt.items()) / n}


def implied_sigma(p_home, mu):
    """Polson & Stern (2015): con P(local) del mercado y la ventaja esperada μ, σ implícita = μ / Φ⁻¹(p)."""
    if not (0.02 < p_home < 0.98):
        return None
    z = _inv_phi(p_home)
    return mu / z if abs(z) > 1e-6 else None


def _inv_phi(p):
    lo, hi = -8.0, 8.0
    for _ in range(80):
        m = (lo + hi) / 2
        if 0.5 * (1 + math.erf(m / math.sqrt(2))) < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def starter_leash(boxscores):
    """Tiempo de falla acelerado: lanzamientos al salir ~ a − β·carreras (aperturas de la liga de los últimos días)."""
    xs, ys = [], []
    for b in boxscores:
        for s in ("away", "home"):
            ps = b[s]["pitchers"]
            if ps and ps[0].get("pitches"):
                xs.append(ps[0].get("r") or 0)
                ys.append(ps[0]["pitches"])
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    beta = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my + beta * mx
    resid = [y - (a - beta * x) for x, y in zip(xs, ys)]
    return {"starts": len(xs), "meanPitches": my, "meanRuns": mx, "beta": beta, "sdResid": statistics.pstdev(resid),
            "intercept": a}
