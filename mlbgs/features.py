"""Agregados de liga, equipos, abridores y bullpen a partir del bundle de datos.

Aquí vive la FASE 1 del algoritmo maestro (regresión a la media / shrinkage) y los
insumos que después usa el modelo de carreras: tasas por entrada de la liga,
dispersión de carreras, park factors, Elo, Pitágoras y uso reciente del bullpen.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import defaultdict

from . import mathlib as M

# Puntos de estabilización (Russell Carleton, tabla 5.7.4 del framework).
K_PITCHER = {"k": 70, "bb": 170, "hr": 1150, "gb": 70}  # HR/FB 400 FB ≈ 1150 BF con ~35% de FB
K_ERA_BF = 1000          # ERA cruda: muy ruidosa, se regresa fuerte
K_XERA_PA = 250          # xERA (basada en xwOBA) estabiliza antes que el ERA
K_TEAM_WPCT = 70         # juegos de .500 que se agregan al win% (Tango)
K_TEAM_RUNS = 45         # juegos para regresar carreras/juego del equipo
K_SPLIT_PA = 700         # PA para regresar splits vs mano
K_BULLPEN_IP = 150       # entradas para regresar el bullpen
K_START_IP = 6           # aperturas para regresar entradas por apertura
PREV_WEIGHT = 0.5        # la temporada anterior pesa la mitad


def f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def ip(stat: dict | None) -> float:
    return M.ip_to_float((stat or {}).get("inningsPitched"))


def add_stats(a: dict | None, b: dict | None, wb: float = 1.0) -> dict:
    """Suma dos bloques de stats (b ponderado), convirtiendo IP a decimal."""
    out: dict[str, float] = {}
    for src, w in ((a or {}, 1.0), (b or {}, wb)):
        for k, v in src.items():
            if k == "inningsPitched":
                out["ip"] = out.get("ip", 0.0) + M.ip_to_float(v) * w
            elif isinstance(v, (int, float)):
                out[k] = out.get(k, 0.0) + v * w
    return out


# ============================================================ liga

class League:
    def __init__(self, bundle: dict):
        self.season = bundle["meta"]["season"]
        results = bundle.get("results", [])
        self.n_games = len(results)
        ts = bundle.get("teamStats", {})

        # --- totales de pitcheo de la liga (suma de los 30 equipos)
        tot = {}
        sp = {}
        rp = {}
        for t in ts.values():
            tot = add_stats(tot, t.get("pitching"))
            sp = add_stats(sp, t.get("sp"))
            rp = add_stats(rp, t.get("rp"))
        self.tot, self.sp, self.rp = tot, sp, rp
        lg_ip = tot.get("ip", 1.0) or 1.0
        self.era = 9 * tot.get("earnedRuns", 0) / lg_ip
        self.ra9 = 9 * tot.get("runs", 0) / lg_ip
        self.c_fip = M.fip_constant(self.era, tot.get("homeRuns", 0), tot.get("baseOnBalls", 0),
                                    tot.get("hitByPitch", 0), tot.get("strikeOuts", 0), lg_ip)
        bf = tot.get("battersFaced", 1.0) or 1.0
        self.bf_per_ip = bf / lg_ip
        self.k_rate = tot.get("strikeOuts", 0) / bf
        self.bb_rate = tot.get("baseOnBalls", 0) / bf
        self.hbp_rate = tot.get("hitByPitch", 0) / bf
        self.hr_rate = tot.get("homeRuns", 0) / bf
        self.whip = (tot.get("hits", 0) + tot.get("baseOnBalls", 0)) / lg_ip
        go, ao = tot.get("groundOuts", 0), tot.get("airOuts", 0)
        self.gb_rate = go / (go + ao) if go + ao else 0.44
        self.sp_ip_per_gs = sp.get("ip", 0) / sp.get("gamesStarted", 1) if sp.get("gamesStarted") else 5.2
        self.sp_ra9 = 9 * sp.get("runs", 0) / sp.get("ip", 1) if sp.get("ip") else self.ra9
        self.rp_ra9 = 9 * rp.get("runs", 0) / rp.get("ip", 1) if rp.get("ip") else self.ra9
        self.rp_era = 9 * rp.get("earnedRuns", 0) / rp.get("ip", 1) if rp.get("ip") else self.era
        self.ra_per_era = self.ra9 / self.era if self.era else 1.08

        # --- bateo de liga (total y vs mano)
        hit = {}
        vl = {}
        vr = {}
        for t in ts.values():
            hit = add_stats(hit, t.get("hitting"))
            vl = add_stats(vl, t.get("hit_vl"))
            vr = add_stats(vr, t.get("hit_vr"))
        self.hit = hit
        self.obp, self.slg = obp_slg(hit)
        self.obpslg = self.obp * self.slg
        self.obpslg_vs = {"L": prod(obp_slg(vl)) or self.obpslg, "R": prod(obp_slg(vr)) or self.obpslg}
        pa = hit.get("plateAppearances", 1.0) or 1.0
        self.bat_k_rate = hit.get("strikeOuts", 0) / pa
        self.bat_bb_rate = hit.get("baseOnBalls", 0) / pa
        self.iso = self.slg - (hit.get("hits", 0) / hit.get("atBats", 1) if hit.get("atBats") else 0.245)

        # --- carreras por entrada, por lado (FASE 3)
        self._innings(results)
        self._home_edge(results)
        self._dispersion(results)

    # carreras esperadas por partido en la entrada i para cada lado (incluye los 9nos no jugados como 0)
    def _innings(self, results):
        sums = {"away": [0.0] * 10, "home": [0.0] * 10}
        extra = {"away": 0.0, "home": 0.0}
        scored = {"away": [0] * 10, "home": [0] * 10}
        played = {"away": [0] * 10, "home": [0] * 10}
        scoreless_first = played_first = 0
        n = 0
        for g in results:
            inn = g.get("inn")
            if not inn:
                continue
            n += 1
            for i, (a, h) in enumerate(inn):
                idx = i + 1
                for side, v in (("away", a), ("home", h)):
                    if v is None:
                        continue
                    if idx <= 9:
                        sums[side][idx] += v
                        played[side][idx] += 1
                        scored[side][idx] += v > 0
                    else:
                        extra[side] += v
                if idx == 1:
                    for v in (a, h):
                        if v is not None:
                            played_first += 1
                            scoreless_first += v == 0
        n = max(n, 1)
        self.inning_runs = {s: [x / n for x in sums[s]] for s in sums}  # índice 1..9
        self.extra_runs = {s: extra[s] / n for s in extra}
        self.rpg = {s: sum(self.inning_runs[s]) + self.extra_runs[s] for s in sums}
        self.p_scoreless_half1 = scoreless_first / played_first if played_first else 0.72
        # P(anotar al menos una carrera en la media entrada i | se jugó), por lado: calibra NRFI y "quién anota primero"
        self.p_score_half = {s: [scored[s][i] / played[s][i] if played[s][i] else 0.28 for i in range(10)] for s in sums}
        # carreras promedio en la media entrada i cuando se jugó (para escalar por equipo)
        self.runs_half_played = {s: [sums[s][i] / played[s][i] if played[s][i] else 0.5 for i in range(10)] for s in sums}
        self.lam1 = (self.inning_runs["away"][1] + self.inning_runs["home"][1]) / 2

    def _home_edge(self, results):
        hw = sum(1 for g in results if g.get("hr", 0) > g.get("ar", 0))
        self.home_win = hw / len(results) if results else 0.53
        ex = [g for g in results if g.get("inn") and len(g["inn"]) > 9]
        self.extra_share = len(ex) / len(results) if results else 0.09
        self.extra_home_win = (sum(1 for g in ex if g["hr"] > g["ar"]) / len(ex)) if ex else 0.52
        self.home_odds_ratio = self.home_win / (1 - self.home_win) if 0 < self.home_win < 1 else 1.12

    def _dispersion(self, results):
        """Var/Media de carreras por juego dentro de cada equipo (sobredispersión, 5.7.5)."""
        by_team = defaultdict(list)
        for g in results:
            by_team[g["away"]].append(g["ar"])
            by_team[g["home"]].append(g["hr"])
        ratios = []
        for runs in by_team.values():
            if len(runs) >= 20:
                m = statistics.fmean(runs)
                if m > 0:
                    ratios.append(statistics.pvariance(runs) / m)
        self.var_ratio = statistics.fmean(ratios) if ratios else 2.0


def obp_slg(h: dict) -> tuple[float, float]:
    ab = h.get("atBats", 0)
    pa_obp = ab + h.get("baseOnBalls", 0) + h.get("hitByPitch", 0) + h.get("sacFlies", 0)
    if not ab or not pa_obp:
        return 0.0, 0.0
    obp = (h.get("hits", 0) + h.get("baseOnBalls", 0) + h.get("hitByPitch", 0)) / pa_obp
    slg = h.get("totalBases", 0) / ab
    return obp, slg


def prod(t):
    return t[0] * t[1]


# ============================================================ equipos: Pitágoras, Elo, historial

class Teams:
    def __init__(self, bundle: dict, lg: League, parks: dict):
        self.lg = lg
        self.info = {int(k): v for k, v in bundle.get("teams", {}).items()}
        self.standings = {int(k): v for k, v in bundle.get("standings", {}).items()}
        self.stats = {int(k): v for k, v in bundle.get("teamStats", {}).items()}
        self.results = sorted(bundle.get("results", []), key=lambda g: (g["date"], g["pk"]))
        self.prev = sorted(bundle.get("prevResults", []), key=lambda g: (g["date"], g["pk"]))
        self.parks = parks
        self._season_totals()
        self._elo()

    def abbr(self, tid):
        return (self.info.get(tid) or {}).get("abbr") or str(tid)

    def _season_totals(self):
        agg = defaultdict(lambda: {"g": 0, "rs": 0, "ra": 0, "w": 0, "l": 0, "f3rs": 0, "f3ra": 0, "f5rs": 0,
                                   "f5ra": 0, "hg": 0, "ag": 0})
        for g in self.results:
            for side, opp in (("away", "home"), ("home", "away")):
                t = agg[g[side]]
                rs, ra = g["ar" if side == "away" else "hr"], g["hr" if side == "away" else "ar"]
                t["g"] += 1
                t["rs"] += rs
                t["ra"] += ra
                t["w" if rs > ra else "l"] += 1
                t["hg" if side == "home" else "ag"] += 1
                inn = g.get("inn") or []
                si, oi = (0, 1) if side == "away" else (1, 0)
                t["f3rs"] += sum((x[si] or 0) for x in inn[:3])
                t["f3ra"] += sum((x[oi] or 0) for x in inn[:3])
                t["f5rs"] += sum((x[si] or 0) for x in inn[:5])
                t["f5ra"] += sum((x[oi] or 0) for x in inn[:5])
        self.agg = agg

        prev = defaultdict(lambda: {"g": 0, "rs": 0, "ra": 0})
        for g in self.prev:
            for side in ("away", "home"):
                t = prev[g[side]]
                t["g"] += 1
                t["rs"] += g["ar" if side == "away" else "hr"]
                t["ra"] += g["hr" if side == "away" else "ar"]
        self.prev_agg = prev

    def pythag(self, tid) -> dict:
        a = self.agg[tid]
        n = M.pythagenpat_exponent(a["rs"], a["ra"], a["g"])
        p = M.pythagorean(a["rs"], a["ra"], n)
        pv = self.prev_agg.get(tid)
        prior = 0.5
        if pv and pv["g"]:
            pp = M.pythagorean(pv["rs"], pv["ra"], M.pythagenpat_exponent(pv["rs"], pv["ra"], pv["g"]))
            prior = 0.5 + PREV_WEIGHT * (pp - 0.5)
        true = M.shrink(p, a["g"], K_TEAM_WPCT, prior)
        wpct = a["w"] / a["g"] if a["g"] else 0.5
        return {"exp": n, "pyth": p, "wpct": wpct, "prior": prior, "true": true, "luck": wpct - p,
                "g": a["g"], "rs": a["rs"], "ra": a["ra"]}

    # --- Elo (5.7.7): K=4, ventaja de local 24, multiplicador por margen, regresión de 1/3 entre temporadas
    ELO_K = 4.0
    ELO_HFA = 24.0

    def _elo(self):
        r: dict[int, float] = defaultdict(lambda: 1500.0)
        season = None
        self.elo_history: dict[int, list] = defaultdict(list)
        for g in self.prev + self.results:
            if season is not None and g["season"] != season:
                for t in list(r):
                    r[t] = 1500 + (r[t] - 1500) * (2 / 3)
            season = g["season"]
            ra, rh = r[g["away"]], r[g["home"]]
            neutral = self.is_neutral(g)
            eh = M.elo_expected(rh + (0 if neutral else self.ELO_HFA), ra)
            sh = 1.0 if g["hr"] > g["ar"] else 0.0
            margin = abs(g["hr"] - g["ar"])
            win_diff = (rh - ra) if sh else (ra - rh)
            mult = math.log(margin + 1) * 2.2 / (win_diff * 0.001 + 2.2)
            delta = self.ELO_K * mult * (sh - eh)
            r[g["home"]] = rh + delta
            r[g["away"]] = ra - delta
            if g["season"] == self.lg.season:
                self.elo_history[g["home"]].append([g["date"], round(r[g["home"]], 1)])
                self.elo_history[g["away"]].append([g["date"], round(r[g["away"]], 1)])
        self.elo = dict(r)

    def is_neutral(self, g) -> bool:
        home_venue = (self.info.get(g["home"]) or {}).get("venue")
        return bool(g.get("venue") and home_venue and g["venue"] != home_venue)

    # --- historial (sección 2)
    def last_games(self, tid, side: str | None = None, n: int = 10, opp: int | None = None) -> list[dict]:
        rows = []
        for g in reversed(self.prev + self.results):
            if tid not in (g["away"], g["home"]):
                continue
            if side == "home" and g["home"] != tid:
                continue
            if side == "away" and g["away"] != tid:
                continue
            if opp is not None and opp not in (g["away"], g["home"]):
                continue
            rows.append(g)
            if len(rows) >= n:
                break
        return rows

    def offense(self, tid) -> dict:
        """Índice ofensivo neutral de parque y regresado (1.00 = promedio de liga)."""
        a = self.agg[tid]
        lg_rpg = (self.lg.rpg["away"] + self.lg.rpg["home"]) / 2
        rpg = a["rs"] / a["g"] if a["g"] else lg_rpg
        home_park = self.parks.get((self.info.get(tid) or {}).get("venue"), 1.0)
        neutral = rpg / ((1 + home_park) / 2)
        idx = M.shrink(neutral / lg_rpg, a["g"], K_TEAM_RUNS, 1.0) if lg_rpg else 1.0
        st = self.stats.get(tid, {})
        hit = st.get("hitting", {})
        obp, slg = obp_slg(hit)
        pa = hit.get("plateAppearances", 0) or 1
        vs = {}
        for hand, key in (("L", "hit_vl"), ("R", "hit_vr")):
            h = st.get(key, {})
            o, s = obp_slg(h)
            raw = (o * s) / (obp * slg) if obp * slg and o * s else 1.0
            lg_ratio = self.lg.obpslg_vs[hand] / self.lg.obpslg if self.lg.obpslg else 1.0
            rel = raw / lg_ratio if lg_ratio else 1.0
            vs[hand] = {"ratio": M.shrink(rel, h.get("plateAppearances", 0), K_SPLIT_PA, 1.0) * lg_ratio,
                        "ops": o + s, "obp": o, "slg": s, "pa": h.get("plateAppearances", 0),
                        "k": h.get("strikeOuts", 0) / h["plateAppearances"] if h.get("plateAppearances") else None}
        return {"rpg": rpg, "neutralRpg": neutral, "idx": idx, "park": home_park, "obp": obp, "slg": slg,
                "ops": obp + slg, "k": hit.get("strikeOuts", 0) / pa, "bb": hit.get("baseOnBalls", 0) / pa,
                "iso": slg - (hit.get("hits", 0) / hit["atBats"] if hit.get("atBats") else 0),
                "hrpa": hit.get("homeRuns", 0) / pa, "vs": vs, "g": a["g"],
                "f3": a["f3rs"] / a["g"] if a["g"] else None, "f5": a["f5rs"] / a["g"] if a["g"] else None,
                "f3a": a["f3ra"] / a["g"] if a["g"] else None, "f5a": a["f5ra"] / a["g"] if a["g"] else None,
                "rapg": a["ra"] / a["g"] if a["g"] else None}


# ============================================================ abridores

def pitcher_profile(p: dict | None, lg: League, savant: dict, team_sp: dict | None) -> dict:
    """Perfil del abridor con shrinkage bayesiano (5.7.4) y FIP con constante de liga (3.4)."""
    if not p or not p.get("season"):
        return {"missing": True, "name": (p or {}).get("name") or "Por anunciar", "id": (p or {}).get("id")}
    cur = add_stats(p["season"], None)
    prev = add_stats(p.get("prevSeason"), None)
    comb = add_stats(p["season"], p.get("prevSeason"), PREV_WEIGHT)
    bf_c, bf_all = cur.get("battersFaced", 0), comb.get("battersFaced", 0)
    ip_c = cur.get("ip", 0)

    def rate(key, stats):
        return stats.get(key, 0) / stats["battersFaced"] if stats.get("battersFaced") else 0.0

    k_s = M.shrink(rate("strikeOuts", comb), bf_all, K_PITCHER["k"], lg.k_rate)
    bb_s = M.shrink(rate("baseOnBalls", comb), bf_all, K_PITCHER["bb"], lg.bb_rate)
    hbp_s = M.shrink(rate("hitByPitch", comb), bf_all, K_PITCHER["bb"], lg.hbp_rate)
    hr_s = M.shrink(rate("homeRuns", comb), bf_all, K_PITCHER["hr"], lg.hr_rate)
    go, ao = comb.get("groundOuts", 0), comb.get("airOuts", 0)
    gb_s = M.shrink(go / (go + ao) if go + ao else lg.gb_rate, go + ao, K_PITCHER["gb"], lg.gb_rate)
    fip_raw = M.fip(cur.get("homeRuns", 0), cur.get("baseOnBalls", 0), cur.get("hitByPitch", 0),
                    cur.get("strikeOuts", 0), ip_c, lg.c_fip)
    fip_true = (13 * hr_s + 3 * (bb_s + hbp_s) - 2 * k_s) * lg.bf_per_ip + lg.c_fip
    # xFIP aproximado: HR esperados con la tasa de HR de la liga por elevado (airOuts + HR como proxy de FB)
    fb_proxy = cur.get("airOuts", 0) + cur.get("homeRuns", 0)
    lg_fb = lg.tot.get("airOuts", 0) + lg.tot.get("homeRuns", 0)
    lg_hr_fb = lg.tot.get("homeRuns", 0) / lg_fb if lg_fb else 0.1
    xfip = ((13 * fb_proxy * lg_hr_fb + 3 * (cur.get("baseOnBalls", 0) + cur.get("hitByPitch", 0))
             - 2 * cur.get("strikeOuts", 0)) / ip_c + lg.c_fip) if ip_c else None
    era_raw = M.era(cur.get("earnedRuns", 0), ip_c)
    era_comb = M.era(comb.get("earnedRuns", 0), comb.get("ip", 0)) or lg.era
    era_s = M.shrink(era_comb, bf_all, K_ERA_BF, lg.era)
    sv = (savant.get("expected") or {}).get(str(p["id"])) or {}
    xera = sv.get("xera")
    xera_s = M.shrink(xera, sv.get("pa") or 0, K_XERA_PA, lg.era) if xera is not None else None
    if xera_s is not None:
        est_era = 0.45 * fip_true + 0.35 * xera_s + 0.20 * era_s
        blend = "0.45·FIP regresado + 0.35·xERA regresada + 0.20·ERA regresada"
    else:
        est_era = 0.65 * fip_true + 0.35 * era_s
        blend = "0.65·FIP regresado + 0.35·ERA regresada (sin xERA)"
    ra9 = est_era * lg.ra_per_era
    sc = (savant.get("pitcher") or {}).get(str(p["id"])) or {}

    # aperturas (game logs)
    starts = [r for r in p.get("log", []) if r["stat"].get("gamesStarted")]
    prev_starts = [r for r in p.get("prevLog", []) if r["stat"].get("gamesStarted")]
    ips = [M.ip_to_float(r["stat"].get("inningsPitched")) for r in starts]
    ip_per_start = statistics.fmean(ips) if ips else None
    recent_ips = ips[-8:]
    base_ip = statistics.fmean(recent_ips) if recent_ips else lg.sp_ip_per_gs
    exp_ip = M.shrink(base_ip, len(recent_ips), K_START_IP, lg.sp_ip_per_gs)
    if not starts and cur.get("gamesStarted", 0) == 0:
        exp_ip = min(exp_ip, 2.0)  # relevista usado como opener
    pitches = [r["stat"].get("numberOfPitches") or r["stat"].get("pitchesThrown") for r in starts[-5:]]

    last5 = starts[-5:]
    l5 = add_stats(None, None)
    for r in last5:
        l5 = add_stats(l5, r["stat"])
    l5_bf = l5.get("battersFaced", 0)
    l5_fip = M.fip(l5.get("homeRuns", 0), l5.get("baseOnBalls", 0), l5.get("hitByPitch", 0),
                   l5.get("strikeOuts", 0), l5.get("ip", 0), lg.c_fip) if l5.get("ip") else None
    form_ratio = (l5_fip / fip_raw) if (l5_fip and fip_raw and fip_raw > 0) else 1.0
    form = M.shrink(max(0.5, min(form_ratio, 1.8)), l5_bf, 600, 1.0)

    splits = {}
    for code in ("h", "a"):
        s = add_stats(p.get("splits", {}).get(code), None)
        s_ip = s.get("ip", 0)
        s_fip = M.fip(s.get("homeRuns", 0), s.get("baseOnBalls", 0), s.get("hitByPitch", 0),
                      s.get("strikeOuts", 0), s_ip, lg.c_fip) if s_ip else None
        ratio = (s_fip / fip_raw) if (s_fip and fip_raw and fip_raw > 0) else 1.0
        splits[code] = {"ip": s_ip, "bf": s.get("battersFaced", 0), "era": M.era(s.get("earnedRuns", 0), s_ip),
                        "whip": M.whip(s.get("hits", 0), s.get("baseOnBalls", 0), s_ip), "fip": s_fip,
                        "k": rate("strikeOuts", s), "bb": rate("baseOnBalls", s), "gs": s.get("gamesStarted", 0),
                        "w": s.get("wins", 0), "l": s.get("losses", 0),
                        "factor": M.shrink(max(0.6, min(ratio, 1.6)), s.get("battersFaced", 0), 600, 1.0)}

    team_fip = None
    if team_sp and team_sp.get("inningsPitched"):
        ts = add_stats(team_sp, None)
        team_fip = M.fip(ts.get("homeRuns", 0), ts.get("baseOnBalls", 0), ts.get("hitByPitch", 0),
                         ts.get("strikeOuts", 0), ts["ip"], lg.c_fip)

    return {
        "missing": False, "id": p["id"], "name": p["name"], "hand": p.get("hand"), "age": p.get("age"),
        "w": cur.get("wins", 0), "l": cur.get("losses", 0), "g": cur.get("gamesPlayed", 0),
        "gs": cur.get("gamesStarted", 0), "ip": ip_c, "bf": bf_c, "prevBf": prev.get("battersFaced", 0),
        "era": era_raw, "whip": M.whip(cur.get("hits", 0), cur.get("baseOnBalls", 0), ip_c),
        "k": rate("strikeOuts", cur), "bb": rate("baseOnBalls", cur),
        "kbb": rate("strikeOuts", cur) - rate("baseOnBalls", cur),
        "hr9": 9 * cur.get("homeRuns", 0) / ip_c if ip_c else None,
        "gb": (cur.get("groundOuts", 0) / (cur.get("groundOuts", 0) + cur.get("airOuts", 0))
               if cur.get("groundOuts", 0) + cur.get("airOuts", 0) else None),
        "fip": fip_raw, "xfip": xfip, "xera": xera, "xwoba": sv.get("xwoba"),
        "barrel": sc.get("barrel"), "hardhit": sc.get("hardhit"), "bbe": sc.get("bbe"),
        "shrunk": {"k": k_s, "bb": bb_s, "kbb": k_s - bb_s, "hr": hr_s, "gb": gb_s, "fip": fip_true,
                   "era": era_s, "xera": xera_s, "weightCur": bf_all / (bf_all + K_PITCHER["k"]) if bf_all else 0},
        "estEra": est_era, "estRa9": ra9, "blend": blend, "factor": ra9 / lg.ra9,
        "ipPerStart": ip_per_start, "expIp": exp_ip, "pitchesLast5": pitches,
        "expBf": exp_ip * (bf_c / ip_c if ip_c else lg.bf_per_ip),
        "last5": [log_row(r) for r in reversed(last5)], "l5fip": l5_fip, "l5bf": l5_bf, "form": form,
        "starts": starts, "prevStarts": prev_starts, "splits": splits, "teamRotationFip": team_fip,
    }


def log_row(r: dict) -> dict:
    s = r["stat"]
    return {"date": r["date"], "pk": r.get("pk"), "opp": r.get("opp"), "home": r.get("home"), "win": r.get("win"),
            "ip": s.get("inningsPitched"), "h": s.get("hits", 0), "er": s.get("earnedRuns", 0),
            "r": s.get("runs", 0), "bb": s.get("baseOnBalls", 0), "k": s.get("strikeOuts", 0),
            "hr": s.get("homeRuns", 0), "bf": s.get("battersFaced", 0),
            "pitches": s.get("numberOfPitches") or s.get("pitchesThrown"), "gs": s.get("gamesStarted", 0)}


# ============================================================ bullpen

def bullpen_table(bundle: dict, lg: League, today: dt.date) -> dict:
    """Calidad (split 'rp' de la temporada) + fatiga de los últimos 3 días + ERA últimos 7 días."""
    ts = bundle.get("teamStats", {})
    out = {}
    for tid_s, st in ts.items():
        rp = add_stats(st.get("rp"), None)
        ipv = rp.get("ip", 0)
        bf = rp.get("battersFaced", 0) or 1
        h, bb, hr, k = rp.get("hits", 0), rp.get("baseOnBalls", 0), rp.get("homeRuns", 0), rp.get("strikeOuts", 0)
        hbp, r, er = rp.get("hitByPitch", 0), rp.get("runs", 0), rp.get("earnedRuns", 0)
        ab, sf = rp.get("atBats", 0), rp.get("sacFlies", 0)
        fipv = M.fip(hr, bb, hbp, k, ipv, lg.c_fip) if ipv else None
        era = M.era(er, ipv)
        babip_den = ab - k - hr + sf
        lob_den = h + bb + hbp - 1.4 * hr
        est = (0.6 * (fipv or lg.era) + 0.4 * (era or lg.era))
        lg_rp_fip = lg.rp_era
        out[int(tid_s)] = {
            "ip": ipv, "era": era, "whip": M.whip(h, bb, ipv), "fip": fipv, "k": k / bf, "bb": bb / bf,
            "kbb": (k - bb) / bf, "hr9": 9 * hr / ipv if ipv else None, "h9": 9 * h / ipv if ipv else None,
            "babip": (h - hr) / babip_den if babip_den > 0 else None,
            "lob": (h + bb + hbp - r) / lob_den if lob_den > 0 else None,
            "sv": rp.get("saves", 0), "bs": rp.get("blownSaves", 0), "hld": rp.get("holds", 0),
            "estEra": M.shrink(est, ipv, K_BULLPEN_IP, lg_rp_fip),
        }
        out[int(tid_s)]["factor"] = out[int(tid_s)]["estEra"] * lg.ra_per_era / lg.ra9

    # --- uso reciente desde box scores
    usage: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    last7: dict[int, dict] = defaultdict(lambda: {"ip": 0.0, "er": 0, "g": set()})
    names: dict[int, str] = {}
    staff: dict[int, set] = defaultdict(set)
    for b in bundle.get("boxscores", []):
        d = dt.date.fromisoformat(b["date"])
        days_ago = (today - d).days
        for side in ("away", "home"):
            t = b[side]
            staff[t["team"]].update(r["id"] for r in t["pitchers"])
            staff[t["team"]].update(t.get("bullpen") or [])
            for row in t["pitchers"]:
                names[row["id"]] = row.get("name")
                if row["order"] == 0:
                    continue  # el abridor no cuenta como uso de bullpen
                usage[t["team"]][row["id"]].append({"date": b["date"], "daysAgo": days_ago, "pitches": row["pitches"] or 0,
                                                    "ip": M.ip_to_float(row["ip"]), "sv": row.get("sv", 0)})
                if days_ago <= 7:
                    last7[t["team"]]["ip"] += M.ip_to_float(row["ip"])
                    last7[t["team"]]["er"] += row.get("er", 0) or 0
                    last7[t["team"]]["g"].add(b["pk"])

    fatigue = {}
    for tid in out:
        load = 0.0
        for apps in usage.get(tid, {}).values():
            for a in apps:
                w = {1: 1.0, 2: 0.66, 3: 0.33}.get(a["daysAgo"], 0.0 if a["daysAgo"] > 3 else 1.0)
                load += w * a["pitches"]
        fatigue[tid] = load
        l7 = last7.get(tid)
        out[tid]["era7"] = M.era(l7["er"], l7["ip"]) if l7 and l7["ip"] else None
        out[tid]["ip7"] = l7["ip"] if l7 else 0
        out[tid]["fatigue"] = load

    vals = list(fatigue.values())
    for tid in out:
        out[tid]["zFatigue"] = M.zscore(fatigue[tid], vals)
        out[tid]["fatigueMult"] = 1 + 0.02 * max(-2.0, min(2.0, out[tid]["zFatigue"]))

    # RiesgoBullpen = z(ERA)+z(WHIP)+z(BB%)+z(HR/9)+z(Fatiga)  (ecuación 7)
    cols = {c: [v[c] for v in out.values() if v.get(c) is not None] for c in ("era", "whip", "bb", "hr9")}
    for tid, v in out.items():
        parts = {c: M.zscore(v[c], cols[c]) if v.get(c) is not None else 0.0 for c in cols}
        parts["fatiga"] = v["zFatigue"]
        v["riskParts"] = parts
        v["risk"] = sum(parts.values())
    ranked = sorted(out, key=lambda t: out[t]["risk"])
    for i, tid in enumerate(ranked):
        out[tid]["rank"] = i + 1  # 1 = bullpen más confiable
    return {"teams": out, "usage": usage, "names": names, "staff": staff}


def relievers(tid: int, bundle: dict, bp: dict, today: dt.date, lg: League) -> list[dict]:
    """Relevistas clave del equipo: cerrador (más salvamentos), setup (más holds) y brazos de más uso."""
    staff = bp.get("staff", {}).get(tid)
    rows = [p for p in bundle.get("playersPitching", []) if p.get("team") == tid and (not staff or p["id"] in staff)]
    rel = []
    for p in rows:
        s = add_stats(p["stat"], None)
        gp = s.get("gamesPlayed", 0)
        if gp < 5 or s.get("gamesStarted", 0) > 0.3 * gp:
            continue
        rel.append((p, s))
    if not rel:
        return []
    used = bp["usage"].get(tid, {})
    by_sv = sorted(rel, key=lambda x: (x[1].get("saves", 0), x[1].get("holds", 0)), reverse=True)
    by_hld = sorted(rel, key=lambda x: x[1].get("holds", 0), reverse=True)
    by_ip = sorted(rel, key=lambda x: x[1].get("ip", 0), reverse=True)
    chosen: list[tuple[str, tuple]] = []
    seen = set()

    def take(role, cand):
        for c in cand:
            if c[0]["id"] not in seen:
                seen.add(c[0]["id"])
                chosen.append((role, c))
                return

    take("Cerrador", by_sv)
    take("Setup", by_hld)
    take("Setup", by_hld)
    take("Alto uso", by_ip)
    take("Alto uso", by_ip)
    out = []
    for role, (p, s) in chosen:
        apps = sorted(used.get(p["id"], []), key=lambda a: a["date"], reverse=True)
        last = apps[0] if apps else None
        g3 = [a for a in apps if 1 <= a["daysAgo"] <= 3]
        consecutive = {a["daysAgo"] for a in apps} >= {1, 2}
        yday = sum(a["pitches"] for a in apps if a["daysAgo"] == 1)
        p3 = sum(a["pitches"] for a in g3)
        if len(g3) >= 3 or (consecutive and yday >= 15) or yday >= 35:
            avail, risk = "Probablemente no", "Alto"
        elif consecutive or yday >= 20 or p3 >= 45:
            avail, risk = "Dudoso", "Medio"
        else:
            avail, risk = "Sí", "Bajo"
        ipv = s.get("ip", 0)
        bf = s.get("battersFaced", 0) or 1
        out.append({
            "id": p["id"], "name": p["name"], "role": role, "ip": ipv,
            "era": M.era(s.get("earnedRuns", 0), ipv), "whip": M.whip(s.get("hits", 0), s.get("baseOnBalls", 0), ipv),
            "fip": M.fip(s.get("homeRuns", 0), s.get("baseOnBalls", 0), s.get("hitByPitch", 0), s.get("strikeOuts", 0),
                         ipv, lg.c_fip) if ipv else None,
            "k": s.get("strikeOuts", 0) / bf, "bb": s.get("baseOnBalls", 0) / bf,
            "sv": s.get("saves", 0), "hld": s.get("holds", 0), "bs": s.get("blownSaves", 0),
            "lastDate": last["date"] if last else None, "lastPitches": last["pitches"] if last else None,
            "g3": len(g3), "ip3": sum(a["ip"] for a in g3), "p3": p3, "consecutive": consecutive,
            "available": avail, "risk": risk, "idle7": not apps,
        })
    return out


# ============================================================ park factors

def park_factors(bundle: dict) -> tuple[dict, dict]:
    """Índice de carreras 3 años de Baseball Savant (100 = neutro) → multiplicador."""
    raw = (bundle.get("savant") or {}).get("park") or {}
    pf = {}
    for vid, r in raw.items():
        if r.get("runs"):
            pf[int(vid)] = r["runs"] / 100.0
    return pf, {int(k): v for k, v in raw.items()}
