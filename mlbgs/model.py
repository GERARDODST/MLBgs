"""Análisis partido por partido siguiendo el Framework MLB Picks v2 (secciones 1-10).

Cada partido produce un diccionario con las 10 secciones del framework, listo para
serializarse a JSON y mostrarse en la página. Los números salen solo de datos reales
(MLB Stats API, Baseball Savant y, si hay clave, The Odds API); lo que no se puede
obtener queda marcado como faltante y activa el gate de completitud.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from collections import defaultdict

from . import features as F
from . import mathlib as M

SIDES = ("away", "home")
TOTAL_LINES = [7.5, 8.0, 8.5, 9.0, 9.5]
EDGE_MIN = 0.03          # edge mínimo (3 pp) para considerar que la cuota tiene valor
MODEL_MIN = 0.55         # probabilidad mínima para que el modelo "apoye" un lado sin momios
SCORE_WEIGHTS = [        # tabla 3.3 del framework
    ("Calidad de temporada", 0.20, "ERA y WHIP regresados a la media"),
    ("Métricas esperadas", 0.15, "FIP con constante de liga y xERA de Statcast, regresados"),
    ("K-BB%, BB%, control", 0.15, "K-BB% regresado (K% 70 BF, BB% 170 BF)"),
    ("Matchup vs lineup rival", 0.15, "Ofensiva rival vs la mano del abridor"),
    ("Arsenal vs perfil rival", 0.15, "Proxy Statcast: Barrel% permitido vs contacto del rival"),
    ("Local/visita/parque", 0.10, "Split casa/visita del abridor regresado"),
    ("Últimas 5 salidas", 0.10, "FIP de las últimas 5 aperturas (forma, no factor dominante)"),
]


def r2(x, n=2):
    return None if x is None else round(float(x), n)


def pct(x, n=1):
    return None if x is None else round(100 * float(x), n)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================ contexto compartido

class Context:
    def __init__(self, bundle: dict):
        self.b = bundle
        self.today = dt.date.fromisoformat(bundle["meta"]["today"])
        self.lg = F.League(bundle)
        self.pf, self.park_raw = F.park_factors(bundle)
        self.T = F.Teams(bundle, self.lg, self.pf)
        self.bp = F.bullpen_table(bundle, self.lg, self.today)
        self.savant = bundle.get("savant") or {}
        self.hands = {int(k): v for k, v in (bundle.get("hands") or {}).items()}
        self.bats = {int(k): v for k, v in (bundle.get("bats") or {}).items()}
        self.hitters: dict[int, dict] = {}
        for p in bundle.get("playersHitting", []):
            prev = self.hitters.get(p["id"])
            self.hitters[p["id"]] = {"name": p["name"], "team": p["team"],
                                     "stat": F.add_stats(prev["stat"] if prev else None, p["stat"])}
        self.venue_names = {int(k): v.get("name") for k, v in self.park_raw.items()}
        for t in self.T.info.values():
            if t.get("venue"):
                self.venue_names.setdefault(t["venue"], t.get("venueName"))
        self.odds = index_odds(bundle.get("odds"), self.T.info)
        tk = [F.obp_slg(st.get("hitting", {})) for st in bundle.get("teamStats", {}).values()]
        self.team_k = sorted((st.get("hitting", {}).get("strikeOuts", 0) / st["hitting"]["plateAppearances"])
                             for st in bundle.get("teamStats", {}).values()
                             if st.get("hitting", {}).get("plateAppearances"))
        self.k_high = self.team_k[int(len(self.team_k) * 0.67)] if self.team_k else 0.23
        self.lg_barrel = self._lg_barrel()
        self.lg_kbb = self.lg.k_rate - self.lg.bb_rate

    def _lg_barrel(self):
        rows = (self.savant.get("batter") or {}).values()
        num = sum((r.get("barrel") or 0) * (r.get("bbe") or 0) for r in rows)
        den = sum((r.get("bbe") or 0) for r in rows)
        return num / den if den else 7.5

    def venue_name(self, vid):
        return self.venue_names.get(vid) or "Sede alterna"


# ============================================================ momios (sección 7)

def index_odds(raw, teams: dict) -> dict:
    """Agrupa los momios de The Odds API por (visitante, local)."""
    if not raw:
        return {}
    by_club = {}
    for t in teams.values():
        by_club[t["club"].lower()] = t["id"]
        by_club[t["name"].lower()] = t["id"]

    def tid(name):
        n = name.lower()
        if n in by_club:
            return by_club[n]
        for club, i in by_club.items():
            if n.endswith(club):
                return i
        return None

    out = defaultdict(list)
    for ev in raw:
        a, h = tid(ev.get("away_team", "")), tid(ev.get("home_team", ""))
        if not a or not h:
            continue
        books = []
        for bk in ev.get("bookmakers", []):
            row = {"book": bk.get("title"), "ml": {}, "rl": {}, "total": {}}
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    side = "home" if tid(o.get("name", "")) == h else "away" if tid(o.get("name", "")) == a else None
                    if m["key"] == "h2h" and side:
                        row["ml"][side] = o["price"]
                    elif m["key"] == "spreads" and side:
                        row["rl"][side] = {"price": o["price"], "point": o.get("point")}
                    elif m["key"] == "totals":
                        row["total"][o["name"].lower()] = {"price": o["price"], "point": o.get("point")}
            books.append(row)
        out[(a, h)].append({"commence": ev.get("commence_time"), "books": books})
    return out


def odds_for_game(ctx: Context, g: dict) -> dict | None:
    evs = ctx.odds.get((g["away"], g["home"]))
    if not evs:
        return None
    t0 = dt.datetime.fromisoformat(g["time"].replace("Z", "+00:00"))

    def gap(ev):
        try:
            return abs((dt.datetime.fromisoformat(ev["commence"].replace("Z", "+00:00")) - t0).total_seconds())
        except Exception:  # noqa: BLE001
            return 9e9

    ev = min(evs, key=gap)
    if gap(ev) > 6 * 3600:
        return None
    books = [b for b in ev["books"] if b["ml"] or b["total"]]
    lines = [b["total"]["over"]["point"] for b in books if b["total"].get("over")]
    main_line = statistics.mode(lines) if lines else None

    def best(get):
        vals = [(get(b), b["book"]) for b in books if get(b) is not None]
        return max(vals, key=lambda v: M.american_to_decimal(v[0])) if vals else None

    return {
        "books": books, "nBooks": len(books), "totalLine": main_line,
        "bestMl": {s: best(lambda b, s=s: b["ml"].get(s)) for s in SIDES},
        "bestRl": {s: best(lambda b, s=s: (b["rl"].get(s) or {}).get("price")
                            if abs(((b["rl"].get(s) or {}).get("point") or 0)) == 1.5 else None) for s in SIDES},
        "rlPoint": {s: next(((b["rl"].get(s) or {}).get("point") for b in books if b["rl"].get(s)), None) for s in SIDES},
        "bestTotal": {k: best(lambda b, k=k: b["total"][k]["price"]
                              if b["total"].get(k) and b["total"][k]["point"] == main_line else None)
                      for k in ("over", "under")},
    }


# ============================================================ utilidades del partido

def parse_weather(w: dict, roof: str | None) -> dict:
    cond = (w or {}).get("condition")
    temp = F.f((w or {}).get("temp"), None) if (w or {}).get("temp") else None
    wind = (w or {}).get("wind") or ""
    m = re.match(r"(\d+)\s*mph,?\s*(.*)", wind)
    speed = int(m.group(1)) if m else None
    direction = (m.group(2) if m else "").strip()
    closed = bool(re.search(r"Dome|Roof Closed", cond or "", re.I)) or (roof or "").lower() == "dome"
    out_ = bool(re.search(r"^Out", direction))
    in_ = bool(re.search(r"^In", direction))
    return {"condition": cond, "temp": temp, "windText": wind or None, "speed": speed, "direction": direction or None,
            "closed": closed, "out": out_ and not closed, "in": in_ and not closed, "available": bool(w)}


def starter_fracs(exp_ip: float, sd: float = 1.2) -> list[float]:
    """Fracción esperada de cada entrada (1..9) cubierta por el abridor, con salida ~ Normal(exp_ip, sd)."""
    pts = [exp_ip + sd * z for z in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)]
    w = [math.exp(-0.5 * z * z) for z in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)]
    sw = sum(w)
    out = [0.0]
    for i in range(1, 10):
        out.append(sum(wi * clip(x - (i - 1), 0, 1) for x, wi in zip(pts, w)) / sw)
    return out


def tbd_pitcher(ctx: Context, tid: int) -> dict:
    """Abridor no anunciado: se usa el promedio de la rotación del equipo (y se marca como faltante)."""
    sp = F.add_stats(ctx.b.get("teamStats", {}).get(str(tid), {}).get("sp"), None)
    lg = ctx.lg
    ipv = sp.get("ip", 0)
    fipv = M.fip(sp.get("homeRuns", 0), sp.get("baseOnBalls", 0), sp.get("hitByPitch", 0), sp.get("strikeOuts", 0),
                 ipv, lg.c_fip) if ipv else lg.era
    erav = M.era(sp.get("earnedRuns", 0), ipv) if ipv else lg.era
    est = 0.6 * fipv + 0.4 * erav
    bf = sp.get("battersFaced", 0) or 1
    return {"missing": True, "name": "Por anunciar", "id": None, "hand": None, "estEra": est,
            "estRa9": est * lg.ra_per_era, "factor": est * lg.ra_per_era / lg.ra9, "expIp": lg.sp_ip_per_gs,
            "form": 1.0, "splits": {}, "shrunk": {"k": sp.get("strikeOuts", 0) / bf, "bb": sp.get("baseOnBalls", 0) / bf,
                                                  "kbb": (sp.get("strikeOuts", 0) - sp.get("baseOnBalls", 0)) / bf,
                                                  "hr": sp.get("homeRuns", 0) / bf, "fip": fipv, "era": erav, "xera": None},
            "k": None, "bb": None, "kbb": None, "era": None, "fip": None, "whip": None, "starts": [], "prevStarts": [],
            "last5": [], "note": "Sin abridor probable oficial: se usa la rotación del equipo (ERA/FIP de abridores)."}


def lineup_profile(ctx: Context, ids: list[int], names: dict, team_off: dict) -> dict | None:
    """Lineup confirmado: OBP·SLG, K%, BB% y Barrel% de los 9 bateadores (regresados a la liga)."""
    if len(ids) < 9:
        return None
    lg = ctx.lg
    wts = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]
    tot_w = obp = slg = k = bb = 0.0
    brl_num = brl_den = 0.0
    rows = []
    for i, pid in enumerate(ids[:9]):
        h = (ctx.hitters.get(pid) or {}).get("stat", {})
        pa = h.get("plateAppearances", 0)
        o, s = F.obp_slg(h)
        o_s = M.shrink(o, pa, 300, lg.obp) if pa else lg.obp
        s_s = M.shrink(s, pa, 320, lg.slg) if pa else lg.slg
        k_s = M.shrink(h.get("strikeOuts", 0) / pa, pa, 60, lg.bat_k_rate) if pa else lg.bat_k_rate
        bb_s = M.shrink(h.get("baseOnBalls", 0) / pa, pa, 120, lg.bat_bb_rate) if pa else lg.bat_bb_rate
        sc = (ctx.savant.get("batter") or {}).get(str(pid)) or {}
        if sc.get("barrel") is not None and sc.get("bbe"):
            brl_num += sc["barrel"] * sc["bbe"]
            brl_den += sc["bbe"]
        w = wts[i]
        tot_w += w
        obp += w * o_s
        slg += w * s_s
        k += w * k_s
        bb += w * bb_s
        rows.append({"id": pid, "name": names.get(str(pid)) or (ctx.hitters.get(pid) or {}).get("name"),
                     "bats": ctx.bats.get(pid), "pa": pa, "obp": r2(o, 3), "slg": r2(s, 3), "ops": r2(o + s, 3),
                     "hr": h.get("homeRuns", 0), "k": pct(h.get("strikeOuts", 0) / pa) if pa else None,
                     "barrel": sc.get("barrel")})
    obp, slg, k, bb = obp / tot_w, slg / tot_w, k / tot_w, bb / tot_w
    team_os = team_off["obp"] * team_off["slg"] or lg.obpslg
    ratio = M.shrink((obp * slg) / team_os, 1, 1, 1.0)  # regresión 50% hacia la ofensiva habitual del equipo
    lefties = sum(1 for r in rows if r["bats"] == "L") + 0.5 * sum(1 for r in rows if r["bats"] == "S")
    return {"rows": rows, "obp": obp, "slg": slg, "k": k, "bb": bb, "ratio": ratio,
            "barrel": brl_num / brl_den if brl_den else None, "lefties": lefties}


def team_barrel(ctx: Context, tid: int) -> float | None:
    num = den = 0.0
    for pid, h in ctx.hitters.items():
        if h["team"] != tid:
            continue
        sc = (ctx.savant.get("batter") or {}).get(str(pid))
        if sc and sc.get("barrel") is not None and sc.get("bbe"):
            num += sc["barrel"] * sc["bbe"]
            den += sc["bbe"]
    return num / den if den else None


# ============================================================ modelo de carreras (secciones 5.3 y 6.7)

def lambdas(ctx: Context, bat_side: str, off: dict, lineup: dict | None, opp_sp: dict, opp_bp: dict,
            park: float, adj: dict, opp_is_home: bool) -> dict:
    lg = ctx.lg
    hand = opp_sp.get("hand")
    if hand in ("L", "R"):
        hand_ratio = off["vs"][hand]["ratio"]
    else:
        hand_ratio = 0.72 * off["vs"]["R"]["ratio"] + 0.28 * off["vs"]["L"]["ratio"]
    split = (opp_sp.get("splits") or {}).get("h" if opp_is_home else "a", {}).get("factor", 1.0)
    sp_mult = opp_sp["factor"] * opp_sp.get("form", 1.0) * split
    bp_mult = opp_bp["factor"] * opp_bp.get("fatigueMult", 1.0)
    lineup_ratio = lineup["ratio"] if lineup else 1.0
    fr = starter_fracs(opp_sp["expIp"])
    team_adj = adj["team"][bat_side]
    per = [0.0]
    for i in range(1, 10):
        base = lg.inning_runs[bat_side][i]
        pitch = fr[i] * hand_ratio * sp_mult * adj["sp"] + (1 - fr[i]) * bp_mult * adj["bp"]
        per.append(base * off["idx"] * lineup_ratio * park * pitch * adj["all"] * team_adj)
    extra = lg.extra_runs[bat_side] * off["idx"] * lineup_ratio * park * bp_mult * adj["bp"] * adj["all"] * team_adj
    base_per = [0.0] + [lg.inning_runs[bat_side][i] for i in range(1, 10)]
    return {
        "per": per, "f1": per[1], "f3": sum(per[1:4]), "f5": sum(per[1:6]), "g9": sum(per[1:10]),
        "full": sum(per[1:10]) + extra, "extra": extra, "starterFracs": fr,
        "vsStarter": sum(per[i] * fr[i] for i in range(1, 10)),
        "vsBullpen": sum(per[i] * (1 - fr[i]) for i in range(1, 10)) + extra,
        "lgFull": sum(base_per) + lg.extra_runs[bat_side],
        "components": {"offense": off["idx"], "handRatio": hand_ratio, "hand": hand, "lineup": lineup_ratio,
                       "park": park, "starter": opp_sp["factor"], "form": opp_sp.get("form", 1.0), "split": split,
                       "bullpen": opp_bp["factor"], "fatigue": opp_bp.get("fatigueMult", 1.0)},
    }


def adjustments(ctx: Context, pa: dict, ph: dict, offs: dict, lineups: dict, bps: dict, park_idx: dict,
                weather: dict, umpire: str | None) -> tuple[dict, list[dict]]:
    """Tabla de ajustes multiplicativos de la sección 5.3."""
    lg = ctx.lg
    rows = []
    good_kbb = ctx.lg_kbb + 0.04

    def row(cond, applies, factor, scope, impact, why, in_base=None, note=None):
        """in_base: explicación de por qué el efecto ya está en λ base (se evalúa, pero no se duplica)."""
        rows.append({"cond": cond, "applies": bool(applies), "factor": factor, "scope": scope, "impact": impact,
                     "why": why, "inBase": in_base, "applied": bool(applies) and not in_base, "note": note})

    kbb_a, kbb_h = pa["shrunk"]["kbb"], ph["shrunk"]["kbb"]
    row("Ambos abridores con buen K-BB%", kbb_a >= good_kbb and kbb_h >= good_kbb and not (pa["missing"] or ph["missing"]),
        0.90, "sp", "Baja total", "Menos tráfico y menos rallies.",
        in_base="El FIP regresado de cada abridor ya pondera K y BB; cuenta en el filtro de ponches (6.3)",
        note=f"K-BB% regresado: {pct(kbb_a)}% y {pct(kbb_h)}% (umbral {pct(good_kbb)}%)")
    for side, opp_sp in (("away", ph), ("home", pa)):
        k_rate = (lineups[side] or {}).get("k") or offs[side]["k"]
        row(f"Rival con alto K% ({ctx.T.abbr(offs[side]['tid'])} batea)", k_rate >= ctx.k_high, 0.90, f"team:{side}",
            "Baja total", "Menos bolas en juego.",
            in_base="Las carreras y el OBP·SLG del equipo ya reflejan su K%; cuenta en el filtro de ponches (6.3)",
            note=f"K% {pct(k_rate)}% vs umbral {pct(ctx.k_high)}% (tercio superior)")
    both_strong = bps["away"]["rank"] <= 10 and bps["home"]["rank"] <= 10
    row("Ambos bullpens fuertes", both_strong, 0.90, "bp", "Baja total completo", "Menor probabilidad de carreras tardías.",
        in_base="La calidad y la fatiga de cada bullpen ya multiplican las entradas de relevo en λ base",
        note=f"Ranking RiesgoBullpen: {bps['away']['rank']} y {bps['home']['rank']} de 30")
    runs_idx = park_idx.get("runs")
    row("Parque pitcher-friendly", runs_idx is not None and runs_idx <= 95, 0.92, "all", "Baja HR y extra bases",
        "Penaliza Over.", in_base="El park factor real de Savant (índice 3 años) ya multiplica λ base",
        note=f"Índice de carreras {runs_idx}")
    for side in SIDES:
        confirmed = lineups[side] is not None
        row(f"Lineup proyectado, no confirmado ({ctx.T.abbr(offs[side]['tid'])})", not confirmed, 0.95, f"team:{side}",
            "Baja confianza ofensiva", "No se debe sobreestimar el ataque.",
            in_base=("Sin lineup se usa la ofensiva habitual del equipo (no se sobreestima); la falta de lineup "
                     "bloquea el Verde en el gate en lugar de restar carreras") if not confirmed else None)
    hr_park = (park_idx.get("hr") or 100) >= 105
    for side, sp in (("home", pa), ("away", ph)):  # el lado que batea contra ese abridor
        brl = (lineups[side] or {}).get("barrel") or offs[side].get("barrel")
        high_hr = (sp.get("hr9") or 0) >= 1.35 or sp["shrunk"]["hr"] >= lg.hr_rate * 1.2
        contact = brl is not None and brl >= ctx.lg_barrel
        row(f"Pitcher con HR/9 alto + parque ofensivo ({sp['name']})", high_hr and hr_park and contact and not sp["missing"],
            1.05, f"team:{side}", "Sube total", "Solo si también hay contacto fuerte.",
            note=f"HR/9 {r2(sp.get('hr9'))}, índice HR del parque {park_idx.get('hr')}, Barrel% rival {r2(brl, 1)}")
    wind_out = weather["out"] and (weather["speed"] or 0) >= 10
    row("Viento fuerte hacia afuera", wind_out, 1.05, "all", "Sube HR esperado", "Aumenta riesgo de Over.",
        note=weather["windText"] or "Sin reporte de clima todavía")
    row("Viento fuerte hacia adentro (extensión simétrica)", weather["in"] and (weather["speed"] or 0) >= 10, 0.95, "all",
        "Baja HR esperado", "Contradicción 6.11: el clima favorece a los pitchers.",
        note=weather["windText"] or "Sin reporte de clima todavía")
    row("Umpire de zona amplia", False, 0.93, "all", "Baja total", "Favorece ponches y Under.",
        note=f"Umpire: {umpire or 'no asignado aún'} · tendencias de zona (UmpScorecards) no conectadas: no evaluable")
    row("Umpire de zona pequeña", False, 1.05, "all", "Sube total", "Favorece boletos y tráfico en bases.",
        note="Sin datos de tendencia del umpire: no evaluable")

    # "Nunca subir el total por un solo factor"
    ups = [r for r in rows if r["applied"] and r["factor"] > 1]
    if len(ups) == 1:
        ups[0]["applied"] = False
        ups[0]["note"] = (ups[0]["note"] or "") + " · no aplicado: regla 'nunca subir el total por un solo factor'"

    adj = {"sp": 1.0, "bp": 1.0, "all": 1.0, "team": {"away": 1.0, "home": 1.0}}
    for r in rows:
        if not r["applied"]:
            continue
        if r["scope"].startswith("team:"):
            adj["team"][r["scope"][5:]] *= r["factor"]
        else:
            adj[r["scope"]] *= r["factor"]
    for k in ("sp", "bp", "all"):
        adj[k] = clip(adj[k], 0.80, 1.12)
    for s in SIDES:
        adj["team"][s] = clip(adj["team"][s], 0.80, 1.12)
    return adj, rows


# ============================================================ puntaje de abridores (3.3)

def starter_scores(ctx: Context, sp: dict, opp_off: dict, opp_lineup: dict | None, is_home: bool, opp_tid: int) -> dict:
    lg = ctx.lg
    if sp["missing"]:
        return {"total": None, "rows": [], "note": "Abridor por anunciar"}
    s = sp["shrunk"]
    whip_s = M.shrink(sp["whip"] or lg.whip, sp["bf"], 300, lg.whip)
    z = []
    z.append(((-(s["era"] - lg.era) / 0.6) + (-(whip_s - lg.whip) / 0.12)) / 2)
    xera_part = (-(s["xera"] - lg.era) / 0.6) if s.get("xera") is not None else None
    fip_part = -(s["fip"] - lg.era) / 0.5
    z.append((fip_part + xera_part) / 2 if xera_part is not None else fip_part)
    z.append((s["kbb"] - ctx.lg_kbb) / 0.05)
    hand = sp.get("hand") or "R"
    threat = opp_off["idx"] * opp_off["vs"][hand]["ratio"] * (opp_lineup["ratio"] if opp_lineup else 1.0)
    z.append(-(threat - 1) / 0.08)
    brl_opp = (opp_lineup or {}).get("barrel") or opp_off.get("barrel")
    arsenal_ok = sp.get("barrel") is not None and brl_opp is not None
    z.append((-(sp["barrel"] - ctx.lg_barrel) / 2.0 - (brl_opp - ctx.lg_barrel) / 2.0) / 2 if arsenal_ok else None)
    z.append(-((sp["splits"].get("h" if is_home else "a") or {}).get("factor", 1.0) - 1) / 0.08)
    z.append(-((sp.get("l5fip") or s["fip"]) - lg.era) / 1.0 if sp.get("last5") else None)
    rows = []
    tot_w = tot = 0.0
    for (name, w, what), zi in zip(SCORE_WEIGHTS, z):
        sc = None if zi is None else clip(50 + 15 * zi, 0, 100)
        rows.append({"factor": name, "weight": w, "what": what, "score": r2(sc, 0)})
        if sc is not None:
            tot += w * sc
            tot_w += w
    return {"total": tot / tot_w if tot_w else None, "rows": rows, "threat": threat,
            "note": None if tot_w == 1.0 else "Factores sin dato: su peso se redistribuye entre los demás"}


def result_for(ctx: Context, pk: int, tid: int):
    for g in ctx.T.results + ctx.T.prev:
        if g["pk"] == pk:
            won = (g["ar"] > g["hr"]) if g["away"] == tid else (g["hr"] > g["ar"])
            return won
    return None


# ============================================================ análisis de un partido

def analyze(ctx: Context, g: dict) -> dict:
    lg, T = ctx.lg, ctx.T
    a_id, h_id = g["away"], g["home"]
    ab, hb = T.abbr(a_id), T.abbr(h_id)
    ai, hi = T.info.get(a_id, {}), T.info.get(h_id, {})
    neutral = T.is_neutral(g)
    park = ctx.pf.get(g["venue"], 1.0)
    park_idx = ctx.park_raw.get(g["venue"], {})
    weather = parse_weather(g.get("weather"), g.get("roof"))
    ump = next((o["name"] for o in g.get("officials", []) if o["type"] == "Home Plate"), None)

    sps = {}
    for side, tid in (("away", a_id), ("home", h_id)):
        pid = g["probable"].get(side)
        raw = ctx.b.get("pitchers", {}).get(str(pid)) if pid else None
        prof = F.pitcher_profile(raw, lg, ctx.savant, ctx.b.get("teamStats", {}).get(str(tid), {}).get("sp"))
        if prof.get("missing"):
            prof = dict(tbd_pitcher(ctx, tid), name=prof.get("name") or "Por anunciar")
        sps[side] = prof
    offs = {}
    for side, tid in (("away", a_id), ("home", h_id)):
        o = T.offense(tid)
        o["tid"] = tid
        o["barrel"] = team_barrel(ctx, tid)
        offs[side] = o
    lineups = {s: lineup_profile(ctx, g["lineups"][s], g["lineupNames"][s], offs[s]) for s in SIDES}
    bps = {s: ctx.bp["teams"].get(tid) for s, tid in (("away", a_id), ("home", h_id))}
    opp = {"away": "home", "home": "away"}

    adj, adj_rows = adjustments(ctx, sps["away"], sps["home"], offs, lineups, bps, park_idx, weather, ump)
    neutral_adj = {"sp": 1.0, "bp": 1.0, "all": 1.0, "team": {"away": 1.0, "home": 1.0}}
    lam = {s: lambdas(ctx, s, offs[s], lineups[s], sps[opp[s]], bps[opp[s]], park, adj, opp[s] == "home") for s in SIDES}
    lam_base = {s: lambdas(ctx, s, offs[s], lineups[s], sps[opp[s]], bps[opp[s]], park, neutral_adj, opp[s] == "home")
                for s in SIDES}

    # --- distribuciones (5.4, 5.7.5, 6.8)
    vr = lg.var_ratio
    pmf9 = {s: M.negbin_pmf(lam[s]["g9"], vr) for s in SIDES}
    pmf_full = {s: M.negbin_pmf(lam[s]["full"], vr) for s in SIDES}
    w9, tie9, l9 = M.outcome_probs(pmf9["home"], pmf9["away"])
    x_home = 0.5 if neutral else lg.extra_home_win
    x_home = 0.5 * x_home + 0.5 * (lam["home"]["extra"] / (lam["home"]["extra"] + lam["away"]["extra"]))
    p_home_lambda = w9 + tie9 * x_home
    margin = M.margin_probs(pmf9["home"], pmf9["away"])
    total_pmf = M.sum_pmf(pmf_full["away"], pmf_full["home"])
    f5 = {s: M.poisson_pmf(lam[s]["f5"]) for s in SIDES}
    f3 = {s: M.poisson_pmf(lam[s]["f3"]) for s in SIDES}
    f5h, f5t, f5a = M.outcome_probs(f5["home"], f5["away"])
    f3h, f3t, f3a = M.outcome_probs(f3["home"], f3["away"])
    f5_total = M.sum_pmf(f5["away"], f5["home"])
    f3_total = M.sum_pmf(f3["away"], f3["home"])

    # NRFI calibrado con la frecuencia real de primeras entradas en blanco
    def p_score(side, i):
        base_p = lg.p_score_half[side][i]
        ratio = lam[side]["per"][i] / lg.inning_runs[side][i] if lg.inning_runs[side][i] else 1.0
        return 1 - math.exp(math.log(1 - base_p) * ratio)

    p1 = {s: p_score(s, 1) for s in SIDES}
    nrfi = (1 - p1["away"]) * (1 - p1["home"])
    first = {"away": 0.0, "home": 0.0}
    none_yet = 1.0
    for i in range(1, 10):
        for s in SIDES:
            p = p_score(s, i)
            first[s] += none_yet * p
            none_yet *= 1 - p

    # --- triangulación (FASE 2): Log5, Elo con abridor, modelo λ
    py = {s: T.pythag(tid) for s, tid in (("away", a_id), ("home", h_id))}
    p_log5 = M.log5(py["home"]["true"], py["away"]["true"])
    if not neutral:
        p_log5 = M.with_home_edge(p_log5, lg.home_odds_ratio)
    elo_adj = {}
    for s, tid in (("away", a_id), ("home", h_id)):
        sp = sps[s]
        rot = sp.get("teamRotationFip") or lg.era
        runs_saved = (rot * lg.ra_per_era - sp["estRa9"]) * sp["expIp"] / 9
        elo_adj[s] = {"runsSaved": runs_saved, "elo": M.prob_to_elo_diff(clip(0.5 + 0.1 * runs_saved, 0.05, 0.95)),
                      "rotation": rot}
    elo_a = T.elo.get(a_id, 1500) + elo_adj["away"]["elo"]
    elo_h = T.elo.get(h_id, 1500) + elo_adj["home"]["elo"] + (0 if neutral else T.ELO_HFA)
    p_elo = M.elo_expected(elo_h, elo_a)
    methods = {"log5": p_log5, "elo": p_elo, "lambda": p_home_lambda}
    p_tri = statistics.fmean(methods.values())
    spread = max(methods.values()) - min(methods.values())
    confidence = "alta" if spread <= 0.05 else "media" if spread <= 0.10 else "baja"

    # --- abridores
    score = {s: starter_scores(ctx, sps[s], offs[opp[s]], lineups[opp[s]], s == "home",
                               a_id if s == "home" else h_id) for s in SIDES}
    sc_a, sc_h = score["away"]["total"], score["home"]["total"]
    if sc_a is not None and sc_h is not None:
        d = sc_h - sc_a
        adv_side = "home" if d > 0 else "away"
        adv_level = "clara" if abs(d) >= 8 else "ligera" if abs(d) >= 3 else "equilibrado"
    else:
        d, adv_side, adv_level = None, None, "sin dato"

    # --- props de ponches
    kprops = {}
    for s in SIDES:
        sp = sps[s]
        if sp["missing"]:
            continue
        o = offs[opp[s]]
        ok = (lineups[opp[s]] or {}).get("k") or (o["vs"].get(sp.get("hand") or "R", {}).get("k") or o["k"])
        pk_ = sp["shrunk"]["k"]
        odds = (pk_ / (1 - pk_)) * (ok / (1 - ok)) / (lg.k_rate / (1 - lg.k_rate))
        pk_m = odds / (1 + odds)
        exp_k = pk_m * sp["expBf"]
        pmf = M.poisson_pmf(exp_k)
        base = math.floor(exp_k)
        lines = [base - 1.5, base - 0.5, base + 0.5, base + 1.5]
        kprops[s] = {"pitcher": sp["name"], "kRate": pk_m, "expBf": sp["expBf"], "exp": exp_k, "oppK": ok,
                     "lines": [{"line": ln, **{k: r2(v, 4) for k, v in M.over_under(pmf, ln).items()}}
                               for ln in lines if ln > 0]}

    # --- mercado
    odds = odds_for_game(ctx, g)
    proj = {s: lam[s]["full"] for s in SIDES}
    total_proj = proj["away"] + proj["home"]
    markets = market_table(ctx, g, p_tri, methods, margin, tie9, total_pmf, pmf_full, f5h, f5t, f5a, f3h, f3t, f3a,
                           f5_total, f3_total, nrfi, kprops, odds, lam)

    game = {
        "pk": g["pk"], "date": g["date"], "time": g["time"], "gameNumber": g.get("gameNumber"), "dh": g.get("dh"),
        "venue": {"id": g["venue"], "name": g.get("venueName"), "city": g.get("venueCity"), "roof": g.get("roof"),
                  "turf": g.get("turf"), "elevation": g.get("elevation"), "neutral": neutral,
                  "park": {"runs": park_idx.get("runs"), "hr": park_idx.get("hr"), "years": park_idx.get("years"),
                           "factor": park}},
        "teams": {s: team_card(ctx, tid, g, s) for s, tid in (("away", a_id), ("home", h_id))},
        "series": g.get("series"), "weather": weather, "umpire": ump,
        "officials": g.get("officials"),
    }
    sections = {}
    sections["s1"] = section1(ctx, ai, hi, neutral, g)
    sections["s2"] = section2(ctx, g, a_id, h_id, neutral)
    sections["s3"] = section3(ctx, g, sps, score, offs, lineups, a_id, h_id, d, adv_side, adv_level, kprops)
    sections["s4"] = section4(ctx, g, bps, a_id, h_id, offs, lineups)
    sections["s5"] = section5(ctx, g, sps, offs, lineups, bps, lam, lam_base, adj_rows, py, methods, p_tri, spread,
                              confidence, elo_adj, elo_a, elo_h, markets, score, adv_side, neutral, park_idx)
    sections["s6"] = section6(ctx, g, sps, offs, lineups, bps, lam, total_pmf, f3_total, f5_total, pmf_full, nrfi,
                              p1, first, weather, park_idx, kprops, markets, adj_rows)
    sections["s7"] = section7(ctx, g, odds, markets, p_tri, total_proj)
    sections["s9"] = section9(ctx, g, sps, lineups, bps, weather, ump, odds, park_idx)
    sections["s8"] = section8(ctx, g, sections, markets, sps, bps, lam, total_proj, odds, kprops, score, adv_side, offs,
                              lineups, weather, park_idx)
    sections["s10"] = section10(ctx, sections, methods, p_tri, confidence, lam, odds)

    best = sections["s8"]["best"]
    game["summary"] = {
        "pHome": p_tri, "pAway": 1 - p_tri, "confidence": confidence, "spread": spread,
        "proj": {"away": proj["away"], "home": proj["home"], "total": total_proj,
                 "f5": {"away": lam["away"]["f5"], "home": lam["home"]["f5"]},
                 "f3": {"away": lam["away"]["f3"], "home": lam["home"]["f3"]}},
        "nrfi": nrfi, "starterEdge": {"side": adv_side, "level": adv_level, "diff": d},
        "probables": {s: {"name": sps[s]["name"], "hand": sps[s].get("hand"), "era": sps[s].get("era"),
                          "fip": sps[s].get("fip"), "missing": sps[s]["missing"], "w": sps[s].get("w"),
                          "l": sps[s].get("l"), "id": sps[s].get("id")} for s in SIDES},
        "light": best["light"] if best else "Gris", "best": best, "gate": sections["s9"]["gate"],
        "lineupsConfirmed": {s: lineups[s] is not None for s in SIDES}, "hasOdds": odds is not None,
    }
    game["sections"] = sections
    return game


def team_card(ctx: Context, tid: int, g: dict, side: str) -> dict:
    info = ctx.T.info.get(tid, {})
    st = ctx.T.standings.get(tid, {})
    return {"id": tid, "abbr": info.get("abbr"), "name": info.get("name"), "club": info.get("club"),
            "league": info.get("league"), "division": info.get("division"),
            "w": st.get("w"), "l": st.get("l"), "streak": st.get("streak"), "divRank": st.get("divRank"),
            "gb": st.get("gb"), "clinch": st.get("clinch"), "elo": r2(ctx.T.elo.get(tid), 0)}


# ============================================================ tabla de probabilidades por mercado (5.5)

def market_table(ctx, g, p_tri, methods, margin, tie9, total_pmf, pmf_full, f5h, f5t, f5a, f3h, f3t, f3a,
                 f5_total, f3_total, nrfi, kprops, odds, lam):
    T = ctx.T
    ab, hb = T.abbr(g["away"]), T.abbr(g["home"])
    rows = []

    def add(market, key, pick, p, price=None, line=None, extra=None, push=0.0):
        implied = M.american_to_prob(price) if price is not None else None
        p_win = p / (1 - push) if push and push < 1 else p  # sin push para comparar contra el precio
        rows.append({"market": market, "key": key, "pick": pick, "p": p, "pNoPush": p_win, "push": push,
                     "line": line, "price": price, "implied": implied,
                     "edge": (p_win - implied) if implied is not None else None,
                     "fair": M.fair_american(p_win), "minPrice": M.fair_american(clip(p_win - EDGE_MIN, 0.01, 0.99)),
                     **(extra or {})})

    ml_price = {s: (odds["bestMl"][s][0] if odds and odds["bestMl"].get(s) else None) for s in SIDES}
    add("Moneyline", "ml", ab, 1 - p_tri, ml_price["away"], extra={"methods": {k: 1 - v for k, v in methods.items()}})
    add("Moneyline", "ml", hb, p_tri, ml_price["home"], extra={"methods": dict(methods)})
    p_h_rl = sum(p for m, p in margin.items() if m >= 2)
    p_a_rl = sum(p for m, p in margin.items() if m <= -2)
    rl_price = {s: (odds["bestRl"][s][0] if odds and odds["bestRl"].get(s) else None) for s in SIDES}
    rl_point = odds["rlPoint"] if odds else {"away": None, "home": None}
    if (rl_point.get("home") or -1.5) < 0:
        add("Run Line", "rl", f"{hb} -1.5", p_h_rl, rl_price["home"], -1.5)
        add("Run Line", "rl", f"{ab} +1.5", 1 - p_h_rl, rl_price["away"], 1.5)
    else:
        add("Run Line", "rl", f"{ab} -1.5", p_a_rl, rl_price["away"], -1.5)
        add("Run Line", "rl", f"{hb} +1.5", 1 - p_a_rl, rl_price["home"], 1.5)
    line = odds["totalLine"] if odds and odds.get("totalLine") else None
    for ln in sorted(set(TOTAL_LINES + ([line] if line else []))):
        ou = M.over_under(total_pmf, ln)
        is_mkt = ln == line
        add("Total completo", "total", f"Over {ln}", ou["over"],
            odds["bestTotal"]["over"][0] if is_mkt and odds["bestTotal"].get("over") else None, ln,
            {"market_line": is_mkt}, push=ou["push"])
        add("Total completo", "total", f"Under {ln}", ou["under"],
            odds["bestTotal"]["under"][0] if is_mkt and odds["bestTotal"].get("under") else None, ln,
            {"market_line": is_mkt}, push=ou["push"])
    add("F5 Moneyline", "f5ml", f"{ab} F5", f5a, extra={"tie": f5t, "noTie": f5a / (1 - f5t) if f5t < 1 else None},
        push=f5t)
    add("F5 Moneyline", "f5ml", f"{hb} F5", f5h, extra={"tie": f5t, "noTie": f5h / (1 - f5t) if f5t < 1 else None},
        push=f5t)
    add("F5 Run Line", "f5rl", f"{ab} F5 +0.5", f5a + f5t)
    add("F5 Run Line", "f5rl", f"{hb} F5 +0.5", f5h + f5t)
    for ln in (3.5, 4.5, 5.5):
        ou = M.over_under(f5_total, ln)
        add("F5 total", "f5total", f"F5 Over {ln}", ou["over"], line=ln)
        add("F5 total", "f5total", f"F5 Under {ln}", ou["under"], line=ln)
    add("F3 Moneyline", "f3ml", f"{ab} F3", f3a, extra={"tie": f3t}, push=f3t)
    add("F3 Moneyline", "f3ml", f"{hb} F3", f3h, extra={"tie": f3t}, push=f3t)
    for ln in (1.5, 2.5, 3.5):
        ou = M.over_under(f3_total, ln)
        add("F3 total", "f3total", f"F3 Over {ln}", ou["over"], line=ln)
        add("F3 total", "f3total", f"F3 Under {ln}", ou["under"], line=ln)
    for s, abbr in (("away", ab), ("home", hb)):
        for ln in (2.5, 3.5, 4.5, 5.5):
            ou = M.over_under(pmf_full[s], ln)
            add(f"Team total {'visitante' if s == 'away' else 'local'}", f"tt_{s}", f"{abbr} Over {ln}", ou["over"], line=ln)
            add(f"Team total {'visitante' if s == 'away' else 'local'}", f"tt_{s}", f"{abbr} Under {ln}", ou["under"], line=ln)
    add("NRFI/YRFI", "nrfi", "NRFI", nrfi)
    add("NRFI/YRFI", "nrfi", "YRFI", 1 - nrfi)
    for s, kp in kprops.items():
        for l in kp["lines"]:
            add("Props principales", f"k_{s}", f"{kp['pitcher']} Over {l['line']} K", l["over"], line=l["line"])
            add("Props principales", f"k_{s}", f"{kp['pitcher']} Under {l['line']} K", l["under"], line=l["line"])
    return rows


# ============================================================ secciones 1 y 2

def section1(ctx, ai, hi, neutral, g):
    same_league = ai.get("leagueId") == hi.get("leagueId")
    same_div = ai.get("divisionId") == hi.get("divisionId")
    if same_div:
        interp = ("Partido divisional: los equipos se conocen bien y el historial directo gana relevancia, "
                  "pero NO se asume automáticamente más carreras; conviene un análisis más conservador del total "
                  "y esperar un manejo más agresivo del bullpen.")
    elif same_league:
        interp = "Misma liga, distinta división: contexto competitivo normal; el historial directo es de muestra corta."
    else:
        interp = "Interliga: pocos enfrentamientos previos, el historial directo casi no aporta información."
    if neutral:
        interp += " Sede neutra: ningún split de casa/visita de temporada aplica a este partido."
    return {
        "rows": [
            {"q": "¿Ambos equipos pertenecen a la misma liga?", "a": "Sí" if same_league else "No",
             "d": f"{ai.get('league')} vs {hi.get('league')}"},
            {"q": "¿Ambos equipos pertenecen a la misma división?", "a": "Sí" if same_div else "No",
             "d": f"{ai.get('division')} / {hi.get('division')}"},
            {"q": "¿El partido es divisional?", "a": "Sí" if same_div else "No", "d": ""},
            {"q": "División del equipo visitante", "a": ai.get("division"), "d": ""},
            {"q": "División del equipo local", "a": hi.get("division"), "d": ""},
            {"q": "¿Se juega en sede neutra?", "a": "Sí" if neutral else "No",
             "d": f"{g.get('venueName')} ({g.get('venueCity')})"},
        ],
        "divisional": same_div, "sameLeague": same_league, "neutral": neutral, "interpretation": interp,
    }


def game_row(ctx, g, tid=None, current_venue=None):
    T = ctx.T
    winner = g["away"] if g["ar"] > g["hr"] else g["home"]
    row = {"date": g["date"], "away": T.abbr(g["away"]), "home": T.abbr(g["home"]), "venue": ctx.venue_name(g.get("venue")),
           "score": f"{g['ar']}-{g['hr']}", "winner": T.abbr(winner), "total": g["ar"] + g["hr"],
           "innings": len(g.get("inn") or []),
           "f5": (sum((x[0] or 0) for x in (g.get("inn") or [])[:5]), sum((x[1] or 0) for x in (g.get("inn") or [])[:5]))}
    if tid is not None:
        row["won"] = winner == tid
    if current_venue is not None:
        row["sameVenue"] = g.get("venue") == current_venue
    return row


def section2(ctx, g, a_id, h_id, neutral):
    T = ctx.T
    sa, sh = T.standings.get(a_id, {}), T.standings.get(h_id, {})

    def wl(x):
        return f"{x[0]}-{x[1]}" if x else "—"

    rec = []
    for s, tid, st in (("Visitante", a_id, sa), ("Local", h_id, sh)):
        sp = st.get("splits", {})
        rec.append({"team": T.abbr(tid), "role": s, "overall": f"{st.get('w')}-{st.get('l')}",
                    "home": wl(sp.get("home")), "away": wl(sp.get("away")), "last10": wl(sp.get("lastTen")),
                    "streak": st.get("streak"),
                    "comment": ("Rendimiento fuera de casa: " + wl(sp.get("away"))) if s == "Visitante"
                    else ("Rendimiento en casa: " + wl(sp.get("home")))})
    home_l10 = [game_row(ctx, x, h_id) for x in T.last_games(h_id, "home", 10)]
    away_l10 = [game_row(ctx, x, a_id) for x in T.last_games(a_id, "away", 10)]
    h2h = [game_row(ctx, x, h_id, g["venue"]) for x in T.last_games(h_id, None, 10, opp=a_id)]
    all_h2h = T.last_games(h_id, None, 60, opp=a_id)
    at_venue = [x for x in all_h2h if x.get("venue") == g["venue"]]

    def avg(rows, key_fn):
        return statistics.fmean(key_fn(r) for r in rows) if rows else None

    def rs(r, tid):
        return int(r["score"].split("-")[0 if r["away"] == T.abbr(tid) else 1])

    def ra(r, tid):
        return int(r["score"].split("-")[1 if r["away"] == T.abbr(tid) else 0])

    hw = sum(r["won"] for r in home_l10)
    aw = sum(r["won"] for r in away_l10)
    h2h_h = sum(1 for r in h2h if r["winner"] == T.abbr(h_id))
    venue_h = sum(1 for x in at_venue if (x["hr"] > x["ar"]) == (x["home"] == h_id))
    tot_avg = avg(home_l10 + away_l10, lambda r: r["total"])
    h2h_tot = avg(h2h, lambda r: r["total"])
    lg_tot = ctx.lg.rpg["away"] + ctx.lg.rpg["home"]
    lean_side = "Local" if hw - aw >= 3 else "Visitante" if aw - hw >= 3 else None
    ref_tot = statistics.fmean([x for x in (tot_avg, h2h_tot) if x is not None]) if (tot_avg or h2h_tot) else None
    lean_tot = None if ref_tot is None else "Over" if ref_tot >= lg_tot + 1 else "Under" if ref_tot <= lg_tot - 1 else None
    summary = [
        {"k": "Récord últimos 10 del local en casa", "v": f"{hw}-{len(home_l10) - hw}"},
        {"k": "Récord últimos 10 del visitante fuera", "v": f"{aw}-{len(away_l10) - aw}"},
        {"k": "Récord últimos 10 enfrentamientos directos",
         "v": f"{T.abbr(h_id)} {h2h_h} – {T.abbr(a_id)} {len(h2h) - h2h_h}" if h2h else "Sin enfrentamientos"},
        {"k": "Récord entre ambos en el estadio actual",
         "v": f"{T.abbr(h_id)} {venue_h}-{len(at_venue) - venue_h} ({len(at_venue)} juegos, 2 temporadas)" if at_venue else "—"},
        {"k": "Promedio de carreras anotadas por el local (L10 casa)", "v": r2(avg(home_l10, lambda r: rs(r, h_id)))},
        {"k": "Promedio de carreras permitidas por el local (L10 casa)", "v": r2(avg(home_l10, lambda r: ra(r, h_id)))},
        {"k": "Promedio de carreras anotadas por el visitante (L10 fuera)", "v": r2(avg(away_l10, lambda r: rs(r, a_id)))},
        {"k": "Promedio de carreras permitidas por el visitante (L10 fuera)", "v": r2(avg(away_l10, lambda r: ra(r, a_id)))},
        {"k": "Total promedio (L10 casa/fuera) · H2H · liga", "v": f"{r2(tot_avg)} · {r2(h2h_tot)} · {r2(lg_tot)}"},
        {"k": "Tendencia favorece", "v": " / ".join(x for x in (lean_side, lean_tot) if x) or "Sin tendencia clara"},
    ]
    warn = ("El historial directo es apoyo, no argumento suficiente para pick final: no debe dominar si cambiaron "
            "pitchers, lineups, bullpen o contexto de estadio.")
    if len(h2h) < 6:
        warn += f" Muestra H2H corta ({len(h2h)} juegos en 2 temporadas): baja confianza histórica."
    if neutral:
        warn += " Sede neutra: los splits de casa/visita no aplican."
    return {"records": rec, "homeL10": home_l10, "awayL10": away_l10, "h2h": h2h, "summary": summary, "warning": warn,
            "h2hCount": len(h2h)}


# ============================================================ sección 3: abridores

def section3(ctx, g, sps, score, offs, lineups, a_id, h_id, d, adv_side, adv_level, kprops):
    T, lg = ctx.T, ctx.lg
    teams = {"away": a_id, "home": h_id}
    ident = []
    comp = []
    blocks = {}
    for s in SIDES:
        sp = sps[s]
        tid = teams[s]
        opp_tid = teams["home" if s == "away" else "away"]
        ident.append({"side": s, "name": sp["name"], "team": T.abbr(tid),
                      "hand": {"R": "Derecho", "L": "Zurdo"}.get(sp.get("hand"), "—"),
                      "status": "Por anunciar" if sp["missing"] else "Confirmado (probable oficial MLB)",
                      "cond": "Visitante" if s == "away" else "Local"})
        if sp["missing"]:
            comp.append({"side": s, "name": sp["name"], "team": T.abbr(tid), "missing": True,
                         "estEra": r2(sp["estEra"]), "note": sp.get("note")})
            blocks[s] = None
            continue
        comp.append({
            "side": s, "name": sp["name"], "team": T.abbr(tid), "cond": "V" if s == "away" else "L",
            "hand": sp.get("hand"), "wl": f"{int(sp['w'])}-{int(sp['l'])}", "era": r2(sp["era"]), "whip": r2(sp["whip"]),
            "ip": r2(sp["ip"], 1), "gs": int(sp["gs"]), "k": pct(sp["k"]), "bb": pct(sp["bb"]), "kbb": pct(sp["kbb"]),
            "fip": r2(sp["fip"]), "xfip": r2(sp["xfip"]), "siera": None, "xera": r2(sp["xera"]), "hr9": r2(sp["hr9"]),
            "gb": pct(sp["gb"]), "barrel": sp.get("barrel"), "hardhit": sp.get("hardhit"), "bf": int(sp["bf"]),
            "shrunk": {k: (pct(v) if k in ("k", "bb", "kbb", "hr", "gb") else r2(v)) for k, v in sp["shrunk"].items()
                       if k != "weightCur"},
            "estEra": r2(sp["estEra"]), "estRa9": r2(sp["estRa9"]), "blend": sp["blend"], "expIp": r2(sp["expIp"], 1),
            "ipPerStart": r2(sp["ipPerStart"], 1),
        })
        cond_code = "a" if s == "away" else "h"
        cond_starts = [x for x in sp["starts"] if bool(x["home"]) == (s == "home")]
        vs_rival = [x for x in sp["prevStarts"] + sp["starts"] if x["opp"] == opp_tid][-3:]
        opp_sp = sps["home" if s == "away" else "away"]
        faced = None
        if not opp_sp["missing"]:
            mine = {x["pk"] for x in sp["starts"] + sp["prevStarts"]}
            theirs = {x["pk"]: x for x in opp_sp["starts"] + opp_sp["prevStarts"]}
            common = sorted((theirs[pk]["date"] for pk in mine & set(theirs)), reverse=True)
            faced = {"count": len(common), "dates": common[:3]}
        spl = sp["splits"].get(cond_code, {})
        team_record = [result_for(ctx, x["pk"], tid) for x in sp["starts"]]
        tw = sum(1 for x in team_record if x)
        tl = sum(1 for x in team_record if x is False)
        ks = kprops.get(s)
        k_value = ks and (ks["exp"] >= 6.0 or sp["shrunk"]["k"] >= 0.26)
        blocks[s] = {
            "season": {"era": r2(sp["era"]), "whip": r2(sp["whip"]), "fip": r2(sp["fip"]), "k": pct(sp["k"]),
                       "bb": pct(sp["bb"]), "ip": r2(sp["ip"], 1), "bf": int(sp["bf"])},
            "condition": {"label": "fuera de casa" if s == "away" else "en casa", "era": r2(spl.get("era")),
                          "whip": r2(spl.get("whip")), "fip": r2(spl.get("fip")), "k": pct(spl.get("k")),
                          "bb": pct(spl.get("bb")), "ip": r2(spl.get("ip"), 1), "bf": int(spl.get("bf") or 0),
                          "factor": r2(spl.get("factor"), 3)},
            "last5cond": [F.log_row(x) for x in reversed(cond_starts[-5:])],
            "last5": sp["last5"],
            "vsRival": [F.log_row(x) for x in reversed(vs_rival)],
            "facedOpp": faced,
            "teamRecordInStarts": f"{tw}-{tl}",
            "kValue": bool(k_value), "kProj": r2(ks["exp"], 1) if ks else None,
            "kNote": ("Tiene valor en ponches: revisar primero F5 Under o Under antes que Over." if k_value
                      else "Sin valor especial en ponches."),
            "bf": {"season": int(sp["bf"]), "cond": int(spl.get("bf") or 0),
                   "last5": int(sp.get("l5bf") or 0), "vsRival": int(sum(x["stat"].get("battersFaced", 0) for x in vs_rival))},
            "form": r2(sp["form"], 3), "l5fip": r2(sp.get("l5fip")),
        }
    for s in SIDES:
        for r in (blocks.get(s) or {}).get("last5cond", []) + (blocks.get(s) or {}).get("last5", []) + (blocks.get(s) or {}).get("vsRival", []):
            r["oppAbbr"] = T.abbr(r["opp"])
    real = [s for s in SIDES if not sps[s]["missing"]]

    def best(fn, low=True, label=None):
        if len(real) < 2:
            return None
        vals = {s: fn(sps[s]) for s in real}
        if any(v is None for v in vals.values()):
            return None
        s = min(vals, key=vals.get) if low else max(vals, key=vals.get)
        return {"side": s, "name": sps[s]["name"], "vals": {k: r2(v, 3) for k, v in vals.items()}, "label": label}

    interp = []
    if len(real) == 2:
        items = [
            ("Pitcher con mejor temporada", best(lambda p: p["shrunk"]["era"] * 0.5 + p["shrunk"]["fip"] * 0.5), "ERA/FIP regresados"),
            ("Pitcher que mejora por localía/visita", best(lambda p: (p["splits"].get("a" if p is sps["away"] else "h") or {}).get("factor", 1.0)), "factor de split (<1 mejora)"),
            ("Pitcher en mejor forma reciente (tras shrinkage)", best(lambda p: p["form"]), "factor de forma regresado"),
            ("Mayor riesgo de HR", best(lambda p: p["shrunk"]["hr"], low=False), "HR/BF regresado"),
            ("Mayor riesgo de bases por bolas", best(lambda p: p["shrunk"]["bb"], low=False), "BB% regresado"),
            ("Mayor riesgo de colapso temprano", best(lambda p: p["estRa9"] / max(p["expIp"], 1), low=False), "RA9 estimada / entradas esperadas"),
            ("Pitcher más confiable para F3", best(lambda p: p["estRa9"] + 3 * p["shrunk"]["bb"]), "RA9 estimada + control"),
            ("Pitcher más confiable para F5", best(lambda p: p["estRa9"] - 0.3 * p["expIp"]), "RA9 estimada y profundidad"),
        ]
        vs_best = None
        rival_era = {}
        for s in real:
            rows = (blocks[s] or {}).get("vsRival") or []
            ipv = sum(M.ip_to_float(x["ip"]) for x in rows)
            if ipv >= 5:
                rival_era[s] = 9 * sum(x["er"] for x in rows) / ipv
        if len(rival_era) == 2:
            s = min(rival_era, key=rival_era.get)
            vs_best = {"side": s, "name": sps[s]["name"], "vals": {k: r2(v) for k, v in rival_era.items()}}
        items.insert(3, ("Mejor historial contra el rival", vs_best, "ERA en los últimos 3 enfrentamientos (muestra chica)"))
        for label, b, basis in items:
            interp.append({"k": label, "v": b["name"] if b else "Sin dato suficiente", "basis": basis,
                           "vals": b["vals"] if b else None})
    k_side = [s for s in real if blocks[s] and blocks[s]["kValue"]]
    if k_side:
        interp.append({"k": "Valor en Ks", "v": ", ".join(sps[s]["name"] for s in k_side),
                       "basis": "Revisar primero F5 Under o Under antes que Over", "vals": None})
    if adv_side:
        who = sps[adv_side]["name"]
        team = T.abbr(teams[adv_side])
        adv_text = f"{team} / {who} — ventaja {adv_level} ({r2(abs(d), 1)} pts de 100)"
    else:
        adv_text = "Sin abridores confirmados: no se puede evaluar la ventaja."
    early = ("La ventaja se expresa mejor en F3/F5, NRFI/YRFI y props de ponches, donde el bullpen pesa menos."
             if adv_side and adv_level != "equilibrado" else
             "Abridores parejos: la ventaja de abridor no justifica por sí sola un mercado temprano.")
    return {"ident": ident, "comparison": comp, "weights": [{"f": n, "w": w, "what": wh} for n, w, wh in SCORE_WEIGHTS],
            "scores": {s: score[s] for s in SIDES}, "blocks": blocks, "interpretation": interp,
            "advantage": {"side": adv_side, "level": adv_level, "diff": r2(d, 1), "text": adv_text, "markets": early},
            "fipConstant": r2(ctx.lg.c_fip, 3), "lgEra": r2(ctx.lg.era)}


# ============================================================ sección 4: bullpen

def section4(ctx, g, bps, a_id, h_id, offs, lineups):
    T, lg = ctx.T, ctx.lg
    teams = {"away": a_id, "home": h_id}
    stats = []
    for s in SIDES:
        b = bps[s]
        stats.append({"side": s, "team": T.abbr(teams[s]), "era": r2(b["era"]), "whip": r2(b["whip"]), "fip": r2(b["fip"]),
                      "xfip": None, "k": pct(b["k"]), "bb": pct(b["bb"]), "kbb": pct(b["kbb"]), "hr9": r2(b["hr9"]),
                      "h9": r2(b["h9"]), "lob": pct(b["lob"]), "babip": r2(b["babip"], 3), "bs": int(b["bs"]),
                      "sv": int(b["sv"]), "hld": int(b["hld"]), "ip": r2(b["ip"], 1), "era7": r2(b["era7"]),
                      "ip7": r2(b["ip7"], 1), "fatigue": round(b["fatigue"]), "zFatigue": r2(b["zFatigue"]),
                      "risk": r2(b["risk"]), "riskParts": {k: r2(v) for k, v in b["riskParts"].items()},
                      "rank": b["rank"], "estEra": r2(b["estEra"])})

    def pick_team(fn, low=True):
        vals = {s: fn(bps[s]) for s in SIDES}
        if any(v is None for v in vals.values()):
            return "—"
        s = min(vals, key=vals.get) if low else max(vals, key=vals.get)
        return T.abbr(teams[s])

    misleading = []
    for s in SIDES:
        b = bps[s]
        if b["era"] is not None and b["fip"] is not None and b["fip"] - b["era"] >= 0.40:
            misleading.append(f"{T.abbr(teams[s])} (ERA {r2(b['era'])} vs FIP {r2(b['fip'])}: posible regresión negativa)")
    interp = [
        {"k": "Bullpen que permite más tráfico en bases", "v": pick_team(lambda b: b["whip"], low=False)},
        {"k": "Bullpen con mejor control", "v": pick_team(lambda b: b["bb"])},
        {"k": "Bullpen que poncha más", "v": pick_team(lambda b: b["k"], low=False)},
        {"k": "Bullpen que permite más HR", "v": pick_team(lambda b: b["hr9"], low=False)},
        {"k": "Bullpen con ERA posiblemente engañosa", "v": "; ".join(misleading) or "Ninguno (ERA y FIP alineados)"},
        {"k": "Bullpen con mayor riesgo de colapso tardío", "v": pick_team(lambda b: b["risk"], low=False)},
    ]
    both_strong = bps["away"]["rank"] <= 10 and bps["home"]["rank"] <= 10
    if both_strong:
        interp.append({"k": "Ambos bullpens fuertes", "v": "Sí: penalizar Over completo y Run Line del favorito"})
    rel = {}
    reading = []
    for s in SIDES:
        tid = teams[s]
        rows = F.relievers(tid, ctx.b, ctx.bp, ctx.today, lg)
        for r in rows:
            r["hand"] = ctx.hands.get(r["id"])
            opp = offs["home" if s == "away" else "away"]
            if r["hand"] in ("L", "R"):
                ratio = opp["vs"][r["hand"]]["ratio"]
                r["matchup"] = "Relevista" if ratio <= 0.97 else "Lineup" if ratio >= 1.03 else "Neutral"
                r["oppRatio"] = r2(ratio, 3)
            for k in ("era", "whip", "fip"):
                r[k] = r2(r[k])
            r["k"], r["bb"], r["ip"], r["ip3"] = pct(r["k"]), pct(r["bb"]), r2(r["ip"], 1), r2(r["ip3"], 1)
        rel[s] = rows
        closer = next((r for r in rows if r["role"] == "Cerrador"), None)
        ab = T.abbr(tid)
        if closer:
            if closer["consecutive"]:
                reading.append(f"{ab}: el cerrador {closer['name']} lanzó días consecutivos ({closer['available']}).")
            elif closer["available"] != "Sí":
                reading.append(f"{ab}: el cerrador {closer['name']} tiene carga reciente ({closer['p3']} pitcheos en 3 días).")
            if closer["idle7"]:
                reading.append(f"{ab}: el cerrador {closer['name']} no aparece en 7 días (posible lesión o descanso): verificar.")
        setups = [r for r in rows if r["role"] == "Setup" and r["available"] != "Sí"]
        if setups:
            reading.append(f"{ab}: setup con uso intenso — " + ", ".join(f"{r['name']} ({r['available']})" for r in setups) + ".")
        last = [bx for bx in ctx.b.get("boxscores", []) if tid in (bx["away"]["team"], bx["home"]["team"])]
        last.sort(key=lambda bx: bx["date"])
        if last:
            bx = last[-1]
            side_bx = "away" if bx["away"]["team"] == tid else "home"
            st = bx[side_bx]["pitchers"][0] if bx[side_bx]["pitchers"] else None
            if st and M.ip_to_float(st["ip"]) < 4.5:
                reading.append(f"{ab}: su abridor anterior ({st['name']}) salió temprano ({st['ip']} IP) el {bx['date']}: más carga al bullpen.")
        extras = [x for x in ctx.T.results if tid in (x["away"], x["home"]) and x.get("inn") and len(x["inn"]) > 9
                  and (ctx.today - dt.date.fromisoformat(x["date"])).days <= 3]
        if extras:
            reading.append(f"{ab}: jugó extra innings en los últimos 3 días ({extras[-1]['date']}).")
        if bps[s]["zFatigue"] >= 1.0:
            reading.append(f"{ab}: bullpen sobrecargado (fatiga z = {r2(bps[s]['zFatigue'])}, {round(bps[s]['fatigue'])} pitcheos ponderados en 3 días).")
    has_usage = all(any(tid in (bx["away"]["team"], bx["home"]["team"]) for bx in ctx.b.get("boxscores", []))
                    for tid in teams.values())
    if not has_usage:
        reading.append("Falta información de uso del bullpen de los últimos 3 días: el pick de juego completo no puede ser verde.")
    if not reading:
        reading.append("Sin señales de fatiga relevantes: brazos clave disponibles en ambos bullpens.")
    ra, rh = bps["away"]["risk"], bps["home"]["risk"]
    adv = "home" if rh < ra else "away"
    gap = abs(ra - rh)
    if gap < 1.0:
        concl = "Bullpens parejos: el juego completo se puede analizar sin castigo especial por relevo."
    else:
        weak = "away" if adv == "home" else "home"
        concl = (f"Ventaja de bullpen para {T.abbr(teams[adv])} (RiesgoBullpen {r2(bps[adv]['risk'])} vs "
                 f"{r2(bps[weak]['risk'])}). Si el pick depende de {T.abbr(teams[weak])} en juego completo, preferir F5.")
    if both_strong:
        concl += " Ambos bullpens top-10: buscar Under si los relevistas fuertes están descansados."
    return {"stats": stats, "interpretation": interp, "relievers": rel, "fatigueReading": reading,
            "advantage": {"side": adv if gap >= 1.0 else None, "team": T.abbr(teams[adv]) if gap >= 1.0 else None,
                          "gap": r2(gap)}, "conclusion": concl, "hasUsage": has_usage, "bothStrong": both_strong,
            "formula": "RiesgoBullpen = z(ERA) + z(WHIP) + z(BB%) + z(HR/9) + z(Fatiga)  (z entre los 30 bullpens)"}


# ============================================================ sección 5: modelo

def section5(ctx, g, sps, offs, lineups, bps, lam, lam_base, adj_rows, py, methods, p_tri, spread, confidence,
             elo_adj, elo_a, elo_h, markets, score, adv_side, neutral, park_idx):
    T, lg = ctx.T, ctx.lg
    ab, hb = T.abbr(g["away"]), T.abbr(g["home"])
    lam_tbl = {}
    for s in SIDES:
        c = lam[s]["components"]
        lam_tbl[s] = {
            "team": ab if s == "away" else hb,
            "offense": r2(c["offense"], 3), "handRatio": r2(c["handRatio"], 3), "hand": c["hand"],
            "lineup": r2(c["lineup"], 3), "park": r2(c["park"], 3), "starter": r2(c["starter"], 3), "form": r2(c["form"], 3),
            "split": r2(c["split"], 3), "bullpen": r2(c["bullpen"], 3), "fatigue": r2(c["fatigue"], 3),
            "f1": r2(lam[s]["f1"], 3), "f3": r2(lam[s]["f3"]), "f5": r2(lam[s]["f5"]), "g9": r2(lam[s]["g9"]),
            "full": r2(lam[s]["full"]), "vsStarter": r2(lam[s]["vsStarter"]), "vsBullpen": r2(lam[s]["vsBullpen"]),
            "base": {"f3": r2(lam_base[s]["f3"]), "f5": r2(lam_base[s]["f5"]), "full": r2(lam_base[s]["full"])},
            "lgFull": r2(lam[s]["lgFull"]), "perInning": [r2(x, 3) for x in lam[s]["per"][1:]],
            "starterFracs": [r2(x, 2) for x in lam[s]["starterFracs"][1:]],
        }
    sa, sh = score["away"]["total"], score["home"]["total"]
    factors = [
        {"f": "Pitcher abridor", "a": r2(sa, 0), "h": r2(sh, 0), "adv": T.abbr(g[adv_side]) if adv_side else "—",
         "m": "F3/F5/props"},
        {"f": "Pitágoras (win% verdadero)", "a": pct(py["away"]["true"]), "h": pct(py["home"]["true"]),
         "adv": ab if py["away"]["true"] > py["home"]["true"] else hb, "m": "ML"},
        {"f": "Elo (con ajuste de abridor)", "a": r2(elo_a, 0), "h": r2(elo_h, 0),
         "adv": ab if elo_a > elo_h else hb, "m": "ML"},
        {"f": "Ofensiva vs mano del abridor", "a": r2(offs["away"]["idx"] * lam["away"]["components"]["handRatio"], 3),
         "h": r2(offs["home"]["idx"] * lam["home"]["components"]["handRatio"], 3),
         "adv": ab if offs["away"]["idx"] * lam["away"]["components"]["handRatio"] > offs["home"]["idx"] * lam["home"]["components"]["handRatio"] else hb,
         "m": "Team total / F5"},
        {"f": "Bullpen (RiesgoBullpen, menor = mejor)", "a": r2(bps["away"]["risk"]), "h": r2(bps["home"]["risk"]),
         "adv": ab if bps["away"]["risk"] < bps["home"]["risk"] else hb, "m": "ML / total tardío"},
        {"f": "Carreras esperadas λ", "a": r2(lam["away"]["full"]), "h": r2(lam["home"]["full"]),
         "adv": ab if lam["away"]["full"] > lam["home"]["full"] else hb, "m": "ML/RL/Total"},
        {"f": "Estadio (índice carreras Savant, 3 años)", "a": park_idx.get("runs"), "h": park_idx.get("runs"),
         "adv": "Over" if (park_idx.get("runs") or 100) >= 105 else "Under" if (park_idx.get("runs") or 100) <= 95 else "Neutro",
         "m": "Total"},
    ]
    luck = []
    for s, tid in (("away", g["away"]), ("home", g["home"])):
        p = py[s]
        txt = (f"{T.abbr(tid)}: win% real {pct(p['wpct'])}% vs Pitagórico {pct(p['pyth'])}% (exponente {r2(p['exp'], 3)}). ")
        if p["luck"] >= 0.03:
            txt += "Récord inflado por suerte en juegos cerrados: no debe pesar tanto como sugiere la tabla."
        elif p["luck"] <= -0.03:
            txt += "Récord por debajo de su diferencial: el equipo es mejor de lo que dice su récord."
        else:
            txt += "Récord alineado con su diferencial de carreras."
        luck.append({"team": T.abbr(tid), "wpct": pct(p["wpct"]), "pyth": pct(p["pyth"]), "true": pct(p["true"]),
                     "prior": pct(p["prior"]), "g": p["g"], "rs": p["rs"], "ra": p["ra"], "text": txt})
    tri = {
        "log5": {"pHome": methods["log5"], "detail": f"Log5({pct(py['home']['true'])}%, {pct(py['away']['true'])}%)"
                 + ("" if neutral else f" + ventaja de local (odds × {r2(lg.home_odds_ratio, 3)}, win% local liga {pct(lg.home_win)}%)")},
        "elo": {"pHome": methods["elo"], "detail": f"Elo {hb} {r2(T.elo.get(g['home'], 1500), 0)}"
                f" {'+' if elo_adj['home']['elo'] >= 0 else ''}{r2(elo_adj['home']['elo'], 0)} abridor"
                + ("" if neutral else f" +{int(T.ELO_HFA)} local")
                + f" vs {ab} {r2(T.elo.get(g['away'], 1500), 0)} {'+' if elo_adj['away']['elo'] >= 0 else ''}{r2(elo_adj['away']['elo'], 0)} abridor",
                "starterAdj": {s: {"runsSaved": r2(v["runsSaved"]), "elo": r2(v["elo"], 0), "rotationFip": r2(v["rotation"])}
                               for s, v in elo_adj.items()}},
        "lambda": {"pHome": methods["lambda"], "detail": f"Binomial Negativa (Var/Media = {r2(lg.var_ratio)}) con λ {ab} "
                   f"{r2(lam['away']['g9'])} · {hb} {r2(lam['home']['g9'])} en 9 entradas + extra innings"},
        "pHome": p_tri, "spread": spread, "confidence": confidence,
        "rule": "≤5 pp entre métodos → confianza alta; >10 pp → confianza baja (divergencia, no edge real)",
    }
    probs = []
    for key in ("f3ml", "f3total", "f5ml", "f5rl", "f5total", "ml", "rl", "total", "tt_away", "tt_home", "nrfi"):
        rows = [m for m in markets if m["key"] == key]
        if key in ("total",):
            rows = [m for m in rows if m["line"] in (8.5,) or m.get("market_line")]
        if key in ("f5total", "f3total"):
            rows = [m for m in rows if m["line"] in (4.5, 2.5)]
        if key.startswith("tt_"):
            rows = [m for m in rows if m["line"] in (3.5, 4.5)]
        for m in rows:
            probs.append({"market": m["market"], "pick": m["pick"], "p": m["p"], "push": m.get("push") or 0,
                          "tie": m.get("tie")})
    for s in SIDES:
        rows = [m for m in markets if m["key"] == f"k_{s}"]
        if rows:
            mid = rows[len(rows) // 2 - 1: len(rows) // 2 + 1]
            for m in mid:
                probs.append({"market": m["market"], "pick": m["pick"], "p": m["p"], "push": 0})
    cancel = cancel_reasons(ctx, g, sps, bps, lineups, adj_rows, score, adv_side, lam, py)
    # 5.1 Ventaja Abridor-Lineup = Calidad del abridor − Amenaza ofensiva rival
    svl = []
    for s in SIDES:
        sp = sps[s]
        o = "home" if s == "away" else "away"
        threat = lam[o]["components"]["offense"] * lam[o]["components"]["handRatio"] * lam[o]["components"]["lineup"]
        quality = -(sp["estEra"] - lg.era) / 0.5
        threat_z = (threat - 1) / 0.08
        diff = quality - threat_z
        cls = ("ventaja clara del pitcher" if diff >= 1 else "ventaja ligera del pitcher" if diff >= 0.35 else
               "equilibrado" if diff > -0.35 else "ventaja ligera del lineup" if diff > -1 else "ventaja clara del lineup")
        svl.append({"pitcher": sp["name"], "lineup": T.abbr(g[o]), "quality": r2(quality), "threat": r2(threat_z),
                    "diff": r2(diff), "class": cls, "estEra": r2(sp["estEra"]), "threatIdx": r2(threat, 3),
                    "missing": sp["missing"]})
    lineup_view = {}
    for s in SIDES:
        lu = lineups[s]
        lineup_view[s] = None if not lu else {
            "rows": lu["rows"], "obp": r2(lu["obp"], 3), "slg": r2(lu["slg"], 3), "k": pct(lu["k"]), "bb": pct(lu["bb"]),
            "barrel": r2(lu["barrel"], 1), "ratio": r2(lu["ratio"], 3), "lefties": lu["lefties"]}
    shrink_notes = []
    for s in SIDES:
        sp = sps[s]
        if sp["missing"]:
            continue
        shrink_notes.append(
            f"{sp['name']}: K% {pct(sp['k'])}% → {pct(sp['shrunk']['k'])}%, BB% {pct(sp['bb'])}% → {pct(sp['shrunk']['bb'])}%, "
            f"FIP {r2(sp['fip'])} → {r2(sp['shrunk']['fip'])}, ERA {r2(sp['era'])} → {r2(sp['shrunk']['era'])} "
            f"({int(sp['bf'])} BF esta temporada + {int(sp['prevBf'])} BF de la anterior al 50%).")
    return {
        "factors": factors, "lambda": lam_tbl, "adjustments": adj_rows, "triangulation": tri, "luck": luck,
        "starterVsLineup": svl, "lineups": lineup_view,
        "probabilities": probs, "cancel": cancel, "shrinkage": shrink_notes,
        "league": {"era": r2(lg.era), "ra9": r2(lg.ra9), "cFip": r2(lg.c_fip, 3), "varRatio": r2(lg.var_ratio, 3),
                   "homeWin": pct(lg.home_win), "rpgAway": r2(lg.rpg["away"]), "rpgHome": r2(lg.rpg["home"]),
                   "extraShare": pct(lg.extra_share), "extraHomeWin": pct(lg.extra_home_win),
                   "pScoreless1": pct(lg.p_scoreless_half1), "games": lg.n_games},
        "correlation": [
            "F5 Under y Under completo comparten la misma causa (abridores dominantes): no multiplicar sus probabilidades.",
            "ML del favorito y Over de carreras permitidas del abridor rival dependen del mismo colapso.",
            "NRFI y Over de ponches del abridor comparten el supuesto de un primer episodio dominante.",
        ],
    }


def cancel_reasons(ctx, g, sps, bps, lineups, adj_rows, score, adv_side, lam, py):
    T = ctx.T
    out = []

    def add(cond, text):
        out.append({"on": bool(cond), "text": text})

    if adv_side:
        opp_side = "home" if adv_side == "away" else "away"
        add(bps[adv_side]["risk"] > bps[opp_side]["risk"] + 1.0,
            f"Abridor superior ({sps[adv_side]['name']}), pero bullpen débil detrás")
    add(adv_side and bps[adv_side]["rank"] > 20, "Buen F5, pero mal juego completo")
    add(False, "Buen historial, pero muestra pequeña (el H2H no entra al modelo)")
    for s in SIDES:
        sp = sps[s]
        if not sp["missing"] and sp["era"] is not None and sp["fip"] is not None:
            add(sp["era"] < sp["fip"] - 0.6 or (sp.get("xera") and sp["era"] < sp["xera"] - 0.6),
                f"Buena ERA de {sp['name']} ({r2(sp['era'])}), pero FIP/xERA malos ({r2(sp['fip'])}/{r2(sp.get('xera'))})")
    add(any(lineups[s] is None for s in SIDES), "Lineup proyectado, no confirmado")
    add(any(bps[s]["zFatigue"] >= 1.0 for s in SIDES), "Bullpen cansado")
    add(False, "Closer no disponible (ver 4.3)")
    add(False, "Umpire contrario (sin datos de tendencia)")
    wind = next((r for r in adj_rows if r["cond"] == "Viento fuerte hacia afuera"), None)
    add(wind and wind["applies"], "Clima contrario (viento hacia afuera ≥10 mph)")
    for s in SIDES:
        add(py[s]["luck"] >= 0.04, f"Equipo {T.abbr(g[s])} inflado (win% {pct(py[s]['wpct'])}% vs Pitágoras {pct(py[s]['pyth'])}%)")
    add(any(r["cond"].startswith("Rival con alto K%") and r["applies"] for r in adj_rows) or
        any(r["cond"].startswith("Ambos abridores") and r["applies"] for r in adj_rows), "Pick de Over contradicho por alto K%")
    total = lam["away"]["full"] + lam["home"]["full"]
    add(total < 8.0, f"Run Line favorito contradicho por total bajo ({r2(total)})")
    return out


# ============================================================ sección 6: Over/Under

def section6(ctx, g, sps, offs, lineups, bps, lam, total_pmf, f3_total, f5_total, pmf_full, nrfi, p1, first, weather,
             park_idx, kprops, markets, adj_rows):
    T, lg = ctx.T, ctx.lg
    ab, hb = T.abbr(g["away"]), T.abbr(g["home"])
    lg_total = lg.rpg["away"] + lg.rpg["home"]
    total = lam["away"]["full"] + lam["home"]["full"]

    def ou_lean(cond_over, cond_under):
        return "Over" if cond_over else "Under" if cond_under else "Neutro"

    base = []
    for s in SIDES:
        sp = sps[s]
        name = f"Abridor {'visitante' if s == 'away' else 'local'}: {sp['name']}"
        if sp["missing"]:
            base.append({"f": name, "lean": "Neutro", "c": "Por anunciar"})
            continue
        base.append({"f": name, "lean": ou_lean(sp["estEra"] >= lg.era + 0.35, sp["estEra"] <= lg.era - 0.35),
                     "c": f"ERA est. {r2(sp['estEra'])} (liga {r2(lg.era)}), K-BB% {pct(sp['shrunk']['kbb'])}%"})
    for s in SIDES:
        o = offs[s]
        lu = lineups[s]
        base.append({"f": f"Ofensiva {T.abbr(g[s])} ({'confirmado' if lu else 'proyectado'})",
                     "lean": ou_lean(o["idx"] * lam[s]["components"]["handRatio"] >= 1.04,
                                     o["idx"] * lam[s]["components"]["handRatio"] <= 0.96),
                     "c": f"Índice {r2(o['idx'], 3)} × vs mano {r2(lam[s]['components']['handRatio'], 3)}; OBP {r2(o['obp'], 3)}, ISO {r2(o['iso'], 3)}"})
    for s in SIDES:
        b = bps[s]
        base.append({"f": f"Bullpen {T.abbr(g[s])} (temporada / 7 días)", "lean": ou_lean(b["rank"] >= 21, b["rank"] <= 10),
                     "c": f"ERA {r2(b['era'])} / {r2(b['era7'])} últimos 7 días, rank {b['rank']}/30"})
    runs_idx = park_idx.get("runs")
    base.append({"f": "Estadio", "lean": ou_lean((runs_idx or 100) >= 105, (runs_idx or 100) <= 95),
                 "c": f"Índice carreras {runs_idx}, HR {park_idx.get('hr')} ({park_idx.get('years')})"})
    base.append({"f": "Clima", "lean": ou_lean(weather["out"] and (weather["speed"] or 0) >= 10,
                                               weather["in"] and (weather["speed"] or 0) >= 10),
                 "c": ("Techo cerrado / domo" if weather["closed"] else
                       f"{weather['condition'] or 'Sin reporte'}, {weather['temp'] or '—'}°F, viento {weather['windText'] or '—'}")})
    base.append({"f": "Umpire", "lean": "Neutro", "c": f"{g.get('officials') and next((o['name'] for o in g['officials'] if o['type'] == 'Home Plate'), None) or 'No asignado aún'} (sin datos de zona)"})
    kp = [kprops[s]["exp"] for s in kprops]
    base.append({"f": "Proyección de ponches", "lean": ou_lean(False, sum(kp) >= 12),
                 "c": f"{r2(sum(kp), 1)} K proyectados de los abridores" + (" — cuidado con Over" if sum(kp) >= 12 else "")})

    # filtro de ponches (6.3)
    kf = []
    for s in SIDES:
        sp = sps[s]
        kf.append({"c": f"Abridor con alto K% ({sp['name']})", "on": (not sp["missing"]) and sp["shrunk"]["k"] >= 0.26,
                   "v": pct(sp["shrunk"]["k"])})
    for s in SIDES:
        k_rate = (lineups[s] or {}).get("k") or offs[s]["k"]
        kf.append({"c": f"Rival con alto K% ({T.abbr(g[s])})", "on": k_rate >= ctx.k_high, "v": pct(k_rate)})
    kf.append({"c": "Umpire de zona amplia", "on": False, "v": "sin dato"})
    for s in SIDES:
        bb = (lineups[s] or {}).get("bb") or offs[s]["bb"]
        kf.append({"c": f"Lineup con bajo BB% ({T.abbr(g[s])})", "on": bb <= lg.bat_bb_rate - 0.01, "v": pct(bb)})
    kf.append({"c": "Parque pitcher-friendly", "on": (runs_idx or 100) <= 95, "v": runs_idx})
    k_count = sum(1 for x in kf if x["on"])
    park_ok = [
        {"c": "Contacto fuerte (Barrel% de ambos lineups ≥ liga)",
         "on": all(((lineups[s] or {}).get("barrel") or offs[s].get("barrel") or 0) >= ctx.lg_barrel for s in SIDES)},
        {"c": "HR/9 alto de algún abridor (≥1.35)", "on": any((sps[s].get("hr9") or 0) >= 1.35 for s in SIDES)},
        {"c": "Viento favorable (hacia afuera ≥10 mph)", "on": weather["out"] and (weather["speed"] or 0) >= 10},
        {"c": "Lineups confirmados", "on": all(lineups[s] is not None for s in SIDES)},
    ]
    park_text = (f"Índice de carreras {runs_idx}: " +
                 ("parque ofensivo, pero solo sube el total si se cumplen las 4 condiciones." if (runs_idx or 100) >= 105 else
                  "parque que favorece a los pitchers." if (runs_idx or 100) <= 95 else "parque neutro."))
    lines = []
    for m in markets:
        if m["key"] in ("total", "f3total", "f5total") or m["key"].startswith("tt_"):
            lines.append(m)
    table = []
    for key, label in (("f3total", "F3 total"), ("f5total", "F5 total"), ("total", "Total completo"),
                       ("tt_away", f"Team total {ab}"), ("tt_home", f"Team total {hb}")):
        by_line = defaultdict(dict)
        for m in lines:
            if m["key"] == key:
                by_line[m["line"]]["over" if "Over" in m["pick"] else "under"] = m
        for ln in sorted(by_line):
            o, u = by_line[ln].get("over"), by_line[ln].get("under")
            po, pu = o["pNoPush"], u["pNoPush"]
            if po >= 0.56:
                read, dec = f"Over {ln} con {pct(po)}%", "Lean Over"
            elif pu >= 0.56:
                read, dec = f"Under {ln} con {pct(pu)}%", "Lean Under"
            else:
                read, dec = "Línea ajustada", "Sin ventaja"
            table.append({"market": label, "line": ln, "over": o["p"], "under": u["p"], "push": o.get("push") or 0,
                          "read": read, "decision": dec, "fairOver": o["fair"], "fairUnder": u["fair"],
                          "marketLine": bool(o.get("market_line"))})

    def best_line(side):
        rows = [t for t in table if t["market"] == "Total completo"]
        if side == "over":
            ok = [t for t in rows if t["over"] / (1 - t["push"]) >= 0.56]
            return max((t["line"] for t in ok), default=None)
        ok = [t for t in rows if t["under"] / (1 - t["push"]) >= 0.56]
        return min((t["line"] for t in ok), default=None)

    bo, bu = best_line("over"), best_line("under")
    contr = []

    def c(on, text):
        contr.append({"on": bool(on), "text": text})

    sp_under = all((not sps[s]["missing"]) and sps[s]["estEra"] <= lg.era - 0.2 for s in SIDES)
    bp_over = any(bps[s]["rank"] >= 21 for s in SIDES)
    c(sp_under and bp_over, "Abridores favorecen Under, pero bullpen favorece Over")
    c(any(lineups[s] is None for s in SIDES), "Abridor vulnerable, pero lineup rival no confirmado")
    c(weather["in"] and (weather["speed"] or 0) >= 10, "Over depende de HR, pero el clima favorece pitchers (viento hacia adentro)")
    c(any(bps[s]["zFatigue"] >= 1.0 for s in SIDES), "Under depende de bullpen, pero relevistas están cansados")
    c(False, "Umpire de zona pequeña/amplia (sin datos)")
    c((park_idx.get("hr") or 100) <= 95, "El estadio reduce HR")
    c(k_count >= 2, "Los props de Ks contradicen el Over")
    c(bps["away"]["rank"] <= 10 and bps["home"]["rank"] <= 10, "Ambos bullpens están descansados y son fuertes")
    lead = max(abs(lam["away"]["full"] - lam["home"]["full"]), 0)
    c(lead >= 1.2 and total < 8.5, "El favorito puede ganar sin necesidad de muchas carreras")
    over_block = k_count >= 3
    development = {
        "f3": {"away": r2(lam["away"]["f3"], 1), "home": r2(lam["home"]["f3"], 1)},
        "f5": {"away": r2(lam["away"]["f5"], 1), "home": r2(lam["home"]["f5"], 1)},
        "final": {"away": r2(lam["away"]["full"], 1), "home": r2(lam["home"]["full"], 1)},
        "firstToScore": {"away": r2(first["away"], 3), "home": r2(first["home"], 3),
                         "none": r2(1 - first["away"] - first["home"], 3)},
        "vsStarter": {s: r2(lam[s]["vsStarter"]) for s in SIDES}, "vsBullpen": {s: r2(lam[s]["vsBullpen"]) for s in SIDES},
        "perInning": {s: [r2(x, 2) for x in lam[s]["per"][1:]] for s in SIDES},
    }
    share_f5 = (lam["away"]["f5"] + lam["home"]["f5"]) / total if total else 0
    dev_notes = [
        f"Debería anotar primero: {ab if first['away'] > first['home'] else hb} "
        f"({pct(max(first['away'], first['home']))}% vs {pct(min(first['away'], first['home']))}%).",
        f"El {pct(share_f5, 0)}% de las carreras esperadas llega en las primeras 5 entradas "
        f"({'el total depende más de F5' if share_f5 >= 0.58 else 'las carreras tardías pesan'}).",
    ]
    for s in SIDES:
        o = "home" if s == "away" else "away"
        dmg = "el abridor" if lam[s]["vsStarter"] >= lam[s]["vsBullpen"] else "el bullpen"
        dev_notes.append(f"{T.abbr(g[s])}: el daño probable viene contra {dmg} de {T.abbr(g[o])} "
                         f"({r2(lam[s]['vsStarter'])} vs abridor, {r2(lam[s]['vsBullpen'])} vs bullpen).")
    pmf_view = [r2(x, 4) for x in total_pmf[:21]]
    near = min((abs(total - ln) for ln in TOTAL_LINES))
    classification = ("No bet" if over_block and bo else "Esperar información" if any(lineups[s] is None for s in SIDES)
                      else "Lean" if (bo or bu) else "No bet")
    return {
        "base": base, "kFilter": kf, "kCount": k_count, "overBlocked": over_block, "parkChecks": park_ok,
        "parkText": park_text, "lineTable": table, "contradictions": contr, "development": development,
        "devNotes": dev_notes, "totalPmf": pmf_view, "nrfi": {"p": nrfi, "yrfi": 1 - nrfi,
                                                             "pAway1": p1["away"], "pHome1": p1["home"],
                                                             "lgScoreless": lg.p_scoreless_half1},
        "segments": {"f3": r2(lam["away"]["f3"] + lam["home"]["f3"]), "f5": r2(lam["away"]["f5"] + lam["home"]["f5"]),
                     "full": r2(total), "lg": r2(lg_total)},
        "conclusion": {
            "bestOver": f"Over {bo}" if bo and not over_block else ("Bloqueado por filtro de ponches" if bo else "Ninguna línea"),
            "bestUnder": f"Under {bu}" if bu else "Ninguna línea",
            "noValue": [t["line"] for t in table if t["market"] == "Total completo" and t["decision"] == "Sin ventaja"],
            "market": ("F5 total" if share_f5 >= 0.6 else "Team total" if abs(lam['away']['full'] - lam['home']['full']) >= 1.2
                       else "Total completo"),
            "missing": [x for x, on in (("lineups confirmados", any(lineups[s] is None for s in SIDES)),
                                        ("umpire asignado", not any(o["type"] == "Home Plate" for o in g.get("officials", []))),
                                        ("clima del día", not weather["available"] and not weather["closed"]),
                                        ("momios (línea real)", True)) if on],
            "classification": classification,
            "nearLine": near < 0.3,
            "rule": "Si el total proyectado cae muy cerca de la línea, no forzar pick: buscar F5 total, team total o props.",
        },
        "kprops": {s: {"pitcher": v["pitcher"], "exp": r2(v["exp"], 2), "kRate": pct(v["kRate"]), "expBf": r2(v["expBf"], 1),
                       "oppK": pct(v["oppK"]), "lines": v["lines"]} for s, v in kprops.items()},
    }


# ============================================================ sección 7: cuotas y valor

def section7(ctx, g, odds, markets, p_tri, total_proj):
    T = ctx.T
    ab, hb = T.abbr(g["away"]), T.abbr(g["home"])
    books = []
    if odds:
        for b in odds["books"]:
            books.append({"book": b["book"], "mlAway": b["ml"].get("away"), "mlHome": b["ml"].get("home"),
                          "rlAway": b["rl"].get("away"), "rlHome": b["rl"].get("home"),
                          "over": b["total"].get("over"), "under": b["total"].get("under")})
    priced = [m for m in markets if m["price"] is not None]
    value = sorted(priced, key=lambda m: m["edge"], reverse=True)
    fav_side = "home" if p_tri >= 0.5 else "away"
    fav_price = None
    if odds and odds["bestMl"].get(fav_side):
        fav_price = odds["bestMl"][fav_side][0]
    expensive = fav_price is not None and fav_price <= -170
    low_total = total_proj < 8.0
    fair = []
    for key in ("ml", "rl", "f5ml", "total", "nrfi"):
        for m in markets:
            if m["key"] != key:
                continue
            if key == "total" and m["line"] not in (8.5,) and not m.get("market_line"):
                continue
            fair.append({"market": m["market"], "pick": m["pick"], "p": m["pNoPush"], "fair": m["fair"],
                         "minPrice": m["minPrice"], "price": m["price"], "edge": m["edge"]})
    return {
        "hasOdds": odds is not None, "nBooks": odds["nBooks"] if odds else 0, "books": books,
        "value": value[:12], "fair": fair, "edgeMin": EDGE_MIN,
        "expensiveFav": {"on": expensive and low_total, "favPrice": fav_price, "lowTotal": low_total,
                         "text": "Favorito caro + total bajo = buscar Under, F5 Under, props de pitcher o no bet"},
        "questions": [
            {"q": "¿El ML está inflado?", "a": "Sí" if expensive else "No" if fav_price is not None else "Sin momio",
             "adj": "Evitar ML · bajar confianza"},
            {"q": "¿El total proyectado es bajo?", "a": "Sí" if low_total else "No", "adj": "Evitar Run Line · revisar Under o F5"},
        ],
        "note": (None if odds else
                 "Sin momios conectados (agrega el secreto ODDS_API_KEY de The Odds API): sin edge calculable → no bet. "
                 "Usa el momio justo y el momio mínimo aceptable para comparar con tu casa de apuestas."),
    }


# ============================================================ sección 9: completitud

def section9(ctx, g, sps, lineups, bps, weather, ump, odds, park_idx):
    T = ctx.T
    h2h = len(T.last_games(g["home"], None, 10, opp=g["away"]))
    fields = []

    def add(sec, field, status, source, blocks):
        fields.append({"sec": sec, "field": field, "status": status, "source": source, "blocks": blocks})

    add("1-2", "División de ambos equipos", "disponible", "MLB Stats API /teams", "Baja confianza histórica")
    add("1-2", "Récord L10 local en casa / visitante fuera", "disponible", "MLB Stats API /schedule (resultados)",
        "Baja confianza histórica")
    add("1-2", "H2H últimos 10", "disponible" if h2h >= 6 else "parcial" if h2h else "faltante",
        f"MLB Stats API (2 temporadas: {h2h} juegos)", "Baja confianza histórica")
    for s in SIDES:
        sp = sps[s]
        who = f"abridor {'visitante' if s == 'away' else 'local'}"
        if sp["missing"]:
            add("3", f"Abridor probable ({who})", "faltante", "MLB Stats API probablePitcher", "Bloquea Verde en ML/F5/props")
            continue
        add("3", f"ERA, WHIP, K-BB% ({sp['name']})", "disponible", "MLB Stats API /people stats", "Bloquea Verde en ML/F5/props")
        add("3", f"FIP ({sp['name']})", "derivado", f"Cálculo propio con constante de liga {r2(ctx.lg.c_fip, 3)}",
            "Bloquea Verde en ML/F5/props")
        add("3", f"xERA ({sp['name']})", "disponible" if sp.get("xera") is not None else "faltante",
            "Baseball Savant (expected stats)", "Bloquea Verde en ML/F5/props")
        add("3", f"xFIP / SIERA ({sp['name']})", "sustituido",
            "xFIP aproximado con airOuts; SIERA (FanGraphs) sustituido por FIP según 3.4", "—")
        add("3", f"Últimas 5 aperturas ({sp['name']})", "disponible" if len(sp["last5"]) >= 5 else "parcial",
            "MLB Stats API gameLog", "Bloquea Verde en ML/F5/props")
        add("3", f"Split casa/visita ({sp['name']})", "disponible" if sp["splits"] else "faltante",
            "MLB Stats API statSplits h/a", "Bloquea Verde en ML/F5/props")
    add("4", "ERA/FIP del bullpen", "disponible", "MLB Stats API teams/stats (split relievers)", "Bloquea Verde en total y ML tardío")
    has_usage = all(any(tid in (bx["away"]["team"], bx["home"]["team"]) for bx in ctx.b.get("boxscores", []))
                    for tid in (g["away"], g["home"]))
    add("4", "Uso del bullpen últimos 3 días", "derivado" if has_usage else "faltante",
        "Box scores MLB Stats API (últimos 7 días)", "Bloquea Verde en total y ML tardío")
    add("4", "Disponibilidad del cerrador", "derivado" if has_usage else "faltante",
        "Box scores + líder de salvamentos del equipo", "Bloquea Verde en total y ML tardío")
    add("5", "Park factor", "disponible" if park_idx else "faltante", "Baseball Savant (índice 3 años)", "Bloquea Verde en total")
    add("5", "Ajustes multiplicativos con dato real", "derivado", "Cálculo propio (tabla 5.3)", "Bloquea Verde en total")
    for s in SIDES:
        add("6", f"Lineup confirmado ({T.abbr(g[s])})", "disponible" if lineups[s] else "faltante",
            "MLB Stats API lineups", "Bloquea Verde en total y NRFI/YRFI")
    add("6", "Clima/viento", "no aplica" if weather["closed"] else "disponible" if weather["available"] else "faltante",
        "MLB Stats API weather (reporte del estadio)", "Bloquea Verde en total y NRFI/YRFI")
    add("6", "Umpire asignado", "disponible" if ump else "faltante", "MLB Stats API officials",
        "Bloquea Verde en total y NRFI/YRFI")
    add("7", "Momios de al menos 2 casas", "disponible" if odds and odds["nBooks"] >= 2 else "faltante",
        "The Odds API (ODDS_API_KEY)", "Sin edge calculable → no bet")
    missing = [f for f in fields if f["status"] == "faltante"]

    def blocked(tag):
        return any(tag in f["blocks"] for f in missing)

    gate = {
        "missing": len(missing), "total": len(fields),
        "ok": sum(1 for f in fields if f["status"] in ("disponible", "derivado", "no aplica", "sustituido")),
        "blocks": {
            "ml": blocked("ML/F5/props") or blocked("ML tardío") or blocked("no bet"),
            "f5": blocked("ML/F5/props") or blocked("no bet"),
            "props": blocked("ML/F5/props") or blocked("no bet"),
            "total": blocked("total") or blocked("no bet"),
            "nrfi": blocked("NRFI") or blocked("ML/F5/props") or blocked("no bet"),
        },
        "campoFaltante": bool(missing),
    }
    return {"fields": fields, "gate": gate,
            "rule": "Si falta un campo obligatorio, el mercado afectado nunca puede ser Verde (máximo Amarillo)."}


# ============================================================ sección 8: auditoría y decisión

def section8(ctx, g, S, markets, sps, bps, lam, total_proj, odds, kprops, score, adv_side, offs, lineups, weather,
             park_idx):
    T, lg = ctx.T, ctx.lg
    ab, hb = T.abbr(g["away"]), T.abbr(g["home"])
    gate = S["s9"]["gate"]
    k_count = S["s6"]["kCount"]
    both_strong = S["s4"]["bothStrong"]
    p_home = S["s5"]["triangulation"]["pHome"]
    conf = S["s5"]["triangulation"]["confidence"]
    fav = "home" if p_home >= 0.5 else "away"
    dog = "away" if fav == "home" else "home"
    k_heavy = any(kp["exp"] >= 7 for kp in kprops.values())
    missing_info = any(lineups[s] is None for s in SIDES) or not S["s4"]["hasUsage"] or not g.get("officials")

    def mk(key, pick):
        return next((m for m in markets if m["key"] == key and m["pick"] == pick), None)

    def best_of(key, cond=None):
        rows = [m for m in markets if m["key"] == key and (cond is None or cond(m))]
        if not rows:
            return None
        priced = [m for m in rows if m["edge"] is not None]
        if priced:
            return max(priced, key=lambda m: m["edge"])
        return max(rows, key=lambda m: m["pNoPush"])

    decisions = []

    def decide(label, m, guion, guion_why, contra, block_key, assumption):
        if m is None:
            decisions.append({"market": label, "pick": "—", "modelo": "No", "guion": "No", "cuota": "No",
                              "contradiction": "Alta", "light": "Rojo", "decision": "No bet", "why": "Sin dato",
                              "assumption": assumption})
            return
        p = m["pNoPush"]
        methods = m.get("methods")
        if m["edge"] is not None:
            modelo = m["edge"] > 0 and (not methods or all(
                (v - m["implied"]) > 0 for v in methods.values()))
        elif label == "Run Line":
            modelo = p >= 0.5
        else:
            modelo = p >= MODEL_MIN and conf != "baja" if methods else p >= MODEL_MIN + 0.01
        cuota = m["edge"] is not None and m["edge"] >= EDGE_MIN
        n_contra = sum(1 for x in contra if x)
        level = "Alta" if n_contra >= 2 else "Media" if n_contra == 1 else "Baja"
        blocked = gate["blocks"].get(block_key, False)
        if not modelo or level == "Alta":
            light, dec = "Rojo", "No bet"
        elif m["price"] is None:
            light, dec = "Gris", "Esperar cuota"
        elif cuota and guion and level == "Baja":
            light, dec = "Verde", "Pick"
        elif cuota and (guion or level == "Media"):
            light, dec = "Amarillo", "Lean"
        else:
            light, dec = "Rojo", "No bet"
        if blocked and light == "Verde":
            light, dec = "Amarillo", "Lean"
        stake = None
        if light in ("Verde", "Amarillo") and m["price"] is not None:
            f = M.kelly(p, m["price"])
            stake = {"kelly": f, "quarter": f / 4, "eighth": f / 8}
        decisions.append({
            "conviction": m["edge"] if m["edge"] is not None else p - 0.5,
            "market": label, "pick": m["pick"], "p": p, "price": m["price"], "implied": m["implied"], "edge": m["edge"],
            "fair": m["fair"], "minPrice": m["minPrice"], "modelo": "Sí" if modelo else "No",
            "guion": "Sí" if guion else "No", "guionWhy": guion_why, "cuota": "Sí" if cuota else ("—" if m["price"] is None else "No"),
            "contradiction": level, "contraList": [t for t, on in zip(contra_labels(label), contra) if on],
            "light": light, "decision": dec, "blocked": blocked, "stake": stake, "assumption": assumption,
        })

    # ML
    ml = best_of("ml") if odds else mk("ml", T.abbr(g[fav]))
    ml_side = "home" if ml and ml["pick"] == hb else "away"
    opp_ml = "home" if ml_side == "away" else "away"
    guion_ml = (adv_side != opp_ml or S["s3"]["advantage"]["level"] == "ligera") and bps[ml_side]["rank"] <= 20
    decide("ML", ml, guion_ml, f"Abridor/bullpen de {T.abbr(g[ml_side])} no contradicen el juego completo",
           [bps[ml_side]["rank"] > 20, adv_side == opp_ml and S["s3"]["advantage"]["level"] == "clara",
            conf == "baja", total_proj < 7.5 and S["s7"]["expensiveFav"]["on"]], "ml",
           f"{T.abbr(g[ml_side])} supera al rival en el juego completo")
    # Run line: sin momios se evalúa el favorito -1.5 (el lado que el mercado cotiza como "interesante")
    if odds:
        rl = best_of("rl")
    else:
        rl = next((m for m in markets if m["key"] == "rl" and "-1.5" in m["pick"]), None)
    fav_rl = rl and "-1.5" in rl["pick"]
    decide("Run Line", rl, (not fav_rl) or (total_proj >= 8.5 and not both_strong),
           "Favorito -1.5 solo con total proyectado alto y bullpen rival no top-10",
           [fav_rl and total_proj < 8.0, fav_rl and both_strong, fav_rl and k_count >= 3], "ml",
           "Margen de 2+ carreras")
    # F5
    f5 = max((m for m in markets if m["key"] == "f5ml"), key=lambda m: m["p"])
    f5_side = "away" if f5["pick"].startswith(ab + " ") else "home"
    decide("F5", f5, adv_side == f5_side, "Ventaja del abridor del mismo lado",
           [adv_side is not None and adv_side != f5_side, sps[f5_side]["missing"], (f5.get("tie") or 0) > 0.2], "f5",
           f"Abridor de {T.abbr(g[f5_side])} domina las primeras 5")
    # Total
    if odds and odds.get("totalLine"):
        tot = best_of("total", lambda m: m.get("market_line"))
    else:
        tot = median_line_pick([m for m in markets if m["key"] == "total"])
    is_over = tot and "Over" in tot["pick"]
    guion_tot = (k_count < 3 and not both_strong) if is_over else not any(bps[s]["zFatigue"] >= 1.0 for s in SIDES)
    decide("Total", tot, guion_tot, "Over: sin filtro de ponches ni bullpens top-10 · Under: bullpens descansados",
           [is_over and k_count >= 3, is_over and both_strong, (not is_over) and weather["out"] and (weather["speed"] or 0) >= 10,
            abs(total_proj - (tot["line"] if tot else 8.5)) < 0.3], "total",
           "Ambos abridores permiten tráfico" if is_over else "Abridores/bullpens dominan")
    # Team total (el más fuerte de los dos equipos)
    tt_c = [median_line_pick([m for m in markets if m["key"] == k]) for k in ("tt_away", "tt_home")]
    tt_c = [m for m in tt_c if m]
    tt = max(tt_c, key=lambda m: m["pNoPush"]) if tt_c else None
    tt_side = "away" if tt and tt["key"] == "tt_away" else "home"
    tt_over = tt and "Over" in tt["pick"]
    produce = offs[tt_side]["obp"] >= lg.obp and offs[tt_side]["iso"] >= lg.iso - 0.01
    decide("Team total", tt, produce if tt_over else True, "Over solo si el equipo produce carreras (OBP/ISO ≥ liga), no solo hits",
           [tt_over and not produce, tt_over and lineups[tt_side] is None], "total",
           f"{T.abbr(g[tt_side])} {'anota' if tt_over else 'no anota'}")
    # Props de pitcher (ponches)
    kp_c = [median_line_pick([m for m in markets if m["key"] == k]) for k in ("k_away", "k_home")]
    kp_c = [m for m in kp_c if m]
    kp = max(kp_c, key=lambda m: m["pNoPush"]) if kp_c else None
    kp_side = kp and ("away" if kp["key"] == "k_away" else "home")
    decide("Prop pitcher", kp, bool(kp) and sps[kp_side]["expIp"] >= 5.0 if kp and "Over" in kp["pick"] else True,
           "Over de K solo si el abridor proyecta 5+ entradas",
           [bool(kp) and lineups["home" if kp_side == "away" else "away"] is None], "props", "Abridor lanza profundo")
    decisions.append({"market": "Prop bateador", "pick": "—", "modelo": "No", "guion": "No", "cuota": "No",
                      "contradiction": "—", "light": "Gris", "decision": "No bet",
                      "why": "Sin props de bateadores conectados", "assumption": "—"})
    nr = max((m for m in markets if m["key"] == "nrfi"), key=lambda m: m["p"])
    decide("NRFI/YRFI", nr, True, "Primera entrada calibrada con la frecuencia real de ceros de la liga",
           [lineups["away"] is None or lineups["home"] is None], "nrfi", "Primer episodio dominante" if nr["pick"] == "NRFI"
           else "Tráfico temprano")

    # dependencia de supuestos (8.2)
    deps = [
        {"pick": f"ML {T.abbr(g[fav])}", "assumption": f"El abridor rival ({sps[dog]['name']}) permite daño",
         "fails": "Pierde valor", "obs": "Depende del colapso rival", "risk": "Alto"},
        {"pick": f"Over ER {sps[dog]['name']}", "assumption": "El pitcher permite carreras", "fails": "Pierde valor",
         "obs": "Mismo supuesto que ML", "risk": "Alto"},
        {"pick": "Over total", "assumption": "Ambos abridores permiten tráfico", "fails": "Pierde valor",
         "obs": "Requiere daño temprano", "risk": "Alto"},
        {"pick": "F5 Under", "assumption": "Abridores dominan", "fails": "Pierde valor", "obs": "Depende de picheo fuerte",
         "risk": "Medio"},
    ]
    strong = [d for d in decisions if d["light"] in ("Verde", "Amarillo")]
    shared = defaultdict(list)
    for d in strong:
        shared[d.get("assumption")].append(d["market"])
    corr = [f"{', '.join(v)} dependen de «{k}»: elegir solo uno o bajar stake." for k, v in shared.items() if len(v) > 1]

    audit = audit_checks(ctx, g, S, sps, bps, lam, total_proj, odds, k_count, both_strong, missing_info, decisions,
                         k_heavy, lineups, corr, gate)

    rank = {"Verde": 0, "Amarillo": 1, "Gris": 2, "Rojo": 3}
    cands = [d for d in decisions if d.get("p") is not None]
    crank = {"Baja": 0, "Media": 1, "Alta": 2}
    best = min(cands, key=lambda d: (rank[d["light"]], crank.get(d["contradiction"], 3), -(d.get("conviction") or 0))) if cands else None
    rec = recommendation(ctx, g, decisions, S, sps, fav, total_proj)
    return {"decisions": decisions, "dependencies": deps, "correlation": corr, "audit": audit, "best": best,
            "recommendation": rec,
            "rule": "Solo puede ser pick fuerte si cumple: Modelo + Guion + Cuota + Baja contradicción",
            "central": ("No elijas el mercado que predice al ganador; elige el mercado que mejor representa el guion del "
                        "partido con menor contradicción. Antes de ML, Run Line, Over o team total Over, revisar: "
                        "Under, F5 Under, props de Ks, no bet.")}


def median_line_pick(rows: list[dict]) -> dict | None:
    """Sin momios: la línea que pondría una casa es la más cercana a la mediana (P(Over) ≈ 50%).
    Se toma esa línea y el lado que el modelo prefiere en ella."""
    lines = defaultdict(dict)
    for m in rows:
        lines[m["line"]]["over" if "Over" in m["pick"] else "under"] = m
    best_line, best_gap = None, 9.0
    for ln, pair in lines.items():
        if "over" in pair and "under" in pair:
            gap = abs(pair["over"]["pNoPush"] - 0.5)
            if gap < best_gap:
                best_line, best_gap = ln, gap
    if best_line is None:
        return None
    pair = lines[best_line]
    return max(pair.values(), key=lambda m: m["pNoPush"])


def contra_labels(label):
    return {
        "ML": ["Bullpen del lado elegido en el tercio inferior", "Abridor rival con ventaja clara",
               "Métodos divergen (>10 pp)", "Favorito caro con total bajo"],
        "Run Line": ["Total proyectado bajo (<8)", "Ambos bullpens top-10", "Filtro de ponches activo"],
        "F5": ["Ventaja de abridor para el otro lado", "Abridor por anunciar", "Empate F5 probable (>20%)"],
        "Total": ["Filtro de ponches bloquea Over", "Ambos bullpens top-10", "Viento hacia afuera contra el Under",
                  "Proyección pegada a la línea"],
        "Team total": ["El equipo no produce carreras (OBP/ISO)", "Lineup no confirmado"],
        "Prop pitcher": ["Lineup rival no confirmado"],
        "NRFI/YRFI": ["Lineups no confirmados"],
    }.get(label, [])


def audit_checks(ctx, g, S, sps, bps, lam, total_proj, odds, k_count, both_strong, missing_info, decisions, k_heavy,
                 lineups, corr, gate):
    T = ctx.T
    p_home = S["s5"]["triangulation"]["pHome"]
    fav_p = max(p_home, 1 - p_home)
    out = []

    def a(n, title, status, detail):
        out.append({"n": n, "title": title, "status": status, "detail": detail})

    exp = S["s7"]["expensiveFav"]
    a(1, "No confundir predicción con valor", "Alerta" if fav_p >= 0.6 and (exp["on"] or not odds) else "OK",
      f"Favorito al {pct(fav_p)}%. " + ("Sin momios: la probabilidad no es valor por sí sola." if not odds
                                        else "Precio del favorito " + str(exp["favPrice"])))
    a(2, "No sobreponderar últimas 5 salidas", "OK",
      "Últimas 5 pesan 10% del puntaje y la forma se regresa a la media (≈120 BF frente a k=600).")
    collapse = [s for s in SIDES if not sps[s]["missing"] and (sps[s].get("l5fip") or 0) > (sps[s]["shrunk"]["fip"] + 1.0)]
    a(3, "Pick que depende de que un abridor colapse", "Alerta" if collapse else "OK",
      ("Mala racha reciente de " + ", ".join(sps[s]["name"] for s in collapse) +
       ": verificar parque, bullpen detrás, lineup, BB%, HR/9 y Barrel% antes de apostar a su colapso.") if collapse
      else "Ningún abridor con racha reciente mucho peor que su talento regresado.")
    a(4, "Muchos ponches proyectados", "Alerta" if k_heavy else "OK",
      "Revisar primero F5 Under y Under completo." if k_heavy else "Ningún abridor proyecta 7+ K.")
    a(5, "El park factor no decide solo", "OK", S["s6"]["parkText"])
    a(6, "Ambos bullpens top 10 o descansados", "Alerta" if both_strong else "OK",
      "Penalizar Over completo y Run Line del favorito." if both_strong else "No aplica.")
    a(7, "Diferenciar hits de carreras", "OK", "La ofensiva se mide con carreras, OBP×SLG, BB% e ISO, no con AVG.")
    a(8, "Tabla de dependencia de supuestos", "OK", "Ver tabla 8.2.")
    rl = next((d for d in decisions if d["market"] == "Run Line"), None)
    a(9, "Run Line con total bajo", "Alerta" if total_proj < 8.0 else "OK",
      f"Total proyectado {r2(total_proj)}: " + ("evitar favoritos -1.5." if total_proj < 8.0 else "sin restricción."))
    a(10, "Team total Over requiere producir carreras", "OK", "El guion del team total exige OBP e ISO ≥ liga.")
    a(11, "Divisional no implica más carreras", "OK", "No se aplica ningún ajuste de carreras por ser divisional.")
    a(12, "Información faltante (bullpen 3 días, umpire, lineup, pitch count)", "Alerta" if missing_info else "OK",
      "Picks limitados a Gris/Amarillo, nunca Verde." if missing_info else "Información completa.")
    a(13, "Tres filtros: estadístico, guion y cuota", "OK", "Aplicados en la tabla 8.4 (Modelo · Guion · Cuota).")
    tot = next((d for d in decisions if d["market"] == "Total"), None)
    contra14 = bool(tot and "Under" in (tot.get("pick") or "") and rl and "-1.5" in (rl.get("pick") or "")
                    and rl["light"] in ("Verde", "Amarillo"))
    a(14, "Under + Run Line favorito", "Alerta" if contra14 else "OK",
      "Contradicción: revisar." if contra14 else "Sin contradicción.")
    any_edge = any(d.get("edge") is not None and d["edge"] >= EDGE_MIN for d in decisions)
    a(15, "Sin edge claro → no bet", "Alerta" if not any_edge else "OK",
      "No hay edge ≥3 pp con momios reales: no bet." if not any_edge else "Hay al menos un mercado con edge.")
    a(16, "Correlación entre picks del mismo partido", "Alerta" if corr else "OK", " ".join(corr) or "Sin picks correlacionados.")
    a(17, "Gate de completitud (sección 9)", "Alerta" if gate["campoFaltante"] else "OK",
      f"{gate['missing']} campo(s) obligatorio(s) faltante(s): ningún mercado afectado puede ser Verde."
      if gate["campoFaltante"] else "Todos los campos obligatorios presentes.")
    return out


def recommendation(ctx, g, decisions, S, sps, fav, total_proj):
    T = ctx.T
    by = {d["market"]: d for d in decisions}
    rank = {"Verde": 0, "Amarillo": 1, "Gris": 2, "Rojo": 3}
    valued = [d for d in decisions if d.get("edge") is not None]
    best_val = max(valued, key=lambda d: d["edge"]) if valued else None

    def fmt(d):
        if not d or d.get("p") is None:
            return "—"
        s = f"{d['pick']} ({pct(d['p'])}%"
        if d.get("price") is not None:
            s += f", momio {int(d['price']):+d}, edge {pct(d['edge'])} pp"
        else:
            s += f", momio justo {fmt_odds(d['fair'])}, mínimo aceptable {fmt_odds(d['minPrice'])}"
        return s + f") · {d['light']}"

    probable_no_value = next((d for d in decisions if d.get("p") and d["p"] >= 0.6 and d.get("edge") is not None
                              and d["edge"] < 0), None)
    risky = next((d for d in valued if d["edge"] >= EDGE_MIN and d["contradiction"] != "Baja"), None)
    avoid = [d for d in decisions if d["light"] == "Rojo" and d.get("p")]
    stakes = [f"{d['pick']}: {pct(d['stake']['quarter'], 2)}% (¼ Kelly) · {pct(d['stake']['eighth'], 2)}% (⅛ Kelly)"
              for d in decisions if d.get("stake")]
    return [
        {"k": "Mejor pick por valor", "v": fmt(best_val) if best_val and best_val["edge"] >= EDGE_MIN else "Ninguno con edge ≥ 3 pp"},
        {"k": "Mejor pick F5", "v": fmt(by.get("F5"))},
        {"k": "Mejor total", "v": fmt(by.get("Total"))},
        {"k": "Mejor prop", "v": fmt(by.get("Prop pitcher"))},
        {"k": "Pick con valor pero alto riesgo", "v": fmt(risky) if risky else "—"},
        {"k": "Pick que parece probable pero no tiene valor", "v": fmt(probable_no_value) if probable_no_value else "—"},
        {"k": "Pick que se debe evitar", "v": ", ".join(f"{d['market']}: {d['pick']}" for d in avoid) or "—"},
        {"k": "Mercado donde conviene esperar mejor cuota",
         "v": ", ".join(f"{d['market']} ({d['pick']}, mínimo {fmt_odds(d['minPrice'])})" for d in decisions
                        if d["light"] == "Gris" and d.get("minPrice") is not None) or "—"},
        {"k": "Stake sugerido (Kelly fraccional)", "v": "; ".join(stakes) or "Sin stake: ningún pick Verde/Amarillo con momio real"},
    ]


def fmt_odds(m):
    if m is None:
        return "—"
    return f"{int(round(m)):+d}"


# ============================================================ sección 10: algoritmo maestro

def section10(ctx, S, methods, p_tri, confidence, lam, odds):
    gate = S["s9"]["gate"]
    return [
        {"phase": "FASE 0 — Ingesta con gate de completitud", "status": "Alerta" if gate["campoFaltante"] else "OK",
         "detail": f"{gate['ok']}/{gate['total']} campos obligatorios; campo_faltante = {str(gate['campoFaltante']).lower()}"},
        {"phase": "FASE 1 — Ajuste estadístico (regresión a la media)", "status": "OK",
         "detail": "Shrinkage por punto de estabilización (K% 70 BF, BB% 170 BF, HR 1150 BF, ERA 1000 BF) y Pitágoras vs récord"},
        {"phase": "FASE 2 — Triangulación de P(victoria)", "status": "OK" if confidence != "baja" else "Alerta",
         "detail": f"Log5 {pct(methods['log5'])}% · Elo {pct(methods['elo'])}% · λ {pct(methods['lambda'])}% → "
                   f"{pct(p_tri)}% local (confianza {confidence})"},
        {"phase": "FASE 3 — Modelo de carreras y total", "status": "OK",
         "detail": f"λ F3 {r2(lam['away']['f3'] + lam['home']['f3'])} · F5 {r2(lam['away']['f5'] + lam['home']['f5'])} · "
                   f"juego {r2(lam['away']['full'] + lam['home']['full'])}; Binomial Negativa en 9 entradas, Poisson en F3/F5"},
        {"phase": "FASE 4 — Valor de mercado", "status": "OK" if odds else "Alerta",
         "detail": "Edge, momio justo, mínimo aceptable y Kelly fraccional" if odds
         else "Sin momios: se publican momio justo y mínimo aceptable; edge no calculable"},
        {"phase": "FASE 5 — Auditoría", "status": "OK",
         "detail": f"{sum(1 for x in S['s8']['audit'] if x['status'] == 'Alerta')} alertas de 17 filtros; correlación revisada"},
        {"phase": "FASE 6 — Salida", "status": "OK",
         "detail": "Semáforo, pick, cuota mínima, edge, stake y dato faltante por mercado"},
    ]
