"""Picks por partido con Índice de Confianza (IC, 0-100) y su desglose matemático.

IC = 100 · (0.35·Fuerza + 0.25·Consenso + 0.15·Datos + 0.15·Estabilidad + 0.10·Contradicciones)

  Fuerza          ventaja de la probabilidad del modelo sobre el punto de equilibrio del precio
                  (momio real si existe; si no, precio de referencia típico del mercado), a escala de 12 pp
  Consenso        acuerdo entre métodos independientes del framework (Log5, Elo, λ-Binomial Negativa,
                  Poisson, Monte Carlo): 1 − divergencia/12 pp; con un solo método vale 0.5
  Datos           gate de completitud (sección 9.3) para ese mercado y lineups confirmados
  Estabilidad     peso de la muestra tras la regresión a la media (5.7.4) y error estándar del Monte Carlo
  Contradicciones 1 − (contradicciones activas del mercado)/3 (secciones 5.6, 6.11, 7.7)

Niveles: ≥70 Alta · 55-69 Media · 40-54 Baja · <40 Muy baja.
"""
from __future__ import annotations

from . import mathlib as M

WEIGHTS = {"fuerza": 0.35, "consenso": 0.25, "datos": 0.15, "estabilidad": 0.15, "contradicciones": 0.10}
# precio de referencia típico cuando no hay momio real (se muestra como referencia, no como cuota)
REF_PRICE = {"ml": -110, "rl_fav": 135, "rl_dog": -160, "total": -110, "f5ml": -110, "f5total": -115,
             "tt": -115, "nrfi": -120, "yrfi": 100, "k": -115}
MODEL_NAMES = {
    "log5": "Log5 (5.7.2)", "elo": "Elo + abridor (5.7.7)", "lambda": "λ + Binomial Negativa (5.3 · 5.7.5)",
    "poisson": "λ + Poisson (5.4)", "calib": "λ calibrado a ceros reales (6.8)", "mc": "DIAMANTE-24 Monte Carlo (5.7.3 · 5.7.9)",
    "prisma": "PRISMA bayesiano (5.7.4 · 5.7.6 · 5.7.9)",
}


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def level(ic: float) -> str:
    return "Alta" if ic >= 70 else "Media" if ic >= 55 else "Baja" if ic >= 40 else "Muy baja"


def score_pick(p: float, methods: dict, ref_price: float, real_price: float | None, blocked: bool,
               lineups_ok: bool, stability: float, n_contra: int, contra: list[str]) -> dict:
    price = real_price if real_price is not None else ref_price
    be = M.american_to_prob(price)
    fuerza = clip((p - be) / 0.12)
    vals = [v for v in methods.values() if v is not None]
    spread = (max(vals) - min(vals)) if len(vals) >= 2 else None
    consenso = 0.5 if spread is None else clip(1 - spread / 0.12)
    datos = (0.4 if blocked else 1.0) * (1.0 if lineups_ok else 0.8)
    contradicciones = clip(1 - n_contra / 3)
    parts = {"fuerza": fuerza, "consenso": consenso, "datos": datos, "estabilidad": clip(stability),
             "contradicciones": contradicciones}
    ic = 100 * sum(WEIGHTS[k] * v for k, v in parts.items())
    return {"ic": ic, "level": level(ic), "parts": parts, "breakeven": be, "price": price, "priceIsReal": real_price is not None,
            "edge": p - be, "spread": spread, "contra": contra,
            "formula": " + ".join(f"{WEIGHTS[k]:.2f}·{v:.2f}" for k, v in parts.items()) + f" = {ic / 100:.3f}"}


def build(game: dict, extra: dict | None = None) -> list[dict]:
    """Picks de un partido ya analizado por model.analyze. `extra` = probabilidades del Monte Carlo del Pro-Lab."""
    S = game["sections"]
    extra = extra or {}
    MK_ = extra.get("methodKey", "mc")
    a, h = game["teams"]["away"]["abbr"], game["teams"]["home"]["abbr"]
    tri = S["s5"]["triangulation"]
    gate = S["s9"]["gate"]["blocks"]
    gate_odds = S["s9"]["gate"].get("blocksWithOdds") or gate
    block_key_of = {}
    lu_ok = all(game["summary"]["lineupsConfirmed"].values())
    s3 = S["s3"]
    comp = {c["side"]: c for c in s3["comparison"]}
    stab_sp = []
    for side in ("away", "home"):
        c = comp.get(side) or {}
        bf = c.get("bf") or 0
        stab_sp.append(bf / (bf + 400) if bf else 0.3)
    stab_sp = sum(stab_sp) / 2
    mc_n = extra.get("n")
    mc_stab = 1.0 if not mc_n else clip(1 - 0.5 / (mc_n ** 0.5) * 20)
    markets = game.get("markets") or []
    by_key = {}
    for m in markets:
        by_key.setdefault(m["key"], []).append(m)
    decisions = {d["market"]: d for d in S["s8"]["decisions"]}
    picks = []

    def contra_for(label):
        d = decisions.get(label) or {}
        return d.get("contraList") or []

    fam_to_decision = {"ML": "ML", "RL": "Run Line", "Total": "Total", "F5 total": "Total", "Team total": "Team total",
                       "F5": "F5", "NRFI": "NRFI/YRFI", "K": "Prop pitcher"}

    def add(family, label, pick, p, methods, ref_key, real_price, blocked, stability, contra, how, line=None):
        sc = score_pick(p, methods, REF_PRICE[ref_key], real_price, blocked, lu_ok, stability, len(contra), contra)
        d = decisions.get(fam_to_decision.get(family)) or {}
        bkey = {"ML": "ml", "RL": "ml", "Total": "total", "Team total": "total", "F5 total": "f5", "F5": "f5",
                "NRFI": "nrfi", "K": "props"}[family]
        blocked_odds = bool(gate_odds.get(bkey))
        picks.append({"family": family, "market": label, "pick": pick, "p": p, "line": line,
                      "guion": d.get("guion"), "guionWhy": d.get("guionWhy"), "contradiction": d.get("contradiction"),
                      "blocked": bool(blocked), "blockedWithOdds": blocked_odds,
                      "datosWithOdds": (0.4 if blocked_odds else 1.0) * (1.0 if lu_ok else 0.8),
                      "methods": {MODEL_NAMES[k]: v for k, v in methods.items() if v is not None},
                      "fair": M.fair_american(p), "minPrice": M.fair_american(clip(p - 0.03, 0.01, 0.99)),
                      "refPrice": REF_PRICE[ref_key], "how": how, **sc})

    # --- Moneyline
    p_home = tri["pHome"]
    meth = {"log5": tri["log5"]["pHome"], "elo": tri["elo"]["pHome"], "lambda": tri["lambda"]["pHome"]}
    if extra.get("pHome") is not None:
        meth[MK_] = extra["pHome"]
        p_home = sum(meth.values()) / len(meth)
    side_home = p_home >= 0.5
    ml_methods = {k: (v if side_home else 1 - v) for k, v in meth.items()}
    ml_price = _real(by_key.get("ml"), h if side_home else a)
    add("ML", "Moneyline", h if side_home else a, p_home if side_home else 1 - p_home, ml_methods, "ml", ml_price,
        gate["ml"], (stab_sp + mc_stab) / 2 if mc_n else stab_sp, contra_for("ML"),
        f"Promedio de {len(ml_methods)} métodos: " + " · ".join(f"{MODEL_NAMES[k]} {v:.1%}" for k, v in ml_methods.items()))

    # --- Run line (favorito -1.5 o perro +1.5, el que más supere su precio de referencia)
    rl = by_key.get("rl") or []
    best_rl = None
    for m in rl:
        fav = "-1.5" in m["pick"]
        p = m["pNoPush"]
        mc_p = extra.get("rl", {}).get(m["pick"])
        methods = {"lambda": p, **({MK_: mc_p} if mc_p is not None else {})}
        pp = sum(methods.values()) / len(methods)
        gap = pp - M.american_to_prob(REF_PRICE["rl_fav" if fav else "rl_dog"])
        if best_rl is None or gap > best_rl[0]:
            best_rl = (gap, m, methods, pp, fav)
    if best_rl:
        _, m, methods, pp, fav = best_rl
        add("RL", "Run Line", m["pick"], pp, methods, "rl_fav" if fav else "rl_dog", m.get("price"), gate["ml"],
            stab_sp, contra_for("Run Line"), "Distribución del margen (Binomial Negativa por equipo" +
            (" y Monte Carlo" if len(methods) > 1 else "") + ")")

    # --- Totales: línea de mercado o la más cercana a la mediana
    def line_pick(key, family, label, ref_key, block_key, how, mc_dist=None, real_line=None):
        rows = by_key.get(key) or []
        if not rows:
            return
        lines = {}
        for m in rows:
            lines.setdefault(m["line"], {})["over" if "Over" in m["pick"] else "under"] = m
        if real_line is not None and real_line in lines:
            ln = real_line
        else:
            ln = min((l for l, v in lines.items() if "over" in v and "under" in v),
                     key=lambda l: abs(lines[l]["over"]["pNoPush"] - 0.5))
        best = None
        for side in ("over", "under"):
            m = lines[ln][side]
            methods = {"lambda" if key in ("total",) or key.startswith("tt_") else "poisson": m["pNoPush"]}
            if mc_dist is not None:
                methods[MK_] = _ou(mc_dist, ln, side)
            pp = sum(methods.values()) / len(methods)
            if best is None or pp > best[0]:
                best = (pp, m, methods)
        pp, m, methods = best
        add(family, label, m["pick"], pp, methods, ref_key, m.get("price"), gate[block_key],
            stab_sp, contra_for(family if family != "Total" else "Total"), how, line=ln)

    line_pick("total", "Total", "Total completo", "total", "total",
              "Suma de dos Binomiales Negativas (Var/Media medida en la temporada)" +
              (" + Monte Carlo" if extra.get("total") else ""), extra.get("total"),
              S["s7"].get("marketTotal"))
    line_pick("f5total", "F5 total", "Total primeras 5", "f5total", "f5", "Poisson con λ de las primeras 5 entradas" +
              (" + Monte Carlo" if extra.get("f5total") else ""), extra.get("f5total"))
    for side, abbr in (("away", a), ("home", h)):
        line_pick(f"tt_{side}", "Team total", f"Team total {abbr}", "tt", "total",
                  f"Binomial Negativa de carreras de {abbr}" + (" + Monte Carlo" if extra.get(f"runs_{side}") else ""),
                  extra.get(f"runs_{side}"))

    # --- F5 moneyline (sin empate)
    f5 = by_key.get("f5ml") or []
    if f5:
        m = max(f5, key=lambda x: x["pNoPush"])
        side = "away" if m["pick"].startswith(a + " ") else "home"
        methods = {"poisson": m["pNoPush"]}
        if extra.get("f5"):
            f = extra["f5"]
            methods[MK_] = f[side] / (f["away"] + f["home"]) if (f["away"] + f["home"]) else None
        pp = sum(v for v in methods.values() if v is not None) / len([v for v in methods.values() if v is not None])
        add("F5", "F5 Moneyline", m["pick"], pp, methods, "f5ml", None, gate["f5"], stab_sp, contra_for("F5"),
            "Poisson de las primeras 5 entradas, empates como push" + (" + Monte Carlo" if extra.get("f5") else ""))

    # --- NRFI / YRFI
    nr = by_key.get("nrfi") or []
    if nr:
        p_n = next(m["p"] for m in nr if m["pick"] == "NRFI")
        methods = {"calib": p_n}
        if extra.get("nrfi") is not None:
            methods[MK_] = extra["nrfi"]
        pn = sum(methods.values()) / len(methods)
        edge_n = pn - M.american_to_prob(REF_PRICE["nrfi"])
        edge_y = (1 - pn) - M.american_to_prob(REF_PRICE["yrfi"])
        pick, pp = ("NRFI", pn) if edge_n >= edge_y else ("YRFI", 1 - pn)
        add("NRFI", "NRFI/YRFI", pick, pp, {k: (v if pick == "NRFI" else 1 - v) for k, v in methods.items()},
            "nrfi" if pick == "NRFI" else "yrfi", None, gate["nrfi"], stab_sp, contra_for("NRFI/YRFI"),
            "Primera entrada calibrada con la frecuencia real de ceros de la liga" + (" + Monte Carlo" if MK_ in methods else ""))

    # --- props de ponches (línea mediana)
    for side in ("away", "home"):
        rows = by_key.get(f"k_{side}") or []
        if not rows:
            continue
        lines = {}
        for m in rows:
            lines.setdefault(m["line"], {})["over" if "Over" in m["pick"] else "under"] = m
        ln = min((l for l, v in lines.items() if len(v) == 2), key=lambda l: abs(lines[l]["over"]["pNoPush"] - 0.5))
        best = None
        for o_u in ("over", "under"):
            m = lines[ln][o_u]
            methods = {"poisson": m["pNoPush"]}
            if extra.get(f"k_{side}"):
                methods[MK_] = _ou(extra[f"k_{side}"], ln, o_u)
            pp = sum(methods.values()) / len(methods)
            if best is None or pp > best[0]:
                best = (pp, m, methods)
        pp, m, methods = best
        c = comp.get(side) or {}
        bf = c.get("bf") or 0
        add("K", "Prop de ponches", m["pick"], pp, methods, "k", None, gate["props"], bf / (bf + 70 * 4) if bf else 0.3,
            contra_for("Prop pitcher"), "K% del abridor vs K% del rival (razón de momios) × bateadores esperados, Poisson" +
            (" + Monte Carlo" if MK_ in methods else ""), line=ln)

    picks.sort(key=lambda x: -x["ic"])
    for i, p in enumerate(picks):
        p["rank"] = i + 1
    return picks


def _real(rows, pick):
    for m in rows or []:
        if m["pick"] == pick and m.get("price") is not None:
            return m["price"]
    return None


def _ou(dist: list[float], line: float, side: str) -> float:
    over = sum(p for k, p in enumerate(dist) if k > line)
    under = sum(p for k, p in enumerate(dist) if k < line)
    push = 1 - over - under
    base = 1 - push if push < 1 else 1
    return (over if side == "over" else under) / base


def grade(p: dict, away_abbr: str, home_abbr: str, a: int, h: int, innings: list, starter_k: dict | None = None):
    """True/False si el pick ganó o perdió con el resultado oficial; None si fue push o no hay dato."""
    pk, fam = p["pick"], p["family"]
    f5a = sum((x[0] or 0) for x in innings[:5])
    f5h = sum((x[1] or 0) for x in innings[:5])
    first = innings[0] if innings else [0, 0]
    if fam == "ML":
        return (pk == home_abbr) == (h > a)
    if fam == "RL":
        team, line = pk.rsplit(" ", 1)
        margin = (h - a) if team == home_abbr else (a - h)
        return margin + float(line) > 0
    if fam in ("Total", "F5 total", "Team total"):
        val = (a + h) if fam == "Total" else (f5a + f5h) if fam == "F5 total" else (a if pk.startswith(away_abbr + " ") else h)
        return None if val == p["line"] else (val > p["line"]) == ("Over" in pk)
    if fam == "F5":
        return None if f5a == f5h else (f5h > f5a) == pk.startswith(home_abbr + " ")
    if fam == "NRFI":
        scored = (first[0] or 0) + (first[1] or 0) > 0
        return (not scored) if pk == "NRFI" else scored
    if fam == "K":
        name = pk.split(" Over ")[0].split(" Under ")[0]
        k = (starter_k or {}).get(name)
        if k is None:
            return None
        return None if k == p["line"] else (k > p["line"]) == ("Over" in pk)
    return None
