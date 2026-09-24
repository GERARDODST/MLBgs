"""Stake 1–10 por confianza: cuánto apostar en un pick, de $500 a $1,500.

  stake 1 = $500 · stake 5 = $1,000 · stake 10 = $1,500
  (de 1 a 5 sube $125 por nivel; de 5 a 10, $100 por nivel)

El nivel lo decide la CONFIANZA del pick: el índice de confianza (IC) recalculado con el momio
que te dan (a mejor momio, más Fuerza y más IC), ajustado por los demás modelos y el historial.

  1. Solo si el pick queda en semáforo Verde con ese momio: edge ≥ 3 pp, IC ≥ 55, sin dato
     obligatorio faltante (lineup, umpire, clima, abridor), guion que acompaña y contradicción no alta.
  2. Nivel base = 1 + 9 · (IC − 55) / 30        (IC 55 → 1 · IC 70 → 5.5 · IC 85 o más → 10)
  3. Nivel = redondeo(base · A · H), entre 1 y 10
       A = acuerdo con el modelo del Pro-Lab: 1.00 si coincide; baja a 0.70 si lo ve ≥ 10 pp peor
       H = historial del modelo: acierto real − esperado de sus picks principales calificados,
           encogido n/(n+60) y acotado a [0.85, 1.05]

Sin momio capturado, el stake es el del IC que muestra el pick y vale solo desde su momio mínimo
(escalera: desde qué momio llega a stake 1, 5 y 10). Con tu momio, el IC y el stake se recalculan.
Tope sugerido de cartera: $7,500 por día (la página avisa si se pasa).
"""
from __future__ import annotations

from . import mathlib as M
from . import picks as PK

AMOUNTS = {1: 500, 2: 625, 3: 750, 4: 875, 5: 1000, 6: 1100, 7: 1200, 8: 1300, 9: 1400, 10: 1500}
EDGE_MIN = 0.03
IC_MIN, IC_TOP = 55, 85
LADDER = (1, 5, 10)
GAME_CAP, DAY_CAP = 2_000, 7_500
CONFIG = {"amounts": AMOUNTS, "edgeMin": EDGE_MIN, "icMin": IC_MIN, "icTop": IC_TOP, "ladder": list(LADDER),
          "gameCap": GAME_CAP, "dayCap": DAY_CAP}
LAB_METHODS = {"diamante": PK.MODEL_NAMES["mc"], "prisma": PK.MODEL_NAMES["prisma"], "kronos": PK.MODEL_NAMES["kronos"]}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def agreement(pick: dict, lab_key: str | None) -> float:
    """1.0 si no hay modelo del Pro-Lab o si coincide con el framework; baja si el modelo lo ve peor."""
    name = LAB_METHODS.get(lab_key or "")
    meth = pick.get("methods") or {}
    if not name or name not in meth:
        return 1.0
    fw = [v for k, v in meth.items() if k != name]
    if not fw:
        return 1.0
    gap = sum(fw) / len(fw) - meth[name]          # > 0: el modelo del Pro-Lab es más pesimista
    return 1.0 - 0.30 * clip((gap - 0.03) / 0.07, 0.0, 1.0)


def track(tickets) -> dict:
    """Factor H por modelo con los picks principales ya calificados del historial."""
    rows: dict[str, dict] = {}
    for t in (tickets.values() if isinstance(tickets, dict) else tickets):
        for p in t["picks"]:
            if not p.get("principal") or p.get("res") not in ("ganado", "perdido"):
                continue
            r = rows.setdefault(t["model"], {"n": 0, "won": 0, "exp": 0.0})
            r["n"] += 1
            r["won"] += p["res"] == "ganado"
            r["exp"] += p["p"]
    out = {}
    for m, r in rows.items():
        hit, exp = r["won"] / r["n"], r["exp"] / r["n"]
        out[m] = {"n": r["n"], "hit": hit, "exp": exp, "h": clip(1 + (hit - exp) * r["n"] / (r["n"] + 60), 0.85, 1.05)}
    return out


def ic_at(pick: dict, dec: float) -> float:
    """Índice de confianza con un momio dado (misma cuenta que la página al capturar momios)."""
    fuerza = clip((pick["p"] - 1 / dec) / 0.12, 0.0, 1.0)
    datos = pick.get("datosWithOdds", pick["parts"]["datos"])
    return pick["ic"] + 100 * PK.WEIGHTS["fuerza"] * (fuerza - pick["parts"]["fuerza"]) \
        + 100 * PK.WEIGHTS["datos"] * (datos - pick["parts"]["datos"])


def blocked_why(pick: dict) -> str | None:
    if pick.get("blockedWithOdds"):
        return "falta un dato obligatorio (lineup, umpire, clima o abridor)"
    if pick.get("guion") == "No":
        return "el guion del partido no acompaña"
    if pick.get("contradiction") == "Alta":
        return "contradicciones altas entre secciones"
    return None


def level_from_ic(ic: float, a: float = 1.0, h: float = 1.0) -> int:
    if ic < IC_MIN:
        return 0
    base = 1 + 9 * (ic - IC_MIN) / (IC_TOP - IC_MIN)
    return int(clip(round(base * a * h), 1, 10))


def stake_at(pick: dict, dec: float, a: float = 1.0, h: float = 1.0) -> dict:
    """Nivel (1–10) y monto a un momio decimal `dec`; nivel 0 = no apostar (con el motivo)."""
    why = blocked_why(pick)
    if why:
        return {"level": 0, "stake": 0, "why": why}
    if pick["p"] - 1 / dec < EDGE_MIN:
        return {"level": 0, "stake": 0, "why": "edge menor a 3 pp con ese momio"}
    ic = ic_at(pick, dec)
    lvl = level_from_ic(ic, a, h)
    if not lvl:
        return {"level": 0, "stake": 0, "why": "confianza menor a 55 con ese momio", "ic": ic}
    return {"level": lvl, "stake": AMOUNTS[lvl], "ic": ic}


def _min_dec(pick: dict, level: int, a: float, h: float) -> float | None:
    lo, hi = 1.01, 50.0
    if stake_at(pick, hi, a, h)["level"] < level:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if stake_at(pick, mid, a, h)["level"] >= level:
            hi = mid
        else:
            lo = mid
    return hi


def american_floor(dec: float) -> int:
    """El momio americano entero más bajo que ya cumple (siempre hacia el lado que paga más)."""
    am = (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)
    return int(-(-am // 1))    # techo: −145.3 → −145, +104.2 → +105


def plan(pick: dict, a: float = 1.0, h: float = 1.0) -> dict:
    """Stake por la confianza que muestra el pick (su IC) y la escalera: desde qué momio llega a stake 1, 5 y 10."""
    out = {"a": a, "h": h, "block": blocked_why(pick), "ladder": [], "level": 0}
    if out["block"]:
        return out
    out["level"] = level_from_ic(pick["ic"], a, h)
    # momio mínimo de cada nivel 1..10 (None si no se alcanza): sirve para calificar después con tu momio
    out["steps"] = [(american_floor(d) if (d := _min_dec(pick, lv, a, h)) else None) for lv in range(1, 11)]
    for level in LADDER:
        frm = out["steps"][level - 1]
        if frm is None:
            break
        step = {"level": level, "stake": AMOUNTS[level], "from": frm}
        if out["ladder"] and out["ladder"][-1]["from"] == step["from"]:
            out["ladder"][-1] = step      # mismo momio para dos niveles: queda el mayor
        else:
            out["ladder"].append(step)
    if pick.get("priceIsReal") and pick.get("price") is not None:     # con momio real conectado, manda ese precio
        out["level"] = stake_at(pick, M.american_to_decimal(pick["price"]), a, h)["level"]
    if not out["ladder"]:
        out["level"] = 0
    return out


def attach(analyses: list[dict], labs: list[dict], tickets) -> dict:
    """Agrega el plan de stake a cada pick del framework y de los Pro-Labs. Devuelve la configuración para la página."""
    tr = track(tickets)
    hf = lambda m: (tr.get(m) or {}).get("h", 1.0)  # noqa: E731
    for a in analyses:
        for p in a.get("picks") or []:
            p["stake"] = plan(p, 1.0, hf("framework"))
    for lab in labs:
        key = lab.get("modelKey")
        for p in (lab.get("picks") or []) + (lab.get("topPicks") or []):
            p["stake"] = plan(p, agreement(p, key), hf(key))
    return {**CONFIG, "track": tr}
