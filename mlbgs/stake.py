"""STAKE-K: cuánto apostar en cada pick, entre $500 y $1,500.

Kelly fraccional sobre una probabilidad encogida hacia el mercado según qué tanto confiamos en ella.

  1. Certeza c ∈ [0, 1] = las partes del índice de confianza que NO dependen del precio
         c = (0.25·Consenso + 0.15·Datos + 0.15·Estabilidad + 0.10·Sin contradicciones) / 0.65
     (la Fuerza —la ventaja sobre el precio— ya entra por Kelly; así no se cuenta dos veces)
  2. Peso w = (0.30 + 0.60·c) · A · H, acotado a [0.20, 0.95]
         A = acuerdo con los demás modelos: si el partido tiene modelo del Pro-Lab y su probabilidad
             queda por debajo de la del framework, A baja de 1.00 (≤ 3 pp) a 0.70 (≥ 10 pp)
         H = historial del modelo: (acierto real − esperado) de sus picks principales calificados,
             encogido n/(n+60) y acotado a [0.85, 1.05]
  3. Probabilidad para apostar p* = q + w·(p − q), con q = 1/d la probabilidad del momio.
     Kelly con p*: f = (p*·d − 1)/(d − 1) = w · (p·d − 1)/(d − 1)
  4. Stake = ¼ · f · B con B = $50,000 de banca de referencia ($500 = 1 %, $1,500 = 3 %),
     redondeado a $50 y acotado a [$500, $1,500]. Si ¼ Kelly da menos de $300 no se apuesta
     (subirlo al mínimo sería apostar casi el doble de lo que justifica la ventaja).
  5. Solo con semáforo Verde a ese momio: edge ≥ 3 pp, IC ≥ 55, sin dato obligatorio faltante,
     guion ≠ No y contradicción ≠ Alta (las mismas reglas de la sección 8 del framework).

Cartera: tope de $2,000 por partido (los picks de un mismo juego están correlacionados) y
$7,500 por día (15 % de la banca de referencia).
"""
from __future__ import annotations

from . import mathlib as M
from . import picks as PK

BANK = 50_000
MIN, MAX, STEP = 500, 1_500, 50
KELLY = 0.25
FLOOR = 300
EDGE_MIN = 0.03
IC_MIN = 55
GAME_CAP, DAY_CAP = 2_000, 7_500
LEVELS = (500, 1_000, 1_500)
CONFIG = {"bank": BANK, "min": MIN, "max": MAX, "step": STEP, "kelly": KELLY, "floor": FLOOR, "edgeMin": EDGE_MIN,
          "icMin": IC_MIN, "gameCap": GAME_CAP, "dayCap": DAY_CAP, "levels": list(LEVELS)}
LAB_METHODS = {"diamante": PK.MODEL_NAMES["mc"], "prisma": PK.MODEL_NAMES["prisma"], "kronos": PK.MODEL_NAMES["kronos"]}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def certainty(parts: dict) -> float:
    return (0.25 * parts["consenso"] + 0.15 * parts["datos"] + 0.15 * parts["estabilidad"]
            + 0.10 * parts["contradicciones"]) / 0.65


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


def track(tickets: dict | list) -> dict:
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
        out[m] = {"n": r["n"], "hit": hit, "exp": exp,
                  "h": clip(1 + (hit - exp) * r["n"] / (r["n"] + 60), 0.85, 1.05)}
    return out


def weight(pick: dict, agree: float = 1.0, h: float = 1.0) -> float:
    return clip((0.30 + 0.60 * certainty(pick["parts"])) * agree * h, 0.20, 0.95)


def ic_at(pick: dict, dec: float) -> float:
    """Índice de confianza con un momio dado (misma cuenta que la página al capturar momios)."""
    edge = pick["p"] - 1 / dec
    fuerza = clip(edge / 0.12, 0.0, 1.0)
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


def stake_at(pick: dict, w: float, dec: float) -> dict:
    """Stake recomendado a un momio decimal `dec`."""
    p = pick["p"]
    edge = p - 1 / dec
    why = blocked_why(pick)
    if why:
        return {"stake": 0, "why": why}
    if edge < EDGE_MIN:
        return {"stake": 0, "why": "edge menor a 3 pp"}
    if ic_at(pick, dec) < IC_MIN:
        return {"stake": 0, "why": "índice de confianza menor a 55 con ese momio"}
    f = w * max(0.0, (p * dec - 1) / (dec - 1))
    raw = KELLY * f * BANK
    if raw < FLOOR:
        return {"stake": 0, "why": "ventaja chica para el mínimo de $500", "raw": raw}
    return {"stake": int(clip(round(raw / STEP) * STEP, MIN, MAX)), "raw": raw, "f": f}


def _min_dec(pick: dict, w: float, level: int) -> float | None:
    lo, hi = 1.01, 50.0
    if stake_at(pick, w, hi)["stake"] < level:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if stake_at(pick, w, mid)["stake"] >= level:
            hi = mid
        else:
            lo = mid
    return hi


def _american_floor(dec: float) -> int:
    """El momio americano entero más bajo que ya cumple (siempre hacia el lado que paga más)."""
    am = (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)
    return int(-(-am // 1))    # techo: −145.3 → −145, +104.2 → +105


def plan(pick: dict, agree: float = 1.0, h: float = 1.0) -> dict:
    """Escalera de stake: desde qué momio conviene apostar $500, $1,000 y $1,500."""
    w = weight(pick, agree, h)
    out = {"w": w, "cert": certainty(pick["parts"]), "agree": agree, "h": h, "block": blocked_why(pick), "ladder": []}
    if out["block"]:
        return out
    for level in LEVELS:
        d = _min_dec(pick, w, level)
        if d is None:
            break
        step = {"stake": level, "from": _american_floor(d)}
        if out["ladder"] and out["ladder"][-1]["from"] == step["from"]:
            out["ladder"][-1] = step      # mismo momio para dos montos: queda el mayor
        else:
            out["ladder"].append(step)
    if pick.get("priceIsReal") and pick.get("price") is not None:
        out["atMarket"] = stake_at(pick, w, M.american_to_decimal(pick["price"]))["stake"]
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
