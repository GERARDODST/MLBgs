"""Boleto tipo casino de la decisión única de cada partido: cuánto se apostó, a qué momio y cuánto se cobró.

Un boleto por partido (la decisión «apostar»; si el partido tiene ticket del Pro-Lab, manda ese):

  momio decimal = 1 + 100/|m| si m < 0 · 1 + m/100 si m > 0, a 2 decimales como en el boleto (−137 → 1.73)
  pago potencial = stake · momio decimal          ganancia potencial = stake · (decimal − 1)
  ganado  → cobro = stake · decimal, neto = stake · (decimal − 1)
  perdido → cobro = 0,               neto = −stake
  push o anulado → cobro = stake (se devuelve), neto = 0

Momio del boleto, en este orden:
  1. el que registraste (momio real que te dieron, americano o decimal, y el monto si fue otro);
  2. si no, el de referencia: el momio más bajo con el que valía el stake recomendado
     (escalera guardada antes del juego). Es conservador: con un momio mejor se cobra más.
Con tu momio, el stake sale de la misma escalera (a mejor momio, más stake), salvo que registres el monto.
La página (`site/template.html`) repite estas cuentas en JavaScript: si se cambia una, cambiar la otra.
"""
from __future__ import annotations

from . import mathlib as M
from . import stake as ST

MODEL_ORDER = ("eigen", "kronos", "prisma", "diamante", "framework")   # si hay dos tickets, manda el del Pro-Lab
SETTLED = ("ganado", "perdido", "push", "anulado")


def dec_of(am: float) -> float:
    """Momio decimal a 2 cifras, el que se ve en el boleto y con el que se calcula el pago."""
    return round(M.american_to_decimal(am), 2)


def parse_odds(raw) -> dict | None:
    """'-137', '+120' o '120' (americano) y '1.73' (decimal, con punto) → {am, dec}. Igual que parseOdds de la página."""
    t = str(raw if raw is not None else "").strip().replace("−", "-").replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return None
    if "." in t and 1 < v < 30:
        return {"am": round((v - 1) * 100 if v >= 2 else -100 / (v - 1)), "dec": v}
    if abs(v) >= 100:
        return {"am": round(v), "dec": dec_of(v)}
    return None


def level_at(steps: list | None, dec: float) -> int:
    """Nivel de stake (0–10) que da la escalera a un momio decimal."""
    lv = 0
    for i, frm in enumerate(steps or []):
        if frm is not None and dec >= M.american_to_decimal(frm) - 1e-9:
            lv = i + 1
    return lv


def _decision_pick(t: dict) -> dict | None:
    d = t.get("decision") or {}
    return next((p for p in t["picks"] if p["pick"] == d.get("pick")), None)


def reference_odds(t: dict) -> int | None:
    """El momio más bajo con el que valía el stake recomendado."""
    d = t.get("decision") or {}
    p = _decision_pick(t) or {}
    steps = (p.get("stake") or {}).get("steps")
    lv = d.get("level") or 0
    if steps and 1 <= lv <= len(steps) and steps[lv - 1] is not None:
        return steps[lv - 1]
    ladder = [x for x in d.get("ladder") or [] if x.get("level", 0) <= lv]
    return ladder[-1]["from"] if ladder else d.get("minPrice")


def slip(t: dict, reg: dict | None = None) -> dict | None:
    """Boleto de la decisión del ticket; None si la decisión no fue «apostar»."""
    d = t.get("decision") or {}
    if d.get("status") != "apostar" or not d.get("level"):
        return None
    p = _decision_pick(t) or {}
    reg = reg or {}
    mine = parse_odds(reg.get("odds"))
    if mine:
        o, source = mine, "tuyo"
        level = level_at((p.get("stake") or {}).get("steps"), o["dec"]) if (p.get("stake") or {}).get("steps") else d["level"]
    else:
        ref = reference_odds(t)
        if ref is None:
            return None
        o, source, level = {"am": ref, "dec": dec_of(ref)}, "referencia", d["level"]
    amount = float(reg["amount"]) if reg.get("amount") else float(ST.AMOUNTS.get(level, 0))
    res = p.get("res") if t.get("status") in ("calificado", "anulado") else None
    if t.get("status") == "anulado":
        res = "anulado"
    payout = amount * o["dec"]
    if res == "ganado":
        cobro, neto = payout, payout - amount
    elif res == "perdido":
        cobro, neto = 0.0, -amount
    elif res in ("push", "anulado"):
        cobro, neto = amount, 0.0
    else:
        cobro = neto = None
    return {"pk": t["pk"], "date": t["date"], "model": t["model"], "game": f"{t['away']} @ {t['home']}",
            "pick": d["pick"], "market": d.get("market"), "am": o["am"], "dec": round(o["dec"], 4), "source": source,
            "level": level, "amount": amount, "payout": round(payout, 2), "profit": round(payout - amount, 2),
            "res": res or "pendiente", "cobro": None if cobro is None else round(cobro, 2), "neto": None if neto is None else round(neto, 2),
            "played": reg.get("played", True) is not False}


def slips(tickets, regs: dict | None = None) -> list[dict]:
    """Un boleto por partido (manda el ticket del Pro-Lab si hay dos)."""
    regs = regs or {}
    items = list(tickets.values()) if isinstance(tickets, dict) else list(tickets)
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    seen, out = set(), []
    for t in sorted(items, key=lambda t: (t["date"], t.get("time") or "", t["pk"], rank.get(t["model"], 9))):
        if t["pk"] in seen or not t.get("decision"):
            continue
        seen.add(t["pk"])
        s = slip(t, regs.get(str(t["pk"])))
        if s:
            out.append(s)
    return out


def ledger(bs: list[dict]) -> dict:
    """Cartera: lo apostado, lo cobrado y el neto de los boletos jugados y ya calificados."""
    done = [b for b in bs if b["played"] and b["res"] in SETTLED]
    staked = sum(b["amount"] for b in done)
    cobro = sum(b["cobro"] for b in done)
    net = sum(b["neto"] for b in done)
    return {"n": len(done), "pending": sum(1 for b in bs if b["played"] and b["res"] not in SETTLED),
            "win": sum(b["res"] == "ganado" for b in done), "lose": sum(b["res"] == "perdido" for b in done),
            "push": sum(b["res"] in ("push", "anulado") for b in done),
            "staked": round(staked, 2), "cobro": round(cobro, 2), "net": round(net, 2), "roi": net / staked if staked else None}


def main():
    import argparse
    from . import historial as HI
    ap = argparse.ArgumentParser(description="Boletos de la decisión única de un día (momio decimal, stake, cobro y neto)")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    bs = slips([t for t in HI.load().values() if t["date"] == args.date])
    money = lambda x: "" if x is None else f"${x:,.2f}"  # noqa: E731
    for b in bs:
        neto = "" if b["neto"] is None else f"{b['neto']:+,.2f}"
        print(f"{b['game']:10s} {b['pick']:12s} {b['am']:+5d} = {b['dec']:.3f} · stake {b['level']} ${b['amount']:,.0f}"
              f" · pago {money(b['payout'])} · {b['res']:9s} · cobro {money(b['cobro'])} · neto {neto} ({b['source']})")
    L = ledger(bs)
    print(f"Apostado ${L['staked']:,.2f} · cobrado ${L['cobro']:,.2f} · neto {L['net']:+,.2f} · ROI {L['roi'] or 0:.1%} · {L['win']}-{L['lose']}-{L['push']}")


if __name__ == "__main__":
    main()
