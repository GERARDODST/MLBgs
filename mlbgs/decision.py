"""Decisión del partido: UNA opción (pick + stake 1–10) que conecta todo el análisis.

Orden de trabajo (el mismo que sigue la conclusión de Claude en la página):
  1. Framework completo: contexto y motivación, forma, abridores, bullpen y fatiga, modelo de carreras
     y triangulación, guion, parque/clima/umpire, mercado, contradicciones y gate de datos (secciones 1-10).
  2. Modelo del Pro-Lab, si el partido lo tiene: sus probabilidades entran a los picks y su acuerdo con
     el framework mueve el stake.
  3. Cuotas: el stake final depende del momio que te den (escalera de stake 1, 5 y 10).
  4. Historial: si el modelo acierta menos de lo que promete, su stake baja.

Regla de elección: entre los picks del partido (framework, o framework + modelo del Pro-Lab), el de
mayor índice de confianza que pasa las reglas del semáforo (guion, contradicciones, datos). Si el mejor
solo espera datos obligatorios (lineup, umpire…) la decisión es «esperar»; si ninguno pasa, «no apostar».
"""
from __future__ import annotations

from . import stake as ST


def _side(pick: dict, a: dict) -> str | None:
    """Equipo al que favorece el pick (away/home) o None para totales y NRFI."""
    txt = pick["pick"]
    t = a["teams"]
    for side in ("away", "home"):
        if txt == t[side]["abbr"] or txt.startswith(t[side]["abbr"] + " "):
            return None if pick["family"] == "Team total" and "Under" in txt else side
    if pick["family"] == "K":
        for side in ("away", "home"):
            if (a["summary"]["probables"].get(side) or {}).get("name", "") and txt.startswith(a["summary"]["probables"][side]["name"]):
                return side if "Over" in txt else ("home" if side == "away" else "away")
    return None


def _tone(side: str | None, adv_side: str | None, level: str | None = None) -> str:
    if not side or not adv_side or level in ("equilibrado", "sin dato"):
        return "neutral"
    return "pro" if side == adv_side else "contra"


def checklist(a: dict, pick: dict, lab: dict | None, track: dict) -> list[dict]:
    S, s = a["sections"], a["summary"]
    t = a["teams"]
    side = _side(pick, a)
    name = {k: t[k]["abbr"] for k in ("away", "home")}
    out = []

    # 1. contexto y motivación
    imp = (S["s1"].get("importance") or {}).get("rows") or []
    ctx = "; ".join(f"{r['team']}: {r.get('status', '—')} (playoffs {100 * (r.get('pPlayoffs') or 0):.0f}%)" for r in imp)
    ser = a.get("series") or {}
    extra = f" · serie: juego {ser.get('gameNumber')} de {ser.get('totalGames')}{', ' + ser['result'] if ser.get('result') else ''}" if ser.get("gameNumber") else ""
    tone = "neutral"
    if side and imp:
        mine = next((r for r in imp if r["team"] == name[side]), None)
        other = next((r for r in imp if r["team"] != name[side]), None)
        if mine and other:
            dm = max(((mine.get("leverage") or {}).get(k) or {}).get("delta", 0) for k in ("po", "div"))
            do = max(((other.get("leverage") or {}).get(k) or {}).get("delta", 0) for k in ("po", "div"))
            tone = "pro" if dm - do > 0.05 else "contra" if do - dm > 0.05 else "neutral"
    out.append({"k": "Contexto y motivación", "v": (ctx or "sin dato") + ("; divisional" if S["s1"].get("divisional") else "") + extra, "tone": tone})

    # 2. forma reciente
    summ = S["s2"].get("summary") or []
    out.append({"k": "Forma e historial", "v": "; ".join(f"{x['k']}: {x['v']}" for x in summ[:3]) or "sin dato", "tone": "neutral"})

    # 3. abridores
    adv = S["s3"].get("advantage") or {}
    out.append({"k": "Abridores", "v": adv.get("text") or "sin dato", "tone": _tone(side, adv.get("side"), adv.get("level"))})

    # 4. bullpen y fatiga
    bp = S["s4"].get("advantage") or {}
    out.append({"k": "Bullpen y fatiga", "v": S["s4"].get("conclusion") or "sin dato",
                "tone": "neutral" if pick["family"] in ("F5", "F5 total", "NRFI") else _tone(side, bp.get("side"))})

    # 5. modelo de carreras y triangulación
    tri = S["s5"]["triangulation"]
    proj = s["proj"]
    tri_txt = (f"P({name['home']}) Log5 {tri['log5']['pHome']:.0%} · Elo {tri['elo']['pHome']:.0%} · λ {tri['lambda']['pHome']:.0%}"
               f" · triangulación {s.get('confidence')} · marcador esperado {name['away']} {proj['away']:.1f}–{proj['home']:.1f} {name['home']}")
    fav = "home" if s["pHome"] >= 0.5 else "away"
    out.append({"k": "Modelo de carreras", "v": tri_txt, "tone": _tone(side, fav)})

    # 6. guion y 7. contradicciones del mercado del pick
    g = pick.get("guion")
    out.append({"k": "Guion del partido", "v": f"{g or '—'}: {pick.get('guionWhy') or 'sin detalle'}",
                "tone": "pro" if g == "Sí" else "contra" if g == "No" else "neutral"})
    c = pick.get("contradiction")
    cl = pick.get("contra") or []
    out.append({"k": "Contradicciones", "v": f"{c or '—'}" + (": " + "; ".join(cl) if cl else ""),
                "tone": "pro" if c == "Baja" else "contra" if c == "Alta" else "neutral"})

    # 8. parque, clima y umpire
    w = a.get("weather") or {}
    wtxt = f"{w.get('condition', '')} {w.get('temp', '')}°F, viento {w.get('windText', '—')}" if w.get("available") else "clima sin dato"
    out.append({"k": "Parque, clima y umpire", "v": f"{S['s6'].get('parkText', '')} {wtxt} · umpire {a.get('umpire') or 'sin anunciar'}", "tone": "neutral"})

    # 9. datos obligatorios
    missing = [f["field"] for f in S["s9"]["fields"] if f.get("status") == "faltante"]
    blocked = pick.get("blockedWithOdds")
    lu = s.get("lineupsConfirmed") or {}
    miss_txt = ("Falta: " + ", ".join(missing) + ("" if blocked else " (no frena este mercado o lo cubre tu momio)")) if missing else "Completos"
    out.append({"k": "Datos obligatorios", "v": miss_txt + f" · lineups {'confirmados' if lu and all(lu.values()) else 'pendientes'}",
                "tone": "contra" if blocked else "pro"})

    # 10. mercado y cuotas
    st = pick.get("stake") or {}
    lad = " · ".join(f"stake {x['level']} desde {x['from']:+d}" for x in st.get("ladder") or [])
    out.append({"k": "Cuotas", "v": f"probabilidad {pick['p']:.1%}, momio justo {pick['fair']:+.0f}, mínimo aceptable {pick['minPrice']:+.0f}"
                + (f"; {lad}" if lad else "") + ("" if S["s7"].get("hasOdds") else "; sin momios conectados: el stake final depende del momio que te den"),
                "tone": "neutral"})

    # 11. modelo del Pro-Lab
    if lab:
        mk = lab.get("modelKey")
        mname = ST.LAB_METHODS.get(mk)
        pm = (pick.get("methods") or {}).get(mname)
        agree = st.get("a", 1.0)
        out.append({"k": f"Modelo {lab.get('model') or mk}", "v": (f"lo ve en {pm:.1%}" if pm is not None else "no evalúa este mercado")
                    + f" · acuerdo con el framework {agree:.2f}", "tone": "pro" if pm is not None and agree >= 1 else "contra" if agree < 1 else "neutral"})

    # 12. historial del modelo
    key = (lab or {}).get("modelKey") or "framework"
    tr = track.get(key)
    if tr:
        out.append({"k": "Historial", "v": f"picks principales {tr['hit']:.0%} de acierto contra {tr['exp']:.0%} esperado (n={tr['n']}) → factor {tr['h']:.3f}",
                    "tone": "pro" if tr["h"] > 1.0 else "contra" if tr["h"] < 0.995 else "neutral"})
    return out


def decide(a: dict, picks: list[dict], lab: dict | None = None, track: dict | None = None) -> dict:
    track = track or {}
    cands = sorted(picks or [], key=lambda p: -p["ic"])
    if not cands:
        return {"status": "no apostar", "why": "sin picks evaluados", "pick": None}
    ok = [p for p in cands if not (p.get("stake") or {}).get("block") and (p.get("stake") or {}).get("ladder")]
    wait = [p for p in cands if "dato obligatorio" in ((p.get("stake") or {}).get("block") or "")]
    if ok:
        p = ok[0]
        lvl = (p.get("stake") or {}).get("level", 0)
        status = "apostar" if lvl else "no apostar"
        why = ("el pick con más confianza que pasa guion, contradicciones y datos obligatorios" if lvl else
               f"confianza {p['ic']:.0f} (< 55): solo entra con un momio mejor (stake 1 desde {p['stake']['ladder'][0]['from']:+d})")
    elif wait:
        p, status = wait[0], "esperar"
        why = "falta un dato obligatorio (lineup, umpire, clima o abridor): se confirma cerca del primer lanzamiento"
    else:
        p, status = cands[0], "no apostar"
        why = (p.get("stake") or {}).get("block") or "ningún pick alcanza stake con un momio razonable"
    st = p.get("stake") or {}
    return {
        "status": status, "why": why,
        "source": f"Framework v2 + {lab.get('model')}" if lab else "Framework v2",
        "pick": {k: p.get(k) for k in ("pick", "market", "family", "p", "ic", "level", "fair", "minPrice", "line")},
        "stake": {"level": st.get("level", 0) if status == "apostar" else 0, "amount": ST.AMOUNTS.get(st.get("level", 0), 0) if status == "apostar" else 0,
                  "minPrice": (st.get("ladder") or [{}])[0].get("from"), "ladder": st.get("ladder") or [], "a": st.get("a", 1.0), "h": st.get("h", 1.0)},
        "alternatives": [{"pick": q["pick"], "market": q["market"], "ic": q["ic"]} for q in cands if q is not p][:3],
        "checklist": checklist(a, p, lab, track),
    }


def attach(analyses: list[dict], labs: list[dict], track: dict) -> None:
    """Decisión final de cada partido (con los picks del Pro-Lab si existe) y la del framework solo."""
    by_pk = {lab["pk"]: lab for lab in labs}
    for a in analyses:
        fw = decide(a, a.get("picks") or [], None, track)
        lab = by_pk.get(a["pk"])
        a["decisionFramework"] = fw
        a["decision"] = decide(a, lab.get("picks") or [], lab, track) if lab else fw
    for lab in labs:
        if lab.get("base"):
            lab["decision"] = decide(lab["base"], lab.get("picks") or [], lab, track)
