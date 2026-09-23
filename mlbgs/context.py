"""Contexto del partido que no es estadística pura del juego: lineup y bullpen (confirmados o
proyectados), importancia en la tabla (simulación del resto de la temporada), noticias oficiales,
matchup del arsenal de pitcheos y park factor propio cuando Savant no lo publica.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

from . import features as F
from . import mathlib as M


# ============================================================ park factor propio

def own_park_factors(results: list[dict], teams: dict) -> dict[int, dict]:
    """PF = carreras/juego en el estadio ÷ carreras/juego del mismo equipo de visita, regresado (k = 80 juegos)."""
    home_venue = {int(t): v.get("venue") for t, v in teams.items()}
    at_venue = defaultdict(lambda: [0, 0])  # carreras, juegos
    road = defaultdict(lambda: [0, 0])
    for g in results:
        v = g.get("venue")
        if v and v == home_venue.get(g["home"]):
            at_venue[v][0] += g["ar"] + g["hr"]
            at_venue[v][1] += 1
        road[g["away"]][0] += g["ar"] + g["hr"]
        road[g["away"]][1] += 1
    out = {}
    for tid, v in home_venue.items():
        if not v or not at_venue[v][1] or not road[tid][1]:
            continue
        raw = (at_venue[v][0] / at_venue[v][1]) / (road[tid][0] / road[tid][1])
        n = at_venue[v][1]
        out[v] = {"runs": round(100 * M.shrink(raw, n, 80, 1.0), 1), "raw": round(100 * raw, 1), "games": n}
    return out


# ============================================================ lineup proyectado

def team_boxes(bundle: dict, tid: int) -> list[dict]:
    rows = []
    for b in bundle.get("boxscores", []):
        for side, opp in (("away", "home"), ("home", "away")):
            if b[side]["team"] == tid and b[side].get("lineup"):
                opp_sp = (b[opp]["pitchers"] or [{}])[0].get("id") if b[opp].get("pitchers") else None
                rows.append({"date": b["date"], "pk": b["pk"], "lineup": b[side]["lineup"], "oppStarter": opp_sp,
                             "pitchers": b[side]["pitchers"]})
    return sorted(rows, key=lambda r: (r["date"], r["pk"]), reverse=True)


def projected_lineup(bundle: dict, tid: int, opp_hand: str | None, hands: dict, roster_ids: set | None,
                     n_games: int = 10) -> dict | None:
    """Lineup más probable: aperturas recientes (peso 0.85^i), doble peso contra abridores de la misma mano."""
    games = team_boxes(bundle, tid)[:n_games]
    if not games:
        return None
    w_start = defaultdict(float)
    slot_sum = defaultdict(float)
    starts = defaultdict(int)
    same_starts = defaultdict(int)
    pos_count = defaultdict(lambda: defaultdict(float))
    names = {}
    n_same = 0
    for i, g in enumerate(games):
        same = opp_hand is not None and hands.get(g["oppStarter"]) == opp_hand
        n_same += same
        w = (0.85 ** i) * (2.0 if same else 1.0)
        for slot, p in enumerate(g["lineup"][:9]):
            pid = p["id"]
            if roster_ids and pid not in roster_ids:
                continue
            names[pid] = p.get("name")
            w_start[pid] += w
            slot_sum[pid] += w * (slot + 1)
            starts[pid] += 1
            same_starts[pid] += same
            pos_count[pid][p.get("pos") or "?"] += w
    chosen = sorted(w_start, key=w_start.get, reverse=True)[:9]
    has_c = any(max(pos_count[p], key=pos_count[p].get) == "C" for p in chosen)
    if not has_c:
        catchers = [p for p in w_start if max(pos_count[p], key=pos_count[p].get) == "C" and p not in chosen]
        if catchers and chosen:
            chosen[-1] = max(catchers, key=w_start.get)
    chosen.sort(key=lambda p: slot_sum[p] / w_start[p])
    total_w = sum((0.85 ** i) * (2.0 if (opp_hand and hands.get(g["oppStarter"]) == opp_hand) else 1.0)
                  for i, g in enumerate(games))
    rows = []
    for p in chosen:
        rows.append({"id": p, "name": names.get(p), "pos": max(pos_count[p], key=pos_count[p].get),
                     "slot": round(slot_sum[p] / w_start[p], 1), "starts": starts[p], "games": len(games),
                     "sameStarts": same_starts[p], "sameGames": n_same, "prob": min(1.0, w_start[p] / total_w)})
    conf = sum(r["prob"] for r in rows) / 9 if rows else 0
    return {"ids": [r["id"] for r in rows], "rows": rows, "confidence": conf, "games": len(games), "sameGames": n_same}


# ============================================================ bullpen completo

def full_bullpen(bundle: dict, ctx_bp: dict, tid: int, starter_id: int | None, today, lg, roster: dict | None,
                 official_bullpen: list | None = None) -> list[dict]:
    """Todos los pitchers del roster activo (menos el abridor del día) con rol, fatiga y disponibilidad."""
    pitchers = [p for p in (roster or {}).get("players", []) if p.get("pos") == "P" and p["id"] != starter_id]
    if not pitchers:
        staff = ctx_bp.get("staff", {}).get(tid, set())
        by_id = {p["id"]: p for p in bundle.get("playersPitching", []) if p.get("team") == tid}
        pitchers = [{"id": i, "name": by_id[i]["name"], "pit": by_id[i]["stat"], "throws": (bundle.get("hands") or {}).get(str(i)),
                     "pitVs": {}} for i in staff if i in by_id and i != starter_id]
    used = ctx_bp["usage"].get(tid, {})
    out = []
    for p in pitchers:
        s = F.add_stats(p.get("pit"), None)
        gp = s.get("gamesPlayed", 0)
        gs = s.get("gamesStarted", 0)
        ipv = s.get("ip", 0)
        bf = s.get("battersFaced", 0) or 1
        apps = sorted(used.get(p["id"], []), key=lambda a: a["date"], reverse=True)
        g3 = [a for a in apps if 1 <= a["daysAgo"] <= 3]
        consecutive = {a["daysAgo"] for a in apps} >= {1, 2}
        yday = sum(a["pitches"] for a in apps if a["daysAgo"] == 1)
        p3 = sum(a["pitches"] for a in g3)
        starter_like = gp and gs / gp >= 0.5
        if starter_like:
            avail, risk = "Rotación", "—"
        elif len(g3) >= 3 or (consecutive and yday >= 15) or yday >= 35:
            avail, risk = "No disponible", "Alto"
        elif consecutive or yday >= 20 or p3 >= 45:
            avail, risk = "Dudoso", "Medio"
        else:
            avail, risk = "Disponible", "Bajo"
        k_s = M.shrink(s.get("strikeOuts", 0) / bf, bf, 70, lg.k_rate)
        bb_s = M.shrink(s.get("baseOnBalls", 0) / bf, bf, 170, lg.bb_rate)
        hr_s = M.shrink(s.get("homeRuns", 0) / bf, bf, 1150, lg.hr_rate)
        fip_s = (13 * hr_s + 3 * (bb_s + lg.hbp_rate) - 2 * k_s) * lg.bf_per_ip + lg.c_fip
        out.append({
            "id": p["id"], "name": p.get("name"), "throws": p.get("throws") or (bundle.get("hands") or {}).get(str(p["id"])),
            "g": int(gp), "gs": int(gs), "ip": ipv, "era": M.era(s.get("earnedRuns", 0), ipv) if ipv else None,
            "whip": M.whip(s.get("hits", 0), s.get("baseOnBalls", 0), ipv) if ipv else None,
            "fip": M.fip(s.get("homeRuns", 0), s.get("baseOnBalls", 0), s.get("hitByPitch", 0), s.get("strikeOuts", 0),
                         ipv, lg.c_fip) if ipv else None,
            "fipShrunk": fip_s, "k": s.get("strikeOuts", 0) / bf, "bb": s.get("baseOnBalls", 0) / bf,
            "sv": int(s.get("saves", 0)), "hld": int(s.get("holds", 0)), "bs": int(s.get("blownSaves", 0)),
            "ipPerG": ipv / gp if gp else 0, "lastDate": apps[0]["date"] if apps else None,
            "lastPitches": apps[0]["pitches"] if apps else None, "g3": len(g3), "p3": p3, "consecutive": consecutive,
            "available": avail, "risk": risk, "starterLike": bool(starter_like),
            "official": (p["id"] in official_bullpen) if official_bullpen is not None else None,
            "stats": s, "vs": p.get("pitVs") or {},
        })
    rel = [r for r in out if not r["starterLike"] and r["g"] > 0]
    if rel:
        closer = max(rel, key=lambda r: (r["sv"], r["hld"]))
        if closer["sv"] >= 3:
            closer["role"] = "Cerrador"
        for r in sorted([r for r in rel if not r.get("role")], key=lambda r: r["hld"], reverse=True)[:2]:
            if r["hld"] >= 3:
                r["role"] = "Setup"
        for r in rel:
            if not r.get("role"):
                r["role"] = "Largo" if r["ipPerG"] >= 1.6 else "Intermedio"
    for r in out:
        r.setdefault("role", "Rotación" if r["starterLike"] else "Sin uso")
        base = {"Cerrador": 0.42, "Setup": 0.48, "Intermedio": 0.30, "Largo": 0.16, "Rotación": 0.02, "Sin uso": 0.05}[r["role"]]
        mult = {"Disponible": 1.0, "Dudoso": 0.45, "No disponible": 0.08, "Rotación": 1.0}[r["available"]]
        r["pUse"] = min(0.95, base * mult)
    order = {"Cerrador": 0, "Setup": 1, "Intermedio": 2, "Largo": 3, "Sin uso": 4, "Rotación": 5}
    return sorted(out, key=lambda r: (order[r["role"]], -(r["pUse"])))


# ============================================================ importancia: simulación del resto de la temporada

def playoff_sim(bundle: dict, T, lg, game: dict | None = None, n: int = 6000, seed: int = 7) -> dict:
    """P(playoffs), P(división) y P(bye) por equipo; si se da un partido, el apalancamiento de ganarlo o perderlo.
    Formato actual: 3 campeones de división + 3 comodines por liga; los 2 mejores campeones descansan la 1ª ronda."""
    rng = random.Random(seed)
    st = {int(k): v for k, v in bundle.get("standings", {}).items()}
    info = T.info
    remaining = [g for g in bundle.get("remaining", []) if g["away"] in st and g["home"] in st]
    if not st or not remaining:
        return {"available": False}
    talent = {t: T.pythag(t)["true"] for t in st}
    p_home = {}
    for g in remaining:
        p = M.with_home_edge(M.log5(talent[g["home"]], talent[g["away"]]), lg.home_odds_ratio)
        p_home[g["pk"]] = p
    leagues = defaultdict(list)
    for t in st:
        leagues[(info.get(t) or {}).get("leagueId")].append(t)
    divisions = defaultdict(list)
    for t in st:
        divisions[(info.get(t) or {}).get("divisionId")].append(t)

    def run(force: dict | None):
        cnt = {k: defaultdict(int) for k in ("po", "div", "bye")}
        for _ in range(n):
            w = {t: st[t]["w"] for t in st}
            for g in remaining:
                if force and g["pk"] in force:
                    home_wins = force[g["pk"]]
                else:
                    home_wins = rng.random() < p_home[g["pk"]]
                w[g["home"] if home_wins else g["away"]] += 1
            tb = {t: rng.random() for t in st}
            div_win = set()
            for d, ts in divisions.items():
                div_win.add(max(ts, key=lambda t: (w[t], tb[t])))
            for lgid, ts in leagues.items():
                champs = sorted([t for t in ts if t in div_win], key=lambda t: (w[t], tb[t]), reverse=True)
                wc = sorted([t for t in ts if t not in div_win], key=lambda t: (w[t], tb[t]), reverse=True)[:3]
                for t in champs:
                    cnt["div"][t] += 1
                    cnt["po"][t] += 1
                for t in champs[:2]:
                    cnt["bye"][t] += 1
                for t in wc:
                    cnt["po"][t] += 1
        return {k: {t: v[t] / n for t in st} for k, v in cnt.items()}

    base = run(None)
    out = {"available": True, "sims": n, "remainingGames": len(remaining),
           "teams": {t: {"po": base["po"][t], "div": base["div"][t], "bye": base["bye"][t]} for t in st}}
    if game and any(g["pk"] == game["pk"] for g in remaining):
        hw = run({game["pk"]: True})
        aw = run({game["pk"]: False})
        lev = {}
        for side in ("away", "home"):
            t = game[side]
            win, lose = (hw, aw) if side == "home" else (aw, hw)
            lev[side] = {k: {"win": win[k][t], "lose": lose[k][t], "delta": win[k][t] - lose[k][t]}
                         for k in ("po", "div", "bye")}
        out["leverage"] = lev
    return out


def importance(ctx, g: dict, sim: dict | None) -> dict:
    T = ctx.T
    rows = []
    notes = []
    top = 0.0
    for side in ("away", "home"):
        tid = g[side]
        s = T.standings.get(tid, {})
        info = T.info.get(tid, {})
        p = (sim or {}).get("teams", {}).get(tid) if sim and sim.get("available") else None
        lev = ((sim or {}).get("leverage") or {}).get(side)
        eliminated = s.get("elim") == "E" and s.get("wcElim") in ("E", None)
        status = "Clasificado" if s.get("clinched") else "Eliminado" if eliminated else "En la pelea"
        if lev:
            top = max(top, abs(lev["po"]["delta"]), abs(lev["div"]["delta"]) * 0.5, abs(lev["bye"]["delta"]) * 0.5)
        rows.append({"team": info.get("abbr"), "division": info.get("division"), "w": s.get("w"), "l": s.get("l"),
                     "divRank": s.get("divRank"), "gb": s.get("gb"), "wcRank": s.get("wcRank"), "wcgb": s.get("wcgb"),
                     "elim": s.get("elim"), "wcElim": s.get("wcElim"), "magic": s.get("magic"), "clinch": s.get("clinch"),
                     "status": status, "pPlayoffs": p["po"] if p else None, "pDivision": p["div"] if p else None,
                     "pBye": p["bye"] if p else None, "leverage": lev})
        if status == "Eliminado":
            notes.append(f"{info.get('abbr')} está eliminado: suele dar descanso a titulares y probar novatos o relevistas "
                         "de septiembre; el lineup proyectado pierde confiabilidad.")
        elif status == "Clasificado":
            notes.append(f"{info.get('abbr')} ya clasificó: posible manejo de carga (titulares descansando, bullpen protegido), "
                         "salvo que aún pelee por siembra (bye).")
    level = "Alta" if top >= 0.05 else "Media" if top >= 0.01 else "Baja"
    if all(r["status"] == "Eliminado" for r in rows):
        level = "Nula"
        notes.append("Ninguno de los dos se juega nada en la tabla: partido de baja motivación competitiva.")
    return {"rows": rows, "level": level, "topLeverage": top, "notes": notes,
            "method": (f"Simulación Monte Carlo del resto de la temporada ({(sim or {}).get('sims', 0)} temporadas, "
                       f"{(sim or {}).get('remainingGames', 0)} juegos restantes): Log5 con win% verdadero + ventaja de local; "
                       "3 campeones de división + 3 comodines por liga.") if sim and sim.get("available") else "Sin simulación"}


# ============================================================ noticias oficiales

def news(bundle: dict, tid: int, lineup_ids: set, limit: int = 12) -> dict:
    tx = [t for t in bundle.get("transactions", []) if t.get("team") == tid]
    injured = ((bundle.get("rosters") or {}).get(str(tid)) or {}).get("injured", [])
    hitters = {p["id"]: p for p in bundle.get("playersHitting", [])}
    items = []
    for t in tx[:limit]:
        pa = F.add_stats((hitters.get(t.get("person")) or {}).get("stat"), None).get("plateAppearances", 0)
        impact = "Alta" if t.get("code") == "SC" and pa >= 250 else "Media" if t.get("code") in ("SC", "TR", "DES") else "Baja"
        items.append({**t, "impact": impact, "pa": int(pa)})
    return {"items": items, "injured": injured}


# ============================================================ arsenal vs perfil del rival (3.3, 5.1)

def arsenal_matchup(bundle: dict, pitcher_id: int | None, batter_ids: list[int]) -> dict | None:
    """Σ uso × (run value/100 del pitcheo − run value/100 de los bateadores contra ese pitcheo), regresado.
    Positivo = ventaja del pitcher (carreras por 100 pitcheos)."""
    ars = bundle.get("arsenal") or {}
    rows = (ars.get("pitcher") or {}).get(str(pitcher_id)) if pitcher_id else None
    if not rows:
        return None
    bat = ars.get("batter") or {}
    detail = []
    total = 0.0
    use_sum = sum((r.get("usage") or 0) for r in rows) or 1
    for r in rows:
        u = (r.get("usage") or 0) / use_sum
        p_rv = M.shrink(r.get("rv100") or 0, r.get("pitches") or 0, 300, 0.0)
        vals = []
        for b in batter_ids:
            for x in bat.get(str(b), []):
                if x.get("type") == r.get("type"):
                    vals.append(M.shrink(x.get("rv100") or 0, x.get("pa") or 0, 60, 0.0))
        b_rv = sum(vals) / len(batter_ids) if batter_ids else 0.0
        edge = p_rv - b_rv
        total += u * edge
        detail.append({"type": r.get("type"), "name": r.get("name"), "usage": r.get("usage"), "pitcherRv": r.get("rv100"),
                       "pitcherRvShrunk": p_rv, "batterRv": b_rv, "batters": len(vals), "whiff": r.get("whiff"),
                       "woba": r.get("woba"), "edge": edge})
    return {"score": total, "rows": sorted(detail, key=lambda d: -(d["usage"] or 0))}
