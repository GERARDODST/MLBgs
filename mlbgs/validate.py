"""Cotejo de datos: cada fuente se verifica contra otra independiente antes de usarla.

Cada chequeo devuelve OK / Alerta / Falla, con el detalle de lo que no cuadra. Las fallas
no detienen el modelo (los datos pueden venir con retraso de minutos entre endpoints),
pero quedan visibles en la página y en el gate de cada partido.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from . import mathlib as M
from .features import add_stats


def _check(cid, title, status, detail, bad=None, source=""):
    return {"id": cid, "title": title, "status": status, "detail": detail, "bad": bad or [], "source": source}


def validate(bundle: dict) -> dict:
    teams = bundle.get("teams", {})
    abbr = lambda t: (teams.get(str(t)) or {}).get("abbr") or str(t)  # noqa: E731
    results = bundle.get("results", [])
    st = bundle.get("standings", {})
    ts = bundle.get("teamStats", {})
    checks = []

    # 1. duplicados en el calendario
    seen = defaultdict(int)
    for g in results:
        seen[g["pk"]] += 1
    dups = [pk for pk, n in seen.items() if n > 1]
    checks.append(_check("dup", "Juegos duplicados en el calendario", "Falla" if dups else "OK",
                         f"{len(dups)} gamePk repetidos" if dups else
                         f"{len(results)} juegos finales, sin duplicados (los juegos suspendidos y reanudados se deduplican por gamePk)",
                         dups, "MLB Stats API /schedule"))

    # 2. standings vs resultados
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for g in results:
        aw = g["ar"] > g["hr"]
        agg[g["away"]][0 if aw else 1] += 1
        agg[g["home"]][1 if aw else 0] += 1
        agg[g["away"]][2] += g["ar"]
        agg[g["away"]][3] += g["hr"]
        agg[g["home"]][2] += g["hr"]
        agg[g["home"]][3] += g["ar"]
    bad = []
    for tid, s in st.items():
        c = agg[int(tid)]
        if (s.get("w"), s.get("l"), s.get("rs"), s.get("ra")) != tuple(c):
            bad.append(f"{abbr(tid)}: standings {s.get('w')}-{s.get('l')} ({s.get('rs')}/{s.get('ra')}) vs resultados "
                       f"{c[0]}-{c[1]} ({c[2]}/{c[3]})")
    checks.append(_check("standings", "Standings = suma de resultados (W-L, carreras a favor y en contra)",
                         "Alerta" if bad else "OK",
                         f"{len(bad)} equipo(s) no cuadran (suele ser un juego terminado después de la última "
                         "actualización de standings)" if bad else f"Los 30 equipos cuadran exactamente",
                         bad, "/standings vs /schedule"))

    # 3. linescore
    ls_bad = [f"{abbr(g['away'])} @ {abbr(g['home'])} {g['date']}" for g in results if g.get("inn") and (
        sum(x[0] or 0 for x in g["inn"]) != g["ar"] or sum(x[1] or 0 for x in g["inn"]) != g["hr"])]
    checks.append(_check("linescore", "Suma de carreras por entrada = marcador final",
                         "Falla" if ls_bad else "OK", f"{len(ls_bad)} juegos no cuadran" if ls_bad else
                         f"{sum(1 for g in results if g.get('inn'))} linescores cuadran", ls_bad, "/schedule?hydrate=linescore"))

    # 4. splits por equipo completos y consistentes (abridores + relevistas = total)
    missing = [abbr(t) for t, v in ts.items() if not all(k in v for k in ("sp", "rp", "hit_vl", "hit_vr"))]
    sprp = []
    for t, v in ts.items():
        if "sp" in v and "rp" in v:
            tot, sp, rp = add_stats(v["pitching"], None), add_stats(v["sp"], None), add_stats(v["rp"], None)
            if abs(tot.get("ip", 0) - sp.get("ip", 0) - rp.get("ip", 0)) > 0.01 or \
                    tot.get("runs", 0) != sp.get("runs", 0) + rp.get("runs", 0):
                sprp.append(f"{abbr(t)}: IP {tot.get('ip', 0):.1f} vs {sp.get('ip', 0) + rp.get('ip', 0):.1f}")
    live_teams = {t for g in bundle.get("live", []) for t in (g["away"], g["home"])}
    if live_teams and sprp:
        sprp = [x + (" · juego en curso: la API actualiza el total antes que los splits" if any(
            x.startswith(abbr(t) + ":") for t in live_teams) else "") for x in sprp]
    checks.append(_check("splits", "Splits abridores/relevistas y vs mano para los 30 equipos",
                         "Falla" if missing else "Alerta" if sprp else "OK",
                         (f"Faltan splits de {', '.join(missing)}" if missing else
                          f"{len(sprp)} equipos: abridores + relevistas ≠ total" if sprp else
                          "30 equipos completos; abridores + relevistas = total (IP y carreras)"),
                         missing + sprp, "/teams/stats?stats=statSplits"))

    # 5. carreras anotadas por la liga = carreras permitidas
    rs = sum((v.get("hitting") or {}).get("runs", 0) for v in ts.values())
    ra = sum((v.get("pitching") or {}).get("runs", 0) for v in ts.values())
    checks.append(_check("league", "Carreras anotadas de la liga = carreras permitidas", "OK" if rs == ra else "Alerta",
                         f"{rs} anotadas vs {ra} permitidas", [], "/teams/stats hitting vs pitching"))

    # 6. carreras de cada equipo (stats de bateo) = standings
    rs_bad = [f"{abbr(t)}: bateo {v.get('hitting', {}).get('runs')} vs standings {st.get(t, {}).get('rs')}"
              for t, v in ts.items() if st.get(t) and v.get("hitting", {}).get("runs") != st[t].get("rs")]
    checks.append(_check("teamruns", "Carreras del equipo (stats de bateo) = standings", "Alerta" if rs_bad else "OK",
                         f"{len(rs_bad)} equipos difieren" if rs_bad else "Los 30 equipos cuadran", rs_bad,
                         "/teams/stats vs /standings"))

    # 7-9. abridores probables
    era_bad, log_bad, team_bad = [], [], []
    sav = (bundle.get("savant") or {}).get("expected") or {}
    games_by_pitcher = {}
    for g in bundle.get("upcoming", []):
        for side in ("away", "home"):
            if g["probable"].get(side):
                games_by_pitcher[str(g["probable"][side])] = g[side]
    for pid, p in (bundle.get("pitchers") or {}).items():
        s = p.get("season") or {}
        ipv = M.ip_to_float(s.get("inningsPitched"))
        api = M.era(s.get("earnedRuns", 0), ipv) if ipv else None
        sv = sav.get(pid)
        if api is not None and sv and sv.get("era") is not None and abs(api - sv["era"]) > 0.015:
            era_bad.append(f"{p['name']}: API {api:.2f} vs Savant {sv['era']:.2f}")
        logs = p.get("log", [])
        lip = sum(M.ip_to_float(r["stat"].get("inningsPitched")) for r in logs)
        ler = sum(r["stat"].get("earnedRuns", 0) for r in logs)
        lk = sum(r["stat"].get("strikeOuts", 0) for r in logs)
        if logs and (abs(lip - ipv) > 0.01 or ler != s.get("earnedRuns") or lk != s.get("strikeOuts")):
            log_bad.append(f"{p['name']}: temporada {ipv:.1f} IP/{s.get('earnedRuns')} ER/{s.get('strikeOuts')} K vs "
                           f"game logs {lip:.1f}/{ler}/{lk}")
        last_team = logs[-1].get("team") if logs else None
        if last_team and games_by_pitcher.get(pid) and last_team != games_by_pitcher[pid]:
            team_bad.append(f"{p['name']}: última apertura con {abbr(last_team)}, hoy con {abbr(games_by_pitcher[pid])} (¿traspaso?)")
    checks.append(_check("era", "ERA del abridor: MLB Stats API vs Baseball Savant", "Alerta" if era_bad else "OK",
                         f"{len(era_bad)} difieren (Savant se actualiza de madrugada)" if era_bad else
                         f"{len(bundle.get('pitchers') or {})} abridores cuadran", era_bad, "/people vs Savant expected stats"))
    checks.append(_check("gamelog", "Temporada del abridor = suma de sus game logs (IP, ER, K)", "Falla" if log_bad else "OK",
                         f"{len(log_bad)} no cuadran" if log_bad else "Todos cuadran", log_bad, "/people season vs gameLog"))
    checks.append(_check("pitcherteam", "El abridor probable pertenece al equipo del partido", "Alerta" if team_bad else "OK",
                         f"{len(team_bad)} casos" if team_bad else "Todos coinciden", team_bad, "gameLog vs /schedule"))

    # 10. lineups: cada bateador en el roster activo del equipo
    rosters = bundle.get("rosters") or {}
    lu_bad = []
    n_lu = 0
    for g in bundle.get("upcoming", []):
        for side in ("away", "home"):
            ids = g["lineups"][side]
            ros = {p["id"] for p in (rosters.get(str(g[side])) or {}).get("players", [])}
            if not ids or not ros:
                continue
            n_lu += 1
            out = [g["lineupNames"][side].get(str(i), str(i)) for i in ids if i not in ros]
            if out:
                lu_bad.append(f"{abbr(g[side])}: {', '.join(out)} no aparecen en el roster activo")
    checks.append(_check("lineup", "Lineups confirmados contra el roster activo", "Alerta" if lu_bad else "OK",
                         f"{len(lu_bad)} lineups con jugadores fuera del roster" if lu_bad else
                         f"{n_lu} lineups confirmados, todos en su roster activo", lu_bad, "lineups vs /teams/{id}/roster"))

    # 11. park factors
    parks = (bundle.get("savant") or {}).get("park") or {}
    pk_bad = sorted({g.get("venueName") or str(g["venue"]) for g in bundle.get("upcoming", []) if str(g["venue"]) not in parks})
    checks.append(_check("park", "Park factor de Savant para cada estadio", "Alerta" if pk_bad else "OK",
                         f"Sin park factor: {', '.join(pk_bad)} (se usa 100)" if pk_bad else "Todos los estadios tienen índice",
                         pk_bad, "Savant statcast-park-factors"))

    # 12. cobertura de box scores (uso del bullpen)
    boxes = bundle.get("boxscores") or []
    dates = sorted({b["date"] for b in boxes})
    cov_bad = []
    if dates:
        in_window = [g for g in results if dates[0] <= g["date"] <= dates[-1]]
        have = {b["pk"] for b in boxes}
        miss = [g for g in in_window if g["pk"] not in have]
        if miss:
            cov_bad = [f"{abbr(g['away'])} @ {abbr(g['home'])} {g['date']}" for g in miss]
    checks.append(_check("boxes", "Box scores de la ventana de uso del bullpen", "Alerta" if cov_bad else "OK",
                         f"Faltan {len(cov_bad)} box scores" if cov_bad else
                         f"{len(boxes)} box scores ({dates[0] if dates else '—'} a {dates[-1] if dates else '—'}), completos",
                         cov_bad, "/game/{pk}/boxscore"))

    # 13. récord del calendario = standings
    rec_bad = []
    for g in bundle.get("upcoming", []):
        for side in ("away", "home"):
            rec = (g.get("records") or {}).get(side) or {}
            s = st.get(str(g[side])) or {}
            if rec and s and (rec.get("wins"), rec.get("losses")) != (s.get("w"), s.get("l")):
                rec_bad.append(f"{abbr(g[side])}: calendario {rec.get('wins')}-{rec.get('losses')} vs standings {s.get('w')}-{s.get('l')}")
    checks.append(_check("records", "Récord en el calendario = standings", "Alerta" if rec_bad else "OK",
                         f"{len(set(rec_bad))} diferencias (juegos del mismo día ya terminados)" if rec_bad else "Coinciden",
                         sorted(set(rec_bad)), "/schedule leagueRecord vs /standings"))

    # 14. momios: casas fuera del consenso
    odds_bad = []
    for ev in bundle.get("odds") or []:
        probs = []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m["key"] == "h2h" and len(m["outcomes"]) == 2:
                    o = sorted(m["outcomes"], key=lambda x: x["name"])
                    probs.append((bk["title"], M.american_to_prob(o[0]["price"])))
        if len(probs) >= 3:
            med = statistics.median(p for _, p in probs)
            for book, p in probs:
                if abs(p - med) > 0.05:
                    odds_bad.append(f"{ev.get('away_team')} @ {ev.get('home_team')}: {book} {p:.3f} vs mediana {med:.3f}")
    checks.append(_check("odds", "Momios: casas fuera del consenso (>5 pp de la mediana)",
                         "Alerta" if odds_bad else "OK" if bundle.get("odds") else "N/A",
                         f"{len(odds_bad)} casos" if odds_bad else ("Consenso sin atípicos" if bundle.get("odds")
                                                                    else "Sin momios automáticos conectados"),
                         odds_bad, "The Odds API"))

    counts = defaultdict(int)
    for c in checks:
        counts[c["status"]] += 1
    return {"checks": checks, "counts": dict(counts)}
