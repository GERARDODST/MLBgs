"""Pro-Lab: prueba del modelo DIAMANTE-24 sobre un partido con los datos congelados antes del primer lanzamiento.

Uso:
    python -m mlbgs.prolab            # WSH @ DET, 23-sep-2026 (gamePk 824223)

Entrada (carpeta prolab/, capturada por .github/workflows/prolab-snapshot.yml antes del partido):
  * bundle_fixed_{pk}_pre.json.gz  bundle completo corregido (16:55Z)
  * bundle_1609_{pk}_pre.json.gz   bundle de las 16:09Z con el partido en "Preview" y sus abridores
  * snapshot_{pk}_pre.json.gz      feed del partido (lineups y bullpen oficiales), rosters con splits,
                                   transacciones, historial bateador vs pitcher, box scores recientes
  * result_{pk}.json               (opcional) resultado oficial, para comparar después del partido
Salida: prolab/prolab_{pk}.json (se embebe en la página).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import os
import statistics
import sys
import time

from . import context as C
from . import features as F
from . import fetch as FE
from . import markov as MK
from . import mathlib as M
from . import model
from . import picks as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "prolab")
MODEL_NAME = "DIAMANTE-24"
SIDES = ("away", "home")


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ============================================================ ensamblar los datos congelados

def assemble(pk: int) -> tuple[dict, dict, dict]:
    base = load(os.path.join(DIR, f"bundle_fixed_{pk}_pre.json.gz"))
    pre = load(os.path.join(DIR, f"bundle_1609_{pk}_pre.json.gz"))   # 16:09Z: el partido aún en "Preview"
    snap = load(os.path.join(DIR, f"snapshot_{pk}_pre.json.gz"))
    game = next(g for g in pre["upcoming"] if g["pk"] == pk)
    feed = snap["feed"]
    box = feed["liveData"]["boxscore"]["teams"]
    teams = (game["away"], game["home"])
    # lineups y bullpen oficiales del feed del partido (publicados por el club antes del juego)
    for side in SIDES:
        order = box[side].get("battingOrder") or []
        if order:
            game["lineups"][side] = order
            pl = box[side]["players"]
            game["lineupNames"][side] = {str(i): pl[f"ID{i}"]["person"]["fullName"] for i in order if f"ID{i}" in pl}
    game["officialBullpen"] = {s: box[s].get("bullpen") or [] for s in SIDES}
    game["officialBench"] = {s: box[s].get("bench") or [] for s in SIDES}
    game["weather"] = feed["gameData"].get("weather") or game.get("weather")
    game["officials"] = [{"type": o["officialType"], "name": o["official"]["fullName"]}
                         for o in feed["liveData"]["boxscore"].get("officials", [])] or game.get("officials")
    base["upcoming"] = [game]
    base["pitchers"] = {**base.get("pitchers", {}), **{k: v for k, v in pre["pitchers"].items()
                                                       if int(k) in game["probable"].values()}}
    base["rosters"] = {str(t): FE.parse_roster(snap[f"roster_{t}"], snap.get(f"roster40_{t}") or {}) for t in teams}
    rows = [t for tid in teams for t in (snap.get(f"transactions_{tid}") or {}).get("transactions", [])]
    base["transactions"] = FE.parse_transactions(rows, set(teams))
    base["meta"] = {**base["meta"], "generatedAt": snap["takenAt"], "today": game["date"]}
    return base, snap, game


# ============================================================ tasas por turno

def rates_batter(stat, vs_stat, pitcher_hand, L):
    c, n = MK.event_counts(stat)
    overall = MK.shrunk_rates(c, n, L, MK.K_BAT)
    if vs_stat:
        cv, nv = MK.event_counts(vs_stat)
        if nv:
            return MK.shrunk_rates(cv, nv, overall, {e: 3 * k for e, k in MK.K_BAT.items()}), overall
    return overall, overall


def rates_pitcher(stat, prev, vs_stat, L):
    comb = F.add_stats(stat, prev, 0.5)
    comb["battersFaced"] = comb.get("battersFaced", 0)
    c, n = MK.event_counts(comb, pitcher=True)
    overall = MK.shrunk_rates(c, n, L, MK.K_PIT)
    if vs_stat:
        cv, nv = MK.event_counts(vs_stat, pitcher=True)
        if nv:
            return MK.shrunk_rates(cv, nv, overall, {e: 3 * k for e, k in MK.K_PIT.items()}), overall
    return overall, overall


def gb_share(stat, lg_gb):
    go, ao = (stat or {}).get("groundOuts", 0), (stat or {}).get("airOuts", 0)
    return M.shrink(go / (go + ao) if go + ao else lg_gb, go + ao, 70, lg_gb)


# ============================================================ corrida principal

def run(pk: int = 824223, n_sims: int = 50000, n_morning: int = 20000) -> dict:
    t0 = time.time()
    bundle, snap, game = assemble(pk)
    ctx = model.Context(bundle)
    lg = ctx.lg
    base = model.analyze(ctx, game)            # las 10 secciones del framework (modelo MLBgs)
    L = MK.league_rates(lg.tot)
    e_rate = MK.calibrate_residual(L, lg.runs_per_half, lg.gb_rate)
    re_lg = MK.re24(MK.with_residual(L, e_rate), lg.gb_rate)
    re_raw = MK.re24(L, lg.gb_rate)

    # --- parque y clima
    park = ctx.park_raw.get(game["venue"], {})

    def idx(key, fallback):
        v = park.get(key) or fallback
        return 1 + 0.5 * ((v or 100) / 100 - 1)       # índice de 3 años, regresado a la mitad

    wx = model.parse_weather(game.get("weather"), game.get("roof"))
    wind_hr = 1.0
    if wx["out"] and (wx["speed"] or 0) >= 10:
        wind_hr = 1.10 if wx["speed"] < 16 else 1.15
    elif wx["in"] and (wx["speed"] or 0) >= 10:
        wind_hr = 0.90 if wx["speed"] < 16 else 0.85
    runs_i = park.get("runs") or 100
    mult = {"HR": idx("hr", runs_i) * wind_hr, "K": idx("so", 100), "BB": idx("bb", 100),
            "1B": idx("1b", runs_i), "2B": idx("2b", runs_i), "3B": idx("3b", runs_i)}

    # --- jugadores
    rosters = {int(t): {p["id"]: p for p in r["players"]} for t, r in bundle["rosters"].items()}
    hitters = {p["id"]: p for p in bundle.get("playersHitting", [])}
    sps = {}
    for side in SIDES:
        pid = game["probable"][side]
        raw = bundle["pitchers"][str(pid)]
        ros = rosters[game[side]].get(pid, {})
        sps[side] = {"id": pid, "name": raw["name"], "hand": raw.get("hand"), "season": raw["season"],
                     "prev": raw.get("prevSeason"), "vs": ros.get("pitVs") or {}, "gb": gb_share(raw["season"], lg.gb_rate),
                     "log": raw.get("log", [])}

    def batter(pid, tid, name=None):
        ros = rosters[tid].get(pid) or {}
        st = ros.get("hit") or (hitters.get(pid) or {}).get("stat") or {}
        return {"id": pid, "name": name or ros.get("name") or (hitters.get(pid) or {}).get("name"),
                "bats": ros.get("bats") or ctx.bats.get(pid), "stat": st, "vs": ros.get("hitVs") or {},
                "pos": ros.get("pos")}

    lineups = {s: [batter(pid, game[s], game["lineupNames"][s].get(str(pid))) for pid in game["lineups"][s]] for s in SIDES}

    def pa_probs(b, p, tto=2):
        hand = p.get("hand") or "R"
        b_rates, _ = rates_batter(b["stat"], b["vs"].get("vl" if hand == "L" else "vr"), hand, L)
        bats = b.get("bats") or "R"
        eff = ("L" if hand == "R" else "R") if bats == "S" else bats
        p_rates, _ = rates_pitcher(p["season"], p.get("prev"), p["vs"].get("vl" if eff == "L" else "vr"), L)
        r = MK.combine(b_rates, p_rates, L)
        r = MK.scale(r, mult)
        if p.get("starter"):
            f = MK.TTO[min(tto, 4)]
            r = MK.scale(r, {"1B": f, "2B": f, "3B": f, "HR": f, "BB": f})
        return MK.with_residual(r, e_rate)

    # --- bullpens (roles, disponibilidad y lista oficial del partido)
    pens = {s: C.full_bullpen(bundle, ctx.bp, game[s], game["probable"][s], ctx.today, lg,
                              bundle["rosters"][str(game[s])], game["officialBullpen"][s]) for s in SIDES}

    def reliever_spec(r):
        return {"id": r["id"], "name": r["name"], "hand": r.get("throws"), "season": r["stats"], "prev": None,
                "vs": r.get("vs") or {}, "gb": gb_share(r["stats"], lg.gb_rate), "role": r["role"],
                "fip": r["fipShrunk"], "pUse": r["pUse"], "ipPerG": r.get("ipPerG") or 1.0,
                "avail": {"Disponible": 1.0, "Dudoso": 0.5, "No disponible": 0.0}.get(r["available"], 0.0)}

    def team_rest(tid):
        rp = (bundle["teamStats"].get(str(tid)) or {}).get("rp") or {}
        return {"id": f"rest{tid}", "name": "Resto del bullpen", "hand": None, "season": rp, "prev": None, "vs": {},
                "gb": gb_share(rp, lg.gb_rate), "role": "Resto", "fip": lg.era, "pUse": 0.1, "avail": 1.0, "ipPerG": 1.0}

    # --- abridores: bateadores esperados
    def starter_bf(sp):
        starts = [r for r in sp["log"] if r["stat"].get("gamesStarted")][-10:]
        bfs = [r["stat"].get("battersFaced", 0) for r in starts]
        prof = next(c for c in base["sections"]["s3"]["comparison"] if c["name"] == sp["name"])
        exp_ip = prof.get("expIp") or lg.sp_ip_per_gs
        recent = bfs[-6:]
        lg_bf = lg.sp_ip_per_gs * (lg.bf_per_ip + 0.05)
        mu = M.shrink(statistics.fmean(recent), len(recent), 2, lg_bf) if recent else exp_ip * (lg.bf_per_ip + 0.05)
        sd = max(2.0, statistics.pstdev(recent)) if len(recent) >= 3 else 4.0
        return mu, sd, bfs

    # simulate() usa game[lado] = equipo con SU lineup, SU abridor y SU bullpen
    bf_info = {}
    sim_game = {}
    for side in SIDES:
        mu, sd, bfs = starter_bf(sps[side])
        bf_info[side] = {"mean": mu, "sd": sd, "recent": bfs}
        bull = [reliever_spec(r) for r in pens[side] if r["role"] != "Rotación"]
        t = MK.Team(bundle["teams"][str(game[side])]["abbr"], lineups[side], dict(sps[side], starter=True), bull, (mu, sd))
        t.bullpen_rest = team_rest(game[side])
        sim_game[side] = t

    def prob(b, p, tto):
        return MK.cumulative(pa_probs(b, p, tto))

    sim_game["prob"] = prob
    t1 = time.time()
    stats = MK.simulate(sim_game, n=n_sims, seed=24)
    sim_secs = time.time() - t1

    # --- pasada de la mañana (lineups proyectados) vs tarde (confirmados): actualización bayesiana 5.7.6
    morning = None
    proj = {}
    for side in SIDES:
        opp_hand = sps["home" if side == "away" else "away"]["hand"]
        pre_boxes = dict(bundle, boxscores=[b for b in bundle.get("boxscores", []) if b["date"] < game["date"]])
        pl = C.projected_lineup(pre_boxes, game[side], opp_hand, ctx.hands_any, set(rosters[game[side]]))
        proj[side] = pl
    if all(proj.values()):
        mg = {}
        for side in SIDES:
            t = sim_game[side]
            mt = MK.Team(t.name, [batter(r["id"], game[side], r["name"]) for r in proj[side]["rows"]], t.starter, t.bullpen,
                         t.starter_bf)
            mt.bullpen_rest = t.bullpen_rest
            mg[side] = mt
        mg["prob"] = prob
        ms = MK.simulate(mg, n=n_morning, seed=11)
        morning = {"pHome": ms["homeWin"] / ms["n"], "n": ms["n"],
                   "total": sum(k * v for k, v in ms["total"].items()) / ms["n"],
                   "lineups": {s: [{"name": r["name"], "pos": r["pos"], "prob": r["prob"], "confirmed": r["id"] in game["lineups"][s]}
                                   for r in proj[s]["rows"]] for s in SIDES},
                   "hits": {s: sum(1 for r in proj[s]["rows"] if r["id"] in game["lineups"][s]) for s in SIDES}}

    # --- cadenas analíticas: RE24 por equipo y cadena por lineup vs el abridor rival
    chains = {}
    for side in SIDES:
        opp = "home" if side == "away" else "away"
        sp = dict(sps[opp], starter=True)
        probs = [pa_probs(b, sp, 2) for b in lineups[side]]
        avg = {e: sum(p[e] for p in probs) / 9 for e in MK.EVENTS}
        re_team = MK.re24(avg, sp["gb"])
        ch = MK.lineup_chain(probs, sp["gb"])
        proj_inn = MK.innings_projection(ch["R"], ch["T"], 9)
        chains[side] = {
            "vs": sps[opp]["name"], "avgProbs": avg, "re24": re_team["table"], "p1": re_team["p1table"],
            "reEmpty": re_team["RE"][0], "R": ch["R"], "T": ch["T"],
            "innings": [{"inning": r["inning"], "runs": r["runs"], "leadoff": r["leadoff"]} for r in proj_inn],
            "batterProbs": [{"name": b["name"], "bats": b.get("bats"), **{e: p[e] for e in MK.EVENTS},
                             "obp": p["BB"] + p["1B"] + p["2B"] + p["3B"] + p["HR"] + p["E"],
                             "woba": 0.69 * p["BB"] + 0.88 * p["1B"] + 1.25 * p["2B"] + 1.58 * p["3B"] + 2.03 * p["HR"] + 0.88 * p["E"]}
                            for b, p in zip(lineups[side], probs)],
        }

    # --- resultados del Monte Carlo
    n = stats["n"]
    p_home = stats["homeWin"] / n
    se = math.sqrt(p_home * (1 - p_home) / n)
    runs_view = {s: MK.dist_view(stats["runs"][s], 0, 30) for s in SIDES}
    total_view = MK.dist_view(stats["total"], 0, 45)
    margin_keys = sorted(stats["margin"])
    margin_view = {k: stats["margin"][k] / n for k in range(-12, 13)}
    margin_view[-13] = sum(v for k, v in stats["margin"].items() if k <= -13) / n   # colas agrupadas
    margin_view[13] = sum(v for k, v in stats["margin"].items() if k >= 13) / n
    f5v = {k: v / n for k, v in stats["f5"].items()}
    f3v = {k: v / n for k, v in stats["f3"].items()}
    kdist = {s: MK.dist_view(stats["starterK"][s], 0, 15) for s in SIDES}
    outs_dist = {s: MK.dist_view(stats["starterOuts"][s], 0, 27) for s in SIDES}
    rel_use = {}
    for s in SIDES:
        names = {r["id"]: r for r in sim_game[s].bullpen}
        rel_use[s] = sorted([{"name": names[i]["name"] if i in names else "Resto del bullpen", "role": names.get(i, {}).get("role"),
                              "p": c / n} for i, c in stats["relUse"][s].items()], key=lambda x: -x["p"])
    mc_extra = {
        "n": n, "pHome": p_home, "total": total_view, "f5total": MK.dist_view(stats["f5total"], 0, 20),
        "runs_away": runs_view["away"], "runs_home": runs_view["home"], "nrfi": stats["nrfi"] / n,
        "f5": f5v, "k_away": kdist["away"], "k_home": kdist["home"],
        "rl": {}, "f3": f3v,
    }
    ab, hb = sim_game["away"].name, sim_game["home"].name
    mc_extra["rl"] = {f"{hb} -1.5": sum(v for k, v in margin_view.items() if k >= 2),
                      f"{ab} +1.5": 1 - sum(v for k, v in margin_view.items() if k >= 2),
                      f"{ab} -1.5": sum(v for k, v in margin_view.items() if k <= -2),
                      f"{hb} +1.5": 1 - sum(v for k, v in margin_view.items() if k <= -2)}

    picks = P.build(base, mc_extra)
    methods = base["sections"]["s5"]["triangulation"]
    consensus = {"log5": methods["log5"]["pHome"], "elo": methods["elo"]["pHome"], "lambda": methods["lambda"]["pHome"],
                 "mc": p_home}
    p_final = sum(consensus.values()) / 4

    # --- bateador vs pitcher (historial de carrera, solo referencia: muestras mínimas)
    bvp = {}
    for side in SIDES:
        opp = "home" if side == "away" else "away"
        rows = []
        for b in lineups[side] + [batter(i, game[side]) for i in game["officialBench"][side]]:
            raw = (snap.get("vsPlayer") or {}).get(str(b["id"])) or {}
            tot = next((x for x in raw.get("stats", []) if x["type"]["displayName"] == "vsPlayerTotal"), None)
            st = (tot or {}).get("splits", [{}])
            st = st[0].get("stat", {}) if st else {}
            rows.append({"name": b["name"], "bench": b["id"] not in game["lineups"][side], "pa": st.get("plateAppearances", 0),
                         "h": st.get("hits", 0), "hr": st.get("homeRuns", 0), "k": st.get("strikeOuts", 0),
                         "bb": st.get("baseOnBalls", 0), "ops": st.get("ops")})
        bvp[side] = {"vs": sps[opp]["name"], "rows": rows}

    out = {
        "model": MODEL_NAME, "pk": pk, "frozenAt": snap["takenAt"], "firstPitch": game["time"],
        "builtAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "game": {k: base[k] for k in ("pk", "date", "time", "venue", "teams", "weather", "umpire", "series")},
        "base": base,
        "picks": picks, "topPicks": picks[:2],
        "consensus": {"methods": consensus, "pHome": p_final,
                      "spread": max(consensus.values()) - min(consensus.values())},
        "calibration": {"eRate": e_rate, "target": lg.runs_per_half, "rawRE": re_raw["RE"][0], "calRE": re_lg["RE"][0],
                        "leagueRates": L, "gb": lg.gb_rate},
        "re24": {"league": re_lg["table"], "leagueP1": re_lg["p1table"], "raw": re_raw["table"],
                 "Q": re_lg["Q"], "N0": re_lg["N"][0], "expectedPA": re_lg["expectedPA"], "states": [MK.BASE_NAMES[b] + f" {o}"
                                                                                                     for b, o in MK.STATES]},
        "chains": chains, "park": {"name": game.get("venueName"), "index": park, "mult": mult, "windHr": wind_hr,
                                   "weather": wx},
        "mc": {"n": n, "seconds": sim_secs, "pHome": p_home, "se": se, "ci95": [p_home - 1.96 * se, p_home + 1.96 * se],
               "runs": runs_view, "total": total_view, "margin": margin_view, "f5": f5v, "f3": f3v,
               "f5total": MK.dist_view(stats["f5total"], 0, 20), "nrfi": stats["nrfi"] / n,
               "first": {k: v / n for k, v in stats["first"].items()}, "extras": stats["extras"] / n,
               "byInning": {s: [x / n for x in stats["byInning"][s]] for s in SIDES},
               "k": kdist, "outs": outs_dist, "relievers": rel_use, "convergence": stats["convergence"],
               "meanRuns": {s: sum(k * v for k, v in enumerate(runs_view[s])) for s in SIDES},
               "starterBf": bf_info},
        "morning": morning,
        "lineups": {s: [{"name": b["name"], "bats": b.get("bats"), "pos": b.get("pos"), "id": b["id"]} for b in lineups[s]]
                    for s in SIDES},
        "bench": {s: [batter(i, game[s])["name"] for i in game["officialBench"][s]] for s in SIDES},
        "pens": {s: [{k: v for k, v in r.items() if k not in ("stats", "vs")} for r in pens[s]] for s in SIDES},
        "bvp": bvp, "result": None, "secondsTotal": time.time() - t0,
    }
    rpath = os.path.join(DIR, f"result_{pk}.json")
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as f:
            out["result"] = compare(json.load(f), out)
    return out


def result_from_bundle(bundle: dict, pk: int) -> dict | None:
    """Resultado oficial (linescore + ponches de los abridores) tomado de un bundle posterior al partido."""
    g = next((x for x in bundle.get("results", []) if x["pk"] == pk), None)
    if not g:
        return None
    ks = {}
    for b in bundle.get("boxscores", []):
        if b["pk"] == pk:
            for side in SIDES:
                if b[side]["pitchers"]:
                    st = b[side]["pitchers"][0]
                    ks[st["name"]] = st["k"]
    return {"away": g["ar"], "home": g["hr"], "innings": g.get("inn") or [], "starterK": ks, "date": g["date"]}


def compare(res: dict, lab: dict) -> dict:
    """Resultado oficial vs lo que dijo el modelo antes del partido (usa solo lo guardado en el Pro-Lab)."""
    a, h = res["away"], res["home"]
    ab, hb = lab["game"]["teams"]["away"]["abbr"], lab["game"]["teams"]["home"]["abbr"]
    mc = lab["mc"]
    home_won = h > a
    total = a + h
    inn = res.get("innings") or []
    f5a = sum((x[0] or 0) for x in inn[:5])
    f5h = sum((x[1] or 0) for x in inn[:5])
    graded = [{"pick": p["pick"], "market": p["market"], "p": p["p"], "ic": p["ic"], "level": p["level"],
               "won": P.grade(p, ab, hb, a, h, inn, res.get("starterK"))} for p in lab["picks"]]
    tv = mc["total"]
    return {**res, "homeWon": home_won, "total": total, "f5": [f5a, f5h],
            "brier": (lab["consensus"]["pHome"] - (1 if home_won else 0)) ** 2,
            "brierMc": (mc["pHome"] - (1 if home_won else 0)) ** 2,
            "pTotalExact": tv[total] if total < len(tv) else 0.0, "percentileTotal": sum(tv[:total + 1]),
            "graded": graded, "hits": sum(1 for g in graded if g["won"]), "misses": sum(1 for g in graded if g["won"] is False)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pro-Lab: DIAMANTE-24 sobre un partido congelado")
    ap.add_argument("--pk", type=int, default=824223)
    ap.add_argument("--sims", type=int, default=50000)
    args = ap.parse_args()
    out = run(args.pk, args.sims)
    path = os.path.join(DIR, f"prolab_{args.pk}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"), default=float)
    print(f"{MODEL_NAME}: P({out['game']['teams']['home']['abbr']}) = {out['mc']['pHome']:.3f} ± {1.96 * out['mc']['se']:.3f} "
          f"en {out['mc']['n']} simulaciones ({out['mc']['seconds']:.1f} s) → {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
