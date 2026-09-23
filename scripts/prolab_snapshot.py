"""Congela TODOS los datos previos al partido del Pro-Lab (sin fuga: se corre antes del primer lanzamiento)."""
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlbgs import fetch as F  # noqa: E402

PK = int(os.environ.get("PROLAB_PK", "824223"))
LABEL = os.environ.get("PROLAB_LABEL", "pre")        # "manana" / "tarde": dos pasadas (sección 5.7.6)
OUT = "prolab"
os.makedirs(OUT, exist_ok=True)
B = F.STATS
snap = {"pk": PK, "takenAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "errors": {}}


def grab(name, fn):
    try:
        snap[name] = fn()
        F.log("ok", name)
    except Exception as e:  # noqa: BLE001
        snap["errors"][name] = repr(e)
        F.log("ERR", name, repr(e))


grab("feed", lambda: F.get(f"https://statsapi.mlb.com/api/v1.1/game/{PK}/feed/live"))
_gd = (snap.get("feed") or {}).get("gameData") or {}
TEAMS = [_gd["teams"]["away"]["id"], _gd["teams"]["home"]["id"]]
GAME_DATE = (_gd.get("datetime") or {}).get("officialDate") or dt.date.today().isoformat()
snap["status"] = _gd.get("status")
grab("content", lambda: F.get(f"{B}/game/{PK}/content"))
grab("standingsRaw", lambda: F.get(f"{B}/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason&hydrate=team"))
grab("remaining", lambda: F.get(f"{B}/schedule?sportId=1&season=2026&gameType=R&startDate={GAME_DATE}&endDate=2026-10-06"))
for t in TEAMS:
    grab(f"roster_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=active&hydrate=person(stats(type=[season,statSplits],sitCodes=[vl,vr],group=[hitting,pitching],season=2026))"))
    grab(f"roster40_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=40Man"))
    grab(f"transactions_{t}", lambda t=t: F.get(f"{B}/transactions?teamId={t}&startDate=2026-08-20&endDate={GAME_DATE}"))
    grab(f"injuries_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=fullRoster&date={GAME_DATE}"))

feed = snap.get("feed") or {}
box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
gd = (feed.get("gameData") or {})
prob = gd.get("probablePitchers") or {}
pitchers = {s: (prob.get(s) or {}).get("id") for s in ("away", "home")}
vs = {}
for side, opp in (("away", "home"), ("home", "away")):
    order = (box.get(side) or {}).get("battingOrder") or []
    bench = (box.get(side) or {}).get("bench") or []
    pid = pitchers.get(opp)
    if not pid:
        continue
    def one(bid, pid=pid):
        try:
            return bid, F.get(f"{B}/people/{bid}/stats?stats=vsPlayer&group=hitting&opposingPlayerId={pid}&sportId=1")
        except Exception as e:  # noqa: BLE001
            return bid, {"error": repr(e)}
    for bid, r in F.pmap(one, list(order) + list(bench)):
        vs[str(bid)] = r
snap["vsPlayer"] = vs

# últimos 12 juegos de cada equipo con box score completo (orden al bate, bullpen, bancas)
def last_pks():
    end = (dt.date.fromisoformat(GAME_DATE) - dt.timedelta(days=1)).isoformat()
    sched = F.get(f"{B}/schedule?sportId=1&season=2026&gameType=R&startDate=2026-08-25&endDate={end}")
    out = {}
    for d in sched.get("dates", []):
        for g in d["games"]:
            if g["status"].get("abstractGameState") != "Final":
                continue
            for t in TEAMS:
                if t in (g["teams"]["away"]["team"]["id"], g["teams"]["home"]["team"]["id"]):
                    out.setdefault(t, []).append((g["officialDate"], g["gamePk"]))
    return {t: [pk for _, pk in sorted(v)[-12:]] for t, v in out.items()}
grab("lastPks", last_pks)
boxes = {}
pks = sorted({pk for v in (snap.get("lastPks") or {}).values() for pk in v})
def slim_box(pk):
    bx = F.get(f"{B}/game/{pk}/boxscore")
    out = {}
    for side in ("away", "home"):
        t = bx["teams"][side]
        players = {}
        for k, p in t["players"].items():
            st = p.get("stats") or {}
            players[k] = {"id": p["person"]["id"], "name": p["person"].get("fullName"),
                          "pos": (p.get("position") or {}).get("abbreviation"), "order": p.get("battingOrder"),
                          "bat": st.get("batting") or None, "pit": st.get("pitching") or None}
        out[side] = {"team": t["team"]["id"], "battingOrder": t.get("battingOrder"), "pitchers": t.get("pitchers"),
                     "bullpen": t.get("bullpen"), "bench": t.get("bench"), "players": players}
    return pk, out
for pk, b in F.pmap(slim_box, pks):
    boxes[str(pk)] = b
snap["boxes"] = boxes

# Baseball Savant: arsenal de pitcheos y resultados por tipo de pitcheo, xwOBA de bateadores
def csv(url):
    return F.get(url, raw=True).decode("utf-8-sig", "replace")
S = F.SAVANT
grab("arsenalPitcher", lambda: csv(f"{S}/leaderboard/pitch-arsenal-stats?type=pitcher&pitchType=&year=2026&team=&min=1&csv=true"))
grab("arsenalBatter", lambda: csv(f"{S}/leaderboard/pitch-arsenal-stats?type=batter&pitchType=&year=2026&team=&min=1&csv=true"))
grab("expectedBatter", lambda: csv(f"{S}/leaderboard/expected_statistics?type=batter&year=2026&position=&team=&min=1&csv=true"))

# ---------------------------------------------------------------- KRONOS: datos por lanzamiento (Statcast)
# Tablas de transición de la cuenta (bolas-strikes) por jugador y de la liga, agregadas aquí para no guardar
# cientos de miles de filas: cuenta → {bola, strike cantado, strike tirándole, foul, foul de toque, pelotazo, en juego}.
import csv as _csv  # noqa: E402
import io as _io  # noqa: E402

CAT = {"ball": "B", "blocked_ball": "B", "automatic_ball": "B", "pitchout": "B", "intent_ball": "B",
       "called_strike": "CS", "automatic_strike": "CS",
       "swinging_strike": "SS", "swinging_strike_blocked": "SS", "missed_bunt": "SS", "foul_tip": "SS", "bunt_foul_tip": "SS",
       "foul": "F", "foul_pitchout": "F", "foul_bunt": "FB", "hit_by_pitch": "HBP", "hit_into_play": "X"}
PREV_DAY = (dt.date.fromisoformat(GAME_DATE) - dt.timedelta(days=1)).isoformat()
SC = (f"{S}/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfSea=&hfSit="
      "&player_type={pt}&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={d0}&game_date_lt={d1}"
      "&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name"
      "&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details{who}")


def statcast_rows(pt, d0, d1, who=""):
    txt = F.get(SC.format(pt=pt, d0=d0, d1=d1, who=who), raw=True).decode("utf-8-sig", "replace")
    return list(_csv.DictReader(_io.StringIO(txt)))


def agg_pitches(rows, by_tto=False):
    """Conteos por cuenta y categoría + eventos en juego + lanzamientos por turno y por juego."""
    trans, inplay, tto, clock = {}, {}, {}, {"automatic_ball": 0, "automatic_strike": 0}
    pa_len, games = {}, {}
    for r in rows:
        try:
            b, st = int(r["balls"]), int(r["strikes"])
        except (KeyError, ValueError):
            continue
        desc = r.get("description", "")
        cat = CAT.get(desc, "O")
        if desc in clock:
            clock[desc] += 1
        key = f"{b}-{st}"
        d = trans.setdefault(key, {})
        d[cat] = d.get(cat, 0) + 1
        if cat == "X":
            e = r.get("events") or "?"
            inplay[e] = inplay.get(e, 0) + 1
        if by_tto and r.get("n_thruorder_pitcher"):
            k = min(3, int(r["n_thruorder_pitcher"] or 1))
            dd = tto.setdefault(str(k), {})
            dd[cat] = dd.get(cat, 0) + 1
            if cat == "X":
                ev = "HR" if r.get("events") == "home_run" else "H" if r.get("events") in ("single", "double", "triple") else "O"
                dd["X_" + ev] = dd.get("X_" + ev, 0) + 1
        ab = (r.get("game_pk"), r.get("at_bat_number"))
        try:
            pa_len[ab] = max(pa_len.get(ab, 0), int(r.get("pitch_number") or 0))
        except ValueError:
            pass
        g = games.setdefault(r.get("game_pk"), {"date": r.get("game_date"), "n": 0, "pa": set(), "inn": 0})
        g["n"] += 1
        g["pa"].add(r.get("at_bat_number"))
        try:
            g["inn"] = max(g["inn"], int(r.get("inning") or 0))
        except ValueError:
            pass
    hist = {}
    for n in pa_len.values():
        hist[str(n)] = hist.get(str(n), 0) + 1
    glist = sorted(({"pk": k, "date": v["date"], "pitches": v["n"], "bf": len(v["pa"]), "lastInning": v["inn"]}
                    for k, v in games.items()), key=lambda x: x["date"] or "")
    return {"n": len(rows), "trans": trans, "inplay": inplay, "tto": tto, "clock": clock, "paLen": hist, "games": glist}


def player_pitches(args):
    pid, pt = args
    who = f"&pitchers_lookup%5B%5D={pid}" if pt == "pitcher" else f"&batters_lookup%5B%5D={pid}"
    try:
        rows = statcast_rows(pt, "2026-03-20", PREV_DAY, who)
        return f"{pt}:{pid}", agg_pitches(rows, by_tto=(pt == "pitcher"))
    except Exception as e:  # noqa: BLE001
        return f"{pt}:{pid}", {"error": repr(e)}


kron_ids = []
for side, opp in (("away", "home"), ("home", "away")):
    for bid in list((box.get(side) or {}).get("battingOrder") or []) + list((box.get(side) or {}).get("bench") or []):
        kron_ids.append((bid, "batter"))
for t in TEAMS:
    for r in ((snap.get(f"roster_{t}") or {}).get("roster") or []):
        if ((r.get("position") or {}).get("type") == "Pitcher"):
            kron_ids.append((r["person"]["id"], "pitcher"))
for sd in ("away", "home"):
    if pitchers.get(sd) and (pitchers[sd], "pitcher") not in kron_ids:
        kron_ids.append((pitchers[sd], "pitcher"))
snap["pitchData"] = dict(F.pmap(player_pitches, kron_ids, workers=4))
F.log("statcast por jugador:", len(snap["pitchData"]), "errores:", sum(1 for v in snap["pitchData"].values() if "error" in v))


def league_day(d):
    try:
        return d, agg_pitches(statcast_rows("pitcher", d, d), by_tto=True)
    except Exception as e:  # noqa: BLE001
        return d, {"error": repr(e)}


days = [(dt.date.fromisoformat(GAME_DATE) - dt.timedelta(days=k)).isoformat() for k in range(1, 15)]
snap["pitchLeague"] = dict(F.pmap(league_day, days, workers=3))
F.log("statcast liga:", {d: v.get("n") for d, v in snap["pitchLeague"].items()})

# MLB Stats API: códigos de situación (cuentas) y temporada avanzada de los jugadores del partido
grab("situationCodes", lambda: F.get(f"{B}/situationCodes"))
ids = sorted({str(i) for i, _ in kron_ids})
grab("peopleAdvanced", lambda: F.get(f"{B}/people?personIds={','.join(ids)}&hydrate=stats(group=[hitting,pitching],type=[season,seasonAdvanced],season=2026)"))

with gzip.open(f"{OUT}/snapshot_{PK}_{LABEL}.json.gz", "wt", encoding="utf-8") as f:
    json.dump(snap, f, separators=(",", ":"))
F.log("snapshot extra listo; errores:", snap["errors"])

bundle = F.fetch_bundle(dt.date.fromisoformat(GAME_DATE), days=2)
if not any(g["pk"] == PK for g in bundle["upcoming"]):
    F.log("AVISO: el partido ya no está en 'Preview' (¿empezó?)")
with gzip.open(f"{OUT}/bundle_{PK}_{LABEL}.json.gz", "wt", encoding="utf-8") as f:
    json.dump(bundle, f, separators=(",", ":"))
F.log("bundle listo", bundle["meta"]["errors"])
