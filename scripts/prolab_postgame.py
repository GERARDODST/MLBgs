"""Datos de DESPUÉS del partido para el post-mortem de un modelo del Pro-Lab (nunca entran a una predicción).

Guarda prolab/postgame_<pk>.json.gz con:
  * la jugada a jugada oficial (MLB Stats API, feed/live): cada turno con bateador, pitcher, entrada, cuenta,
    resultado, carreras y cada lanzamiento (tipo, resultado, velocidad y, en juego, velocidad y ángulo de salida);
  * el box score (línea de cada pitcher y bateador, orden al bate, decisiones) y el marcador por entrada;
  * Statcast del partido (Baseball Savant): xwOBA y xBA de cada bola en juego y el valor wOBA de cada turno;
  * las demás aperturas de esta postemporada ya jugadas (para seguir midiendo el gancho de octubre).

Uso: PROLAB_POST=<pk> python scripts/prolab_postgame.py
"""
import csv as _csv
import datetime as dt
import gzip
import io as _io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlbgs import fetch as F  # noqa: E402

PK = int(os.environ["PROLAB_POST"])
B = F.STATS
OUT = "prolab"
out = {"pk": PK, "takenAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "errors": {}}


def grab(name, fn):
    try:
        out[name] = fn()
        F.log("ok", name)
    except Exception as e:  # noqa: BLE001
        out["errors"][name] = repr(e)
        F.log("ERR", name, repr(e))


feed = F.get(f"https://statsapi.mlb.com/api/v1.1/game/{PK}/feed/live")
gd, ld = feed.get("gameData") or {}, feed.get("liveData") or {}
out["status"] = gd.get("status")
if (gd.get("status") or {}).get("abstractGameState") != "Final":
    F.log("AVISO: el partido todavía no es final:", gd.get("status"))
GAME_DATE = (gd.get("datetime") or {}).get("officialDate")
out["teams"] = {s: {"id": gd["teams"][s]["id"], "abbr": gd["teams"][s].get("abbreviation")} for s in ("away", "home")}
out["date"] = GAME_DATE


def slim_play(p):
    ab = p.get("about") or {}
    mu = p.get("matchup") or {}
    res = p.get("result") or {}
    pitches = []
    for ev in p.get("playEvents") or []:
        if not ev.get("isPitch"):
            continue
        det, pd_, hd = ev.get("details") or {}, ev.get("pitchData") or {}, ev.get("hitData") or {}
        pitches.append({"type": (det.get("type") or {}).get("code"), "call": (det.get("call") or {}).get("code"),
                        "desc": det.get("description"), "inPlay": det.get("isInPlay"), "strike": det.get("isStrike"),
                        "ball": det.get("isBall"), "count": ev.get("count"), "speed": pd_.get("startSpeed"),
                        "ev": hd.get("launchSpeed"), "la": hd.get("launchAngle"), "traj": hd.get("trajectory")})
    return {"inning": ab.get("inning"), "half": ab.get("halfInning"), "ab": ab.get("atBatIndex"),
            "batter": (mu.get("batter") or {}).get("id"), "pitcher": (mu.get("pitcher") or {}).get("id"),
            "bats": (mu.get("batSide") or {}).get("code"), "throws": (mu.get("pitchHand") or {}).get("code"),
            "event": res.get("eventType"), "desc": res.get("description"), "rbi": res.get("rbi"),
            "awayScore": res.get("awayScore"), "homeScore": res.get("homeScore"), "outs": (p.get("count") or {}).get("outs"),
            "pitches": pitches}


out["plays"] = [slim_play(p) for p in (ld.get("plays") or {}).get("allPlays") or []]
out["linescore"] = {"innings": [{"num": i.get("num"), "away": (i.get("away") or {}).get("runs"), "home": (i.get("home") or {}).get("runs")}
                                for i in (ld.get("linescore") or {}).get("innings") or []],
                    "teams": (ld.get("linescore") or {}).get("teams")}
box = (ld.get("boxscore") or {}).get("teams") or {}
out["box"] = {}
for s in ("away", "home"):
    t = box.get(s) or {}
    players = {}
    for k, p in (t.get("players") or {}).items():
        st = p.get("stats") or {}
        players[str(p["person"]["id"])] = {"name": p["person"].get("fullName"), "pos": (p.get("position") or {}).get("abbreviation"),
                                           "order": p.get("battingOrder"), "bat": st.get("batting") or None,
                                           "pit": st.get("pitching") or None}
    out["box"][s] = {"battingOrder": t.get("battingOrder"), "pitchers": t.get("pitchers"), "players": players}
out["decisions"] = ld.get("decisions")

# Statcast del partido: xwOBA/xBA por bola en juego y valor wOBA por turno (Savant puede tardar horas en publicarlo)
SC = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
      "&hfGT=F%7CD%7CL%7CW%7C&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA="
      "&game_date_gt={d}&game_date_lt={d}&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
      "&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details")
KEEP = ("game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "batter", "pitcher", "pitch_type", "events",
        "description", "estimated_woba_using_speedangle", "estimated_ba_using_speedangle", "woba_value", "woba_denom",
        "launch_speed", "launch_angle", "release_speed", "balls", "strikes", "n_thruorder_pitcher")


def statcast():
    txt = F.get(SC.format(d=GAME_DATE), raw=True).decode("utf-8-sig", "replace")
    rows = [r for r in _csv.DictReader(_io.StringIO(txt)) if r.get("game_pk") == str(PK)]
    return [{k: r.get(k) for k in KEEP} for r in rows]


grab("statcast", statcast)

# aperturas de esta postemporada ya jugadas (gancho de octubre, fuera de muestra)
season = int(GAME_DATE[:4])


def post_starters(pk):
    bx = F.get(f"{B}/game/{pk}/boxscore")
    res = {}
    for side in ("away", "home"):
        t = bx["teams"][side]
        rows = []
        for pid in t.get("pitchers") or []:
            st = ((t["players"].get(f"ID{pid}") or {}).get("stats") or {}).get("pitching") or {}
            rows.append({"id": pid, "ip": st.get("inningsPitched"), "bf": st.get("battersFaced"),
                         "pitches": st.get("numberOfPitches") or st.get("pitchesThrown"), "er": st.get("earnedRuns"),
                         "runs": st.get("runs"), "k": st.get("strikeOuts"), "bb": st.get("baseOnBalls")})
        res[side] = {"team": t["team"]["id"], "pitchers": rows}
    return pk, res


def post_now():
    sch = F.get(f"{B}/schedule?sportId=1&season={season}&gameType=F,D,L,W")
    pks = [g["gamePk"] for d in sch.get("dates", []) for g in d["games"] if g["status"].get("abstractGameState") == "Final"]
    return dict(F.pmap(post_starters, pks, workers=6))


grab("postSeasonGames", post_now)
with gzip.open(f"{OUT}/postgame_{PK}.json.gz", "wt", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
F.log("post-mortem listo:", len(out["plays"]), "turnos,", len(out.get("statcast") or []), "lanzamientos Statcast; errores:", out["errors"])
