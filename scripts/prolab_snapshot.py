"""Congela TODOS los datos previos al partido del Pro-Lab (sin fuga: se corre antes del primer lanzamiento)."""
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlbgs import fetch as F  # noqa: E402

PK = int(os.environ.get("PROLAB_PK", "824223"))
TEAMS = [120, 116]
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
grab("content", lambda: F.get(f"{B}/game/{PK}/content"))
grab("standingsRaw", lambda: F.get(f"{B}/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason&hydrate=team"))
grab("remaining", lambda: F.get(f"{B}/schedule?sportId=1&season=2026&gameType=R&startDate=2026-09-23&endDate=2026-10-06"))
for t in TEAMS:
    grab(f"roster_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=active&hydrate=person(stats(type=[season,statSplits],sitCodes=[vl,vr],group=[hitting,pitching],season=2026))"))
    grab(f"roster40_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=40Man"))
    grab(f"transactions_{t}", lambda t=t: F.get(f"{B}/transactions?teamId={t}&startDate=2026-08-20&endDate=2026-09-23"))
    grab(f"injuries_{t}", lambda t=t: F.get(f"{B}/teams/{t}/roster?rosterType=fullRoster&date=2026-09-23"))

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
    sched = F.get(f"{B}/schedule?sportId=1&season=2026&gameType=R&startDate=2026-08-25&endDate=2026-09-22")
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

with gzip.open(f"{OUT}/snapshot_{PK}_pre.json.gz", "wt", encoding="utf-8") as f:
    json.dump(snap, f, separators=(",", ":"))
F.log("snapshot extra listo; errores:", snap["errors"])

bundle = F.fetch_bundle(dt.date(2026, 9, 23), days=2)
with gzip.open(f"{OUT}/bundle_{PK}_pre.json.gz", "wt", encoding="utf-8") as f:
    json.dump(bundle, f, separators=(",", ":"))
F.log("bundle listo", bundle["meta"]["errors"])
