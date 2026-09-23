"""FASE 0 del algoritmo maestro: ingesta de datos oficiales.

Fuentes (todas de MLB salvo los momios, que son opcionales):
  * MLB Stats API (statsapi.mlb.com): calendario, abridores probables, lineups,
    umpires, clima, standings, box scores, stats de equipos y jugadores.
  * Baseball Savant (baseballsavant.mlb.com): xERA, Barrel%, Hard Hit% y park factors.
  * The Odds API (opcional, variable de entorno ODDS_API_KEY): momios de varias casas.

El resultado es un "bundle" con solo los campos que usa el modelo, para que el
modelo sea una función pura y reproducible de ese bundle.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

STATS = "https://statsapi.mlb.com/api/v1"
SAVANT = "https://baseballsavant.mlb.com"
ODDS = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
UA = {"User-Agent": "MLBgs/1.0 (+https://github.com/GERARDODST/MLBgs)"}

PITCH_KEYS = [
    "gamesPlayed", "gamesStarted", "wins", "losses", "saves", "saveOpportunities", "holds", "blownSaves",
    "inningsPitched", "battersFaced", "hits", "runs", "earnedRuns", "homeRuns", "baseOnBalls",
    "intentionalWalks", "hitByPitch", "strikeOuts", "groundOuts", "airOuts", "flyOuts", "atBats",
    "sacFlies", "numberOfPitches", "pitchesThrown", "doubles", "triples", "inheritedRunners",
    "inheritedRunnersScored", "gamesFinished",
]
HIT_KEYS = [
    "gamesPlayed", "plateAppearances", "atBats", "runs", "hits", "doubles", "triples", "homeRuns",
    "baseOnBalls", "intentionalWalks", "hitByPitch", "strikeOuts", "sacFlies", "stolenBases",
    "avg", "obp", "slg", "ops", "babip", "totalBases", "leftOnBase",
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def get(url: str, raw: bool = False, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            return body if raw else json.loads(body)
        except Exception as e:  # noqa: BLE001 - reintentar cualquier fallo de red
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET {url} falló: {last!r}")


def pick(stat: dict, keys: list[str]) -> dict:
    return {k: stat[k] for k in keys if k in stat}


def pmap(fn, items, workers: int = 8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ calendario

def slim_game(g: dict) -> dict:
    t = g["teams"]
    out = {
        "pk": g["gamePk"],
        "date": g.get("officialDate"),
        "time": g.get("gameDate"),
        "season": int(g.get("season", 0)),
        "type": g.get("gameType"),
        "state": g["status"].get("abstractGameState"),
        "detailed": g["status"].get("detailedState"),
        "away": t["away"]["team"]["id"],
        "home": t["home"]["team"]["id"],
        "venue": (g.get("venue") or {}).get("id"),
        "dh": g.get("doubleHeader"),
        "gameNumber": g.get("gameNumber"),
        "dayNight": g.get("dayNight"),
    }
    if "score" in t["away"] and "score" in t["home"]:
        out["ar"], out["hr"] = t["away"]["score"], t["home"]["score"]
    ls = g.get("linescore") or {}
    if ls.get("innings"):
        out["inn"] = [[(i.get("away") or {}).get("runs"), (i.get("home") or {}).get("runs")] for i in ls["innings"]]
    return out


def is_completed(g: dict) -> bool:
    return (g["status"].get("abstractGameState") == "Final"
            and not re.search(r"Postponed|Cancel|Suspended", g["status"].get("detailedState", ""))
            and "score" in g["teams"]["away"])


def season_schedule(season: int) -> tuple[list[dict], list[dict]]:
    """(resultados finales, juegos restantes). Un juego suspendido y reanudado aparece dos veces
    en el calendario con el mismo gamePk: se deduplica para no contarlo doble (cotejo contra standings)."""
    data = get(f"{STATS}/schedule?sportId=1&season={season}&gameType=R&hydrate=linescore")
    games = [g for d in data.get("dates", []) for g in d["games"]]
    done: dict[int, dict] = {}
    remaining: dict[int, dict] = {}
    for g in games:
        if is_completed(g):
            done[g["gamePk"]] = slim_game(g)
        elif g["status"].get("abstractGameState") != "Final" and not re.search(
                r"Postponed|Cancel", g["status"].get("detailedState", "")):
            remaining[g["gamePk"]] = {"pk": g["gamePk"], "date": g.get("officialDate"),
                                      "away": g["teams"]["away"]["team"]["id"], "home": g["teams"]["home"]["team"]["id"]}
    for pk in done:
        remaining.pop(pk, None)
    return sorted(done.values(), key=lambda g: (g["date"], g["pk"])), sorted(remaining.values(), key=lambda g: g["date"])


def season_results(season: int) -> list[dict]:
    return season_schedule(season)[0]


def upcoming_games(start: str, end: str) -> list[dict]:
    hyd = "probablePitcher,team,venue(location,fieldInfo),lineups,officials,weather,seriesStatus"
    data = get(f"{STATS}/schedule?sportId=1&startDate={start}&endDate={end}&gameType=R,F,D,L,W&hydrate={hyd}")
    out = []
    live = []
    for d in data.get("dates", []):
        for g in d["games"]:
            if g["status"].get("abstractGameState") == "Live":
                live.append({"pk": g["gamePk"], "away": g["teams"]["away"]["team"]["id"],
                             "home": g["teams"]["home"]["team"]["id"], "detailed": g["status"].get("detailedState")})
            if g["status"].get("abstractGameState") != "Preview":
                continue
            s = slim_game(g)
            t = g["teams"]
            venue = g.get("venue") or {}
            loc = venue.get("location") or {}
            fi = venue.get("fieldInfo") or {}
            s.update({
                "venueName": venue.get("name"),
                "venueCity": loc.get("city"),
                "roof": fi.get("roofType"),
                "turf": fi.get("turfType"),
                "elevation": loc.get("elevation"),
                "probable": {side: (t[side].get("probablePitcher") or {}).get("id") for side in ("away", "home")},
                "records": {side: t[side].get("leagueRecord") for side in ("away", "home")},
                "lineups": {
                    "away": [p["id"] for p in (g.get("lineups") or {}).get("awayPlayers", [])],
                    "home": [p["id"] for p in (g.get("lineups") or {}).get("homePlayers", [])],
                },
                "lineupNames": {
                    "away": {str(p["id"]): p.get("fullName") for p in (g.get("lineups") or {}).get("awayPlayers", [])},
                    "home": {str(p["id"]): p.get("fullName") for p in (g.get("lineups") or {}).get("homePlayers", [])},
                },
                "officials": [{"type": o.get("officialType"), "name": o["official"].get("fullName")}
                              for o in g.get("officials", [])],
                "weather": g.get("weather") or {},
                "series": {k: (g.get("seriesStatus") or {}).get(k) for k in ("gameNumber", "totalGames", "result")},
                "gamesInSeries": g.get("gamesInSeries"),
                "seriesGameNumber": g.get("seriesGameNumber"),
            })
            out.append(s)
    upcoming_games.live = live
    return out


def scoreboard(start: str, end: str) -> list[dict]:
    """Marcador de todos los partidos (programados, en vivo y finales) con la situación actual."""
    data = get(f"{STATS}/schedule?sportId=1&startDate={start}&endDate={end}&gameType=R,F,D,L,W&hydrate=linescore,team,lineups,probablePitcher")
    out = []
    for d in data.get("dates", []):
        for g in d["games"]:
            t = g["teams"]
            ls = g.get("linescore") or {}
            off = ls.get("offense") or {}
            dfn = ls.get("defense") or {}
            lu = g.get("lineups") or {}
            out.append({
                "pk": g["gamePk"], "date": g.get("officialDate"), "time": g.get("gameDate"),
                "state": g["status"].get("abstractGameState"), "detailed": g["status"].get("detailedState"),
                "away": t["away"]["team"]["id"], "home": t["home"]["team"]["id"],
                "awayAbbr": t["away"]["team"].get("abbreviation"), "homeAbbr": t["home"]["team"].get("abbreviation"),
                "ar": t["away"].get("score"), "hr": t["home"].get("score"),
                "inning": ls.get("currentInning"), "top": ls.get("isTopInning"), "inningState": ls.get("inningState"),
                "outs": ls.get("outs"), "bases": (1 if off.get("first") else 0) | (2 if off.get("second") else 0) | (4 if off.get("third") else 0),
                "inn": [[(i.get("away") or {}).get("runs"), (i.get("home") or {}).get("runs")] for i in ls.get("innings", [])],
                "venue": (g.get("venue") or {}).get("name"),
                "now": {"batter": (off.get("batter") or {}).get("fullName"), "onDeck": (off.get("onDeck") or {}).get("fullName"),
                        "pitcher": (dfn.get("pitcher") or {}).get("fullName"), "balls": ls.get("balls"), "strikes": ls.get("strikes")},
                "probs": {s: (t[s].get("probablePitcher") or {}).get("fullName") for s in ("away", "home")},
                "lu": ({s: [{"name": p.get("fullName"), "pos": (p.get("primaryPosition") or {}).get("abbreviation")}
                            for p in lu.get(f"{s}Players") or []] for s in ("away", "home")}
                       if (lu.get("awayPlayers") or lu.get("homePlayers")) else None),
            })
    today = end
    live = [g for g in out if g["date"] == today and g["state"] in ("Live", "Final") and "Postponed" not in (g["detailed"] or "")]
    for g, box in zip(live, pmap(lambda g: _live_box(g["pk"]), live)):
        if box:
            g["box"] = box
    return out


def _live_box(pk):
    """Box score reducido: orden al bate actual (con cambios) y pitchers usados, con su línea del juego."""
    try:
        b = get(f"{STATS}/game/{pk}/boxscore")
    except Exception:  # noqa: BLE001 - un box score faltante no debe tumbar la ingesta
        return None
    out = {}
    for sd in ("away", "home"):
        tm = b["teams"][sd]
        P = tm.get("players") or {}
        bat = []
        for pid in tm.get("battingOrder") or []:
            p = P.get(f"ID{pid}") or {}
            st = (p.get("stats") or {}).get("batting") or {}
            bat.append({"name": (p.get("person") or {}).get("fullName"), "pos": (p.get("position") or {}).get("abbreviation"),
                        "ab": st.get("atBats", 0), "h": st.get("hits", 0), "r": st.get("runs", 0), "rbi": st.get("rbi", 0),
                        "bb": st.get("baseOnBalls", 0), "k": st.get("strikeOuts", 0), "hr": st.get("homeRuns", 0),
                        "sub": int(p.get("battingOrder") or 0) % 100 != 0})
        pit = []
        for pid in tm.get("pitchers") or []:
            p = P.get(f"ID{pid}") or {}
            st = (p.get("stats") or {}).get("pitching") or {}
            pit.append({"name": (p.get("person") or {}).get("fullName"), "ip": st.get("inningsPitched", "0.0"),
                        "pitches": st.get("numberOfPitches", st.get("pitchesThrown", 0)), "k": st.get("strikeOuts", 0),
                        "bb": st.get("baseOnBalls", 0), "h": st.get("hits", 0), "er": st.get("earnedRuns", 0)})
        out[sd] = {"bat": bat, "pit": pit}
    return out


# ------------------------------------------------------------------ equipos y standings

def teams(season: int) -> dict:
    data = get(f"{STATS}/teams?sportId=1&season={season}")
    out = {}
    for t in data["teams"]:
        out[str(t["id"])] = {
            "id": t["id"], "abbr": t.get("abbreviation"), "name": t.get("name"),
            "club": t.get("clubName") or t.get("teamName"), "location": t.get("locationName"),
            "league": (t.get("league") or {}).get("name"), "leagueId": (t.get("league") or {}).get("id"),
            "division": (t.get("division") or {}).get("name"), "divisionId": (t.get("division") or {}).get("id"),
            "venue": (t.get("venue") or {}).get("id"), "venueName": (t.get("venue") or {}).get("name"),
        }
    return out


def standings(season: int) -> dict:
    data = get(f"{STATS}/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason")
    out = {}
    for rec in data.get("records", []):
        for tr in rec["teamRecords"]:
            splits = {s["type"]: [s["wins"], s["losses"]] for s in tr.get("records", {}).get("splitRecords", [])}
            out[str(tr["team"]["id"])] = {
                "w": tr.get("wins"), "l": tr.get("losses"), "rs": tr.get("runsScored"), "ra": tr.get("runsAllowed"),
                "gp": tr.get("gamesPlayed"), "streak": (tr.get("streak") or {}).get("streakCode"),
                "divRank": tr.get("divisionRank"), "gb": tr.get("divisionGamesBack"),
                "wcgb": tr.get("wildCardGamesBack"), "clinch": tr.get("clinchIndicator"),
                "elim": tr.get("eliminationNumber"), "splits": splits,
                "wcRank": tr.get("wildCardRank"), "wcElim": tr.get("wildCardEliminationNumber"),
                "magic": tr.get("magicNumber"), "divLeader": tr.get("divisionLeader"), "clinched": tr.get("clinched"),
                "leagueRank": tr.get("leagueRank"), "runDiff": tr.get("runDifferential"),
                "lastUpdated": tr.get("lastUpdated"),
            }
    return out


def team_stats(season: int) -> dict:
    out: dict[str, dict] = {}

    def put(tid, key, stat, keys):
        out.setdefault(str(tid), {})[key] = pick(stat, keys)

    for s in get(f"{STATS}/teams/stats?season={season}&group=pitching&stats=season&sportIds=1&limit=100")["stats"][0]["splits"]:
        put(s["team"]["id"], "pitching", s["stat"], PITCH_KEYS)
    for s in get(f"{STATS}/teams/stats?season={season}&group=pitching&stats=statSplits&sitCodes=sp,rp&sportIds=1&limit=100")["stats"][0]["splits"]:
        put(s["team"]["id"], s["split"]["code"], s["stat"], PITCH_KEYS)
    for s in get(f"{STATS}/teams/stats?season={season}&group=hitting&stats=season&sportIds=1&limit=100")["stats"][0]["splits"]:
        put(s["team"]["id"], "hitting", s["stat"], HIT_KEYS)
    for s in get(f"{STATS}/teams/stats?season={season}&group=hitting&stats=statSplits&sitCodes=vl,vr&sportIds=1&limit=100")["stats"][0]["splits"]:
        put(s["team"]["id"], "hit_" + s["split"]["code"], s["stat"], HIT_KEYS)
    return out


def players(season: int, group: str) -> list[dict]:
    keys = PITCH_KEYS if group == "pitching" else HIT_KEYS
    data = get(f"{STATS}/stats?stats=season&group={group}&season={season}&sportId=1&playerPool=ALL&limit=4000")
    out = []
    for s in data["stats"][0]["splits"]:
        out.append({"id": s["player"]["id"], "name": s["player"].get("fullName"),
                    "team": (s.get("team") or {}).get("id"), "numTeams": s.get("numTeams", 1),
                    "stat": pick(s["stat"], keys)})
    return out


# ------------------------------------------------------------------ abridores probables

def _single_split(splits: list[dict]) -> dict | None:
    """Con traspasos la API devuelve una fila por equipo + la total (sin 'team')."""
    if not splits:
        return None
    total = [s for s in splits if "team" not in s]
    return (total or splits)[0]


def pitcher(pid: int, season: int) -> dict:
    cur = get(f"{STATS}/people/{pid}?hydrate=stats(group=[pitching],type=[season,gameLog,statSplits],"
              f"sitCodes=[h,a],season={season})")["people"][0]
    prev = get(f"{STATS}/people/{pid}?hydrate=stats(group=[pitching],type=[season,gameLog],season={season - 1})")["people"][0]
    out = {"id": pid, "name": cur.get("fullName"), "hand": (cur.get("pitchHand") or {}).get("code"),
           "age": cur.get("currentAge"), "number": cur.get("primaryNumber"),
           "season": None, "splits": {}, "log": [], "prevSeason": None, "prevLog": []}

    def logs(stats_block):
        rows = []
        for s in stats_block["splits"]:
            st = s["stat"]
            rows.append({"date": s.get("date"), "pk": (s.get("game") or {}).get("gamePk"),
                         "opp": (s.get("opponent") or {}).get("id"), "team": (s.get("team") or {}).get("id"),
                         "home": s.get("isHome"), "win": s.get("isWin"), "stat": pick(st, PITCH_KEYS)})
        return rows

    for block in cur.get("stats", []):
        kind = block["type"]["displayName"]
        if kind == "season":
            one = _single_split(block["splits"])
            out["season"] = pick(one["stat"], PITCH_KEYS) if one else None
        elif kind == "gameLog":
            out["log"] = logs(block)
        elif kind == "statSplits":
            for code in ("h", "a"):
                one = _single_split([s for s in block["splits"] if s.get("split", {}).get("code") == code])
                if one:
                    out["splits"][code] = pick(one["stat"], PITCH_KEYS)
    for block in prev.get("stats", []):
        kind = block["type"]["displayName"]
        if kind == "season":
            one = _single_split(block["splits"])
            out["prevSeason"] = pick(one["stat"], PITCH_KEYS) if one else None
        elif kind == "gameLog":
            out["prevLog"] = logs(block)
    return out


def handedness(ids: list[int]) -> tuple[dict, dict]:
    """Mano de lanzar y de batear para una lista de jugadores (en lotes)."""
    throws, bats = {}, {}
    ids = sorted(set(i for i in ids if i))
    chunks = [ids[i:i + 150] for i in range(0, len(ids), 150)]

    def one(chunk):
        return get(f"{STATS}/people?personIds={','.join(map(str, chunk))}")["people"]

    for people in pmap(one, chunks, workers=4):
        for p in people:
            throws[str(p["id"])] = (p.get("pitchHand") or {}).get("code")
            bats[str(p["id"])] = (p.get("batSide") or {}).get("code")
    return throws, bats


# ------------------------------------------------------------------ rosters, lesionados y noticias

def roster(tid: int, season: int) -> dict:
    """Roster activo con stats de temporada y splits vs zurdos/derechos, más la lista de lesionados."""
    hyd = (f"person(stats(type=[season,statSplits],sitCodes=[vl,vr],group=[hitting,pitching],season={season}))")
    act = get(f"{STATS}/teams/{tid}/roster?rosterType=active&hydrate={hyd}")
    try:
        full = get(f"{STATS}/teams/{tid}/roster?rosterType=40Man")
    except Exception as e:  # noqa: BLE001
        log("lesionados", tid, repr(e))
        full = {}
    return parse_roster(act, full)


def parse_roster(act: dict, full: dict) -> dict:
    players = []
    for r in act.get("roster", []):
        per = r.get("person") or {}
        row = {"id": per.get("id"), "name": per.get("fullName"), "pos": (r.get("position") or {}).get("abbreviation"),
               "number": r.get("jerseyNumber"), "status": (r.get("status") or {}).get("description"),
               "bats": (per.get("batSide") or {}).get("code"), "throws": (per.get("pitchHand") or {}).get("code"),
               "hit": None, "pit": None, "hitVs": {}, "pitVs": {}}
        for blk in per.get("stats", []):
            kind, grp = blk["type"]["displayName"], blk["group"]["displayName"]
            for sp in blk.get("splits", []):
                if kind == "season" and "team" in sp and len(blk["splits"]) > 1:
                    continue
                keys = HIT_KEYS if grp == "hitting" else PITCH_KEYS
                st = pick(sp.get("stat", {}), keys)
                if kind == "season":
                    row["hit" if grp == "hitting" else "pit"] = st
                elif kind == "statSplits":
                    code = (sp.get("split") or {}).get("code")
                    if code in ("vl", "vr"):
                        row["hitVs" if grp == "hitting" else "pitVs"][code] = st
        players.append(row)
    injured = []
    for r in (full or {}).get("roster", []):
        code = (r.get("status") or {}).get("code") or ""
        if code.startswith("D") or "Injured" in ((r.get("status") or {}).get("description") or ""):
            injured.append({"id": r["person"]["id"], "name": r["person"].get("fullName"),
                            "pos": (r.get("position") or {}).get("abbreviation"),
                            "status": (r.get("status") or {}).get("description")})
    return {"players": players, "injured": injured}


TRANSACTION_TYPES = {"SC", "CU", "OPT", "DES", "SE", "TR", "REL", "ASG", "SFA", "CLW", "DFA"}


def transactions(team_ids: set[int], start: str, end: str) -> list[dict]:
    """Movimientos oficiales (lista de lesionados, llamados, opciones, cambios) = noticias del equipo."""
    def one(tid):
        return get(f"{STATS}/transactions?teamId={tid}&startDate={start}&endDate={end}").get("transactions", [])

    return parse_transactions([t for rows in pmap(one, sorted(team_ids), workers=6) for t in rows], team_ids)


def parse_transactions(rows: list[dict], team_ids: set[int]) -> list[dict]:
    out = []
    for t in rows:
        if t.get("typeCode") not in TRANSACTION_TYPES:
            continue
        team = (t.get("toTeam") or {}).get("id") or (t.get("fromTeam") or {}).get("id")
        if team not in team_ids:
            continue
        out.append({"id": t.get("id"), "date": t.get("date") or t.get("effectiveDate"), "type": t.get("typeDesc"),
                    "code": t.get("typeCode"), "team": team, "person": (t.get("person") or {}).get("id"),
                    "name": (t.get("person") or {}).get("fullName"), "text": t.get("description")})
    uniq = {t["id"]: t for t in out}
    return sorted(uniq.values(), key=lambda t: (t["date"] or ""), reverse=True)


def arsenal(season: int) -> dict:
    """Arsenal por pitcheo (Savant): uso, run value/100, whiff%, wOBA; para abridores y bateadores."""
    out = {"pitcher": {}, "batter": {}}
    for kind in ("pitcher", "batter"):
        for r in _csv(f"{SAVANT}/leaderboard/pitch-arsenal-stats?type={kind}&pitchType=&year={season}&team=&min=1&csv=true"):
            out[kind].setdefault(r["player_id"], []).append({
                "type": r.get("pitch_type"), "name": r.get("pitch_name"), "usage": _num(r.get("pitch_usage")),
                "rv100": _num(r.get("run_value_per_100")), "pitches": _num(r.get("pitches")), "pa": _num(r.get("pa")),
                "woba": _num(r.get("woba")), "xwoba": _num(r.get("est_woba")), "whiff": _num(r.get("whiff_percent")),
                "k": _num(r.get("k_percent")), "hardhit": _num(r.get("hard_hit_percent"))})
    return out


# ------------------------------------------------------------------ box scores (uso de bullpen)

def boxscore(pk: int) -> dict:
    bx = get(f"{STATS}/game/{pk}/boxscore")
    out = {}
    for side in ("away", "home"):
        t = bx["teams"][side]
        rows = []
        for i, pid in enumerate(t.get("pitchers", [])):
            p = t["players"].get(f"ID{pid}", {})
            st = (p.get("stats") or {}).get("pitching") or {}
            rows.append({"id": pid, "name": (p.get("person") or {}).get("fullName"), "order": i,
                         "pitches": st.get("pitchesThrown", st.get("numberOfPitches", 0)),
                         "ip": st.get("inningsPitched"), "er": st.get("earnedRuns", 0), "r": st.get("runs", 0),
                         "h": st.get("hits", 0), "bb": st.get("baseOnBalls", 0), "k": st.get("strikeOuts", 0),
                         "hr": st.get("homeRuns", 0), "sv": st.get("saves", 0), "hld": st.get("holds", 0),
                         "bs": st.get("blownSaves", 0), "note": st.get("note")})
        lineup = []
        for pid in t.get("battingOrder", []):
            p = t["players"].get(f"ID{pid}", {})
            lineup.append({"id": pid, "name": (p.get("person") or {}).get("fullName"),
                           "pos": (p.get("position") or {}).get("abbreviation")})
        out[side] = {"team": t["team"]["id"], "pitchers": rows, "bullpen": t.get("bullpen", []),
                     "bench": t.get("bench", []), "lineup": lineup}
    return out


# ------------------------------------------------------------------ Baseball Savant

def _csv(url: str) -> list[dict]:
    body = get(url, raw=True).decode("utf-8-sig", "replace")
    return list(csv.DictReader(io.StringIO(body)))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def savant(season: int) -> dict:
    out = {"expected": {}, "pitcher": {}, "batter": {}, "park": {}}
    for r in _csv(f"{SAVANT}/leaderboard/expected_statistics?type=pitcher&year={season}&position=&team=&min=1&csv=true"):
        out["expected"][r["player_id"]] = {"pa": _num(r.get("pa")), "xera": _num(r.get("xera")), "era": _num(r.get("era")),
                                           "woba": _num(r.get("woba")), "xwoba": _num(r.get("est_woba"))}
    for kind in ("pitcher", "batter"):
        for r in _csv(f"{SAVANT}/leaderboard/statcast?type={kind}&year={season}&position=&team=&min=1&csv=true"):
            out[kind][r["player_id"]] = {"bbe": _num(r.get("attempts")), "barrel": _num(r.get("brl_percent")),
                                         "hardhit": _num(r.get("ev95percent")), "ev": _num(r.get("avg_hit_speed"))}
    html = get(f"{SAVANT}/leaderboard/statcast-park-factors?type=year&year={season}&batSide=&stat=index_wOBA"
               f"&condition=All&rolling=3", raw=True).decode("utf-8", "replace")
    m = re.search(r"var data\s*=\s*(\[.*?\]);", html, re.S)
    if m:
        for r in json.loads(m.group(1)):
            out["park"][str(r["venue_id"])] = {
                "name": r.get("venue_name"), "runs": _num(r.get("index_runs")), "hr": _num(r.get("index_hr")),
                "woba": _num(r.get("index_woba")), "so": _num(r.get("index_so")), "bb": _num(r.get("index_bb")),
                "1b": _num(r.get("index_1b")), "2b": _num(r.get("index_2b")), "3b": _num(r.get("index_3b")),
                "hits": _num(r.get("index_hits")),
                "pa": _num(r.get("n_pa")), "years": r.get("year_range"),
            }
    return out


# ------------------------------------------------------------------ momios (opcional)

def odds() -> list[dict] | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals",
                                "oddsFormat": "american"})
    return get(f"{ODDS}?{q}")


# ------------------------------------------------------------------ orquestación

def et_today() -> dt.date:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # noqa: BLE001
        return (dt.datetime.utcnow() - dt.timedelta(hours=4)).date()


def fetch_bundle(today: dt.date | None = None, days: int = 2, bullpen_days: int = 10) -> dict:
    today = today or et_today()
    season = today.year
    errors: dict[str, str] = {}

    def attempt(name, fn, default=None):
        t0 = time.time()
        try:
            v = fn()
            log(f"ok  {name} ({time.time() - t0:.1f}s)")
            return v
        except Exception as e:  # noqa: BLE001
            log(f"ERR {name}: {e!r}")
            errors[name] = repr(e)
            return default

    end = today + dt.timedelta(days=days - 1)
    bundle = {
        "meta": {"generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "today": today.isoformat(), "season": season, "days": days},
        "teams": attempt("teams", lambda: teams(season), {}),
        "upcoming": attempt("upcoming", lambda: upcoming_games(today.isoformat(), end.isoformat()), []),
        "results": [],
        "prevResults": attempt("prevResults", lambda: season_results(season - 1), []),
        "standings": attempt("standings", lambda: standings(season), {}),
        "teamStats": attempt("teamStats", lambda: team_stats(season), {}),
        "playersPitching": attempt("playersPitching", lambda: players(season, "pitching"), []),
        "playersHitting": attempt("playersHitting", lambda: players(season, "hitting"), []),
        "savant": attempt("savant", lambda: savant(season), {}),
        "arsenal": attempt("arsenal", lambda: arsenal(season), {}),
        "odds": attempt("odds", odds, None),
    }
    bundle["results"], bundle["remaining"] = attempt("schedule", lambda: season_schedule(season), ([], []))
    bundle["live"] = getattr(upcoming_games, "live", [])
    bundle["scoreboard"] = attempt("scoreboard", lambda: scoreboard((today - dt.timedelta(days=1)).isoformat(),
                                                                  today.isoformat()), [])

    playing = sorted({t for g in bundle["upcoming"] for t in (g["away"], g["home"])})
    rosters = attempt("rosters", lambda: pmap(lambda t: (t, roster(t, season)), playing, workers=6), [])
    bundle["rosters"] = {str(t): r for t, r in rosters or []}
    bundle["transactions"] = attempt(
        "transactions", lambda: transactions(set(playing), (today - dt.timedelta(days=21)).isoformat(),
                                             today.isoformat()), [])

    pids = sorted({pid for g in bundle["upcoming"] for pid in g["probable"].values() if pid})
    pitchers = attempt("probables", lambda: pmap(lambda p: pitcher(p, season), pids), [])
    bundle["pitchers"] = {str(p["id"]): p for p in pitchers or []}

    ids = [p["id"] for p in bundle["playersPitching"]] + [p["id"] for p in bundle["playersHitting"]]
    ids += [pid for g in bundle["upcoming"] for side in ("away", "home") for pid in g["lineups"][side]]
    throws, bats = attempt("handedness", lambda: handedness(ids), ({}, {}))
    bundle["hands"], bundle["bats"] = throws, bats

    since = (today - dt.timedelta(days=bullpen_days)).isoformat()
    pks = [g["pk"] for g in bundle["results"] if g["date"] and g["date"] >= since]
    boxes = attempt("boxscores", lambda: pmap(boxscore, pks), [])
    by_pk = {g["pk"]: g for g in bundle["results"]}
    bundle["boxscores"] = [dict(b, pk=pk, date=by_pk[pk]["date"]) for pk, b in zip(pks, boxes or [])]
    bundle["meta"]["errors"] = errors
    return bundle


def main() -> None:
    import argparse
    import gzip

    ap = argparse.ArgumentParser(description="Descarga el bundle de datos de MLBgs")
    ap.add_argument("--out", default="data/raw_bundle.json.gz")
    ap.add_argument("--date", help="fecha base YYYY-MM-DD (por defecto hoy, hora del Este)")
    ap.add_argument("--days", type=int, default=2)
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.date) if args.date else None
    bundle = fetch_bundle(today, args.days)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))
    log(f"bundle → {args.out}: {len(bundle['upcoming'])} partidos próximos, "
        f"{len(bundle['results'])} resultados, {len(bundle['boxscores'])} box scores, errores={list(bundle['meta']['errors'])}")


if __name__ == "__main__":
    main()
