"""Descarga muestras reales de la MLB Stats API y Baseball Savant (temporal, para desarrollo)."""
import json, os, sys, urllib.request, datetime

OUT = "samples"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (MLBgs data probe)"}


def get(url, raw=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
    print(f"{len(body):>10} bytes  {url}")
    return body if raw else json.loads(body)


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


def trim(obj, n=3, depth=0):
    if isinstance(obj, list):
        return [trim(x, n, depth + 1) for x in obj[:n]]
    if isinstance(obj, dict):
        return {k: trim(v, n, depth + 1) for k, v in obj.items()}
    return obj


B = "https://statsapi.mlb.com/api/v1"
today = datetime.date(2026, 9, 23)
d0, d1 = today.isoformat(), (today + datetime.timedelta(days=2)).isoformat()
errors = {}

def attempt(name, fn):
    try:
        fn()
    except Exception as e:  # noqa
        print("ERROR", name, repr(e))
        errors[name] = repr(e)

sched = {}
def s1():
    global sched
    sched = get(f"{B}/schedule?sportId=1&startDate={d0}&endDate={d1}&hydrate=probablePitcher,team,venue(location,fieldInfo),lineups,officials,weather,linescore,seriesStatus,decisions")
    save("schedule_upcoming.json", sched)
attempt("schedule_upcoming", s1)

def s2():
    season = get(f"{B}/schedule?sportId=1&season=2026&gameType=R&hydrate=linescore,team")
    dates = season.get("dates", [])
    print("season dates", len(dates), "games", sum(len(d["games"]) for d in dates))
    save("schedule_season_trim.json", {"dates": [dates[-5]] if dates else []})
attempt("schedule_season", s2)

def s2b():
    prev = get(f"{B}/schedule?sportId=1&season=2025&gameType=R")
    dates = prev.get("dates", [])
    print("2025 dates", len(dates), "games", sum(len(d["games"]) for d in dates))
attempt("schedule_2025", s2b)

def s3():
    st = get(f"{B}/standings?leagueId=103,104&season=2026&standingsTypes=regularSeason&hydrate=team(division,league)")
    save("standings.json", trim(st, 2))
attempt("standings", s3)

def s4():
    games = [g for d in sched.get("dates", []) for g in d["games"]]
    g = games[0]
    save("one_game.json", g)
    for side in ("away", "home"):
        pp = g["teams"][side].get("probablePitcher")
        if not pp:
            continue
        pid = pp["id"]
        p = get(f"{B}/people/{pid}?hydrate=stats(group=[pitching],type=[season,gameLog,statSplits],sitCodes=[h,a],season=2026)")
        save(f"pitcher_{side}.json", p)
        p2 = get(f"{B}/people/{pid}/stats?stats=gameLog&group=pitching&season=2025")
        save(f"pitcher_{side}_gamelog2025.json", trim(p2, 3))
        break
attempt("pitcher", s4)

def s5():
    save("teams_pitching_season.json", trim(get(f"{B}/teams/stats?season=2026&group=pitching&stats=season&sportIds=1"), 2))
    save("teams_pitching_spRp.json", trim(get(f"{B}/teams/stats?season=2026&group=pitching&stats=statSplits&sitCodes=sp,rp&sportIds=1"), 4))
    save("teams_hitting_vlvr.json", trim(get(f"{B}/teams/stats?season=2026&group=hitting&stats=statSplits&sitCodes=vl,vr&sportIds=1"), 4))
    save("teams_hitting_season.json", trim(get(f"{B}/teams/stats?season=2026&group=hitting&stats=season&sportIds=1"), 2))
attempt("teams_stats", s5)

def s6():
    x = get(f"{B}/stats?stats=season&group=pitching&season=2026&sportId=1&playerPool=ALL&limit=3000")
    print("players pitching:", len(x.get("stats", [{}])[0].get("splits", [])))
    save("players_pitching.json", trim(x, 3))
attempt("players_pitching", s6)

def s7():
    y = (today - datetime.timedelta(days=1)).isoformat()
    ys = get(f"{B}/schedule?sportId=1&date={y}")
    pk = ys["dates"][0]["games"][0]["gamePk"]
    bx = get(f"{B}/game/{pk}/boxscore")
    t = bx["teams"]["home"]
    slim = {"pitchers": t.get("pitchers"), "bullpen": t.get("bullpen"), "team": t.get("team"),
            "players_sample": {k: v for k, v in list(t["players"].items()) if v.get("stats", {}).get("pitching")}}
    save("boxscore_home_pitchers.json", trim(slim, 4))
attempt("boxscore", s7)

def s8():
    for name, url in {
        "savant_expected_pitcher.csv": "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year=2026&position=&team=&min=1&csv=true",
        "savant_statcast_pitcher.csv": "https://baseballsavant.mlb.com/leaderboard/statcast?type=pitcher&year=2026&position=&team=&min=1&csv=true",
        "savant_statcast_batter.csv": "https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year=2026&position=&team=&min=1&csv=true",
    }.items():
        try:
            body = get(url, raw=True).decode("utf-8-sig", "replace")
            with open(os.path.join(OUT, name), "w") as f:
                f.write("\n".join(body.splitlines()[:6]))
        except Exception as e:  # noqa
            print("ERROR", name, repr(e)); errors[name] = repr(e)
    try:
        html = get("https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year=2026&batSide=&stat=index_wOBA&condition=All&rolling=3", raw=True).decode("utf-8", "replace")
        i = html.find("var data")
        with open(os.path.join(OUT, "savant_parkfactors_snippet.txt"), "w") as f:
            f.write(html[i:i + 4000] if i >= 0 else html[:4000])
    except Exception as e:  # noqa
        print("ERROR parkfactors", repr(e)); errors["parkfactors"] = repr(e)
attempt("savant", s8)

save("errors.json", errors)
print("done; errors:", errors)
