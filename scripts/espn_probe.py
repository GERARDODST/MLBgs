"""Sonda temporal: qué momios publica ESPN en su API pública (sin clave) para los partidos de MLB."""
import datetime as dt
import gzip
import json
import os
import urllib.request

OUT = "dev/espn"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "MLBgs/1.0 (+https://github.com/GERARDODST/MLBgs)", "Accept": "application/json"}


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        return repr(e), None


def save(name, obj):
    with gzip.open(f"{OUT}/{name}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(obj, f)


log = []
today = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
evs = []
for d in (today, today + dt.timedelta(days=1)):
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={d:%Y%m%d}"
    st, js = get(url)
    log.append({"url": url, "status": st})
    if js:
        save(f"scoreboard_{d:%Y%m%d}", js)
        evs += [e["id"] for e in js.get("events", [])]
for eid in evs[:2] + evs[-1:]:
    for name, url in (("core", f"https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}/competitions/{eid}/odds"),
                      ("summary", f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={eid}")):
        st, js = get(url)
        log.append({"url": url, "status": st})
        if js:
            save(f"{name}_{eid}", js if name == "core" else {k: js.get(k) for k in ("pickcenter", "odds", "header")})
    st, js = get(f"https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}/competitions/{eid}/odds")
    for it in (js or {}).get("items", [])[:6]:
        ref = it.get("$ref") if isinstance(it, dict) else None
        if ref and "/odds/" in ref:
            s2, j2 = get(ref.replace("http://", "https://"))
            log.append({"url": ref, "status": s2})
            if j2:
                save(f"coreitem_{eid}_{(j2.get('provider') or {}).get('id', 'x')}", j2)
with open(f"{OUT}/log.json", "w") as f:
    json.dump(log, f, indent=1)
print(json.dumps(log, indent=1))
