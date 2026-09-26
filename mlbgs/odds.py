"""Momios automáticos antes del partido, como SofaScore: apertura, actual y cierre por casa.

SofaScore no raspa las casas: muestra el feed de momios de sus socios de apuestas (datos con licencia)
y guarda el momio de apertura para marcar el movimiento (flechas ▲▼). Aquí se hace lo mismo con APIs
de momios con licencia:

  * odds-api.net (principal, secreto ODDS_API_NET_KEY): 27 casas para MLB; las que operan en México
    se piden a la API (/bookmakers?country_code=MX) y se marcan «MX».
  * The Odds API (respaldo, secreto ODDS_API_KEY): casas de EE. UU. en una sola llamada.

Playdoit, Caliente y Team México bloquean el acceso automático y no tienen API pública: su momio lo
captura el usuario en la página (captura o texto) y manda sobre este.

Cada corrida guarda en data/odds/<fecha>.json, por partido (gamePk) y casa:
  open = el primer momio visto (apertura) · last = el más reciente antes del primer lanzamiento.
Al empezar el partido ya no se pide: `last` queda como cierre. Para gastar pocas llamadas, cada partido
se refresca según lo que falta para el juego (> 6 h: cada 3 h · 1–6 h: cada hora · < 1 h: cada corrida)
y hay topes por corrida y por día. El bundle recibe `odds` con la forma de The Odds API (precios
americanos) más `pk`, `provider` y, por casa, `mx` y `open`, para que model.index_odds los use igual.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

from . import mathlib as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ODDS_DIR = os.path.join(ROOT, "data", "odds")
NET = "https://api.odds-api.net/v1"
THE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
UA = "MLBgs/1.0 (+https://github.com/GERARDODST/MLBgs)"
PROVIDERS = {"odds-api.net": "odds-api.net", "the-odds-api": "The Odds API"}
DAILY = {"odds-api.net": int(os.environ.get("ODDS_API_NET_DAILY", "300")),
         "the-odds-api": int(os.environ.get("ODDS_API_DAILY", "6"))}
PER_RUN = int(os.environ.get("ODDS_API_NET_PER_RUN", "40"))
THE_GAP_MIN = 120          # The Odds API trae todos los partidos en una llamada: como mucho cada 2 h
MATCH_HOURS = 6            # un evento de la API es el partido si empieza a menos de 6 h


# ------------------------------------------------------------------ conversiones

dec_to_am = M.decimal_to_american


am_to_dec = M.american_to_decimal


def _ts(x) -> dt.datetime | None:
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x / 1000 if x > 1e11 else x, dt.timezone.utc)
    try:
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ equipos: nombre de la API → id de MLB

def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().replace(".", "").split())


def team_index(teams: dict):
    names = {}
    for t in (teams or {}).values():
        for k in ("name", "club"):
            if t.get(k):
                names[_norm(t[k])] = t["id"]
        if t.get("location") and t.get("club"):
            names[_norm(f"{t['location']} {t['club']}")] = t["id"]

    def tid(name):
        n = _norm(name)
        if n in names:
            return names[n]
        hits = [(len(k), i) for k, i in names.items() if n.endswith(" " + k) or k.endswith(" " + n)]
        return max(hits)[1] if hits else None
    return tid


# ------------------------------------------------------------------ renglón normalizado por casa
# {"ml": {"away": am, "home": am}, "rl": {"away": {"point", "price"}, "home": {...}},
#  "total": {"over": {"point", "price"}, "under": {...}}}   (precios americanos)

def _pick_rl(cands: dict) -> tuple | None:
    """Run line: el par ±1.5 si está; si no, el par más parejo."""
    pairs = [(hl, v) for hl, v in cands.items() if "home" in v and "away" in v]
    if not pairs:
        return None
    std = [p for p in pairs if abs(p[0]) == 1.5]
    return min(std or pairs, key=lambda p: abs(am_to_dec(p[1]["home"]) - am_to_dec(p[1]["away"])))


def _pick_total(cands: dict) -> tuple | None:
    """Total principal: la línea con Over y Under más parejos."""
    pairs = [(ln, v) for ln, v in cands.items() if "over" in v and "under" in v]
    if not pairs:
        return None
    return min(pairs, key=lambda p: (abs(am_to_dec(p[1]["over"]) - am_to_dec(p[1]["under"])), p[0]))


def _row(ml: dict, rl_c: dict, tot_c: dict) -> dict:
    row = {"ml": {s: ml[s] for s in ("away", "home") if ml.get(s) is not None}, "rl": {}, "total": {}}
    rl = _pick_rl(rl_c)
    if rl:
        hl, v = rl
        row["rl"] = {"home": {"point": hl, "price": v["home"]}, "away": {"point": -hl, "price": v["away"]}}
    tot = _pick_total(tot_c)
    if tot:
        ln, v = tot
        row["total"] = {k: {"point": ln, "price": v[k]} for k in ("over", "under")}
    return row


def rows_from_net(items: list[dict]) -> dict:
    """Snapshot de odds-api.net (renglones por casa y selección, momio decimal) → {casa: renglón}."""
    draw_groups = {(it.get("bookmaker"), it.get("market_group_id")) for it in items if (it.get("side") or "").lower() == "draw"}
    acc = defaultdict(lambda: {"title": None, "ml": {}, "rl": defaultdict(dict), "tot": defaultdict(dict)})
    for it in items:
        if (it.get("is_available") is False or it.get("player_name")
                or (it.get("bookmaker"), it.get("market_group_id")) in draw_groups):   # 1X2 (con empate): no es el ML
            continue
        per = it.get("period")
        if per not in (None, 0, "0", "full time", "full_time", "ft") and it.get("period_str") not in ("full time",):
            continue
        am = dec_to_am(_num(it.get("odds")) or 0)
        if am is None:
            continue
        kind = (it.get("type") or it.get("bet_type") or it.get("market_key") or "").lower()
        side = (it.get("side") or "").lower()
        b = acc[it["bookmaker"]]
        b["title"] = b["title"] or it.get("bookmaker_name")
        line = _num(it.get("line"))
        if kind == "moneyline" and side in ("away", "home"):
            b["ml"][side] = am
        elif kind in ("handicap", "spread", "spreads", "run_line") and side in ("away", "home") and line is not None:
            b["rl"][line if side == "home" else -line][side] = am
        elif kind in ("total", "totals") and side in ("over", "under") and line is not None:
            b["tot"][line][side] = am
    out = {}
    for bk, b in acc.items():
        row = _row(b["ml"], b["rl"], b["tot"])
        if row["ml"] or row["rl"] or row["total"]:
            out[bk] = dict(row, title=b["title"] or bk)
    return out


def rows_from_the(ev: dict, tid) -> dict:
    """Evento de The Odds API (h2h, spreads, totals en americano) → {casa: renglón}."""
    a, h = tid(ev.get("away_team", "")), tid(ev.get("home_team", ""))
    out = {}
    for bk in ev.get("bookmakers", []):
        ml, rl_c, tot_c = {}, defaultdict(dict), defaultdict(dict)
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                t = tid(o.get("name", ""))
                side = "home" if t == h else "away" if t == a else None
                price = o.get("price")
                if price is None:
                    continue
                if m.get("key") == "h2h" and side:
                    ml[side] = price
                elif m.get("key") == "spreads" and side and o.get("point") is not None:
                    rl_c[o["point"] if side == "home" else -o["point"]][side] = price
                elif m.get("key") == "totals" and o.get("point") is not None:
                    tot_c[o["point"]][o.get("name", "").lower()] = price
        row = _row(ml, rl_c, tot_c)
        if row["ml"] or row["rl"] or row["total"]:
            out[bk.get("key") or bk.get("title")] = dict(row, title=bk.get("title") or bk.get("key"))
    return out


def flip(row: dict) -> dict:
    """Voltea visitante/local (la API puede listar al revés un partido en sede neutral)."""
    sw = {"away": "home", "home": "away"}
    return dict(row, ml={sw[k]: v for k, v in (row.get("ml") or {}).items()},
                rl={sw[k]: v for k, v in (row.get("rl") or {}).items()})


# ------------------------------------------------------------------ emparejar eventos con partidos (gamePk)

def match(games: list[dict], events: list[dict], tid) -> dict:
    """{pk: (evento, volteado)}: mismo par de equipos y a menos de MATCH_HOURS del inicio."""
    out, used = {}, set()
    for g in games:
        t0 = _ts(g.get("time"))
        best = None
        for i, ev in enumerate(events):
            if i in used:
                continue
            a, h = tid(ev.get("away_team", "")), tid(ev.get("home_team", ""))
            same, swapped = (a, h) == (g["away"], g["home"]), (a, h) == (g["home"], g["away"])
            if not (same or swapped):
                continue
            te = _ts(ev.get("start_time") or ev.get("commence_time"))
            gap = abs((te - t0).total_seconds()) if te and t0 else 9e9
            if gap <= MATCH_HOURS * 3600 and (best is None or gap < best[0]):
                best = (gap, i, swapped)
        if best:
            used.add(best[1])
            out[g["pk"]] = (events[best[1]], best[2])
    return out


# ------------------------------------------------------------------ almacén de apertura / último momio

def _fill(dst: dict, src: dict) -> dict:
    """Completa en dst lo que falte con src (la apertura de cada mercado es el primero que se vio)."""
    for k, v in src.items():
        if isinstance(v, dict):
            _fill(dst.setdefault(k, {}), v)
        elif k not in dst:
            dst[k] = v
    return dst


def merge(entry: dict, rows: dict, now: dt.datetime, mx: set) -> dict:
    books = entry.setdefault("books", {})
    at = _iso(now)
    for bk, row in rows.items():
        cur = {k: row[k] for k in ("ml", "rl", "total")}
        b = books.setdefault(bk, {"title": row.get("title") or bk})
        b["mx"] = bk in mx
        b["open"] = _fill(b.get("open") or {"at": at}, cur)
        b["last"] = dict(cur, at=at)
    entry["checked"] = at
    return entry


def due(entry: dict, now: dt.datetime) -> bool:
    """¿Toca pedir este partido? Nunca después del primer lanzamiento (el último momio queda de cierre)."""
    start = _ts(entry.get("start"))
    if not start or start <= now:
        return False
    mins = (start - now).total_seconds() / 60
    gap = 180 if mins > 360 else 60 if mins > 60 else 0
    last = _ts(entry.get("checked"))
    return last is None or (now - last).total_seconds() / 60 >= gap - 2


def load_day(date: str, base: str = ODDS_DIR) -> dict:
    path = os.path.join(base, f"{date}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"date": date, "games": {}}


def save_day(day: dict, base: str = ODDS_DIR) -> None:
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, f"{day['date']}.json"), "w", encoding="utf-8") as f:
        json.dump(day, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def load_usage(base: str = ODDS_DIR) -> dict:
    path = os.path.join(base, "usage.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"calls": {}}


def save_usage(u: dict, base: str = ODDS_DIR) -> None:
    keep = sorted(u.get("calls", {}))[-10:]
    u["calls"] = {d: u["calls"][d] for d in keep}
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "usage.json"), "w", encoding="utf-8") as f:
        json.dump(u, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


# ------------------------------------------------------------------ clientes HTTP (se pueden reemplazar en pruebas)

def http_json(url: str, headers: dict | None = None, tries: int = 2):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:   # 401/403/429: no insistir
            raise RuntimeError(f"HTTP {e.code} en {url.split('?')[0]}") from None
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"GET {url.split('?')[0]} falló: {last!r}")


class Budget:
    """Cuenta las llamadas del día (UTC) y de la corrida para no pasarse del plan gratuito."""

    def __init__(self, usage: dict, provider: str, now: dt.datetime, per_run: int):
        self.day = now.strftime("%Y-%m-%d")
        self.calls = usage.setdefault("calls", {}).setdefault(self.day, {})
        self.provider, self.per_run, self.run = provider, per_run, 0

    def ok(self) -> bool:
        return self.run < self.per_run and self.calls.get(self.provider, 0) < DAILY[self.provider]

    def spend(self):
        self.run += 1
        self.calls[self.provider] = self.calls.get(self.provider, 0) + 1


def net_client(key: str, budget: Budget, fetch=http_json):
    def get(path: str, **params):
        if not budget.ok():
            raise RuntimeError("tope de llamadas alcanzado")
        budget.spend()
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        return fetch(f"{NET}{path}" + (f"?{q}" if q else ""), {"X-API-Key": key})
    return get


def _pages(get, path: str, **params) -> list:
    items, cursor = [], None
    for _ in range(10):
        r = get(path, **params, cursor=cursor) if cursor else get(path, **params)
        items += r.get("items") or []
        cursor = r.get("next_cursor")
        if not cursor:
            break
    return items


def net_events(get, t0: dt.datetime, t1: dt.datetime) -> list[dict]:
    rng = {"start_from": int(t0.timestamp()), "start_to": int(t1.timestamp()), "limit": 200}
    evs = _pages(get, "/events", sport="baseball", league="MLB", **rng)
    if not evs:   # por si la liga viene con otro nombre: todo el béisbol y se filtra
        evs = [e for e in _pages(get, "/events", sport="baseball", **rng) if "mlb" in _norm(e.get("league"))
               or "major league" in _norm(e.get("league"))]
    return evs


def net_mx_books(get) -> list[str]:
    r = get("/bookmakers", country_code="MX")
    items = r.get("items") if isinstance(r, dict) else r
    out = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict) and it.get("bookmaker"):
            if not it.get("country_codes") or "MX" in [c.upper() for c in it["country_codes"]]:
                out.append(it["bookmaker"])
    return sorted(set(out))


def net_snapshot(get, event_id: str) -> list[dict]:
    return _pages(get, f"/events/{urllib.parse.quote(str(event_id))}/odds/snapshot",
                  types="moneyline,handicap,total", periods="0", price_fields="odds", limit=2000)


# ------------------------------------------------------------------ corrida

def games_of(bundle: dict) -> list[dict]:
    return [{"pk": g["pk"], "date": g["date"], "time": g["time"], "away": g["away"], "home": g["home"]}
            for g in bundle.get("upcoming", []) if g.get("time")]


def update(bundle: dict, now: dt.datetime | None = None, base: str = ODDS_DIR, env=None, fetch=http_json,
           log=print) -> dict:
    """Pide los momios que tocan, actualiza data/odds/ y devuelve el estado de la corrida."""
    env = os.environ if env is None else env
    now = now or dt.datetime.now(dt.timezone.utc)
    games = games_of(bundle)
    tid = team_index(bundle.get("teams"))
    days = {d: load_day(d, base) for d in sorted({g["date"] for g in games})}
    for g in games:
        e = days[g["date"]]["games"].setdefault(str(g["pk"]), {})
        e.update({"pk": g["pk"], "away": g["away"], "home": g["home"], "start": g["time"]})
    usage = load_usage(base)
    status = {"provider": None, "calls": 0, "refreshed": 0, "error": None}
    key_net, key_the = env.get("ODDS_API_NET_KEY"), env.get("ODDS_API_KEY")
    budget = None
    try:
        if key_net:
            status["provider"] = "odds-api.net"
            budget = Budget(usage, "odds-api.net", now, PER_RUN)
            get = net_client(key_net, budget, fetch)
            mx_cache = usage.get("mx") or {}
            if mx_cache.get("day") != budget.day:
                try:
                    usage["mx"] = {"day": budget.day, "books": net_mx_books(get)}
                except Exception as e:  # noqa: BLE001 - sin la lista, las casas solo no llevan la marca MX
                    log(f"odds: lista MX no disponible ({e})")
            mx = set((usage.get("mx") or {}).get("books") or [])
            todo = [g for g in games if due(days[g["date"]]["games"][str(g["pk"])], now)]
            if todo and any(not days[g["date"]]["games"][str(g["pk"])].get("event") for g in todo):
                t0 = min(_ts(g["time"]) for g in todo) - dt.timedelta(hours=MATCH_HOURS)
                t1 = max(_ts(g["time"]) for g in todo) + dt.timedelta(hours=MATCH_HOURS)
                found = match(todo, net_events(get, t0, t1), tid)
                for g in todo:
                    e = days[g["date"]]["games"][str(g["pk"])]
                    if g["pk"] in found:
                        ev, sw = found[g["pk"]]
                        e.update({"event": str(ev["event_id"]), "swapped": sw, "provider": "odds-api.net"})
                    elif not e.get("event"):
                        e["checked"] = _iso(now)     # aún no lo publica la API: se vuelve a buscar según la cadencia
            todo.sort(key=lambda g: g["time"])    # primero los que empiezan antes
            for g in todo:
                e = days[g["date"]]["games"][str(g["pk"])]
                if not e.get("event") or e.get("provider") not in (None, "odds-api.net"):
                    continue
                if not budget.ok():
                    log("odds: tope de llamadas; el resto queda con su último momio")
                    break
                rows = rows_from_net(net_snapshot(get, e["event"]))
                if e.get("swapped"):
                    rows = {k: flip(v) for k, v in rows.items()}
                if rows:
                    merge(e, rows, now, mx)
                    status["refreshed"] += 1
                else:
                    e["checked"] = _iso(now)
        elif key_the:
            status["provider"] = "the-odds-api"
            budget = Budget(usage, "the-odds-api", now, 1)
            last = _ts(usage.get("theLast"))
            soon = any(due(days[g["date"]]["games"][str(g["pk"])], now) for g in games)
            if soon and budget.ok() and (last is None or (now - last).total_seconds() / 60 >= THE_GAP_MIN):
                budget.spend()
                q = urllib.parse.urlencode({"apiKey": key_the, "regions": "us", "markets": "h2h,spreads,totals",
                                            "oddsFormat": "american"})
                evs = fetch(f"{THE}?{q}") or []
                usage["theLast"] = _iso(now)
                live = [g for g in games if (_ts(g["time"]) or now) > now]
                for pk, (ev, sw) in match(live, evs, tid).items():
                    g = next(x for x in live if x["pk"] == pk)
                    rows = rows_from_the(ev, tid)
                    if sw:
                        rows = {k: flip(v) for k, v in rows.items()}
                    e = days[g["date"]]["games"][str(pk)]
                    e["provider"] = "the-odds-api"
                    if rows:
                        merge(e, rows, now, set())
                        status["refreshed"] += 1
    except Exception as e:  # noqa: BLE001 - sin momios nuevos la página sigue con los guardados
        status["error"] = str(e)
        log(f"odds: {e}")
    status["calls"] = budget.run if budget else 0
    for d in days.values():
        d["games"] = {pk: e for pk, e in d["games"].items() if e.get("books") or e.get("event") or e.get("checked")}
        if d["games"]:
            save_day(d, base)
    if status["calls"]:
        save_usage(usage, base)
    return status


def to_bundle(bundle: dict, base: str = ODDS_DIR, now: dt.datetime | None = None) -> list[dict] | None:
    """Momios guardados de los partidos del bundle, con la forma de The Odds API (lo que lee el modelo)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    teams = {int(t["id"]): t for t in (bundle.get("teams") or {}).values()}
    out = []
    for date in sorted({g["date"] for g in games_of(bundle)}):
        day = load_day(date, base)
        for g in games_of(bundle):
            e = day["games"].get(str(g["pk"])) if g["date"] == date else None
            if not e or not e.get("books"):
                continue
            name = lambda i: (teams.get(int(i)) or {}).get("name") or str(i)  # noqa: E731
            books = []
            for bk, b in sorted(e["books"].items(), key=lambda kv: (not kv[1].get("mx"), kv[1].get("title") or kv[0])):
                books.append({"key": bk, "title": b.get("title") or bk, "mx": bool(b.get("mx")),
                              "last_update": b["last"].get("at"), "markets": _markets(b["last"], name(g["away"]), name(g["home"])),
                              "open": {k: b["open"].get(k) or {} for k in ("ml", "rl", "total")}, "openAt": b["open"].get("at")})
            start = _ts(e.get("start"))
            out.append({"id": e.get("event") or str(g["pk"]), "pk": g["pk"], "provider": PROVIDERS.get(e.get("provider"), e.get("provider")),
                        "commence_time": e.get("start"), "home_team": name(g["home"]), "away_team": name(g["away"]),
                        "checked": e.get("checked"), "closed": bool(start and start <= now), "bookmakers": books})
    return out or None


def _markets(row: dict, away: str, home: str) -> list[dict]:
    ms = []
    if row.get("ml"):
        ms.append({"key": "h2h", "outcomes": [{"name": away if s == "away" else home, "price": p} for s, p in row["ml"].items()]})
    if row.get("rl"):
        ms.append({"key": "spreads", "outcomes": [{"name": away if s == "away" else home, "price": v["price"], "point": v["point"]}
                                                  for s, v in row["rl"].items()]})
    if row.get("total"):
        ms.append({"key": "totals", "outcomes": [{"name": k.capitalize(), "price": v["price"], "point": v["point"]}
                                                 for k, v in row["total"].items()]})
    return ms
