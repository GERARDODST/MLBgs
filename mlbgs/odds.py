"""Momios automáticos antes del partido, como SofaScore: apertura, actual y cierre por casa.

SofaScore no raspa las casas: muestra el feed de momios de sus socios de apuestas (datos con licencia)
y guarda el momio de apertura para marcar el movimiento (flechas ▲▼). Aquí se hace lo mismo:

  * ESPN (siempre, sin clave): el marcador público de ESPN publica los momios de su casa socia
    (DraftKings) con apertura y actual de moneyline, run line y total. Una llamada por fecha.
  * odds-api.net (opcional, secreto ODDS_API_NET_KEY): 27 casas para MLB; las que operan en México
    se piden a la API (/bookmakers?country_code=MX) y se marcan «MX».
  * The Odds API (opcional, secreto ODDS_API_KEY, plan gratis de 500 créditos al mes): solo lo que el feed
    gratis no trae, el F5 (ganador y total de las primeras 5 entradas) de los partidos cuyo pick es F5.
    Cada mercado de un partido cuesta 1 crédito; hay topes por día y por mes.

Playdoit, Caliente y Team México bloquean el acceso automático y no tienen API pública: su momio lo
captura el usuario en la página (captura o texto) y manda sobre este.

Cada corrida guarda en data/odds/<fecha>.json, por partido (gamePk) y casa:
  open = el primer momio visto (apertura) · last = el más reciente antes del primer lanzamiento.
Al empezar el partido ya no se pide: `last` queda como cierre. Para gastar pocas llamadas, cada partido
se refresca según lo que falta para el juego (> 6 h: cada 3 h · 1–6 h: cada hora · < 1 h: cada corrida)
y hay topes por corrida y por día. Un partido con un cambio (abridor, lineup, horario, baja; ver
mlbgs/cambios.py) se revisa en la misma corrida aunque no le toque, y cada llamada a ESPN actualiza todos los
partidos de esa fecha (la llamada ya se pagó).

The Odds API gasta créditos, así que se reparte por prioridad (dentro de los topes por día y por mes):
  0  un mercado ya pedido cuyo abridor cambió después (su momio ya no sirve)
  1  el mercado de la decisión del partido cuando ESPN no lo trae (F5, ponches, team total, NRFI)
  2  verificación: el mercado de la decisión (ML, run line o total) en otras casas, si hay stake o se espera momio
  3  el mercado del otro pick principal
Las prioridades 2 y 3 dejan libres los últimos RESERVE créditos del día para los cambios de última hora.

Verificación entre fuentes (`verify`): DraftKings (ESPN) contra la mediana de las otras casas, por mercado
(moneyline sin vig y total). Si difieren ≥ 3 pp (o la línea del total es otra) se marca «difiere»; el stake ya
usa la mediana de las casas, no una sola. Un momio visto antes de un cambio de abridor se marca `stale` y no
cuenta para el stake (la página lo muestra aparte). El bundle recibe `odds` con la forma de The Odds API (precios
americanos) más `pk`, `provider` y, por casa, `mx` y `open`, para que model.index_odds los use igual.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
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
THE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
PRED_DIR = os.path.join(ROOT, "data", "predictions")
# familia del pick → mercado de The Odds API (lo que ESPN no trae)
F5_MARKETS = {"F5": "h2h_1st_5_innings", "F5 total": "totals_1st_5_innings", "K": "pitcher_strikeouts",
              "Team total": "team_totals", "NRFI": "totals_1st_1_innings"}
MAIN_MARKETS = {"ML": "h2h", "RL": "spreads", "Total": "totals"}       # verificación contra otras casas
MAIN_KEY = {"h2h": "ml", "spreads": "rl", "totals": "total"}
THE_DAY_CREDITS = int(os.environ.get("ODDS_API_DAY_CREDITS", "24"))
RESERVE = int(os.environ.get("ODDS_API_RESERVE", "4"))      # créditos del día guardados para cambios de última hora
VERIFY_PP = 0.03            # DraftKings vs consenso: diferencia de probabilidad sin vig que se marca
ANCHOR = "draftkings"
THE_MONTH_CREDITS = int(os.environ.get("ODDS_API_MONTH_CREDITS", "470"))           # plan gratis: 500
UA = "MLBgs/1.0 (+https://github.com/GERARDODST/MLBgs)"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
PROVIDERS = {"espn": "ESPN", "odds-api.net": "odds-api.net", "the-odds-api": "The Odds API"}
DAILY = {"odds-api.net": int(os.environ.get("ODDS_API_NET_DAILY", "300")),
         "the-odds-api": THE_DAY_CREDITS,
         "espn": int(os.environ.get("ODDS_ESPN_DAILY", "300"))}
PER_RUN = int(os.environ.get("ODDS_API_NET_PER_RUN", "40"))
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
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()   # Rodríguez → rodriguez
    return " ".join(t.lower().replace(".", "").split())


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
        ml, rl_c, tot_c, f5ml, f5_c = {}, defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
        k_c, tt_c, first = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict)), {}
        for m in bk.get("markets", []):
            if any(_norm(o.get("name")) in ("draw", "tie", "empate") for o in m.get("outcomes", [])):
                continue                                   # 3 vías (con empate): no es el F5 que modela el framework
            for o in m.get("outcomes", []):
                t = tid(o.get("name", ""))
                side = "home" if t == h else "away" if t == a else None
                price = o.get("price")
                if price is None:
                    continue
                key = m.get("key")
                if key == "h2h" and side:
                    ml[side] = price
                elif key == "spreads" and side and o.get("point") is not None:
                    rl_c[o["point"] if side == "home" else -o["point"]][side] = price
                elif key == "totals" and o.get("point") is not None:
                    tot_c[o["point"]][o.get("name", "").lower()] = price
                elif key == "h2h_1st_5_innings" and side:
                    f5ml[side] = price
                elif key == "totals_1st_5_innings" and o.get("point") is not None:
                    f5_c[o["point"]][o.get("name", "").lower()] = price
                elif key == "pitcher_strikeouts" and o.get("point") is not None and o.get("description"):
                    k_c[_norm(o["description"])][o["point"]][o.get("name", "").lower()] = price
                elif key == "team_totals" and o.get("point") is not None and o.get("description"):
                    t = tid(o["description"])
                    sd = "home" if t == h else "away" if t == a else None
                    if sd:
                        tt_c[sd][o["point"]][o.get("name", "").lower()] = price
                elif key == "totals_1st_1_innings" and o.get("point") == 0.5:
                    first[{"over": "yrfi", "under": "nrfi"}.get(o.get("name", "").lower(), "x")] = price
        row = _row(ml, rl_c, tot_c)
        f5t = _pick_total(f5_c)
        f5 = {"ml": f5ml if len(f5ml) == 2 else {},
              "total": {k: {"point": f5t[0], "price": f5t[1][k]} for k in ("over", "under")} if f5t else {}}
        if f5["ml"] or f5["total"]:
            row["f5"] = f5
        props = {}
        for who, cands in k_c.items():                      # ponches: la línea más pareja de cada pitcher
            pk_ = _pick_total(cands)
            if pk_:
                props[who] = {k: {"point": pk_[0], "price": pk_[1][k]} for k in ("over", "under")}
        if props:
            row["k"] = props
        tts = {}
        for sd, cands in tt_c.items():
            pk_ = _pick_total(cands)
            if pk_:
                tts[sd] = {k: {"point": pk_[0], "price": pk_[1][k]} for k in ("over", "under")}
        if tts:
            row["tt"] = tts
        if "nrfi" in first and "yrfi" in first:
            row["nrfi"] = {"nrfi": first["nrfi"], "yrfi": first["yrfi"]}
        if row["ml"] or row["rl"] or row["total"] or any(row.get(k) for k in ("f5", "k", "tt", "nrfi")):
            out[bk.get("key") or bk.get("title")] = dict(row, title=short_title(bk.get("title") or bk.get("key")))
    return out


def short_title(t) -> str:
    """Nombre corto de la casa: BetOnline.ag → BetOnline, MyBookie.ag → MyBookie."""
    return re.sub(r"\.(ag|com|eu|lv)$", "", str(t or ""), flags=re.I)


def flip(row: dict) -> dict:
    """Voltea visitante/local (la API puede listar al revés un partido en sede neutral)."""
    sw = {"away": "home", "home": "away"}
    out = dict(row, ml={sw[k]: v for k, v in (row.get("ml") or {}).items()},
               rl={sw[k]: v for k, v in (row.get("rl") or {}).items()})
    if row.get("f5"):
        out["f5"] = dict(row["f5"], ml={sw[k]: v for k, v in (row["f5"].get("ml") or {}).items()})
    if row.get("tt"):
        out["tt"] = {sw[k]: v for k, v in row["tt"].items()}
    return out


# ------------------------------------------------------------------ ESPN (marcador público, sin clave)

def _espn_am(x) -> int | None:
    """'+108', '-112', 'EVEN' → americano entero."""
    t = str(x or "").strip().upper().replace("−", "-")
    if t in ("EVEN", "EV", "PK"):
        return 100
    try:
        v = float(t)
    except ValueError:
        return None
    return round(v) if abs(v) >= 100 else None


def _espn_line(x) -> float | None:
    """'o8.5', 'u8', '+1.5', '-1.5' → número."""
    t = str(x or "").strip().lower().lstrip("ou")
    try:
        return float(t)
    except ValueError:
        return None


def _espn_row(o: dict, when: str) -> dict:
    ml, rl, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}
    get = lambda blk, side: ((blk.get(side) or {}).get(when) or {})  # noqa: E731
    row = {"ml": {}, "rl": {}, "total": {}}
    for s in ("away", "home"):
        am = _espn_am(get(ml, s).get("odds"))
        if am is None and when == "close":     # respaldo: el moneyline suelto del equipo
            am = _espn_am((o.get(f"{s}TeamOdds") or {}).get("moneyLine"))
        if am is not None:
            row["ml"][s] = am
        pt, pr = _espn_line(get(rl, s).get("line")), _espn_am(get(rl, s).get("odds"))
        if pt is not None and pr is not None:
            row["rl"][s] = {"point": pt, "price": pr}
    for k in ("over", "under"):
        pt, pr = _espn_line(get(tot, k).get("line")), _espn_am(get(tot, k).get("odds"))
        if pt is not None and pr is not None:
            row["total"][k] = {"point": pt, "price": pr}
    if len(row["rl"]) < 2:
        row["rl"] = {}
    if len(row["total"]) < 2 or row["total"]["over"]["point"] != row["total"]["under"]["point"]:
        row["total"] = {}
    return row


def espn_events(sb: dict) -> list[dict]:
    """Marcador de ESPN → eventos con renglón actual (close) y apertura (open) por casa."""
    out = []
    for ev in (sb or {}).get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): (c.get("team") or {}).get("displayName") for c in comp.get("competitors", [])}
        rows, opens = {}, {}
        for o in comp.get("odds") or []:
            prov = o.get("provider") or {}
            name = prov.get("displayName") or prov.get("name") or "ESPN"
            key = _norm(name).replace(" ", "")
            cur, opn = _espn_row(o, "close"), _espn_row(o, "open")
            if cur["ml"] or cur["rl"] or cur["total"]:
                rows[key] = dict(cur, title=name)
                opens[key] = opn
        out.append({"id": str(ev.get("id")), "away_team": teams.get("away") or "", "home_team": teams.get("home") or "",
                    "commence_time": comp.get("date") or ev.get("date"), "rows": rows, "opens": opens})
    return out


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


MARKET_KEYS = ("ml", "rl", "total", "f5", "k", "tt", "nrfi")


def merge(entry: dict, rows: dict, now: dt.datetime, mx: set, opens: dict | None = None,
          provider: str | None = None) -> dict:
    """Actualiza el último momio de cada casa. La apertura es la que publica la casa si viene (ESPN);
    si no, el primer momio que se vio."""
    books = entry.setdefault("books", {})
    at = _iso(now)
    for bk, row in rows.items():
        cur = {k: row[k] for k in MARKET_KEYS if row.get(k)}
        b = books.setdefault(bk, {"title": row.get("title") or bk})
        b["mx"] = b.get("mx") or bk in mx
        own = {k: v for k, v in ((opens or {}).get(bk) or {}).items() if k in MARKET_KEYS and v}
        b["open"] = _fill(_fill(dict(own, at=b.get("open", {}).get("at") or at), b.get("open") or {}), cur) if own \
            else _fill(b.get("open") or {"at": at}, cur)
        b["last"] = dict({k: v for k, v in (b.get("last") or {}).items() if k in MARKET_KEYS}, **cur, at=at)
        if provider:
            b["src"] = provider
    if provider and provider not in entry.setdefault("providers", []):
        entry["providers"] = sorted(entry["providers"] + [provider])
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


def market_needs(date: str, pred_dir: str = PRED_DIR) -> dict:
    """{pk: {mercado: prioridad}} de The Odds API según el último análisis guardado (data/predictions).

    1 = mercado de la decisión que ESPN no trae · 2 = verificar el mercado de la decisión (ML, RL, total) cuando
    hay stake o se espera momio · 3 = mercado del otro pick principal."""
    path = os.path.join(pred_dir, f"{date}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows if isinstance(rows, list) else rows.values():
        need = {}
        dec = r.get("decision") or {}
        fam = dec.get("family")
        live = dec.get("status") == "apostar" or (dec.get("status") == "esperar" and dec.get("waitFor") == "momio")
        if fam in F5_MARKETS and dec.get("status") in ("apostar", "esperar"):
            need[F5_MARKETS[fam]] = 1
        elif fam in MAIN_MARKETS and live:
            need[MAIN_MARKETS[fam]] = 2
        for tp in r.get("topPicks") or []:
            if tp.get("family") in F5_MARKETS:
                m = F5_MARKETS[tp["family"]]
                # sin decisión guardada (formato anterior), los dos picks principales cuentan como prioridad 1
                need[m] = min(need.get(m, 9), 3 if dec else 1)
        if need:
            out[r["pk"]] = need
    return out


def f5_needs(date: str, pred_dir: str = PRED_DIR) -> dict:
    """{pk: [mercados]} (compatibilidad): los mercados de market_needs sin la prioridad."""
    return {pk: sorted(m) for pk, m in market_needs(date, pred_dir).items()}


def market_due(entry: dict, market: str, now: dt.datetime, window: int = 360, changed_at: str | None = None) -> bool:
    """Cada mercado de The Odds API: una vez cuando faltan ≤ `window` min (6 h; 3 h para verificar) y un refresco
    en la última hora y media. Si el abridor cambió después de pedirlo, se pide otra vez (una vez por cambio)."""
    start = _ts(entry.get("start"))
    if not start or start <= now:
        return False
    mins = (start - now).total_seconds() / 60
    at, n = (entry.get("theAt") or {}).get(market), (entry.get("theN") or {}).get(market, 0)
    if at is None and market in ("h2h_1st_5_innings", "totals_1st_5_innings") and entry.get("f5at"):
        at, n = entry["f5at"], entry.get("f5n", 1)          # datos guardados antes de llevar la cuenta por mercado
    last = _ts(at)
    if last is None:
        return mins <= window
    chg = _ts(changed_at)
    if chg and last < chg:
        return True                                         # prioridad 0: el momio pedido es de antes del cambio
    return n < 2 and mins <= 90 and (now - last).total_seconds() / 60 >= 60


def f5_due(entry: dict, now: dt.datetime) -> bool:
    return market_due(entry, "h2h_1st_5_innings", now)


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


def probe_summary(items: list[dict], day: str, event_id: str) -> dict:
    """Qué trae un partido completo de odds-api.net (sin filtros): mercados, periodos y casas. Sirve para saber
    si hay F5 (primeras 5 entradas) y cómo lo nombra la API antes de usarlo."""
    combos, books, samples = {}, set(), {}
    for it in items:
        key = "|".join(str(it.get(k) if it.get(k) is not None else "") for k in ("type", "market_key", "period", "period_str"))
        combos[key] = combos.get(key, 0) + 1
        books.add(it.get("bookmaker"))
        if str(it.get("period")) not in ("0", "None", "") and key not in samples:
            samples[key] = {k: it.get(k) for k in ("bookmaker", "selection_key", "selection_name", "line", "side", "odds")}
    return {"day": day, "event": event_id, "items": len(items), "books": sorted(b for b in books if b),
            "combos": dict(sorted(combos.items(), key=lambda kv: -kv[1])[:60]), "samples": samples}


def net_snapshot(get, event_id: str) -> list[dict]:
    return _pages(get, f"/events/{urllib.parse.quote(str(event_id))}/odds/snapshot",
                  types="moneyline,handicap,total", periods="0", price_fields="odds", limit=2000)


# ------------------------------------------------------------------ corrida

def games_of(bundle: dict) -> list[dict]:
    return [{"pk": g["pk"], "date": g["date"], "time": g["time"], "away": g["away"], "home": g["home"]}
            for g in bundle.get("upcoming", []) if g.get("time")]


def update(bundle: dict, now: dt.datetime | None = None, base: str = ODDS_DIR, env=None, fetch=http_json,
           log=print, pred_dir: str = PRED_DIR, hot=None, stale_after: dict | None = None) -> dict:
    """Pide los momios que tocan (ESPN siempre; odds-api.net o The Odds API si hay clave), actualiza
    data/odds/ y devuelve el estado de la corrida.

    hot: partidos con un cambio en esta corrida (se revisan aunque no les toque). stale_after: {pk: hora del
    último cambio de abridor} (un mercado de The Odds API pedido antes se vuelve a pedir, prioridad 0)."""
    env = os.environ if env is None else env
    now = now or dt.datetime.now(dt.timezone.utc)
    games = games_of(bundle)
    tid = team_index(bundle.get("teams"))
    hot = {int(k) for k in (hot or {})}
    stale_after = {int(k): v for k, v in (stale_after or {}).items()}
    days = {d: load_day(d, base) for d in sorted({g["date"] for g in games})}
    for g in games:
        e = days[g["date"]]["games"].setdefault(str(g["pk"]), {})
        e.update({"pk": g["pk"], "away": g["away"], "home": g["home"], "start": g["time"]})
    entry = lambda g: days[g["date"]]["games"][str(g["pk"])]  # noqa: E731
    pending = lambda g: bool(_ts(g["time"]) and _ts(g["time"]) > now)  # noqa: E731
    # primero los que empiezan antes; un partido con cambio se revisa aunque no le toque por horario
    todo = sorted((g for g in games if due(entry(g), now) or (g["pk"] in hot and pending(g))), key=lambda g: g["time"])
    usage = load_usage(base)
    status = {"provider": None, "calls": 0, "refreshed": 0, "error": None, "hot": sorted(hot)}
    used, errors, budgets, fresh = [], [], [], set()
    covered = list(todo)

    # 1) ESPN: momios públicos de su casa socia, con apertura (una llamada por fecha, sin clave)
    if todo and env.get("ODDS_ESPN", "1") != "0":
        budget = Budget(usage, "espn", now, 4)
        budgets.append(budget)
        used.append("espn")
        try:
            for date in sorted({g["date"] for g in todo}):
                if not budget.ok():
                    break
                budget.spend()
                # la llamada trae toda la fecha: se actualizan todos los partidos que no han empezado
                day_todo = [g for g in games if g["date"] == date and pending(g)]
                covered += [g for g in day_todo if g not in covered]
                evs = espn_events(fetch(f"{ESPN}?dates={date.replace('-', '')}"))
                for pk, (ev, sw) in match(day_todo, evs, tid).items():
                    rows, opens = ev["rows"], ev["opens"]
                    if sw:
                        rows, opens = {k: flip(v) for k, v in rows.items()}, {k: flip(v) for k, v in opens.items()}
                    if rows:
                        g = next(x for x in day_todo if x["pk"] == pk)
                        merge(entry(g), rows, now, set(), opens, "espn")
                        entry(g)["espn"] = ev["id"]
                        fresh.add(pk)
        except Exception as e:  # noqa: BLE001
            errors.append(f"ESPN: {e}")

    # 2) odds-api.net (con clave): más casas, las de México marcadas
    key_net, key_the = env.get("ODDS_API_NET_KEY"), env.get("ODDS_API_KEY")
    if todo and key_net:
        used.append("odds-api.net")
        budget = Budget(usage, "odds-api.net", now, PER_RUN)
        budgets.append(budget)
        try:
            get = net_client(key_net, budget, fetch)
            if (usage.get("mx") or {}).get("day") != budget.day:
                try:
                    usage["mx"] = {"day": budget.day, "books": net_mx_books(get)}
                except Exception as e:  # noqa: BLE001 - sin la lista, las casas solo no llevan la marca MX
                    log(f"odds: lista MX no disponible ({e})")
                try:                   # créditos usados y límite del plan (una vez al día)
                    usage["netUsage"] = dict(get("/usage") or {}, day=budget.day)
                except Exception as e:  # noqa: BLE001
                    log(f"odds: uso del plan no disponible ({e})")
            mx = set((usage.get("mx") or {}).get("books") or [])
            if any(not entry(g).get("event") for g in todo):
                t0 = min(_ts(g["time"]) for g in todo) - dt.timedelta(hours=MATCH_HOURS)
                t1 = max(_ts(g["time"]) for g in todo) + dt.timedelta(hours=MATCH_HOURS)
                for pk, (ev, sw) in match(todo, net_events(get, t0, t1), tid).items():
                    g = next(x for x in todo if x["pk"] == pk)
                    entry(g).update({"event": str(ev["event_id"]), "swapped": sw})
            probe = next((entry(g)["event"] for g in todo if entry(g).get("event")), None)
            if probe and (usage.get("netProbe") or {}).get("day") != budget.day and budget.ok():
                try:                   # un partido completo, sin filtros, una vez al día: qué mercados y periodos trae
                    usage["netProbe"] = probe_summary(_pages(get, f"/events/{urllib.parse.quote(probe)}/odds/snapshot",
                                                             price_fields="odds", limit=2000), budget.day, probe)
                except Exception as e:  # noqa: BLE001
                    log(f"odds: muestra completa no disponible ({e})")
            for g in todo:
                e = entry(g)
                if not e.get("event"):
                    continue
                if not budget.ok():
                    log("odds: tope de llamadas de odds-api.net; el resto queda con su último momio")
                    break
                rows = rows_from_net(net_snapshot(get, e["event"]))
                if e.get("swapped"):
                    rows = {k: flip(v) for k, v in rows.items()}
                if rows:
                    merge(e, rows, now, mx, provider="odds-api.net")
                    fresh.add(g["pk"])
        except Exception as e:  # noqa: BLE001
            errors.append(f"odds-api.net: {e}")

    # 3) The Odds API (con clave, plan gratis): solo lo que ESPN no trae (F5, ponches, team total, primera entrada)
    #    y la verificación del mercado de la decisión contra otras casas, por prioridad (ver arriba)
    if key_the:
        used.append("the-odds-api")
        budget = Budget(usage, "the-odds-api", now, 10 ** 6)   # aquí el tope es de créditos, no de llamadas
        budgets.append(budget)
        month = now.strftime("%Y-%m")
        credits = usage.setdefault("theCredits", {})
        try:
            needs = {}
            for d in days:
                needs.update(market_needs(d, pred_dir))
            plan = {}
            for g in games:
                e, chg = entry(g), stale_after.get(g["pk"])
                for m, pr in (needs.get(g["pk"]) or {}).items():
                    if market_due(e, m, now, 180 if pr == 2 else 360, chg):
                        last = _ts((e.get("theAt") or {}).get(m))
                        plan.setdefault(g["pk"], []).append((0 if last and chg and last < _ts(chg) else pr, m))
            want = sorted((g for g in games if plan.get(g["pk"])), key=lambda g: (min(plan[g["pk"]])[0], g["time"]))
            if want and any(not entry(g).get("theId") for g in want):
                evs = fetch(f"{THE}/events?" + urllib.parse.urlencode({"apiKey": key_the, "dateFormat": "iso"})) or []
                for pk, (ev, sw) in match(want, evs, tid).items():   # la lista de partidos no gasta créditos
                    g = next(x for x in want if x["pk"] == pk)
                    entry(g).update({"theId": ev["id"], "theSwapped": sw})
            status["theQueue"] = []
            for g in want:
                e = entry(g)
                if not e.get("theId"):
                    continue
                day_used, month_used = budget.calls.get("the-odds-api", 0), credits.get(month, 0)
                mk, skipped = [], []
                for pr, m in sorted(plan[g["pk"]]):
                    keep = 0 if pr <= 1 else RESERVE          # prioridades 2 y 3 no tocan la reserva del día
                    if day_used + len(mk) + 1 <= THE_DAY_CREDITS - keep and month_used + len(mk) + 1 <= THE_MONTH_CREDITS - keep:
                        mk.append(m)
                    else:
                        skipped.append(m)
                status["theQueue"].append({"pk": g["pk"], "markets": mk, "skipped": skipped,
                                           "priority": min(pr for pr, _ in plan[g["pk"]])})
                if skipped:
                    log(f"odds: tope de créditos de The Odds API; {g['pk']} queda sin {', '.join(skipped)}")
                if not mk:
                    continue
                q = urllib.parse.urlencode({"apiKey": key_the, "regions": "us", "markets": ",".join(mk),
                                            "oddsFormat": "american", "dateFormat": "iso"})
                ev = fetch(f"{THE}/events/{urllib.parse.quote(str(e['theId']))}/odds?{q}") or {}
                got = {m.get("key") for b in ev.get("bookmakers", []) for m in b.get("markets", [])}
                spent = max(1, len(got & set(mk)))
                budget.run += 1
                budget.calls["the-odds-api"] = budget.calls.get("the-odds-api", 0) + spent
                credits[month] = credits.get(month, 0) + spent
                rows = rows_from_the(ev, tid)
                if e.get("theSwapped"):
                    rows = {k: flip(v) for k, v in rows.items()}
                main = {MAIN_KEY[m] for m in mk if m in MAIN_KEY}     # ML/RL/total solo si se pidieron (verificación)
                rows = {k: {kk: vv for kk, vv in v.items() if kk not in ("ml", "rl", "total") or kk in main}
                        for k, v in rows.items()}
                rows = {k: v for k, v in rows.items() if any(v.get(x) for x in ("f5", "k", "tt", "nrfi", *main))}
                for m in mk:
                    e.setdefault("theAt", {})[m] = _iso(now)
                    e.setdefault("theN", {})[m] = e.get("theN", {}).get(m, 0) + 1
                if rows:
                    merge(e, rows, now, set(), provider="the-odds-api")
                    fresh.add(g["pk"])
        except Exception as e:  # noqa: BLE001
            errors.append(f"The Odds API: {e}")

    for g in covered:                  # intentado en esta corrida: se vuelve a pedir según la cadencia
        entry(g)["checked"] = _iso(now)
    for msg in errors:
        log(f"odds: {msg}")
    status.update(provider=" + ".join(PROVIDERS[p] for p in used) or None, calls=sum(b.run for b in budgets),
                  credits=(usage.get("theCredits") or {}).get(now.strftime("%Y-%m")),
                  refreshed=len(fresh), error="; ".join(errors) or None)
    for d in days.values():
        d["games"] = {pk: e for pk, e in d["games"].items() if e.get("books") or e.get("event") or e.get("checked")}
        if d["games"]:
            save_day(d, base)
    if status["calls"]:
        save_usage(usage, base)
    return status


def _no_vig(a, b) -> float | None:
    """Probabilidad sin vig del primer lado de un par de momios americanos."""
    if a is None or b is None:
        return None
    ia, ib = 1 / am_to_dec(a), 1 / am_to_dec(b)
    return ia / (ia + ib)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def verify(books: dict, away: str = "visita", home: str = "local") -> dict:
    """DraftKings (ESPN) contra el consenso de las demás casas, por mercado, con los últimos momios.

    ml: probabilidad sin vig del local · rl: la del local en el mismo hándicap · total: la línea y la probabilidad
    sin vig del Over en esa línea.
    status: ok · difiere (≥ 3 pp o línea distinta) · una fuente (sin otra casa con qué comparar)."""
    out = {}
    ml = {bk: _no_vig((b.get("ml") or {}).get("home"), (b.get("ml") or {}).get("away")) for bk, b in books.items()}
    ml = {k: v for k, v in ml.items() if v is not None}
    if ml:
        others = {k: v for k, v in ml.items() if k != ANCHOR}
        r = {"n": len(ml)}
        if ANCHOR in ml and others:
            cons = _median(others.values())
            diff = ml[ANCHOR] - cons
            r.update(diff=round(diff * 100, 1), status="difiere" if abs(diff) >= VERIFY_PP else "ok",
                     text=f"Moneyline: DraftKings da {home} {ml[ANCHOR] * 100:.1f}% y la mediana de otras {len(others)} "
                          f"casa{'s' if len(others) > 1 else ''} {cons * 100:.1f}% ({abs(diff) * 100:.1f} pp de diferencia)")
        elif len(ml) > 1:
            spread = max(ml.values()) - min(ml.values())
            r.update(diff=round(spread * 100, 1), status="difiere" if spread >= VERIFY_PP else "ok",
                     text=f"Moneyline: {len(ml)} casas entre {min(ml.values()) * 100:.1f}% y {max(ml.values()) * 100:.1f}% para {home}")
        else:
            r.update(status="una fuente", text="Moneyline: una sola casa, sin otra con qué comparar")
        out["ml"] = r
    # run line: la probabilidad sin vig de cubrir del local en el mismo hándicap que DraftKings (o el más común)
    pts = {bk: ((b.get("rl") or {}).get("home") or {}).get("point") for bk, b in books.items()}
    pts = {k: v for k, v in pts.items() if v is not None and ((books[k].get("rl") or {}).get("away") or {}).get("price") is not None}
    if pts:
        vals = list(pts.values())
        pt = pts.get(ANCHOR, max(set(vals), key=vals.count))
        rl = {bk: _no_vig(books[bk]["rl"]["home"]["price"], books[bk]["rl"]["away"]["price"]) for bk, x in pts.items() if x == pt}
        others = {k: v for k, v in rl.items() if k != ANCHOR}
        r = {"n": len(rl)}
        side = f"{home} {pt:+g}"
        if ANCHOR in rl and others:
            cons = _median(others.values())
            diff = rl[ANCHOR] - cons
            r.update(diff=round(diff * 100, 1), status="difiere" if abs(diff) >= VERIFY_PP else "ok",
                     text=f"Run line {side}: DraftKings {rl[ANCHOR] * 100:.1f}% y la mediana de otras {len(others)} "
                          f"casa{'s' if len(others) > 1 else ''} {cons * 100:.1f}% ({abs(diff) * 100:.1f} pp de diferencia)")
        elif len(rl) > 1:
            spread = max(rl.values()) - min(rl.values())
            r.update(diff=round(spread * 100, 1), status="difiere" if spread >= VERIFY_PP else "ok",
                     text=f"Run line {side}: {len(rl)} casas entre {min(rl.values()) * 100:.1f}% y {max(rl.values()) * 100:.1f}%")
        else:
            r.update(status="una fuente", text="Run line: una sola casa, sin otra con qué comparar")
        out["rl"] = r
    lines = {bk: (b.get("total") or {}).get("over", {}).get("point") for bk, b in books.items()}
    lines = {k: v for k, v in lines.items() if v is not None}
    if lines:
        others = {k: v for k, v in lines.items() if k != ANCHOR}
        r = {"n": len(lines)}
        if ANCHOR in lines and others:
            mode = max(set(others.values()), key=lambda x: (list(others.values()).count(x), -abs(x - lines[ANCHOR])))
            if lines[ANCHOR] != mode:
                r.update(status="difiere", text=f"Total: DraftKings pone {lines[ANCHOR]} y {list(others.values()).count(mode)} "
                                                f"de {len(others)} casas {mode}")
            else:
                ov = {bk: _no_vig(books[bk]["total"]["over"]["price"], (books[bk]["total"].get("under") or {}).get("price"))
                      for bk, ln in lines.items() if ln == mode}
                ov = {k: v for k, v in ov.items() if v is not None}
                oth = [v for k, v in ov.items() if k != ANCHOR]
                if ANCHOR in ov and oth:
                    diff = ov[ANCHOR] - _median(oth)
                    r.update(diff=round(diff * 100, 1), status="difiere" if abs(diff) >= VERIFY_PP else "ok",
                             text=f"Total {mode}: misma línea en {len(oth) + 1} casas; el Over de DraftKings "
                                  f"{abs(diff) * 100:.1f} pp {'arriba' if diff > 0 else 'abajo'} de la mediana")
                else:
                    r.update(status="ok", text=f"Total {mode}: misma línea en {len(ov)} casas")
        elif len(lines) > 1:
            r.update(status="ok" if len(set(lines.values())) == 1 else "difiere",
                     text=f"Total: líneas {', '.join(str(x) for x in sorted(set(lines.values())))} en {len(lines)} casas")
        else:
            r.update(status="una fuente", text="Total: una sola casa, sin otra con qué comparar")
        out["total"] = r
    return out


def is_stale(book: dict, since: str | None) -> bool:
    """Momio visto antes del último cambio de abridor: ya no cuenta para el stake."""
    at, chg = _ts((book.get("last") or {}).get("at")), _ts(since)
    return bool(at and chg and at < chg)


def to_bundle(bundle: dict, base: str = ODDS_DIR, now: dt.datetime | None = None,
              stale_after: dict | None = None) -> list[dict] | None:
    """Momios guardados de los partidos del bundle, con la forma de The Odds API (lo que lee el modelo).

    stale_after: {pk: hora del último cambio de abridor}; las casas cuyo último momio es anterior salen con
    `stale` (el modelo no las usa para el stake y la página las muestra aparte)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    stale_after = {int(k): v for k, v in (stale_after or {}).items()}
    teams = {int(t["id"]): t for t in (bundle.get("teams") or {}).values()}
    out = []
    for date in sorted({g["date"] for g in games_of(bundle)}):
        day = load_day(date, base)
        for g in games_of(bundle):
            e = day["games"].get(str(g["pk"])) if g["date"] == date else None
            if not e or not e.get("books"):
                continue
            name = lambda i: (teams.get(int(i)) or {}).get("name") or str(i)  # noqa: E731
            ab = lambda i: (teams.get(int(i)) or {}).get("abbr") or str(i)  # noqa: E731
            since = stale_after.get(g["pk"])
            books, fresh_rows = [], {}
            for bk, b in sorted(e["books"].items(), key=lambda kv: (not kv[1].get("mx"), kv[1].get("title") or kv[0])):
                stale = is_stale(b, since)
                row = {"key": bk, "title": short_title(b.get("title") or bk), "mx": bool(b.get("mx")),
                       "last_update": b["last"].get("at"), "markets": _markets(b["last"], name(g["away"]), name(g["home"])),
                       "open": {k: b["open"].get(k) or {} for k in MARKET_KEYS}, "openAt": b["open"].get("at")}
                if stale:
                    row.update(stale=True, staleSince=since)
                else:
                    fresh_rows[bk] = b["last"]
                books.append(row)
            start = _ts(e.get("start"))
            provs = e.get("providers") or ([e["provider"]] if e.get("provider") else [])
            out.append({"id": e.get("event") or e.get("espn") or str(g["pk"]), "pk": g["pk"],
                        "provider": " + ".join(PROVIDERS.get(p, p) for p in provs) or None,
                        "commence_time": e.get("start"), "home_team": name(g["home"]), "away_team": name(g["away"]),
                        "checked": e.get("checked"), "closed": bool(start and start <= now), "bookmakers": books,
                        "verify": verify(fresh_rows, ab(g["away"]), ab(g["home"])),
                        **({"staleSince": since} if since else {})})
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
    f5 = row.get("f5") or {}
    if f5.get("ml"):
        ms.append({"key": "h2h_1st_5_innings", "outcomes": [{"name": away if s == "away" else home, "price": p}
                                                            for s, p in f5["ml"].items()]})
    if f5.get("total"):
        ms.append({"key": "totals_1st_5_innings", "outcomes": [{"name": k.capitalize(), "price": v["price"], "point": v["point"]}
                                                               for k, v in f5["total"].items()]})
    if row.get("k"):
        ms.append({"key": "pitcher_strikeouts", "outcomes": [{"name": k.capitalize(), "description": who, "price": v["price"],
                                                              "point": v["point"]} for who, ou in row["k"].items()
                                                             for k, v in ou.items()]})
    if row.get("tt"):
        ms.append({"key": "team_totals", "outcomes": [{"name": k.capitalize(), "description": away if sd == "away" else home,
                                                       "price": v["price"], "point": v["point"]}
                                                      for sd, ou in row["tt"].items() for k, v in ou.items()]})
    if row.get("nrfi"):
        ms.append({"key": "totals_1st_1_innings", "outcomes": [{"name": "Over", "price": row["nrfi"]["yrfi"], "point": 0.5},
                                                               {"name": "Under", "price": row["nrfi"]["nrfi"], "point": 0.5}]})
    return ms
