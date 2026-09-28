"""Momio real del mercado para partidos que ya se jugaron, guardado APARTE de los tickets.

Los tickets bloqueados nunca se modifican. Este módulo solo guarda lo que pagó el mercado y evalúa, con
las reglas que el propio ticket congeló (su escalera de stake), qué habría dicho el protocolo con ese
momio real. Sirve para un balance paralelo «con cuota real», separado del balance original.

Fuentes:
  hasta el 26-sep: resumen público de ESPN (pickcenter de su casa socia, DraftKings, con apertura y
                   cierre), bajado una vez con `python -m mlbgs.mercado descargar` desde GitHub Actions
                   (el contenedor de desarrollo no llega a ESPN) y guardado en data/mercado/<fecha>.json.
  desde el 27-sep: data/odds/<fecha>.json (apertura y último momio por casa, congelado al primer
                   lanzamiento), que ya guarda la actualización automática.

Formato de un partido (momios americanos):
  {"fuente": "ESPN · DraftKings", "ml": {"away": {"open": 179, "close": 194}, "home": {...}},
   "rl": {"away": {"point": 1.5, "open": -123, "close": -111}, "home": {...}},
   "total": {"open": 7.5, "close": 8.0, "over": {"open": -110, "close": -118}, "under": {...}}}
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import os

from . import odds as O

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "mercado")
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


# ------------------------------------------------------------------ descarga (solo desde GitHub Actions)

def _dates(a: str, b: str) -> list[str]:
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def descargar(desde: str, hasta: str, out: str, fetch=O.http_json) -> None:
    """Por fecha: los eventos de ESPN con su pickcenter (apertura y cierre por casa), en crudo."""
    os.makedirs(out, exist_ok=True)
    for day in _dates(desde, hasta):
        sb = fetch(f"{O.ESPN}?dates={day.replace('-', '')}")
        evs = []
        for ev in sb.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            teams = {c.get("homeAway"): c.get("team") or {} for c in comp.get("competitors", [])}
            try:
                s = fetch(f"{SUMMARY}?event={ev['id']}")
                pc = [{k: v for k, v in p.items() if k not in ("links", "header", "footer", "link")} for p in s.get("pickcenter") or []]
            except RuntimeError as e:
                pc, s = [], {"error": str(e)}
            evs.append({"id": str(ev.get("id")), "date": comp.get("date") or ev.get("date"),
                        "away_team": (teams.get("away") or {}).get("displayName", ""), "home_team": (teams.get("home") or {}).get("displayName", ""),
                        "pickcenter": pc, "error": s.get("error")})
        with gzip.open(os.path.join(out, f"{day}.json.gz"), "wt", encoding="utf-8") as f:
            json.dump({"date": day, "fetchedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "events": evs}, f)
        print(day, len(evs), "eventos")


def primer_lanzamiento(pks: list[int], out: str, fetch=O.http_json) -> None:
    """Hora programada y hora oficial del primer lanzamiento de cada partido (MLB Stats API)."""
    os.makedirs(out, exist_ok=True)
    res = {}
    for pk in pks:
        g = fetch(FEED.format(pk=pk) + "?fields=gameData,datetime,dateTime,originalDate,gameInfo,firstPitch,gameDurationMinutes,delayDurationMinutes,status,detailedState")
        gd = g.get("gameData") or {}
        res[str(pk)] = {"scheduled": (gd.get("datetime") or {}).get("dateTime"), "firstPitch": (gd.get("gameInfo") or {}).get("firstPitch"),
                        "delayMinutes": (gd.get("gameInfo") or {}).get("delayDurationMinutes"), "status": (gd.get("status") or {}).get("detailedState")}
    with open(os.path.join(out, "primer_lanzamiento.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False))


# ------------------------------------------------------------------ lectura

def _rec_from_rows(opn: dict, cur: dict, fuente: str) -> dict:
    """Renglones de apertura y cierre (formato de odds._espn_row) → registro del partido."""
    rec: dict = {"fuente": fuente}
    ml = {s: {k: v for k, v in (("open", opn.get("ml", {}).get(s)), ("close", cur.get("ml", {}).get(s))) if v is not None} for s in ("away", "home")}
    if any(ml.values()):
        rec["ml"] = ml
    rl = {}
    for s in ("away", "home"):
        c, o = cur.get("rl", {}).get(s), opn.get("rl", {}).get(s)
        if c or o:
            rl[s] = {"point": (c or o)["point"], **({"open": o["price"]} if o and (not c or o["point"] == c["point"]) else {}), **({"close": c["price"]} if c else {})}
    if rl:
        rec["rl"] = rl
    to, tc = opn.get("total") or {}, cur.get("total") or {}
    if to or tc:
        rec["total"] = {"open": (to.get("over") or {}).get("point"), "close": (tc.get("over") or {}).get("point"),
                        **{k: {kk: vv for kk, vv in (("open", (to.get(k) or {}).get("price")), ("close", (tc.get(k) or {}).get("price"))) if vv is not None}
                           for k in ("over", "under")}}
    return rec


def leer_crudo(raw: dict, games: list[dict], tid) -> dict:
    """Eventos crudos de ESPN de un día → {pk: registro} (DraftKings o la primera casa del pickcenter)."""
    events = []
    for ev in raw.get("events", []):
        pcs = ev.get("pickcenter") or []
        pc = next((p for p in pcs if "draftkings" in O._norm((p.get("provider") or {}).get("name", "")).replace(" ", "")), pcs[0] if pcs else None)
        if not pc:
            continue
        events.append({"away_team": ev["away_team"], "home_team": ev["home_team"], "commence_time": ev.get("date"),
                       "rec": _rec_from_rows(O._espn_row(pc, "open"), O._espn_row(pc, "close"), f"ESPN · {(pc.get('provider') or {}).get('name', 'ESPN')}"),
                       "espn": ev["id"]})
    out = {}
    for pk, (ev, swapped) in O.match(games, events, tid).items():
        rec = ev["rec"]
        if swapped:
            rec = {k: ({"away": v.get("home"), "home": v.get("away")} if k in ("ml", "rl") else v) for k, v in rec.items()}
        out[str(pk)] = {**rec, "espn": ev["espn"]}
    return out


def de_odds(entry: dict) -> dict | None:
    """Entrada de data/odds (por casa: open y last) → registro con la casa de ESPN (DraftKings) o la primera."""
    books = (entry or {}).get("books") or {}
    key = next((k for k in books if "draftkings" in k), next(iter(books), None))
    if not key:
        return None
    b = books[key]
    return _rec_from_rows(b.get("open") or {}, b.get("last") or {}, f"{(entry.get('providers') or ['feed'])[0] if isinstance(entry.get('providers'), list) else 'feed'} · {b.get('title') or key}")


def cargar(date: str, base: str = DIR, odds_dir: str = O.ODDS_DIR) -> dict:
    """{pk: registro} de un día: data/mercado si existe; si no, el cierre guardado en data/odds."""
    p = os.path.join(base, f"{date}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    day = O.load_day(date, odds_dir)
    return {pk: r for pk, e in (day.get("games") or {}).items() if (r := de_odds(e))}


def cargar_todo(dates, base: str = DIR, odds_dir: str = O.ODDS_DIR) -> dict:
    return {d: cargar(d, base, odds_dir) for d in sorted(set(dates))}


# ------------------------------------------------------------------ evaluación con las reglas congeladas del ticket

def _dec(am: float) -> float:
    return 1 + 100 / abs(am) if am < 0 else 1 + am / 100


def precio(rec: dict, pick: dict, t: dict, cuando: str = "close") -> int | None:
    """Momio real (americano) del pick en el registro del mercado, o None si ese mercado no está."""
    fam, name = pick.get("family"), pick.get("pick") or ""
    side = "away" if name.split(" ")[0] == t.get("away") else "home" if name.split(" ")[0] == t.get("home") else None
    if fam == "ML" and side:
        return ((rec.get("ml") or {}).get(side) or {}).get(cuando)
    if fam == "RL" and side:
        r = (rec.get("rl") or {}).get(side) or {}
        try:
            pt = float(name.split(" ")[1])
        except (IndexError, ValueError):
            return None
        return r.get(cuando) if r.get("point") == pt else None
    if fam == "Total":
        tot = rec.get("total") or {}
        k = "over" if name.startswith("Over") else "under" if name.startswith("Under") else None
        try:
            line = float(name.split(" ")[1])
        except (IndexError, ValueError):
            return None
        return (tot.get(k) or {}).get(cuando) if k and tot.get(cuando) == line else None
    return None


def nivel_con(pick: dict, am: int) -> int:
    """Nivel 1–10 que la escalera CONGELADA del pick daba a ese momio (0 = no apostar)."""
    st = pick.get("stake") or {}
    steps = st.get("steps")
    if not steps or st.get("block"):
        return 0
    if st.get("maxPrice") is not None and _dec(am) > _dec(st["maxPrice"]) + 1e-9:
        return 0          # filtro contra el mercado: el casino lo ve 10 pp o más distinto
    lvl = 0
    for i, frm in enumerate(steps):
        if frm is not None and _dec(am) >= _dec(frm) - 1e-9:
            lvl = i + 1
    return lvl


def evaluar(t: dict, rec: dict | None, amounts: dict) -> dict | None:
    """Qué habría dicho el protocolo del ticket con el momio real de cierre (sin tocar el ticket)."""
    if not rec:
        return None
    d = t.get("decision") or {}
    out = {"fuente": rec.get("fuente"), "decision": None, "entraria": None}
    by = {p["pick"]: p for p in t.get("picks") or []}
    p = by.get(d.get("pick")) if d.get("pick") else None
    if p:
        am, am0 = precio(rec, p, t, "close"), precio(rec, p, t, "open")
        if am is not None:
            lvl = nivel_con(p, am)
            res = p.get("res")
            neto = None
            if lvl and res in ("ganado", "perdido", "push"):
                stake = amounts[lvl]
                neto = round(stake * (_dec(am) - 1), 2) if res == "ganado" else -stake if res == "perdido" else 0.0
            out["decision"] = {"pick": p["pick"], "open": am0, "close": am, "level": lvl, "amount": amounts.get(lvl, 0),
                               "original": d.get("level") if d.get("status") == "apostar" else 0, "res": res, "neto": neto,
                               "escalera": bool((p.get("stake") or {}).get("steps"))}
    # ¿algún otro pick con momio real habría llegado a stake? (solo informa; no cambia la decisión registrada)
    best = None
    for q in t.get("picks") or []:
        am = precio(rec, q, t, "close")
        if am is None or q is p:
            continue
        lvl = nivel_con(q, am)
        if lvl and (best is None or q.get("ic", 0) > best["ic"]):
            best = {"pick": q["pick"], "close": am, "level": lvl, "ic": q.get("ic", 0), "res": q.get("res")}
    out["entraria"] = best
    return out if out["decision"] or out["entraria"] else None


# ------------------------------------------------------------------ línea de comandos

def construir(raw_dir: str, bundle_path: str, out: str = DIR) -> None:
    """Crudo de ESPN (dev/pm/mercado) + partidos de los tickets → data/mercado/<fecha>.json (una vez por día)."""
    from . import historial as H
    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        teams = json.load(f)["teams"]
    tid = O.team_index(teams)
    abbr = {t["abbr"]: int(i) for i, t in teams.items()}
    tickets = H.load()
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        day = raw["date"]
        games = {t["pk"]: {"pk": t["pk"], "away": abbr.get(t["away"]), "home": abbr.get(t["home"]), "time": t.get("time")}
                 for t in tickets.values() if t.get("date") == day and t.get("time")}
        recs = leer_crudo(raw, [g for g in games.values() if g["away"] and g["home"]], tid)
        dst = os.path.join(out, f"{day}.json")
        if os.path.exists(dst):
            print(day, "ya existe: no se reescribe")
            continue
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")
        print(day, f"{len(recs)} de {len(games)} partidos con momio real")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("descargar")
    a.add_argument("--desde", required=True)
    a.add_argument("--hasta", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("primer")
    b.add_argument("--pks", nargs="+", type=int, required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("construir")
    c.add_argument("--raw", required=True)
    c.add_argument("--bundle", required=True)
    args = ap.parse_args()
    if args.cmd == "descargar":
        descargar(args.desde, args.hasta, args.out)
    elif args.cmd == "primer":
        primer_lanzamiento(args.pks, args.out)
    else:
        construir(args.raw, args.bundle)


if __name__ == "__main__":
    main()
