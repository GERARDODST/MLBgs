"""Expediente de un partido: los DATOS con los que se hicieron sus tickets, separados del ALGORITMO.

data/expedientes/<fecha>/<pk>.json.gz se escribe UNA sola vez, cuando el ticket se bloquea (primer
lanzamiento), y nunca se reescribe. Guarda:
  tickets   huella de cada ticket del partido (framework y Pro-Lab) tal como se bloqueó
  algo      versión del algoritmo y huella del código que lo produjo (ver mlbgs/version.py)
  analysis  el análisis completo del framework tal como se vio (10 secciones, picks, IC, escalera, decisión)
  odds      los momios vistos del partido (data/odds: apertura y último momio por casa)
  prolab    archivos del Pro-Lab del partido con su huella (sus datos congelados ya viven en prolab/)
  raw       datos crudos de la API recortados al partido: calendario, lineups, abridores, umpire y clima,
            los dos equipos (standings, stats, rosters, movimientos), sus jugadores (MLB y Savant), el uso
            del bullpen (box scores) y los resultados de ambos equipos. Los valores de toda la liga que usa
            el modelo (Elo, constantes, Pitágoras) quedan ya calculados en `analysis`.

Mientras el partido no empieza, la versión más reciente espera en .cache/expedientes/<pk>.json.gz (en
GitHub Actions esa carpeta pasa de una corrida a la siguiente con la caché); al bloquearse el ticket, esa
última versión previa al juego pasa a data/expedientes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "expedientes")
PENDING = os.path.join(ROOT, ".cache", "expedientes")
PROLAB_DIR = os.path.join(ROOT, "prolab")


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _game(bundle: dict, pk: int) -> dict | None:
    for k in ("upcoming", "scoreboard", "live"):
        for g in bundle.get(k) or []:
            if g.get("pk") == pk:
                return g
    return None


def raw_slice(bundle: dict, pk: int) -> dict:
    """Los datos crudos del bundle que tocan a este partido (sus dos equipos y sus jugadores)."""
    g = _game(bundle, pk) or {}
    teams = {g.get("away"), g.get("home")} - {None}
    tkeys = {str(t) for t in teams}
    ros = bundle.get("rosters") or {}
    people: set[str] = set()
    for t in tkeys:
        r = ros.get(t) or {}
        for grp in ("players", "injured"):
            for p in r.get(grp) or []:
                if isinstance(p, dict) and p.get("id") is not None:
                    people.add(str(p["id"]))
    for side in ("away", "home"):
        pr = ((g.get("probable") or {}).get(side) or {})
        if isinstance(pr, dict) and pr.get("id") is not None:
            people.add(str(pr["id"]))
        for row in ((g.get("lineups") or {}).get(side) or []):
            if isinstance(row, dict) and row.get("id") is not None:
                people.add(str(row["id"]))
            elif isinstance(row, (int, str)):
                people.add(str(row))
    for key in ("playersPitching", "playersHitting"):
        for p in bundle.get(key) or []:
            if p.get("team") in teams:
                people.add(str(p["id"]))
    mine = lambda x: x.get("away") in teams or x.get("home") in teams  # noqa: E731
    results = [r for r in bundle.get("results") or [] if mine(r)]
    prev = [r for r in bundle.get("prevResults") or [] if mine(r)]
    pks = {r["pk"] for r in results} | {pk}
    boxes = [b for b in bundle.get("boxscores") or [] if b.get("pk") in pks]
    sav = bundle.get("savant") or {}
    ars = bundle.get("arsenal") or {}
    pick = lambda d, keys: {k: v for k, v in (d or {}).items() if k in keys}  # noqa: E731
    odds = bundle.get("odds")
    odds_game = [e for e in (odds if isinstance(odds, list) else []) if e.get("pk") == pk]
    return {
        "meta": bundle.get("meta"),
        "game": g,
        "teams": pick(bundle.get("teams"), tkeys),
        "standings": bundle.get("standings"),
        "remaining": bundle.get("remaining"),
        "teamStats": pick(bundle.get("teamStats"), tkeys),
        "rosters": pick(ros, tkeys),
        "transactions": [x for x in bundle.get("transactions") or [] if x.get("team") in teams],
        "playersPitching": [p for p in bundle.get("playersPitching") or [] if str(p.get("id")) in people],
        "playersHitting": [p for p in bundle.get("playersHitting") or [] if str(p.get("id")) in people],
        "pitchers": pick(bundle.get("pitchers"), people),
        "hands": pick(bundle.get("hands"), people),
        "bats": pick(bundle.get("bats"), people),
        "savant": {"expected": pick(sav.get("expected"), people), "pitcher": pick(sav.get("pitcher"), people),
                   "batter": pick(sav.get("batter"), people),
                   "park": pick(sav.get("park"), {str(g.get("venue")), str(g.get("venueName"))})},
        "arsenal": {"pitcher": pick(ars.get("pitcher"), people), "batter": pick(ars.get("batter"), people)},
        "results": results,
        "prevResults": prev,
        "boxscores": boxes,
        "odds": odds_game,
    }


def prolab_files(pk: int) -> dict:
    """Archivos del Pro-Lab de un partido (modelo, bundle y snapshot congelados) con su huella."""
    import glob
    out = {}
    for path in sorted(glob.glob(os.path.join(PROLAB_DIR, f"*{pk}*"))):
        if os.path.basename(path).startswith("result_"):
            continue      # el resultado llega después del juego
        out[f"prolab/{os.path.basename(path)}"] = file_sha(path)
    return out


def build(a: dict | None, bundle: dict, pk: int, date: str, tickets: dict, odds_entry: dict | None,
          algo: dict, generated: str, ticket_hash) -> dict:
    """Expediente (pendiente) de un partido que todavía no empieza."""
    prolab = prolab_files(pk)
    return {"pk": pk, "date": date, "writtenAt": generated, "algo": algo,
            "tickets": {tid: ticket_hash(t) for tid, t in sorted(tickets.items())},
            "analysis": a, "odds": odds_entry, "prolab": prolab or None, "raw": raw_slice(bundle, pk)}


def _write(path: str, exp: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, path)


def write_pending(exp: dict, pending: str = PENDING) -> str:
    path = os.path.join(pending, f"{exp['pk']}.json.gz")
    _write(path, exp)
    return path


def final_path(date: str, pk: int, base: str = DIR) -> str:
    return os.path.join(base, date, f"{pk}.json.gz")


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def promote(pk: int, date: str, hashes: dict, pending: str = PENDING, base: str = DIR, at: str | None = None) -> dict | None:
    """Al bloquearse un ticket: la última versión previa al juego pasa a data/expedientes (una sola vez).

    Devuelve {"path", "coincide"}: coincide = la huella de cada ticket bloqueado es la del expediente.
    Si ya existe, no se reescribe (se devuelve el que hay). Si no hay versión previa guardada, None."""
    dst = final_path(date, pk, base)
    if os.path.exists(dst):
        with gzip.open(dst, "rt", encoding="utf-8") as f:
            old = json.load(f)
        return {"path": rel(dst), "coincide": all(old.get("tickets", {}).get(k) == v for k, v in hashes.items())}
    src = os.path.join(pending, f"{pk}.json.gz")
    if not os.path.exists(src):
        return None
    with gzip.open(src, "rt", encoding="utf-8") as f:
        exp = json.load(f)
    if exp.get("date") != date:
        return None
    ok = all(exp.get("tickets", {}).get(k) == v for k, v in hashes.items())
    exp["lockedAt"], exp["coincide"] = at, ok
    _write(dst, exp)
    return {"path": rel(dst), "coincide": ok}


def load(path: str) -> dict:
    with gzip.open(os.path.join(ROOT, path) if not os.path.isabs(path) else path, "rt", encoding="utf-8") as f:
        return json.load(f)
