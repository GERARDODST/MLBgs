"""Cada cuánto se vuelve a revisar todo (MLB y momios), según lo cerca que esté el siguiente primer lanzamiento.

La MLB publica los lineups 1–4 h antes y los cambios de último minuto (abridor, bajas) llegan en la última hora y
media; lejos del juego casi nada cambia. Por eso el relevo de update.yml espera menos cuando hay un partido cerca:

  primer lanzamiento en ≤ 2.5 h (o partido en calentamiento)   espera  7 min  → una revisión cada ~12 min
  primer lanzamiento en ≤ 6 h                                  espera 13 min  → cada ~20 min
  más lejos                                                    espera 25 min  → cada ~30 min

La corrida en sí tarda ~5 min. Revisar la MLB no cuesta; los momios de ESPN tampoco (una llamada por fecha) y
los de The Odds API solo se piden por prioridad y con topes de créditos (ver mlbgs/odds.py), así que una
cadencia más corta no gasta créditos de más.

Uso: python -m mlbgs.cadencia  →  imprime los segundos que debe esperar el relevo (lee data/odds y data/cambios).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ((150, 420, "partido en ≤ 2.5 h: lineups y cambios de último minuto"),
         (360, 780, "partido en ≤ 6 h"),
         (None, 1500, "sin partidos cerca"))
RUN_MIN = 5          # lo que tarda una corrida, para decir cada cuánto se revisa


def _ts(x):
    try:
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")) if x else None
    except ValueError:
        return None


def starts(root: str = ROOT, now: dt.datetime | None = None) -> list[dt.datetime]:
    """Horas de primer lanzamiento de hoy y mañana que se conocen (data/odds y data/cambios)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = set()
    for day in (now - dt.timedelta(days=1), now, now + dt.timedelta(days=1)):
        d = day.strftime("%Y-%m-%d")
        for path in glob.glob(os.path.join(root, "data", "odds", f"{d}.json")):
            with open(path, encoding="utf-8") as f:
                for e in (json.load(f).get("games") or {}).values():
                    if _ts(e.get("start")):
                        out.add(_ts(e["start"]))
        for path in glob.glob(os.path.join(root, "data", "cambios", f"{d}.json")):
            with open(path, encoding="utf-8") as f:
                for s in (json.load(f).get("snap") or {}).values():
                    if _ts(s.get("time")):
                        out.add(_ts(s["time"]))
    return sorted(out)


def plan(times: list[dt.datetime], now: dt.datetime | None = None) -> dict:
    """{"sleep": segundos, "every": minutos entre revisiones, "why", "next": siguiente primer lanzamiento}."""
    now = now or dt.datetime.now(dt.timezone.utc)
    # un partido «empezado» hace menos de 30 min puede seguir en calentamiento o retrasado
    ahead = [t for t in times if t > now - dt.timedelta(minutes=30)]
    nxt = min(ahead) if ahead else None
    mins = (nxt - now).total_seconds() / 60 if nxt else None
    for limit, sleep, why in STEPS:
        if limit is None or (mins is not None and mins <= limit):
            return {"sleep": sleep, "every": round(sleep / 60 + RUN_MIN), "why": why,
                    "next": nxt.strftime("%Y-%m-%dT%H:%M:%SZ") if nxt else None,
                    "steps": [{"within": lim, "every": round(sl / 60 + RUN_MIN)} for lim, sl, _ in STEPS]}
    raise AssertionError("inalcanzable")


def main() -> None:
    p = plan(starts())
    print(p["sleep"])


if __name__ == "__main__":
    main()
