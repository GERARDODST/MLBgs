"""Cada cuánto se vuelve a revisar todo (MLB y momios), según lo cerca que esté el siguiente primer lanzamiento.

La MLB publica los lineups 1–4 h antes y los cambios de último minuto (abridor, bajas) llegan en la última hora y
media; lejos del juego casi nada cambia. El relevo de update.yml corre TODO el día mientras haya un partido en las
próximas 36 h (antes se apagaba de 06:00 a 15:00 UTC y dependía del cron de GitHub, que se retrasa horas):

  primer lanzamiento en ≤ 2.5 h (o partido en calentamiento)   espera  7 min  → una revisión cada ~9 min
  primer lanzamiento en ≤ 6 h, o un partido en juego            espera 13 min  → cada ~15 min
  primer lanzamiento en ≤ 12 h                                  espera 25 min  → cada ~27 min
  primer lanzamiento en ≤ 36 h                                  espera 55 min  → cada hora
  sin partidos en 36 h                                          se detiene (el cron de respaldo corre cada hora)

Revisar la MLB no cuesta; los momios de ESPN tampoco (una llamada por fecha) y los de The Odds API solo se piden por
prioridad y con topes de créditos (ver mlbgs/odds.py), así que una cadencia más corta no gasta créditos de más.

Uso: python -m mlbgs.cadencia  →  imprime los segundos que debe esperar el relevo (0: no lanzar otra corrida).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = ((150, 420, "partido en ≤ 2.5 h: lineups y cambios de último minuto"),
         (360, 780, "partido en ≤ 6 h"),
         (720, 1500, "partido en ≤ 12 h"),
         (2160, 3300, "partido en ≤ 36 h: una revisión por hora"),
         (None, 0, "sin partidos en 36 h: el relevo se detiene (lo reanuda el cron de cada hora)"))
LIVE_MIN = 240       # un partido que empezó hace menos de 4 h puede seguir en juego: se revisa como uno cercano
RUN_MIN = 2          # lo que tarda una corrida, para decir cada cuánto se revisa
# el relevo corre todo el día mientras haya un partido en 36 h y el cron de respaldo es cada hora: no hay horas sin corrida
RUN_HOURS = frozenset(range(24))


def first_run_after(t: dt.datetime) -> dt.datetime:
    """La primera corrida de producción desde `t` (hoy hay corrida a toda hora: el relevo o el cron de cada hora)."""
    if t.hour in RUN_HOURS:
        return t
    noon = t.replace(hour=12, minute=0, second=0, microsecond=0)
    return noon if t <= noon else t.replace(hour=15, minute=0, second=0, microsecond=0)


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
    live = [t for t in times if now - dt.timedelta(minutes=LIVE_MIN) < t <= now - dt.timedelta(minutes=30)]
    if live and (mins is None or mins > 360):       # en juego: marcador, finales y calificación de tickets al día
        return {"sleep": 780, "every": round(780 / 60 + RUN_MIN), "why": "partido en juego",
                "next": nxt.strftime("%Y-%m-%dT%H:%M:%SZ") if nxt else None,
                "steps": [{"within": lim, "every": round(sl / 60 + RUN_MIN) if sl else None} for lim, sl, _ in STEPS]}
    for limit, sleep, why in STEPS:
        if limit is None or (mins is not None and mins <= limit):
            return {"sleep": sleep, "every": round(sleep / 60 + RUN_MIN) if sleep else None, "why": why,
                    "next": nxt.strftime("%Y-%m-%dT%H:%M:%SZ") if nxt else None,
                    "steps": [{"within": lim, "every": round(sl / 60 + RUN_MIN) if sl else None} for lim, sl, _ in STEPS]}
    raise AssertionError("inalcanzable")


def main() -> None:
    p = plan(starts())
    print(p["sleep"])


if __name__ == "__main__":
    main()
