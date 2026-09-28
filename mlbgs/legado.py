"""Migración única (28-sep): bloquear los tickets de antes de la regla de inmutabilidad, sin maquillarlos.

Qué hace, con la historia de git como fuente de verdad (hay que correrla en un clon con historia):
  * Tickets que recibieron algo DESPUÉS del juego (los 3 del Pro-Lab del 23-sep: decisión y escalera de
    stake agregadas cuando ya se había jugado): el ticket principal vuelve a su primera versión registrada
    (la original) y la versión posterior se guarda aparte en data/correcciones/<id>.json con sus notas.
  * Tickets reconstruidos después del juego desde un registro previo (la predicción de data/predictions):
    se quedan como están y llevan una nota (data/correcciones) que lo dice.
  * Tickets actualizados después de la hora programada pero antes del primer lanzamiento real (retrasos
    por lluvia): válidos; llevan una nota con la hora real del primer lanzamiento.
  * Todos los tickets ya jugados se bloquean (lock con huella). Los partidos con Pro-Lab reciben un
    expediente armado con sus datos congelados antes del juego (prolab/).

    python -m mlbgs.legado --primer dev/pm/primer_lanzamiento.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import subprocess

from . import expediente as EX
from . import historial as HI
from . import stake as ST

ROOT = HI.ROOT
CORR_DIR = os.path.join(ROOT, "data", "correcciones")
BASE_COMMIT = "9f89fcf"          # primera versión de la base de tickets (24-sep 17:06 UTC)
PROLAB_BUNDLE = {823168: "bundle_823168_manana.json.gz"}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _at(commit: str, path: str) -> list:
    try:
        return json.loads(_git("show", f"{commit}:{path}"))
    except subprocess.CalledProcessError:
        return []


def _commit_time(commit: str) -> str:
    return _git("log", "-1", "--format=%aI", commit).strip()


def _before(path: str, when: str) -> str | None:
    c = _git("rev-list", "-1", f"--before={when}", "HEAD", "--", path).strip()
    return c or None


def _first_version(tid: str, path: str) -> tuple[str, str, dict] | None:
    for line in _git("log", "--format=%h %aI", "--reverse", "HEAD", "--", path).splitlines():
        h, when = line.split()
        v = {t["id"]: t for t in _at(h, path)}
        if tid in v:
            return h, when, v[tid]
    return None


def _norm(t: dict) -> dict:
    c = {k: v for k, v in t.items() if k not in HI.AFTER_GAME}
    c["picks"] = [{k: v for k, v in p.items() if k != "res"} for p in t.get("picks") or []]
    return c


def _hm(iso: str | None) -> str:
    if not iso:
        return "—"
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return t.strftime("%d-%b %H:%M UTC").lower()


def _dec_txt(d: dict | None) -> str:
    if not d:
        return "sin decisión registrada"
    if d.get("status") == "apostar" and d.get("level"):
        return f"{d['pick']} · stake {d['level']} (${d.get('amount') or ST.AMOUNTS.get(d['level'], 0):,})"
    mejor = f" (mejor: {d['pick']})" if d.get("pick") else ""
    return f"{d.get('status')}{mejor}"


def _cambios(orig: dict, corr: dict) -> tuple[list, list, str]:
    """Diferencias entre la versión original y la corregida: qué cambió, qué estadísticas, y si es de fondo."""
    cambios, stats = [], []
    if orig.get("decision") != corr.get("decision"):
        cambios.append({"que": "Decisión única", "original": _dec_txt(orig.get("decision")), "corregido": _dec_txt(corr.get("decision"))})
    po = {p["pick"]: p for p in orig.get("picks") or []}
    pc = {p["pick"]: p for p in corr.get("picks") or []}
    added = [k for k in pc if k not in po]
    gone = [k for k in po if k not in pc]
    if added or gone:
        cambios.append({"que": "Picks", "original": ", ".join(gone) or "—", "corregido": ", ".join(added) or "—"})
    n_stake = sum(1 for k in pc if k in po and pc[k].get("stake") and not po[k].get("stake"))
    if n_stake:
        ej = next(k for k in pc if k in po and pc[k].get("stake") and not po[k].get("stake"))
        lad = (pc[ej].get("stake") or {}).get("ladder") or []
        lad_txt = ", ".join(("$" + format(s["stake"], ",") if s.get("stake") else "stake") + f" desde {s['from']:+d}"
                            for s in lad if s.get("from") is not None)
        cambios.append({"que": "Escalera de stake", "original": "sin escalera (no existía el sistema de stake)",
                        "corregido": f"{n_stake} picks con escalera; p. ej. {ej}: {lad_txt or '—'}"})
    for k in pc:
        if k in po:
            for f, lab in (("p", "probabilidad"), ("ic", "índice de confianza"), ("fair", "momio justo")):
                a, b = po[k].get(f), pc[k].get(f)
                if a != b:
                    stats.append({"que": f"{k}: {lab}", "original": a, "corregido": b})
    for f in ("pHome", "total", "projAway", "projHome", "nrfi"):
        a, b = (orig.get("pred") or {}).get(f), (corr.get("pred") or {}).get(f)
        if a != b:
            stats.append({"que": f"predicción: {f}", "original": a, "corregido": b})
    fondo = bool(added or gone) or any(abs((s["corregido"] or 0) - (s["original"] or 0)) >= 0.02 for s in stats if "probabilidad" in s["que"]) \
        or bool(((corr.get("decision") or {}).get("status") == "apostar" and (corr.get("decision") or {}).get("level"))
                != bool(((orig.get("decision") or {}).get("status") == "apostar" and (orig.get("decision") or {}).get("level"))))
    return cambios, stats, "de fondo" if fondo else "circunstancial"


def migrar(primer: dict, now: str, dry: bool = False) -> dict:
    tickets = HI.load()
    corr: dict[str, dict] = {}
    restored = []
    for tid, t in sorted(tickets.items()):
        if t.get("lock") or t["status"] == "abierto":
            continue
        path = os.path.join("data", "historial", f"{t['date']}.json")
        fp = (primer.get(str(t["pk"])) or {}).get("firstPitch") or t.get("time")
        c = _before(path, fp)
        pre = next((x for x in _at(c, path) if x["id"] == tid), None) if c else None
        if pre is not None:
            if _norm(pre) != _norm(t):
                raise SystemExit(f"{tid}: difiere de su versión previa al primer lanzamiento real; revisar a mano")
            if primer.get(str(t["pk"])) and t.get("registeredAt", "") > (t.get("time") or ""):
                corr[tid] = {"id": tid, "tipo": "nota", "clase": "circunstancial", "fecha": now[:10], "corregido": None, "cambios": [], "estadisticas": [],
                             "resumen": "Actualizado durante un retraso: la versión bloqueada es anterior al primer lanzamiento real.",
                             "notas": [f"Hora programada {_hm(t['time'])}; primer lanzamiento real {_hm(primer[str(t['pk'])]['firstPitch'])} "
                                       f"(retraso de {primer[str(t['pk'])].get('delayMinutes') or '—'} min).",
                                       f"El ticket se registró por última vez a las {_hm(t.get('registeredAt'))}, antes de que empezara el juego: "
                                       "es válido y no se modificó después."]}
            continue
        first = _first_version(tid, path)
        if not first:
            raise SystemExit(f"{tid}: sin versión en git")
        h, when, orig = first
        notes = [f"Este ticket no existía antes del partido: se creó el {_hm(when)} al nacer la base de tickets, "
                 f"con el registro que sí se guardó antes del juego ({'data/predictions' if t['source'] == 'seguimiento' else 'prolab/prolab_' + str(t['pk']) + '.json'}, "
                 f"generado {_hm(t.get('registeredAt'))}; primer lanzamiento {_hm(fp)})."]
        if t["source"] == "pro-lab" and t.get("frozenAt") and t.get("registeredAt", "") > (fp or ""):
            notes.append(f"El modelo del Pro-Lab terminó de construirse {_hm(t['registeredAt'])}, después del primer lanzamiento "
                         f"({_hm(fp)}); sus datos se congelaron antes ({_hm(t['frozenAt'])}).")
        if _norm(orig) == _norm(t):
            corr[tid] = {"id": tid, "tipo": "reconstruido", "clase": "circunstancial", "fecha": now[:10], "corregido": None,
                         "cambios": [], "estadisticas": [], "notas": notes,
                         "resumen": "Reconstruido después del juego desde su registro previo; el contenido es el de ese registro."}
            continue
        cambios, stats, clase = _cambios(orig, t)
        # el original manda: se restaura, conservando lo que llegó después del juego (resultado y calificación)
        res = {p["pick"]: p.get("res") for p in t.get("picks") or []}
        new = {**orig, "status": t["status"], "result": t.get("result"),
               "picks": [{**p, "res": res.get(p["pick"], p.get("res"))} for p in orig.get("picks") or []]}
        notes.append("La versión corregida es la que tenía la base antes de esta revisión: se le agregaron después del juego "
                     "campos que no existían cuando se registró (el sistema de decisión única y stake llegó el 25-sep). "
                     "En el récord oficial cuenta la versión original.")
        if clase == "de fondo" and (t.get("decision") or {}).get("status") == "apostar":
            d = t["decision"]
            pr = next((p for p in t["picks"] if p["pick"] == d["pick"]), {})
            res_txt = {"ganado": "se ganó", "perdido": "se perdió", "push": "fue push"}.get(pr.get("res"), "no se calificó")
            notes.append(f"La corrección agregaba una apuesta ({_dec_txt(d)}) que {res_txt} y contaba en el récord y el balance "
                         "sin haberse registrado antes del juego.")
        if not stats:
            notes.append("Probabilidades, índices de confianza y momios justos de los picks: sin cambios.")
        corr[tid] = {"id": tid, "tipo": "corregido", "clase": clase, "fecha": now[:10], "corregido": t, "cambios": cambios,
                     "estadisticas": stats, "notas": notes,
                     "resumen": "Se le agregó después del juego información que no existía antes: el original manda."}
        tickets[tid] = new
        restored.append(tid)
    # bloqueo + expedientes del Pro-Lab (datos crudos congelados antes del juego)
    exp_paths = {}
    for pk in sorted({t["pk"] for t in tickets.values() if t["source"] == "pro-lab"}):
        lab_path = os.path.join(ROOT, "prolab", f"prolab_{pk}.json")
        bpath = os.path.join(ROOT, "prolab", PROLAB_BUNDLE.get(pk, f"bundle_{pk}_pre.json.gz"))
        if not (os.path.exists(lab_path) and os.path.exists(bpath)):
            continue
        with open(lab_path, encoding="utf-8") as f:
            lab = json.load(f)
        with gzip.open(bpath, "rt", encoding="utf-8") as f:
            bundle = json.load(f)
        mine = {tid: t for tid, t in tickets.items() if t["pk"] == pk}
        date = next(iter(mine.values()))["date"]
        dst = EX.final_path(date, pk)
        if not os.path.exists(dst) and not dry:
            exp = EX.build(lab.get("base"), bundle, pk, date, mine, None, {"version": "anterior al versionado", "code": None}, now, HI.lock_hash)
            exp.update({"legado": True, "lockedAt": now, "coincide": None,
                        "nota": f"Armado el {now[:10]} con los datos congelados del Pro-Lab ({os.path.basename(bpath)}, congelados "
                                f"{_hm(lab.get('frozenAt'))}). El análisis es el del Pro-Lab (`base`); el ticket del framework de ese "
                                "partido se registró con la corrida de su momento, cuyos datos crudos no se guardaron."})
            EX._write(dst, exp)
        exp_paths[pk] = EX.rel(dst)
    locked = 0
    for tid, t in tickets.items():
        if t.get("lock") or t["status"] == "abierto":
            continue
        t = {**t, "lock": {"at": now, "hash": HI.lock_hash(t), "algo": {"version": "anterior al versionado", "code": None},
                           "expediente": exp_paths.get(t["pk"]), "coincide": None, "legado": True}}
        tickets[tid] = t
        locked += 1
    if not dry:
        os.makedirs(CORR_DIR, exist_ok=True)
        for tid, c in corr.items():
            with open(os.path.join(CORR_DIR, f"{tid}.json"), "w", encoding="utf-8") as f:
                json.dump(c, f, ensure_ascii=False, indent=1)
                f.write("\n")
        HI.save(tickets)
    return {"restaurados": restored, "notas": {k: v["tipo"] for k, v in corr.items()}, "bloqueados": locked, "expedientes": exp_paths}


def main() -> None:
    ap = argparse.ArgumentParser(description="Bloquea los tickets de antes de la regla de inmutabilidad")
    ap.add_argument("--primer", required=True, help="JSON con la hora oficial del primer lanzamiento por pk")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    with open(a.primer, encoding="utf-8") as f:
        primer = json.load(f)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(json.dumps(migrar(primer, now, a.dry), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
