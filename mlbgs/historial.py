"""Base de datos de tickets: cada pick que MLBgs registró antes del primer lanzamiento, con el
modelo que lo produjo, los tipos de análisis que usó y su resultado oficial.

Un ticket = un partido × un modelo. Se guarda en `data/historial/AAAA-MM-DD.json` (un archivo por
fecha oficial del juego, con orden estable para que git guarde solo lo que cambia).

Ciclo de vida de un ticket:
  abierto     se registra y se actualiza en cada corrida mientras el partido no empieza
  cerrado     el partido empezó: el ticket queda congelado (cuenta la última versión previa)
  calificado  el partido terminó: cada pick trae ganado / perdido / push con el marcador oficial
  anulado     partido pospuesto o cancelado (los picks no cuentan)

Bloqueo (regla del proyecto): al primer lanzamiento cada ticket recibe `lock` = {at, hash, algo,
expediente}. `hash` es la huella de todo su contenido salvo lo que llega después del juego (estado,
resultado y la calificación de cada pick). Desde ahí el ticket NO cambia: ninguna corrección posterior
de los modelos lo recalcula (las correcciones solo aplican a partidos que no han empezado). `update`
revisa las huellas y se detiene con error si algo intentara modificar un ticket bloqueado. Los datos con
los que se hizo quedan en data/expedientes (ver expediente.py) y la versión del algoritmo en `algo`.
Las notas sobre tickets de antes de esta regla (reconstruidos o completados después del juego) viven
aparte, en data/correcciones: el ticket original no se toca.

Fuentes de un ticket:
  modelo       corrida del framework sobre los próximos partidos (todos los picks, con su explicación)
  seguimiento  predicción guardada en data/predictions/ antes de existir esta base (solo los 2 principales)
  pro-lab      modelo de prueba congelado antes del juego (picks finales framework + modelo)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

from . import expediente as EX
from . import odds as O
from . import picks as PK
from . import version as VER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "historial")
PRED_DIR = os.path.join(ROOT, "data", "predictions")
PROLAB_DIR = os.path.join(ROOT, "prolab")

# Tipos de análisis por modelo: nombre corto (se guarda en el ticket) y qué hace (se muestra en la página).
MODELS = {
    "framework": {
        "name": "Framework v2", "full": "Framework MLB Picks v2", "kind": "Modelo estadístico por secciones",
        "analyses": [
            ["Contexto del partido", "Importancia en la tabla por simulación del resto de la temporada, serie, lesionados, transacciones y noticias (sección 1)."],
            ["Pitágoras + Log5", "Win% pitagórico regresado a la temporada anterior y ventaja de local medida, combinados con Log5 (5.7.2)."],
            ["Elo con abridor", "Elo con K=4 y +24 de local, ajustado por la calidad del abridor (5.7.7)."],
            ["Abridores regresados", "FIP, xERA y ERA con regresión a la media por puntos de estabilización (K% 70 BF, BB% 170 BF) (sección 3)."],
            ["Bullpen y fatiga", "Calidad del bullpen y uso de relevistas en los últimos días (sección 4)."],
            ["λ + Binomial Negativa", "Carreras por entrada de cada equipo y distribución Binomial Negativa del marcador con la sobredispersión medida (5.3 · 5.7.5)."],
            ["Poisson F5 y NRFI", "Primeras 5 y 3 entradas con Poisson; NRFI calibrado con la frecuencia real de ceros (5.4 · 6.8)."],
            ["Parque, clima y umpire", "Park factor de 3 años, viento y temperatura, umpire del plato (ajustes de la tabla 5.3)."],
            ["Valor contra el mercado", "Momio justo, mínimo aceptable, edge contra el momio real o de referencia (sección 7)."],
            ["Contradicciones y semáforo", "Auditoría de contradicciones, gate de datos obligatorios y semáforo (secciones 8 y 9)."],
        ],
    },
    "diamante": {
        "name": "DIAMANTE-24", "full": "Framework v2 + DIAMANTE-24", "kind": "Cadenas de Markov + Monte Carlo",
        "analyses": [
            ["Log5 multinomial por turno", "Resultado de cada turno al bate (out, BB, 1B, 2B, 3B, HR) combinando bateador, pitcher y liga."],
            ["Cadena de 24 estados base-out", "Matriz de transición entre situaciones de corredores y outs; N = (I − Q)⁻¹ y RE24."],
            ["Cadena por lineup (216 estados)", "La cadena sigue el orden al bate real para contar carreras por entrada."],
            ["Monte Carlo del partido", "50,000 partidos simulados con bullpen por rol, openers y corredor en segunda en extras."],
        ],
    },
    "prisma": {
        "name": "PRISMA", "full": "Framework v2 + PRISMA", "kind": "Probabilidad bayesiana",
        "analyses": [
            ["GLM jerárquico", "Ataque y defensa de cada equipo con exposición por entradas (Poisson)."],
            ["Bayes empírico", "Hiperparámetros por máxima verosimilitud marginal (EM) y elegidos por validación fuera de muestra."],
            ["Posterior de Laplace", "Incertidumbre de las fuerzas con Cholesky; intervalo creíble de P(victoria)."],
            ["Predictiva con fragilidad", "Marcador simulado con un efecto común del partido que abre las colas."],
        ],
    },
    "kronos": {
        "name": "KRONOS", "full": "Framework v2 + KRONOS", "kind": "Procesos estocásticos",
        "analyses": [
            ["Cadena de la cuenta", "Cada lanzamiento como paso de una cadena de Markov de 12 cuentas con tablas Statcast; N = (I − Q)⁻¹."],
            ["Cadena base-out calibrada", "Corredores y outs con la carrera extra por base calibrada a las carreras reales por media entrada."],
            ["Salida del abridor", "Proceso de conteo de lanzamientos con límite estimado por carreras permitidas."],
            ["Deriva por vuelta al lineup", "El bateador mejora cada vez que ve al abridor (efecto TTO continuo)."],
            ["Fragilidad del partido", "Efecto gamma común a ambos equipos (clima, umpire, día) medido en 2026."],
            ["Monte Carlo por lanzamiento", "20,000 partidos simulados lanzamiento a lanzamiento con las reglas de 2026."],
        ],
    },
    "eigen": {
        "name": "EIGEN", "full": "Framework v2 + EIGEN", "kind": "Componentes principales (PCA)",
        "analyses": [
            ["PCA de pitchers", "Perfil de 9 métricas Statcast (K, BB, HR, GB, barrel, hard-hit, EV, whiff, velocidad) reducido a componentes; regresión sobre componentes (PCR) para RA9."],
            ["PCA de bateadores", "8 métricas (K, BB, ISO, HR, barrel, hard-hit, EV, SB) reducidas a componentes; PCR para wOBA."],
            ["Talento encogido", "Cada jugador mezcla lo observado con lo que predicen sus componentes según su muestra (600 BF · 300 PA)."],
            ["Lineup por lugar", "Ofensiva del lineup con las apariciones esperadas por lugar en el orden y ajuste de mano."],
            ["Abridor + bullpen", "Carreras por la parte del partido de cada uno (IP esperadas del abridor) y parque a media fuerza."],
            ["Binomial Negativa + bootstrap", "Marcador, F5 y NRFI; intervalo por bootstrap que re-ajusta el PCA y la muestra de cada jugador."],
            ["PCA de equipos y de métodos", "Dónde conviene PCA: equipos (descriptivo) y el acuerdo entre los métodos del framework."],
        ],
    },
}
LAB_EXTRA = ["Picks combinados con el framework"]
ROUND = {"p": 4, "ic": 1, "fair": 0, "pHome": 4, "pModel": 4, "projAway": 2, "projHome": 2, "total": 2, "nrfi": 4}


def _r(obj):
    """Redondeo estable (menos ruido en los diffs de git)."""
    if isinstance(obj, dict):
        return {k: (round(v, ROUND[k]) if k in ROUND and isinstance(v, float) else _r(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_r(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _pick_row(p: dict, rank: int) -> dict:
    rank = p.get("rank") or rank
    return {"rank": rank, "principal": rank <= 2, "family": p["family"], "market": p["market"], "pick": p["pick"],
            "line": p.get("line"), "p": p["p"], "fair": p.get("fair"), "ic": p.get("ic"), "level": p.get("level"),
            "price": p.get("price") if p.get("priceIsReal") else None,
            "methods": sorted((p.get("methods") or {}).keys()), "how": p.get("how"), "res": None,
            **({"stake": {k: p["stake"].get(k) for k in ("level", "ladder", "steps", "block", "a", "h", "maxPrice", "why")
                          if k in p["stake"]}} if p.get("stake") else {})}


def _decision(d: dict | None) -> dict | None:
    """La opción única del partido tal como se registró (pick + stake 1–10)."""
    if not d or not d.get("pick"):
        return None
    return {"status": d["status"], "why": d.get("why"), "source": d.get("source"), "pick": d["pick"]["pick"],
            **({"waitFor": d["waitFor"]} if d.get("waitFor") else {}),
            "market": d["pick"]["market"], "p": d["pick"]["p"], "ic": d["pick"]["ic"], "level": d["stake"]["level"],
            "amount": d["stake"]["amount"], "minPrice": d["stake"]["minPrice"], "ladder": d["stake"]["ladder"]}


def _names(a: dict) -> dict:
    t = a["teams"]
    return {"away": t["away"]["abbr"], "home": t["home"]["abbr"], "awayName": t["away"].get("name"),
            "homeName": t["home"].get("name"), "venue": (a.get("venue") or {}).get("name")}


def _pred(s: dict) -> dict:
    lu = s.get("lineupsConfirmed")
    return {"pHome": s["pHome"], "projAway": s["proj"]["away"], "projHome": s["proj"]["home"], "total": s["proj"]["total"],
            "nrfi": s.get("nrfi"), "light": s.get("light"), "confidence": s.get("confidence"),
            "probables": {k: (v or {}).get("name") for k, v in (s.get("probables") or {}).items()},
            "lineups": ("Confirmados" if isinstance(lu, dict) and all(lu.values()) else "Proyectados" if lu is not None else None)}


def ticket_from_analysis(a: dict, generated: str, algo: dict | None = None) -> dict:
    return {"id": f"{a['date']}-{a['pk']}-framework", "pk": a["pk"], "date": a["date"], "time": a["time"], **_names(a),
            "model": "framework", "modelName": MODELS["framework"]["full"], "kind": MODELS["framework"]["kind"],
            **({"algo": algo} if algo else {}),
            "analyses": [x[0] for x in MODELS["framework"]["analyses"]],
            "source": "modelo", "registeredAt": generated, "status": "abierto", "pred": _pred(a["summary"]),
            "picks": [_pick_row(p, i + 1) for i, p in enumerate(a.get("picks") or [])], "result": None,
            **({"decision": _decision(a.get("decisionFramework"))} if a.get("decisionFramework") else {})}


def ticket_from_prediction(r: dict) -> dict:
    """Predicción del seguimiento (data/predictions) previa a esta base: solo trae los 2 picks principales."""
    return {"id": f"{r['date']}-{r['pk']}-framework", "pk": r["pk"], "date": r["date"], "time": r.get("time"),
            "away": r["away"], "home": r["home"], "awayName": None, "homeName": None, "venue": None,
            "model": "framework", "modelName": MODELS["framework"]["full"], "kind": MODELS["framework"]["kind"],
            "analyses": [x[0] for x in MODELS["framework"]["analyses"]],
            "source": "seguimiento", "registeredAt": r.get("generatedAt"), "status": "abierto",
            "pred": {"pHome": r["pHome"], "projAway": r.get("projAway"), "projHome": r.get("projHome"), "total": r.get("total"),
                     "nrfi": r.get("nrfi"), "light": r.get("light"), "confidence": r.get("confidence"),
                     "probables": r.get("probables") or {}, "lineups": None},
            "picks": [{**_pick_row(tp, i + 1), "rank": i + 1, "principal": True} for i, tp in enumerate(r.get("topPicks") or [])],
            "result": None}


def _lab_total(lab: dict):
    dist = (lab.get("summaryProb") or {}).get("totalDist") or ((lab.get("mc") or {}).get("total"))
    if dist:
        return sum(k * v for k, v in enumerate(dist)) / (sum(dist) or 1)
    return ((lab.get("prisma") or {}).get("predictive") or {}).get("meanTotal")


def ticket_from_lab(lab: dict, algo: dict | None = None) -> dict:
    key = lab.get("modelKey") or ("diamante" if "mc" in lab else "prisma")
    info = MODELS.get(key, {"full": lab.get("model"), "kind": "Modelo del Pro-Lab", "analyses": []})
    base = lab["base"]
    pred = _pred(base["summary"])
    cons = lab.get("consensus") or {}
    pred.update({"pHome": cons.get("pHome", pred["pHome"]), "pModel": (cons.get("methods") or {}).get(key if key != "diamante" else "mc"),
                 "total": _lab_total(lab) or pred["total"]})
    mc = (lab.get("kronos") or {}).get("mc") or lab.get("mc") or {}
    if mc.get("meanRuns"):   # marcador y NRFI del propio modelo (simulación)
        pred.update({"projAway": mc["meanRuns"]["away"], "projHome": mc["meanRuns"]["home"], "nrfi": mc.get("nrfi", pred["nrfi"])})
    top = {p["pick"] for p in lab.get("topPicks") or []}
    picks = [{**_pick_row(p, i + 1), "principal": p["pick"] in top} for i, p in enumerate(lab.get("picks") or [])]
    return {"id": f"{base['date']}-{lab['pk']}-{key}", "pk": lab["pk"], "date": base["date"], "time": base["time"], **_names(base),
            "model": key, "modelName": info["full"], "kind": info["kind"],
            **({"algo": algo} if algo else {}),
            "analyses": [x[0] for x in info["analyses"]] + LAB_EXTRA,
            "source": "pro-lab", "registeredAt": lab.get("builtAt"), "frozenAt": lab.get("frozenAt"), "status": "cerrado",
            "pred": pred, "picks": picks, "result": None,
            **({"decision": _decision(lab.get("decision"))} if lab.get("decision") else {})}


# ------------------------------------------------------------------ resultados oficiales

def finals(bundle: dict) -> dict:
    """pk → estado y marcador oficial, con los ponches de los abridores para calificar props."""
    out: dict = {}
    box_k = {}
    for b in bundle.get("boxscores", []):
        box_k[b["pk"]] = {b[s]["pitchers"][0]["name"]: b[s]["pitchers"][0]["k"] for s in ("away", "home") if b[s]["pitchers"]}
    for g in bundle.get("results", []):
        out[g["pk"]] = {"state": "Final", "date": g.get("date"), "ar": g["ar"], "hr": g["hr"], "inn": g.get("inn") or [],
                        "starterK": box_k.get(g["pk"])}
    for g in bundle.get("scoreboard", []):
        row = {"state": g["state"], "detailed": g.get("detailed"), "date": g.get("date"), "ar": g.get("ar"), "hr": g.get("hr"),
               "inn": g.get("inn") or [], "decisions": g.get("decisions"), "rhe": g.get("rhe")}
        box = g.get("box")
        if box:
            row["starterK"] = {box[s]["pit"][0]["name"]: box[s]["pit"][0]["k"] for s in ("away", "home") if box[s]["pit"]}
        prev = out.get(g["pk"])
        if prev and prev["state"] == "Final" and g["state"] != "Final":
            continue  # el calendario de la temporada ya lo tiene final (p. ej. juego reanudado)
        if prev and not row.get("starterK"):
            row["starterK"] = prev.get("starterK")
        out[g["pk"]] = {**(prev or {}), **{k: v for k, v in row.items() if v is not None}}
    for path in glob.glob(os.path.join(PROLAB_DIR, "result_*.json")):
        pk = int(os.path.basename(path)[7:-5])
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        cur = out.setdefault(pk, {"state": "Final", "date": r.get("date"), "ar": r["away"], "hr": r["home"], "inn": r["innings"]})
        if r.get("starterK"):
            cur["starterK"] = {**(cur.get("starterK") or {}), **r["starterK"]}
    return out


def grade_pick(p: dict, t: dict, fin: dict) -> str:
    won = PK.grade(p, t["away"], t["home"], fin["ar"], fin["hr"], fin.get("inn") or [], fin.get("starterK"))
    if won is None:
        if p["family"] == "K":
            name = p["pick"].split(" Over ")[0].split(" Under ")[0]
            if name not in (fin.get("starterK") or {}):
                return "sin dato"
        elif p["family"] not in ("Total", "F5 total", "Team total", "F5"):
            return "sin dato"
        return "push"
    return "ganado" if won else "perdido"


def _is_off(fin: dict) -> bool:
    return any(w in (fin.get("detailed") or "") for w in ("Postponed", "Cancel"))


def settle(t: dict, fins: dict, now: str) -> dict:
    """Actualiza el estado del ticket con el resultado oficial. No toca picks ya calificados."""
    fin = fins.get(t["pk"])
    if t["status"] == "anulado":
        return t
    if fin and (_is_off(fin) and fin.get("date") == t["date"]):
        return {**t, "status": "anulado", "result": {"state": "Pospuesto", "detailed": fin.get("detailed")},
                "picks": [{**p, "res": "anulado"} for p in t["picks"]]}
    if fin and fin.get("date") and fin["date"] != t["date"] and fin.get("state") == "Preview":
        return {**t, "status": "anulado", "result": {"state": "Pospuesto", "detailed": f"Reprogramado al {fin['date']}"},
                "picks": [{**p, "res": "anulado"} for p in t["picks"]]}
    if fin and fin.get("state") == "Final" and fin.get("ar") is not None:
        picks = [p if p.get("res") not in (None, "sin dato") else {**p, "res": grade_pick(p, t, fin)} for p in t["picks"]]
        if t["status"] == "calificado" and t.get("result"):
            return {**t, "picks": picks}          # el resultado oficial ya registrado no se reescribe
        res = {"state": "Final", "ar": fin["ar"], "hr": fin["hr"], "inn": fin.get("inn") or [],
               "decisions": fin.get("decisions"), "rhe": fin.get("rhe"),
               "gradedAt": (t.get("result") or {}).get("gradedAt") or now}
        return {**t, "status": "calificado", "picks": picks, "result": res}
    started = fin and fin.get("state") == "Live" and (fin.get("detailed") or "") not in ("Warmup", "Pre-Game")
    if t["status"] == "abierto" and (started or (t.get("time") and t["time"] < now)):
        return {**t, "status": "cerrado"}
    return t


# ------------------------------------------------------------------ bloqueo

AFTER_GAME = ("status", "result", "lock")      # lo único que llega después del primer lanzamiento


def lock_hash(t: dict) -> str:
    """Huella del ticket tal como se registró antes del juego (sin estado, resultado ni calificaciones)."""
    c = {k: v for k, v in t.items() if k not in AFTER_GAME}
    c["picks"] = [{k: v for k, v in p.items() if k != "res"} for p in t.get("picks") or []]
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def verify(t: dict) -> bool:
    return bool(t.get("lock")) and lock_hash(t) == t["lock"]["hash"]


def lock_due(t: dict, fins: dict, now: str) -> bool:
    """El ticket se bloquea al primer lanzamiento (o si el partido ya terminó o se anuló)."""
    if t.get("lock"):
        return False
    if t["status"] in ("calificado", "anulado"):
        return True
    fin = fins.get(t["pk"]) or {}
    started = fin.get("state") in ("Live", "Final") and (fin.get("detailed") or "") not in ("Warmup", "Pre-Game")
    return t["status"] != "abierto" and bool(started or (t.get("time") and t["time"] < now))


def lock(t: dict, at: str, exp: dict | None = None) -> dict:
    return {**t, "lock": {"at": at, "hash": lock_hash(t), "algo": t.get("algo"),
                          "expediente": (exp or {}).get("path"), "coincide": (exp or {}).get("coincide")}}


def check_locks(before: dict, after: dict) -> list[str]:
    """Tickets bloqueados que cambiaron (no debe pasar nunca): lista vacía = todo en orden."""
    bad = []
    for tid, o in before.items():
        if not o.get("lock"):
            continue
        n = after.get(tid)
        if n is None:
            bad.append(f"{tid}: desapareció")
        elif n.get("lock") != o["lock"] or lock_hash(n) != o["lock"]["hash"]:
            bad.append(f"{tid}: cambió su contenido bloqueado")
        else:
            olds = {p["pick"]: p.get("res") for p in o.get("picks") or []}
            for p in n.get("picks") or []:
                r0 = olds.get(p["pick"])
                if r0 not in (None, "sin dato") and p.get("res") != r0:
                    bad.append(f"{tid}: la calificación de {p['pick']} cambió ({r0} → {p.get('res')})")
            if o.get("status") == "calificado" and o.get("result") and n.get("result") != o["result"]:
                bad.append(f"{tid}: cambió su resultado oficial")
    return bad


# ------------------------------------------------------------------ almacenamiento

def load(dir_: str = DIR) -> dict:
    tickets = {}
    for path in sorted(glob.glob(os.path.join(dir_, "*.json"))):
        with open(path, encoding="utf-8") as f:
            for t in json.load(f):
                tickets[t["id"]] = t
    return tickets


def _content(t: dict) -> dict:
    return {k: v for k, v in t.items() if k not in ("registeredAt",)}


def save(tickets: dict, dir_: str = DIR) -> list[str]:
    """Escribe un archivo por fecha, solo si su contenido cambió. Devuelve los archivos escritos."""
    os.makedirs(dir_, exist_ok=True)
    by_date: dict[str, list] = {}
    for t in tickets.values():
        by_date.setdefault(t["date"], []).append(t)
    written = []
    for date, rows in by_date.items():
        rows = sorted(rows, key=lambda t: (t.get("time") or "", t["pk"], t["model"] != "framework", t["model"]))
        path = os.path.join(dir_, f"{date}.json")
        text = json.dumps(rows, ensure_ascii=False, indent=1)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                if f.read() == text:
                    continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(path)
    return written


def update(bundle: dict, analyses: list[dict], labs: list[dict], generated: str, save_files: bool = True,
           dir_: str = DIR, pred_dir: str = PRED_DIR, exp_dir: str = EX.DIR, pending_dir: str = EX.PENDING,
           odds_dir: str = O.ODDS_DIR) -> dict:
    """Registra los tickets de esta corrida, los bloquea al primer lanzamiento y califica los terminados.

    Un ticket bloqueado nunca se reescribe: si algo lo intentara, se detiene con error y no guarda nada."""
    tickets = load(dir_)
    before = json.loads(json.dumps(tickets))
    fins = finals(bundle)
    algo = VER.info()

    # 1) framework: la versión más reciente antes del primer lanzamiento
    for a in analyses:
        t = _r(ticket_from_analysis(a, generated, algo))
        old = tickets.get(t["id"])
        if old and (old.get("lock") or old["status"] != "abierto"):
            continue
        if old and _content({**old, "source": t["source"]}) == _content(t):
            continue
        tickets[t["id"]] = t
    # 2) Pro-Lab: congelados antes del juego; su archivo manda hasta que el ticket se bloquea
    for lab in labs:
        t = _r(ticket_from_lab(lab, algo))
        old = tickets.get(t["id"])
        if old and old.get("lock"):
            continue
        if old:
            # la escalera de stake y el resultado ya registrados no cambian con corridas posteriores
            prev = {q["pick"]: q for q in old["picks"]}
            t = {**t, "picks": [{**p, **({"stake": prev[p["pick"]]["stake"]} if prev.get(p["pick"], {}).get("stake") else {}),
                                 "res": (prev.get(p["pick"]) or {}).get("res")} for p in t["picks"]]}
            if old.get("decision"):
                t = {**t, "decision": old["decision"]}
            if old.get("result"):
                t = {**t, "status": old["status"], "result": old["result"]}
        tickets[t["id"]] = t
    # 3) predicción del seguimiento sin ticket: solo si el partido no ha empezado (nunca se reconstruye después)
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                tid = f"{r['date']}-{r['pk']}-framework"
                if tid not in tickets and (r.get("topPicks") or []) and (r.get("time") or "") > generated:
                    tickets[tid] = _r(ticket_from_prediction(r))
    # 4) expediente de cada partido que todavía no empieza (la versión más reciente espera en la caché)
    if save_files:
        for a in analyses:
            mine = {tid: t for tid, t in tickets.items() if t["pk"] == a["pk"] and t["date"] == a["date"] and not t.get("lock")}
            if not mine:
                continue
            entry = ((O.load_day(a["date"], odds_dir).get("games") or {}).get(str(a["pk"])))
            exp = EX.build({k: v for k, v in a.items() if k != "markets"}, bundle, a["pk"], a["date"], mine, entry,
                           algo, generated, lock_hash)
            EX.write_pending(exp, pending_dir)
    # 5) calificar, y bloquear al primer lanzamiento (con su expediente previo al juego)
    for tid, t in list(tickets.items()):
        tickets[tid] = settle(t, fins, generated)
    due = [tid for tid, t in tickets.items() if lock_due(t, fins, generated)]
    for pk in sorted({tickets[tid]["pk"] for tid in due}):
        ids = [tid for tid in due if tickets[tid]["pk"] == pk]
        hashes = {tid: lock_hash(tickets[tid]) for tid in ids}
        exp = EX.promote(pk, tickets[ids[0]]["date"], hashes, pending_dir, exp_dir, generated) if save_files else None
        for tid in ids:
            tickets[tid] = lock(tickets[tid], generated, exp)
    bad = check_locks(before, tickets)
    if bad:
        raise RuntimeError("Tickets bloqueados modificados; no se guarda nada: " + "; ".join(bad[:10]))
    if save_files:
        save(tickets, dir_)
    return tickets


def page_view(tickets: dict, generated: str, full_days: int = 21) -> dict:
    """Lo que se embebe en la página: todo el historial, con la explicación de cada pick solo en los días recientes."""
    rows = sorted(tickets.values(), key=lambda t: (t["date"], t.get("time") or "", t["pk"]), reverse=True)
    dates = sorted({t["date"] for t in rows}, reverse=True)
    recent = set(dates[:full_days])
    out = []
    for t in rows:
        if t["date"] not in recent:
            t = {**t, "picks": [{k: v for k, v in p.items() if k != "how"} for p in t["picks"]]}
        out.append(t)
    from . import mercado as MK
    from . import stake as ST
    corr = load_corrections()
    mkt = MK.cargar_todo([t["date"] for t in rows])
    real = {}
    for t in rows:
        if t["status"] in ("calificado", "cerrado") and t.get("lock"):
            ev = MK.evaluar(t, (mkt.get(t["date"]) or {}).get(str(t["pk"])), ST.AMOUNTS)
            if ev:
                real[t["id"]] = ev
    return {"tickets": out, "models": {k: {kk: v[kk] for kk in ("name", "full", "kind", "analyses")} for k, v in MODELS.items()},
            "asOf": generated, "correcciones": corr, "mercado": real, "algos": VER.CAMBIOS}


CORR_DIR = os.path.join(ROOT, "data", "correcciones")


def load_corrections(dir_: str = CORR_DIR) -> dict:
    """Notas y versiones corregidas de tickets anteriores a la regla de bloqueo (el ticket original no se toca)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(dir_, "*.json"))):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        out[c["id"]] = c
    return out
