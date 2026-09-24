"""Construye la página MLBgs: corre el modelo sobre los próximos partidos, guarda el
seguimiento de predicciones y genera `site/index.html` con los datos embebidos.

Uso:
    python -m mlbgs.build --bundle .cache/raw_bundle.json.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import math
import os
import sys

from . import decision as DE
from . import historial as HI
from . import markov as MK
from . import model
from . import picks as PK
from . import prisma as PRI
from . import prolab as PL
from . import stake as ST
from . import validate as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "data", "predictions")
TEMPLATE = os.path.join(ROOT, "site", "template.html")
OUT_HTML = os.path.join(ROOT, "site", "index.html")


def rounded(obj, nd=4):
    """Redondea floats para que el JSON embebido pese menos."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, nd) for v in obj]
    return obj


def load_bundle(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ seguimiento de predicciones

def prediction_row(a: dict, generated: str) -> dict:
    s = a["summary"]
    tot = next((m for m in a["sections"]["s6"]["lineTable"] if m["market"] == "Total completo" and m["line"] == 8.5), None)
    methods = a["sections"]["s5"]["triangulation"]
    return {
        "pk": a["pk"], "date": a["date"], "time": a["time"], "generatedAt": generated,
        "away": a["teams"]["away"]["abbr"], "home": a["teams"]["home"]["abbr"],
        "pHome": s["pHome"], "methods": {k: methods[k]["pHome"] for k in ("log5", "elo", "lambda")},
        "confidence": s["confidence"], "projAway": s["proj"]["away"], "projHome": s["proj"]["home"],
        "total": s["proj"]["total"], "f5Away": s["proj"]["f5"]["away"], "f5Home": s["proj"]["f5"]["home"],
        "nrfi": s["nrfi"], "pOver85": tot["over"] if tot else None,
        "probables": {k: v["name"] for k, v in s["probables"].items()},
        "light": s["light"], "best": {k: (s["best"] or {}).get(k) for k in ("market", "pick", "p", "price", "edge", "light")},
        "topPicks": [{k: p.get(k) for k in ("family", "market", "pick", "p", "line", "ic", "level")} for p in a.get("picks", [])[:2]],
    }


def save_predictions(analyses: list[dict], generated: str) -> None:
    os.makedirs(PRED_DIR, exist_ok=True)
    by_date: dict[str, list] = {}
    for a in analyses:
        by_date.setdefault(a["date"], []).append(prediction_row(a, generated))
    for date, rows in by_date.items():
        path = os.path.join(PRED_DIR, f"{date}.json")
        cur = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cur = {str(r["pk"]): r for r in json.load(f)}
        changed = False
        for r in rows:
            r = rounded(r)
            old = cur.get(str(r["pk"]))
            strip = lambda x: {k: v for k, v in (x or {}).items() if k != "generatedAt"}  # noqa: E731
            if old is None or strip(old) != strip(r):
                cur[str(r["pk"])] = r  # la última versión antes del partido es la que cuenta
                changed = True
        if not changed and os.path.exists(path):
            continue  # sin cambios reales: no se reescribe (evita commits vacíos cada corrida)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(cur.values(), key=lambda r: (r["time"] or "", r["pk"])), f, ensure_ascii=False, indent=0)


def evaluate(bundle: dict) -> dict:
    """Compara las predicciones guardadas contra los resultados finales oficiales."""
    results = {g["pk"]: g for g in bundle.get("results", [])}
    starter_k = {}
    for b in bundle.get("boxscores", []):
        starter_k[b["pk"]] = {b[s]["pitchers"][0]["name"]: b[s]["pitchers"][0]["k"] for s in ("away", "home") if b[s]["pitchers"]}
    rows = []
    pick_rows = []
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            for p in json.load(f):
                g = results.get(p["pk"])
                if not g:
                    continue
                home_won = g["hr"] > g["ar"]
                inn = g.get("inn") or []
                first = [x for x in inn[:1]]
                nrfi_hit = bool(first) and (first[0][0] or 0) == 0 and (first[0][1] or 0) == 0
                rows.append({**p, "ar": g["ar"], "hr": g["hr"], "homeWon": home_won, "actualTotal": g["ar"] + g["hr"],
                             "nrfiHit": nrfi_hit})
                for tp in p.get("topPicks") or []:
                    won = PK.grade(tp, p["away"], p["home"], g["ar"], g["hr"], inn, starter_k.get(p["pk"]))
                    pick_rows.append({**tp, "date": p["date"], "game": f"{p['away']} @ {p['home']}", "won": won})
    if not rows:
        return {"n": 0, "rows": [], "picks": {"n": 0}}
    by_level = {}
    for r in pick_rows:
        if r["won"] is None:
            continue
        lv = by_level.setdefault(r.get("level") or "—", {"n": 0, "won": 0, "pSum": 0.0})
        lv["n"] += 1
        lv["won"] += bool(r["won"])
        lv["pSum"] += r["p"]
    picks_eval = {"n": sum(v["n"] for v in by_level.values()),
                  "won": sum(v["won"] for v in by_level.values()),
                  "byLevel": [{"level": k, "n": v["n"], "hit": v["won"] / v["n"], "pred": v["pSum"] / v["n"]}
                              for k, v in sorted(by_level.items(), key=lambda kv: ["Alta", "Media", "Baja", "Muy baja"].index(kv[0])
                                                 if kv[0] in ("Alta", "Media", "Baja", "Muy baja") else 9)],
                  "rows": sorted(pick_rows, key=lambda r: r["date"], reverse=True)[:120]}

    def brier(key):
        return sum((r[key] - (1 if r["homeWon"] else 0)) ** 2 for r in rows) / len(rows)

    def logloss(p_fn):
        s = 0.0
        for r in rows:
            p = min(max(p_fn(r), 1e-4), 1 - 1e-4)
            s -= math.log(p if r["homeWon"] else 1 - p)
        return s / len(rows)

    acc = sum(1 for r in rows if (r["pHome"] >= 0.5) == r["homeWon"]) / len(rows)
    tot_rows = [r for r in rows if r.get("total") is not None]
    ou_rows = [r for r in rows if r.get("pOver85") is not None and r["actualTotal"] != 8.5]
    by_light: dict[str, dict] = {}
    for r in rows:
        by_light.setdefault(r["light"], {"n": 0})["n"] += 1
    buckets = []
    for lo in (0.5, 0.55, 0.6, 0.65, 0.7):
        hi = lo + 0.05 if lo < 0.7 else 1.01
        sel = [r for r in rows if lo <= max(r["pHome"], 1 - r["pHome"]) < hi]
        if sel:
            buckets.append({"range": f"{int(lo * 100)}–{int(min(hi, 1) * 100)}%", "n": len(sel),
                            "pred": sum(max(r["pHome"], 1 - r["pHome"]) for r in sel) / len(sel),
                            "hit": sum(1 for r in sel if (r["pHome"] >= 0.5) == r["homeWon"]) / len(sel)})
    return rounded({
        "n": len(rows), "accuracy": acc, "brier": brier("pHome"), "logloss": logloss(lambda r: r["pHome"]),
        "brierMethods": {k: sum((r["methods"][k] - (1 if r["homeWon"] else 0)) ** 2 for r in rows) / len(rows)
                         for k in ("log5", "elo", "lambda")},
        "totalMae": sum(abs(r["total"] - r["actualTotal"]) for r in tot_rows) / len(tot_rows) if tot_rows else None,
        "totalBias": sum(r["total"] - r["actualTotal"] for r in tot_rows) / len(tot_rows) if tot_rows else None,
        "ou85Acc": (sum(1 for r in ou_rows if (r["pOver85"] >= 0.5) == (r["actualTotal"] > 8.5)) / len(ou_rows)
                    if ou_rows else None),
        "nrfiBrier": sum((r["nrfi"] - (1 if r["nrfiHit"] else 0)) ** 2 for r in rows) / len(rows),
        "nrfiRate": sum(1 for r in rows if r["nrfiHit"]) / len(rows),
        "calibration": buckets, "byLight": by_light, "picks": picks_eval,
        "rows": sorted(rows, key=lambda r: (r["date"], r["time"] or ""), reverse=True)[:120],
    })


# ------------------------------------------------------------------ página

def build(bundle: dict, save: bool = True) -> dict:
    ctx = model.Context(bundle)
    analyses = []
    errors = []
    for g in sorted(bundle.get("upcoming", []), key=lambda x: (x["time"] or "", x["pk"])):
        try:
            analyses.append(model.analyze(ctx, g))
        except Exception as e:  # noqa: BLE001 - un partido con datos raros no debe tumbar la página
            errors.append({"pk": g["pk"], "error": repr(e)})
            print(f"ERROR analizando {g['pk']}: {e!r}", file=sys.stderr)
    generated = bundle["meta"]["generatedAt"]
    if save:
        save_predictions(analyses, generated)
    checks = V.validate(bundle)
    labs = load_prolabs(bundle, save)
    lab = next((x for x in labs if x.get("modelKey") == "diamante"), labs[0] if labs else None)
    live = live_view(bundle, ctx, labs)
    track = evaluate(bundle)
    # stake de cada pick (usa el historial ya calificado) antes de registrar los tickets, para que quede guardado
    prev = HI.load() or HI.update(bundle, analyses, labs, generated, save_files=False)
    stake_cfg = ST.attach(analyses, labs, prev)
    DE.attach(analyses, labs, stake_cfg["track"])     # una decisión por partido: framework + modelo + cuotas
    tickets = HI.update(bundle, analyses, labs, generated, save_files=save)
    lg = ctx.lg
    payload = {
        "meta": {**bundle["meta"], "builtAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "analysisErrors": errors, "hasOdds": bool(bundle.get("odds")),
                 "counts": {"games": len(analyses), "results": len(bundle.get("results", [])),
                            "boxscores": len(bundle.get("boxscores", [])), "pitchers": len(bundle.get("pitchers", {}))}},
        "league": {"era": lg.era, "ra9": lg.ra9, "cFip": lg.c_fip, "varRatio": lg.var_ratio, "homeWin": lg.home_win,
                   "rpg": lg.rpg, "kRate": lg.k_rate, "bbRate": lg.bb_rate, "pScoreless1": lg.p_scoreless_half1,
                   "extraShare": lg.extra_share, "extraHomeWin": lg.extra_home_win, "games": lg.n_games,
                   "inningRuns": {s: lg.inning_runs[s][1:] for s in ("away", "home")}},
        "standings": standings_view(ctx),
        "games": analyses,
        "track": track,
        "checks": checks,
        "prolab": lab,
        "prolabs": labs,
        "live": live,
        "historial": HI.page_view(tickets, generated),
        "stakeCfg": stake_cfg,
    }
    for a in payload["games"] + [x["base"] for x in labs]:
        a.pop("markets", None)   # tabla interna de mercados: los picks y las secciones ya la resumen
    payload["prolab"] = None     # compatibilidad: la vista usa `prolabs`
    return rounded(payload)


def live_view(bundle: dict, ctx: model.Context, labs: list[dict] | None = None) -> dict:
    """Partidos de ayer y hoy (programados, en vivo y finales) con sus 2 picks previos y su resultado.

    En los partidos del Pro-Lab los dos picks son los del modelo de prueba (congelados antes del juego)."""
    lab_picks = {lab["pk"]: (lab.get("model") or lab.get("modelKey", "").upper(), lab.get("topPicks") or []) for lab in labs or []}
    sb = bundle.get("scoreboard") or []
    preds = {}
    for date in sorted({g["date"] for g in sb}):
        path = os.path.join(PRED_DIR, f"{date}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                preds.update({r["pk"]: r for r in json.load(f)})
    starter_k = {}
    for b in bundle.get("boxscores", []):
        starter_k[b["pk"]] = {b[s]["pitchers"][0]["name"]: b[s]["pitchers"][0]["k"] for s in ("away", "home") if b[s]["pitchers"]}
    results = bundle.get("results", [])
    hd = PRI.half_inning_dist(results)
    exp_a = sum(sum(1 for x in (g.get("inn") or []) if x[0] is not None) or 9 for g in results) / max(1, len(results))
    exp_h = sum(sum(1 for x in (g.get("inn") or []) if x[1] is not None) or 8.5 for g in results) / max(1, len(results))
    lg = ctx.lg
    rows = []
    for g in sb:
        p = preds.get(g["pk"])
        row = dict(g)
        if p:
            picks = []
            model_name, top = lab_picks.get(g["pk"], (None, p.get("topPicks") or []))
            for tp in top:
                won = None
                if g["state"] == "Final" and g.get("ar") is not None:
                    won = PK.grade(tp, p["away"], p["home"], g["ar"], g["hr"], g.get("inn") or [], starter_k.get(g["pk"]))
                picks.append({**{k: tp.get(k) for k in ("family", "market", "pick", "p", "line", "ic", "level")}, "won": won})
            row["pred"] = {"pHome": p["pHome"], "projAway": p["projAway"], "projHome": p["projHome"], "total": p["total"],
                           "nrfi": p.get("nrfi"), "picks": picks, "generatedAt": p.get("generatedAt"),
                           "probables": p.get("probables"), "model": model_name}
            row["halfAway"] = PRI.scaled_half(hd, p["projAway"] / exp_a)
            row["halfHome"] = PRI.scaled_half(hd, p["projHome"] / exp_h)
        rows.append(row)
    L = MK.league_rates(lg.tot)
    e = MK.calibrate_residual(L, lg.runs_per_half, lg.gb_rate)
    re = MK.re24(MK.with_residual(L, e), lg.gb_rate)
    order = {"Live": 0, "Final": 1, "Preview": 2}
    rows.sort(key=lambda r: (order.get(r["state"], 3), r["date"] if r["state"] != "Final" else "", r["time"] or ""))
    return {"games": rows, "halfLeague": hd, "re24": re["table"], "p1": re["p1table"],
            "xHome": 0.5 * lg.extra_home_win + 0.25, "asOf": bundle["meta"]["generatedAt"]}


def load_prolabs(bundle: dict, save: bool) -> list[dict]:
    labs = []
    for path in sorted(glob.glob(os.path.join(PL.DIR, "prolab_*.json"))):
        lab = load_prolab(bundle, save, path)
        if lab:
            labs.append(lab)
    return labs


def load_prolab(bundle: dict, save: bool, path: str | None = None) -> dict | None:
    """Carga un Pro-Lab congelado; si el partido ya terminó, lo califica con el resultado oficial."""
    paths = [path] if path else sorted(glob.glob(os.path.join(PL.DIR, "prolab_*.json")))
    if not paths:
        return None
    with open(paths[-1], encoding="utf-8") as f:
        lab = json.load(f)
    lab.setdefault("modelKey", "diamante" if "mc" in lab else "prisma")
    pk = lab["pk"]
    rpath = os.path.join(PL.DIR, f"result_{pk}.json")
    res = None
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as f:
            res = json.load(f)
    else:
        res = PL.result_from_bundle(bundle, pk)
        if res and save:
            with open(rpath, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
    if res:
        lab["result"] = PL.compare(res, lab)
    return lab


def standings_view(ctx: model.Context) -> list[dict]:
    rows = []
    for tid, info in ctx.T.info.items():
        st = ctx.T.standings.get(tid, {})
        py = ctx.T.pythag(tid)
        bp = ctx.bp["teams"].get(tid, {})
        rows.append({"id": tid, "abbr": info.get("abbr"), "name": info.get("name"), "league": info.get("league"),
                     "division": info.get("division"), "w": st.get("w"), "l": st.get("l"), "rs": st.get("rs"),
                     "ra": st.get("ra"), "divRank": st.get("divRank"), "gb": st.get("gb"), "clinch": st.get("clinch"),
                     "streak": st.get("streak"), "pyth": py["pyth"], "true": py["true"], "luck": py["luck"],
                     "elo": ctx.T.elo.get(tid), "bpRank": bp.get("rank"), "bpEra": bp.get("era"),
                     "off": ctx.T.offense(tid)["idx"]})
    return sorted(rows, key=lambda r: (r["league"] or "", r["division"] or "", int(r["divRank"] or 99)))


def write_html(payload: dict, out: str = OUT_HTML) -> None:
    with open(TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("__MLBGS_DATA__", data)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera la página MLBgs")
    ap.add_argument("--bundle", default=".cache/raw_bundle.json.gz")
    ap.add_argument("--out", default=OUT_HTML)
    ap.add_argument("--json", help="además escribe el payload como JSON")
    ap.add_argument("--no-save", action="store_true", help="no actualizar data/predictions")
    args = ap.parse_args()
    bundle = load_bundle(args.bundle)
    payload = build(bundle, save=not args.no_save)
    write_html(payload, args.out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    print(f"{len(payload['games'])} partidos → {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB); "
          f"seguimiento: {payload['track']['n']} partidos evaluados", file=sys.stderr)


if __name__ == "__main__":
    main()
