"""Auditoría de un modelo del Pro-Lab después de su partido: qué predijo, qué pasó y qué parte falló, con pruebas
fuera de muestra contra otros algoritmos (nunca con un solo juego).

Hoy cubre OCTUBRE (postemporada). Para cada parte del modelo se hace una pregunta que se puede refutar con datos
que el modelo tenía ANTES del partido (el snapshot congelado), con un corte en el tiempo: se entrena con lo anterior
al corte y se califica lo posterior con una regla de puntuación propia (log-verosimilitud), con intervalo por
bootstrap. Solo entra al modelo lo que mejora fuera de muestra.

  A1  ¿el historial pitcher × rival mejora las tasas por evento? (v1 κ ingenuo · v2 κ sin la forma del día)
  A2  ¿de dónde sale el «efecto del rival»? varianza de juego a juego (ANOVA dentro de la pareja)
  A3  encogimiento de la tasa del pitcher: κ medido contra el 30 fijo
  C1  gancho de octubre: media constante · OLS · Theil-Sen · cociente · pitcheos por out (año contra año)
  C2  la salida del abridor depende de su calidad y de cómo le va en el juego
  K1  distribución de ponches del abridor: Poisson-binomial (v1) contra forma del día acoplada a la salida (v2)
  D1  familiaridad en ponches (reportada, no usada) fuera de muestra

Uso: python -m mlbgs.auditoria --pk 849845   →   prolab/auditoria_849845.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import random
import statistics as stt

from . import octubre as OC
from .version import VERSION

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prolab")
CUTS = ("2026-07-01", "2026-08-01")


# ------------------------------------------------------------------ utilidades

def _tot(rows):
    c = {}
    for r in rows:
        c = OC.add(c, r)
    return c


def _ll(y, n, p):
    p = min(1 - 1e-9, max(1e-9, p))
    return y * math.log(p) + (n - y) * math.log(1 - p)


def _boot(d, weight=None, reps=600, seed=11, stat="mean"):
    """Media (o suma/peso) de las diferencias y su IC90 por bootstrap de aperturas."""
    rng = random.Random(seed)
    if stat == "ratio":
        num = sum(x for x, _ in d)
        den = sum(w for _, w in d)
        est = num / den
        bs = []
        for _ in range(reps):
            smp = [d[rng.randrange(len(d))] for _ in d]
            bs.append(sum(x for x, _ in smp) / sum(w for _, w in smp))
    else:
        est = stt.mean(d)
        bs = [stt.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(reps)]
    bs.sort()
    return {"est": est, "ci90": [bs[int(0.05 * reps)], bs[int(0.95 * reps) - 1]], "n": len(d)}


def _verdict(ci, better_if_positive=True):
    lo, hi = ci
    if lo > 0:
        return "mejora" if better_if_positive else "empeora"
    if hi < 0:
        return "empeora" if better_if_positive else "mejora"
    return "sin diferencia"


class Split:
    """Todo lo que un modelo sabe al corte: liga, equipos, pitchers, parejas y varianzas (v1 y v2)."""

    def __init__(self, rows, cut):
        self.cut = cut
        self.train = [r for r in rows if r["date"] < cut]
        self.test = [r for r in rows if r["date"] >= cut]
        lgc = _tot(self.train)
        self.lg = {e: OC.trials(lgc, e)[0] / OC.trials(lgc, e)[1] for e in OC.EVENTS}
        self.lg["rpa"] = lgc.get("runs", 0) / lgc["bf"]
        by_opp = {}
        for r in self.train:
            by_opp.setdefault(r["opp"], []).append(r)
        self.teams = {}
        for t, rs in by_opp.items():             # equipo como bateador, desde los game logs de los abridores rivales
            c = _tot(rs)
            self.teams[t] = {e: (OC.trials(c, e)[0] + 400 * self.lg[e]) / (OC.trials(c, e)[1] + 400) for e in OC.EVENTS}
            self.teams[t]["rpa"] = (c.get("runs", 0) + 400 * self.lg["rpa"]) / (c["bf"] + 400)
        self.by_pid = {}
        for r in self.train:
            self.by_pid.setdefault(r["pid"], []).append(r)
        self.ptot = {p: _tot(v) for p, v in self.by_pid.items()}
        self.hist = {}
        for r in self.train:
            k = (r["pid"], r["opp"])
            self.hist[k] = OC.add(self.hist.get(k, {}), r)
        self.rho = {e: OC.game_rho(self.train, e) for e in OC.EVENTS}
        self.kp = OC.pitcher_kappa(self.train, self.lg, self.rho)
        self.k1 = {e: OC.kappa(OC.pair_table(self.train, self.teams, self.lg), e)["kappa"] for e in OC.EVENTS}
        pairs2 = OC.pair_table(self.train, self.teams, self.lg, self.kp)
        self.k2 = {e: OC.kappa(pairs2, e, rho_game=self.rho[e]["rho"])["kappa"] for e in OC.EVENTS}
        self.form = OC.form_depth(self.train)

    def rate(self, r, e, version, use_pair=True):
        c = self.ptot[r["pid"]]
        y, n = OC.trials(c, e)
        k0 = OC.pitcher_k0(self.kp, e) if version == 2 else 30.0
        p = OC.odds_ratio((y + k0 * self.lg[e]) / (n + k0), self.teams.get(r["opp"], self.lg)[e], self.lg[e])
        kk = (self.k2 if version == 2 else self.k1)[e]
        h = self.hist.get((r["pid"], r["opp"]))
        if use_pair and h and kk:
            hy, hn = OC.trials(h, e)
            p = OC.shrink_pair(hy, hn, p, kk)["post"]
        return p


# ------------------------------------------------------------------ pruebas

def test_rates(sp: Split) -> dict:
    """A1/A3: log-verosimilitud por intento en las aperturas posteriores al corte."""
    out = {}
    sub = [r for r in sp.test if r["pid"] in sp.ptot]
    with_hist = [r for r in sub if (r["pid"], r["opp"]) in sp.hist]
    for e in OC.EVENTS:
        rows = {}
        for name, a, b in (("v2 − v1", (2, True), (1, True)),
                           ("v1: historial − sin historial", (1, True), (1, False)),
                           ("v2: historial − sin historial", (2, True), (2, False))):
            d = []
            for r in (with_hist if "historial" in name else sub):
                y, n = OC.trials(r, e)
                if n > 0:
                    d.append((_ll(y, n, sp.rate(r, e, *a)) - _ll(y, n, sp.rate(r, e, *b)), n))
            res = _boot([(x * 1000, w) for x, w in d], stat="ratio")
            res["verdict"] = _verdict(res["ci90"])
            rows[name] = res
        out[e] = rows
    return out


def test_kdist(sp: Split) -> dict:
    """K1: distribución de ponches del abridor (con su salida) en las aperturas posteriores al corte."""
    d, calib = [], {"v1": [], "v2": []}
    bf_pred = {"v1": 0.0, "v2": 0.0}
    bf_real = 0
    for r in sp.test:
        tr = sp.by_pid.get(r["pid"])
        if not tr or len(tr) < 6 or not r.get("outs"):
            continue
        outs = [x["outs"] for x in tr if x["outs"]]
        shape, scale = OC.weibull_fit(outs)
        m = sum(outs) / len(outs)
        bpo = sp.ptot[r["pid"]]["bf"] / max(1, sum(outs))
        dist = OC.outs_dist(shape, scale)
        surv1 = [sum(p for k, p in enumerate(dist) if k > int(j / bpo)) for j in range(36)]
        kd1 = OC.k_dist([sp.rate(r, "k", 1)] * 36, surv1)
        kd2 = OC.k_dist_form([sp.rate(r, "k", 2)] * 36, shape, m, bpo, sp.rho["k"]["rho"], sp.form["b"])
        y = r["k"]
        p1 = kd1[y] if y < len(kd1) else 1e-9
        p2 = kd2[y] if y < len(kd2) else 1e-9
        d.append(math.log(max(1e-9, p2)) - math.log(max(1e-9, p1)))
        rng = random.Random(len(d))
        for name, kd in (("v1", kd1), ("v2", kd2)):       # PIT aleatorizado (discreto): uniforme si está calibrado
            below = sum(kd[:y])
            calib[name].append(below + rng.random() * (kd[y] if y < len(kd) else 0.0))
        bf_pred["v1"] += sum(surv1)
        bf_pred["v2"] += sum(sum(p for k, p in enumerate(dist) if k > (j + 0.5) / bpo) for j in range(36))
        bf_real += r["bf"]
    res = _boot(d)
    res["verdict"] = _verdict(res["ci90"])
    n = len(d)
    res["pit"] = {k: {"low10": sum(1 for u in v if u < 0.1) / n, "high10": sum(1 for u in v if u > 0.9) / n} for k, v in calib.items()}
    res["bf"] = {"v1": bf_pred["v1"] / n, "v2": bf_pred["v2"] / n, "real": bf_real / n}
    return res


def test_familiarity(sp: Split) -> dict:
    """D1: ponches con la familiaridad medida en el entrenamiento, fuera de muestra."""
    fam = OC.familiarity(sp.train, OC.pair_table(sp.train, sp.teams, sp.lg, sp.kp))
    slope = (fam.get("k") or {}).get("slope", 0.0)
    seen, prior = {}, {}
    for r in sorted(sp.train + sp.test, key=lambda r: r["date"]):
        k = (r["pid"], r["opp"])
        prior[id(r)] = seen.get(k, 0)
        seen[k] = seen.get(k, 0) + 1
    d = []
    for r in sp.test:
        if r["pid"] not in sp.ptot:
            continue
        p = sp.rate(r, "k", 2)
        pf = min(0.9, p * (1 + slope * min(3, prior[id(r)])))
        d.append(((_ll(r["k"], r["bf"], pf) - _ll(r["k"], r["bf"], p)) * 1000, r["bf"]))
    res = _boot(d, stat="ratio")
    res.update(verdict=_verdict(res["ci90"]), slope=slope, t=(fam.get("k") or {}).get("t"))
    return res


def _hook_rows(post, reg):
    rows = []
    for year, games in (post or {}).items():
        for g in (games or {}).values():
            if not isinstance(g, dict):
                continue
            for side in ("away", "home"):
                t = g.get(side) or {}
                if not t.get("pitchers"):
                    continue
                st = t["pitchers"][0]
                rs = reg.get(f"{year}:{st['id']}") or {}
                gs = rs.get("gamesStarted") or 0
                if gs < 8 or not rs.get("inningsPitched"):
                    continue
                o = OC.outs_of(rs["inningsPitched"])
                rows.append({"year": year, "y": OC.outs_of(st.get("ip")), "x": o / gs, "bfgs": (rs.get("battersFaced") or 0) / gs,
                             "ppo": (rs.get("numberOfPitches") or 0) / max(1, o), "runs": st.get("runs") or 0, "id": st["id"]})
    return rows


def _ols(tr):
    xs, ys = [r["x"] for r in tr], [r["y"] for r in tr]
    mx, my = stt.mean(xs), stt.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def _ts(tr):
    pts = [(r["x"], r["y"]) for r in tr]
    sl = sorted((y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(pts) for (x2, y2) in pts[i + 1:] if abs(x2 - x1) > 1e-9)
    b = sl[len(sl) // 2]
    return stt.median([y - b * x for x, y in pts]), b


def test_hook(post, reg) -> dict:
    """C1/C2: año contra año (entrena un año, califica el otro) con error absoluto y log-score de la distribución."""
    raw = _hook_rows(post, reg)
    clean = [r for r in raw if r["bfgs"] <= 28 and r["x"] <= 21]
    removed = [{"year": r["year"], "id": r["id"], "outsPorApertura": round(r["x"], 1), "outsOctubre": r["y"]}
               for r in raw if r not in clean]
    fits = {
        "media constante (v1)": lambda tr: (lambda r, m=stt.mean(q["y"] for q in tr): m),
        "OLS": lambda tr: (lambda ab: (lambda r: ab[0] + ab[1] * r["x"]))(_ols(tr)),
        "Theil-Sen (v2)": lambda tr: (lambda ab: (lambda r: ab[0] + ab[1] * r["x"]))(_ts(tr)),
        "cociente h·x": lambda tr: (lambda h: (lambda r: h * r["x"]))(math.exp(stt.mean(math.log(max(0.05, q["y"] / q["x"])) for q in tr))),
        "pitcheos por out": lambda tr: (lambda c: (lambda r: c / r["ppo"] if r["ppo"] else 14.0))(
            stt.mean(q["y"] * q["ppo"] for q in tr if q["ppo"])),
    }
    years = sorted({r["year"] for r in clean})
    models = {}
    base_ae, base_ls = None, None
    for name, fit in fits.items():
        ae, ls = [], []
        for yr in years:
            tr = [r for r in clean if r["year"] != yr]
            te = [r for r in clean if r["year"] == yr]
            if not tr or not te:
                continue
            f = fit(tr)
            res = [q["y"] - f(q) for q in tr]
            cv = math.sqrt(sum(x * x for x in res) / max(1, len(res) - 2)) / stt.mean(q["y"] for q in tr)
            shape = max(1.5, min(12.0, cv ** -1.086))
            for r in te:
                mu = max(3.0, f(r))
                d = OC.outs_dist(shape, mu / math.gamma(1 + 1 / shape))
                ae.append(abs(f(r) - r["y"]))
                ls.append(math.log(max(1e-4, d[min(r["y"], len(d) - 1)])))
        models[name] = {"mae": stt.mean(ae), "logScore": stt.mean(ls), "_ls": ls}
        if base_ls is None:
            base_ae, base_ls = ae, ls
    for name, m in models.items():
        m["vsV1"] = _boot([a - b for a, b in zip(m.pop("_ls"), base_ls)])
    bins = []
    for lo, hi in ((0, 15), (15, 16.5), (16.5, 18), (18, 99)):
        ys = [r["y"] for r in clean if lo <= r["x"] < hi]
        if ys:
            bins.append({"temporada": f"{lo}–{hi if hi < 99 else '+'}", "n": len(ys), "mediaOctubre": stt.mean(ys),
                         "p18": sum(1 for y in ys if y >= 18) / len(ys)})
    by_runs = []
    for lo, hi, lab in ((0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 5, "3–4"), (5, 99, "5+")):
        ys = [r["y"] for r in clean if lo <= r["runs"] < hi]
        if ys:
            by_runs.append({"carreras": lab, "n": len(ys), "outs": stt.mean(ys)})
    a, b = _ols(clean)
    a2, b2 = _ts(clean)
    return {"n": len(raw), "nClean": len(clean), "removed": removed, "models": models, "byQuality": bins, "byRuns": by_runs,
            "olsBeta": b, "tsBeta": b2, "v1Beta": _ols([r for r in raw])[1]}


# ------------------------------------------------------------------ el partido

def game_review(v1: dict, v2: dict, pg: dict) -> dict:
    """Lo que predijo cada versión contra lo que pasó (un solo juego: se reporta, no decide)."""
    ls = pg["linescore"]["teams"]
    runs = {"away": ls["away"]["runs"], "home": ls["home"]["runs"]}
    home_won = runs["home"] > runs["away"]
    sp = {}
    for s in ("away", "home"):
        b = pg["box"][s]
        pid = b["pitchers"][0]
        p = b["players"][str(pid)]
        pit = p.get("pit") or {}
        pen = [b["players"][str(x)].get("pit") or {} for x in b["pitchers"][1:]]
        sp[s] = {"name": p["name"], "outs": OC.outs_of(pit.get("inningsPitched")), "k": pit.get("strikeOuts"), "bf": pit.get("battersFaced"),
                 "runs": pit.get("runs"), "pitches": pit.get("numberOfPitches"),
                 "penRuns": sum(x.get("runs") or 0 for x in pen), "penOuts": sum(OC.outs_of(x.get("inningsPitched")) for x in pen)}
    by_inn = [{"n": i["num"], "away": i.get("away"), "home": i.get("home")} for i in pg["linescore"]["innings"]]
    versions = {}
    for name, L in (("v1", v1), ("v2", v2)):
        G, O = L["octubre"]["game"], L["octubre"]
        row = {"pHome": G["pHome"], "ci90": G["ci90"], "lambda": G["lambda"], "logLoss": -math.log(G["pHome"] if home_won else 1 - G["pHome"]),
               "brier": ((1 if home_won else 0) - G["pHome"]) ** 2, "ablation": O["ablation"], "starters": {}}
        for s in ("away", "home"):
            opp = "home" if s == "away" else "away"
            d = O["C"]["detail"][opp]
            kd = G["kDist"].get(s) or []
            st = O["A"]["starters"][s] or {}
            y, o = sp[s]["k"] or 0, sp[s]["outs"]
            row["starters"][s] = {"expOuts": d["expOuts"], "pOutsGE": sum(d["dist"][o:]), "pOuts": d["dist"][o] if o < len(d["dist"]) else 0.0,
                                  "kMean": sum(k * p for k, p in enumerate(kd)), "pK": kd[y] if y < len(kd) else 0.0, "pKGE": sum(kd[y:]),
                                  "kPost": st.get("events", {}).get("k", {}).get("post"), "kW": st.get("events", {}).get("k", {}).get("w"),
                                  "dWobaA": st.get("dWobaA")}
        versions[name] = row
    return {"runs": runs, "homeWon": home_won, "starters": sp, "innings": by_inn, "decisions": pg.get("decisions"), "versions": versions}


def run(pk: int, cuts=CUTS) -> dict:
    snap = json.load(gzip.open(os.path.join(DIR, f"snapshot_{pk}_pre.json.gz"), "rt"))
    rows = OC.start_rows(snap.get("octGameLogs"))
    out = {"pk": pk, "model": "OCTUBRE", "builtAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "algorithm": VERSION, "starts": len(rows), "cuts": {}}
    for cut in cuts:
        sp = Split(rows, cut)
        out["cuts"][cut] = {"train": len(sp.train), "test": len(sp.test),
                            "rhoDay": {e: v["rho"] for e, v in sp.rho.items()},
                            "kappaPitcher": {e: v["kappa"] for e, v in sp.kp.items()},
                            "kappaPairV1": sp.k1, "kappaPairV2": sp.k2, "form": sp.form,
                            "rates": test_rates(sp), "kdist": test_kdist(sp), "familiarity": test_familiarity(sp)}
    out["hook"] = test_hook(snap.get("octPost"), snap.get("octPostReg") or {})
    pg_path = os.path.join(DIR, f"postgame_{pk}.json.gz")
    lab_path = os.path.join(DIR, f"prolab_{pk}.json")
    if os.path.exists(pg_path) and os.path.exists(lab_path):
        from . import prolab as PL
        with open(lab_path, encoding="utf-8") as f:
            v1 = json.load(f)
        v2 = PL.run_octubre(pk, "pre")           # la versión nueva con los MISMOS datos previos (sin fuga)
        with gzip.open(pg_path, "rt") as f:
            pg = json.load(f)
        out["game"] = game_review(v1, v2, pg)
    return out


def main():
    ap = argparse.ArgumentParser(description="Auditoría fuera de muestra de un modelo del Pro-Lab después de su partido")
    ap.add_argument("--pk", type=int, required=True)
    args = ap.parse_args()
    res = run(args.pk)
    path = os.path.join(DIR, f"auditoria_{args.pk}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("auditoría →", path)


if __name__ == "__main__":
    main()
