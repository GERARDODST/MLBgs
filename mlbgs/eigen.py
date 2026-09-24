"""EIGEN: análisis de componentes principales (PCA) para un partido de la MLB.

Pregunta de fondo: ¿dónde sirve PCA en béisbol? Se prueba en cuatro niveles con los datos de 2026 y se
mide qué tan estables y útiles son los componentes en cada uno:

  1. Pitchers (~400 con ≥ 120 bateadores enfrentados): K%, BB%, HR%, GB%, barrel%, hard-hit%, velocidad
     de salida, whiff% y dependencia de la recta. Muchas métricas correlacionadas y muestras chicas por
     pitcher: PCA separa pocos ejes de habilidad (dominio, contacto, control) y la regresión sobre
     componentes (PCR) da una carrera por 9 «de habilidad» mucho más estable que la ERA.
  2. Bateadores (~350 con ≥ 150 turnos): K%, BB%, ISO, HR%, barrel%, hard-hit%, velocidad de salida y
     robos. PCR sobre los componentes estima el wOBA de habilidad de cada bateador del lineup.
  3. Equipos (30): con 30 casos y ~14 métricas los componentes son inestables (bootstrap); sirve para
     describir estilos, no para predecir.
  4. Métodos del framework (Log5, Elo, λ): casi todo cae en un solo componente → no son tres opiniones
     independientes, y el consenso del índice de confianza lo debe tomar en cuenta.

Modelo del partido: carreras esperadas por equipo = carreras de liga × ofensiva del lineup (wOBA de
habilidad por turno) × pitcheo rival (abridor por sus entradas esperadas + bullpen) × parque; marcador
con Binomial Negativa (sobredispersión medida en la liga), F5 y primera entrada con la misma cadena,
e intervalo de P(victoria) por bootstrap de jugadores (se re-estima PCA + PCR en cada réplica).

PCA hecho a mano (Jacobi para matrices simétricas): el proyecto no usa dependencias externas.
"""
from __future__ import annotations

import math
import random

from . import mathlib as M

# pesos lineales de wOBA (escala FanGraphs moderna) y escala para convertir a carreras por turno
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.69, 0.72, 0.88, 1.25, 1.58, 2.03
WOBA_SCALE = 1.23
SLOT_PA = [4.65, 4.55, 4.43, 4.33, 4.24, 4.13, 4.03, 3.93, 3.83]   # turnos esperados por posición en el orden

P_FEATS = [("k", "K%"), ("bb", "BB%"), ("hr", "HR%"), ("gb", "Rodados %"), ("barrel", "Barrel %"),
           ("hardhit", "Hard-hit %"), ("ev", "Velocidad de salida"), ("whiff", "Whiff %"), ("fb", "Uso de rectas")]
B_FEATS = [("k", "K%"), ("bb", "BB%"), ("iso", "ISO"), ("hr", "HR%"), ("barrel", "Barrel %"),
           ("hardhit", "Hard-hit %"), ("ev", "Velocidad de salida"), ("sb", "Robos por turno")]
T_FEATS = [("rpg", "Carreras anotadas/J"), ("obp", "OBP"), ("slg", "SLG"), ("kO", "K% ofensiva"), ("bbO", "BB% ofensiva"),
           ("sbg", "Robos/J"), ("rag", "Carreras permitidas/J"), ("spRa9", "RA9 abridores"), ("spK", "K% abridores"),
           ("spBB", "BB% abridores"), ("rpRa9", "RA9 bullpen"), ("rpK", "K% bullpen"), ("hr9", "HR/9 permitidos"), ("gb", "Rodados % pitcheo")]
FASTBALLS = {"FF", "SI", "FC", "FA"}


# ------------------------------------------------------------------ álgebra

def ip_float(ip) -> float:
    return M.ip_to_float(ip)


def wmean(xs, w):
    s = sum(w)
    return sum(x * wi for x, wi in zip(xs, w)) / s


def standardize(X, w):
    cols = list(zip(*X))
    mu = [wmean(c, w) for c in cols]
    sd = [math.sqrt(max(1e-12, wmean([(x - m) ** 2 for x in c], w))) for c, m in zip(cols, mu)]
    Z = [[(x - m) / s for x, m, s in zip(row, mu, sd)] for row in X]
    return Z, mu, sd


def corr(Z, w):
    p = len(Z[0])
    s = sum(w)
    return [[sum(r[i] * r[j] * wi for r, wi in zip(Z, w)) / s for j in range(p)] for i in range(p)]


def jacobi(A, sweeps=100, tol=1e-12):
    """Valores y vectores propios de una matriz simétrica (vectores en columnas), ordenados de mayor a menor."""
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    vals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    vecs = [[v[r][i] for i in order] for r in range(n)]
    vals = [vals[i] for i in order]
    # signo: la carga más grande de cada componente, positiva
    for j in range(n):
        big = max(range(n), key=lambda r: abs(vecs[r][j]))
        if vecs[big][j] < 0:
            for r in range(n):
                vecs[r][j] = -vecs[r][j]
    return vals, vecs


def solve(A, b):
    n = len(A)
    m = [row[:] + [bi] for row, bi in zip(A, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            m[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def wls(X, y, w, ridge=1e-6):
    """Mínimos cuadrados ponderados con intercepto. X sin columna de unos."""
    Xa = [[1.0] + list(r) for r in X]
    p = len(Xa[0])
    A = [[sum(r[i] * r[j] * wi for r, wi in zip(Xa, w)) + (ridge if i == j and i else 0) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * yi * wi for r, yi, wi in zip(Xa, y, w)) for i in range(p)]
    return solve(A, b)


def predict(beta, x):
    return beta[0] + sum(b * v for b, v in zip(beta[1:], x))


# ------------------------------------------------------------------ PCA + PCR

def pca(X, w):
    Z, mu, sd = standardize(X, w)
    R = corr(Z, w)
    vals, vecs = jacobi(R)
    tot = sum(vals)
    scores = [[sum(z[i] * vecs[i][j] for i in range(len(z))) for j in range(len(vals))] for z in Z]
    return {"mu": mu, "sd": sd, "corr": R, "vals": vals, "vecs": vecs, "explained": [v / tot for v in vals], "scores": scores}


def project(P, x, k):
    z = [(v - m) / s for v, m, s in zip(x, P["mu"], P["sd"])]
    return [sum(z[i] * P["vecs"][i][j] for i in range(len(z))) for j in range(k)]


def kfold_r2(X, y, w, fit_fn, folds=10, seed=7):
    """R² fuera de muestra (validación cruzada ponderada)."""
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    pred = [0.0] * len(y)
    for f in range(folds):
        test = set(idx[f::folds])
        tr = [i for i in idx if i not in test]
        model = fit_fn([X[i] for i in tr], [y[i] for i in tr], [w[i] for i in tr])
        for i in test:
            pred[i] = model(X[i])
    ym = wmean(y, w)
    sse = sum(wi * (yi - pi) ** 2 for yi, pi, wi in zip(y, pred, w))
    sst = sum(wi * (yi - ym) ** 2 for yi, wi in zip(y, w))
    return 1 - sse / sst


def pcr_fit(k):
    def fit(X, y, w):
        P = pca(X, w)
        beta = wls([s[:k] for s in P["scores"]], y, w)
        return lambda x: predict(beta, project(P, x, k))
    return fit


def ols_fit(cols):
    def fit(X, y, w):
        beta = wls([[r[c] for c in cols] for r in X], y, w)
        return lambda x: predict(beta, [x[c] for c in cols])
    return fit


def choose_k(X, y, w, kmax):
    r2 = [kfold_r2(X, y, w, pcr_fit(k)) for k in range(1, kmax + 1)]
    best = max(range(len(r2)), key=lambda i: r2[i])
    # el más simple que queda a 0.005 del mejor
    k = next(i for i in range(len(r2)) if r2[i] >= r2[best] - 0.005) + 1
    return k, r2


def stability(X, w, k, B=120, seed=11):
    """Bootstrap: |coseno| entre las cargas de cada réplica y las de la muestra completa (1 = idénticas)."""
    rng = random.Random(seed)
    P = pca(X, w)
    n = len(X)
    acc = [0.0] * k
    for _ in range(B):
        ids = [rng.randrange(n) for _ in range(n)]
        Q = pca([X[i] for i in ids], [w[i] for i in ids])
        for j in range(k):
            acc[j] += abs(sum(P["vecs"][r][j] * Q["vecs"][r][j] for r in range(len(P["vecs"]))))
    return [a / B for a in acc]


def label(vecs, j, names, top=3):
    """Nombre corto del componente: las variables que más pesan, con su signo."""
    ld = sorted(((vecs[i][j], names[i]) for i in range(len(names))), key=lambda t: -abs(t[0]))[:top]
    return " · ".join(f"{'+' if v > 0 else '−'}{n}" for v, n in ld)


def summary(X, w, names, y, kmax, B=120):
    P = pca(X, w)
    k, r2 = choose_k(X, y, w, kmax)
    beta = wls([s[:k] for s in P["scores"]], y, w)
    stab = stability(X, w, min(kmax, len(names)), B)
    return P, k, r2, beta, stab, {
        "features": names, "n": len(X), "explained": P["explained"], "eig": P["vals"],
        "loadings": [[P["vecs"][i][j] for j in range(len(names))] for i in range(len(names))],
        "labels": [label(P["vecs"], j, names) for j in range(len(names))],
        "corr": P["corr"], "k": k, "cvR2": r2, "stability": stab, "beta": beta, "mu": P["mu"], "sd": P["sd"],
    }


# ------------------------------------------------------------------ datos de jugadores

def pitcher_rows(bundle, min_bf=120):
    sv, ars = bundle["savant"], bundle.get("arsenal", {}).get("pitcher", {})
    rows = []
    for p in bundle["playersPitching"]:
        s = p["stat"]
        bf = s.get("battersFaced") or 0
        ip = ip_float(s.get("inningsPitched", "0"))
        q = sv["pitcher"].get(str(p["id"]))
        if bf < min_bf or ip < 20 or not q or q.get("barrel") is None:
            continue
        a = ars.get(str(p["id"])) or []
        use = sum(x.get("usage") or 0 for x in a) or 1
        whiff = sum((x.get("usage") or 0) * (x.get("whiff") or 0) for x in a) / use if a else None
        fb = sum(x.get("usage") or 0 for x in a if x.get("type") in FASTBALLS) / use * 100 if a else None
        go, ao = s.get("groundOuts") or 0, s.get("airOuts") or 0
        feats = {"k": 100 * (s.get("strikeOuts") or 0) / bf, "bb": 100 * ((s.get("baseOnBalls") or 0) + (s.get("hitByPitch") or 0)) / bf,
                 "hr": 100 * (s.get("homeRuns") or 0) / bf, "gb": 100 * go / max(1, go + ao), "barrel": q["barrel"],
                 "hardhit": q["hardhit"], "ev": q["ev"], "whiff": whiff, "fb": fb}
        if any(v is None for v in feats.values()):
            continue
        rows.append({"id": p["id"], "name": p["name"], "team": p["team"], "bf": bf, "ip": ip, "gs": s.get("gamesStarted") or 0,
                     "g": s.get("gamesPlayed") or 0, "ra9": 9 * (s.get("runs") or 0) / ip, "era": 9 * (s.get("earnedRuns") or 0) / ip,
                     "x": [feats[k] for k, _ in P_FEATS], "feats": feats})
    return rows


def woba(s):
    h, d2, d3, hr = s.get("hits") or 0, s.get("doubles") or 0, s.get("triples") or 0, s.get("homeRuns") or 0
    bb, ibb, hbp = s.get("baseOnBalls") or 0, s.get("intentionalWalks") or 0, s.get("hitByPitch") or 0
    den = (s.get("atBats") or 0) + bb - ibb + (s.get("sacFlies") or 0) + hbp
    if den <= 0:
        return None
    num = W_BB * (bb - ibb) + W_HBP * hbp + W_1B * (h - d2 - d3 - hr) + W_2B * d2 + W_3B * d3 + W_HR * hr
    return num / den


def batter_feats(s, q):
    pa, ab = s.get("plateAppearances") or 0, s.get("atBats") or 0
    if not pa or not ab or not q or q.get("barrel") is None:
        return None
    tb = s.get("totalBases") or 0
    return {"k": 100 * (s.get("strikeOuts") or 0) / pa, "bb": 100 * ((s.get("baseOnBalls") or 0) + (s.get("hitByPitch") or 0)) / pa,
            "iso": (tb - (s.get("hits") or 0)) / ab, "hr": 100 * (s.get("homeRuns") or 0) / pa, "barrel": q["barrel"],
            "hardhit": q["hardhit"], "ev": q["ev"], "sb": 100 * (s.get("stolenBases") or 0) / pa}


def batter_rows(bundle, min_pa=150):
    sv = bundle["savant"]["batter"]
    rows = []
    for p in bundle["playersHitting"]:
        s = p["stat"]
        pa = s.get("plateAppearances") or 0
        if pa < min_pa:
            continue
        f = batter_feats(s, sv.get(str(p["id"])))
        wo = woba(s)
        if not f or wo is None:
            continue
        rows.append({"id": p["id"], "name": p["name"], "team": p["team"], "pa": pa, "woba": wo, "x": [f[k] for k, _ in B_FEATS], "feats": f})
    return rows


def team_rows(bundle):
    rows = []
    for tid, t in bundle["teamStats"].items():
        h, sp, rp, pt = t.get("hitting") or {}, t.get("sp") or {}, t.get("rp") or {}, t.get("pitching") or {}
        g = h.get("gamesPlayed") or pt.get("gamesPlayed") or 0
        pa = h.get("plateAppearances") or 0
        if not g or not pa:
            continue
        ipsp, iprp = ip_float(sp.get("inningsPitched", "0")), ip_float(rp.get("inningsPitched", "0"))
        bfsp, bfrp = sp.get("battersFaced") or 1, rp.get("battersFaced") or 1
        go, ao = pt.get("groundOuts") or 0, pt.get("airOuts") or 0
        st = (bundle.get("standings") or {}).get(str(tid)) or {}
        w, l = st.get("w") or st.get("wins"), st.get("l") or st.get("losses")
        feats = {"rpg": (h.get("runs") or 0) / g, "obp": float(h.get("obp") or 0), "slg": float(h.get("slg") or 0),
                 "kO": 100 * (h.get("strikeOuts") or 0) / pa, "bbO": 100 * (h.get("baseOnBalls") or 0) / pa, "sbg": (h.get("stolenBases") or 0) / g,
                 "rag": (pt.get("runs") or 0) / g, "spRa9": 9 * (sp.get("runs") or 0) / max(1, ipsp), "spK": 100 * (sp.get("strikeOuts") or 0) / bfsp,
                 "spBB": 100 * (sp.get("baseOnBalls") or 0) / bfsp, "rpRa9": 9 * (rp.get("runs") or 0) / max(1, iprp),
                 "rpK": 100 * (rp.get("strikeOuts") or 0) / bfrp, "hr9": 9 * (pt.get("homeRuns") or 0) / max(1, ipsp + iprp),
                 "gb": 100 * go / max(1, go + ao)}
        wp = (w / (w + l)) if w is not None and l else None
        rows.append({"id": int(tid), "x": [feats[k] for k, _ in T_FEATS], "feats": feats, "wp": wp})
    return rows


# ------------------------------------------------------------------ ajuste completo y talento de cada jugador

K_PIT, K_BAT = 600, 300        # peso (BF / PA) del estimado por componentes frente a lo observado


def league(bundle):
    H = {"pa": 0, "runs": 0, "ab": 0, "bb": 0, "ibb": 0, "hbp": 0, "sf": 0, "h": 0, "d2": 0, "d3": 0, "hr": 0}
    ip = runs = rp_ip = rp_runs = 0.0
    for t in bundle["teamStats"].values():
        h = t.get("hitting") or {}
        for k, src in (("pa", "plateAppearances"), ("runs", "runs"), ("ab", "atBats"), ("bb", "baseOnBalls"), ("ibb", "intentionalWalks"),
                       ("hbp", "hitByPitch"), ("sf", "sacFlies"), ("h", "hits"), ("d2", "doubles"), ("d3", "triples"), ("hr", "homeRuns")):
            H[k] += h.get(src) or 0
        p, r = t.get("pitching") or {}, t.get("rp") or {}
        ip += ip_float(p.get("inningsPitched", "0"))
        runs += p.get("runs") or 0
        rp_ip += ip_float(r.get("inningsPitched", "0"))
        rp_runs += r.get("runs") or 0
    lw = woba({"hits": H["h"], "doubles": H["d2"], "triples": H["d3"], "homeRuns": H["hr"], "baseOnBalls": H["bb"],
               "intentionalWalks": H["ibb"], "hitByPitch": H["hbp"], "atBats": H["ab"], "sacFlies": H["sf"]})
    return {"woba": lw, "rpa": H["runs"] / H["pa"], "ra9": 9 * runs / ip, "rpRa9": 9 * rp_runs / rp_ip}


def fit_all(bundle, B=120):
    P, Bt, T = pitcher_rows(bundle), batter_rows(bundle), team_rows(bundle)
    pn, bn, tn = [n for _, n in P_FEATS], [n for _, n in B_FEATS], [n for _, n in T_FEATS]
    Xp, wp, yp = [r["x"] for r in P], [r["bf"] for r in P], [r["ra9"] for r in P]
    Xb, wb, yb = [r["x"] for r in Bt], [r["pa"] for r in Bt], [r["woba"] for r in Bt]
    T = [t for t in T if t["wp"] is not None]
    Xt, wt, yt = [t["x"] for t in T], [1.0] * len(T), [t["wp"] for t in T]
    pP, pk, _, pbeta, _, pS = summary(Xp, wp, pn, yp, len(pn), B)
    bP, bk, _, bbeta, _, bS = summary(Xb, wb, bn, yb, len(bn), B)
    tP, tk, _, tbeta, _, tS = summary(Xt, wt, tn, yt, 6, B)
    # referencias sin PCA: regresión con todas las métricas y con las más usadas
    pS["olsAll"] = kfold_r2(Xp, yp, wp, ols_fit(list(range(len(pn)))))
    pS["olsBase"] = kfold_r2(Xp, yp, wp, ols_fit([0, 1, 2]))            # K%, BB%, HR% (la idea del FIP)
    bS["olsAll"] = kfold_r2(Xb, yb, wb, ols_fit(list(range(len(bn)))))
    bS["olsBase"] = kfold_r2(Xb, yb, wb, ols_fit([0, 1, 2]))            # K%, BB%, ISO
    tS["olsAll"] = kfold_r2(Xt, yt, wt, ols_fit(list(range(len(tn)))), folds=len(T))
    tS["olsBase"] = kfold_r2(Xt, yt, wt, ols_fit([0, 6]), folds=len(T))   # carreras anotadas y permitidas (Pitágoras)
    tS["cvR2"] = [kfold_r2(Xt, yt, wt, pcr_fit(k), folds=len(T)) for k in range(1, 7)]
    ids = {int(t["id"]): t for t in T}
    tS["teams"] = [{"id": t["id"], "wp": t["wp"], "scores": s[:3]} for t, s in zip(T, tP["scores"])]
    return {"pitchers": {"rows": P, "pca": pP, "k": pk, "beta": pbeta, "summary": pS},
            "batters": {"rows": Bt, "pca": bP, "k": bk, "beta": bbeta, "summary": bS},
            "teams": {"rows": ids, "pca": tP, "k": tk, "beta": tbeta, "summary": tS},
            "lg": league(bundle)}


def pitcher_talent(pid, bundle, F, lg):
    pid = int(pid)
    row = next((r for r in F["pitchers"]["rows"] if r["id"] == pid), None)
    if row is None:   # muestra chica: se proyecta igual si hay datos mínimos
        tmp = pitcher_rows({**bundle, "playersPitching": [p for p in bundle["playersPitching"] if p["id"] == pid]}, min_bf=25)
        row = tmp[0] if tmp else None
    if row is None:
        return {"id": pid, "bf": 0, "ra9Obs": None, "fit": lg["ra9"] + 0.3, "talent": lg["ra9"] + 0.3, "scores": [0, 0, 0], "ipStart": 5.0}
    Pm = F["pitchers"]
    sc = project(Pm["pca"], row["x"], Pm["k"])
    fit = predict(Pm["beta"], sc)
    talent = (row["bf"] * row["ra9"] + K_PIT * fit) / (row["bf"] + K_PIT)
    ip_start = row["ip"] / row["gs"] if row["gs"] >= 3 else 5.0
    return {"id": pid, "name": row["name"], "bf": row["bf"], "ra9Obs": row["ra9"], "era": row["era"], "fit": fit, "talent": talent,
            "scores": project(Pm["pca"], row["x"], 3), "feats": row["feats"], "ipStart": min(6.5, max(4.0, ip_start))}


def batter_talent(pid, bundle, F, lg):
    pid = int(pid)
    Bm = F["batters"]
    row = next((r for r in Bm["rows"] if r["id"] == pid), None)
    if row is None:
        p = next((x for x in bundle["playersHitting"] if x["id"] == pid), None)
        s = (p or {}).get("stat") or {}
        f = batter_feats(s, bundle["savant"]["batter"].get(str(pid))) if (s.get("plateAppearances") or 0) >= 30 else None
        if f:
            row = {"id": pid, "name": p["name"], "pa": s["plateAppearances"], "woba": woba(s) or lg["woba"], "x": [f[k] for k, _ in B_FEATS], "feats": f}
    if row is None:
        return {"id": pid, "pa": 0, "wobaObs": None, "fit": lg["woba"] - 0.015, "talent": lg["woba"] - 0.015, "scores": [0, 0, 0]}
    sc = project(Bm["pca"], row["x"], Bm["k"])
    fit = predict(Bm["beta"], sc)
    talent = (row["pa"] * row["woba"] + K_BAT * fit) / (row["pa"] + K_BAT)
    return {"id": pid, "name": row["name"], "pa": row["pa"], "wobaObs": row["woba"], "fit": fit, "talent": talent,
            "scores": project(Bm["pca"], row["x"], 3), "feats": row["feats"]}


def platoon(bat, throw):
    if not bat or not throw:
        return 0.0
    if bat == "S":
        return 0.004
    return 0.008 if bat != throw else -0.008


# ------------------------------------------------------------------ partido

def team_side(lineup_ids, opp_sp, opp_pen_ra9, bundle, F, lg, hands, bats, jitter=None):
    bt = [batter_talent(i, bundle, F, lg) for i in lineup_ids]
    if jitter:
        bt = [{**b, "talent": b["talent"] + jitter(b)} for b in bt]
    thr = hands.get(str(opp_sp["id"])) if opp_sp.get("id") else None
    for b in bt:
        b["bats"] = bats.get(str(b["id"]))
        b["platoon"] = platoon(b["bats"], thr)
    w = SLOT_PA[:len(bt)] or [1]
    wo_sp = sum(wi * (b["talent"] + b["platoon"]) for wi, b in zip(w, bt)) / sum(w)
    wo_bp = sum(wi * b["talent"] for wi, b in zip(w, bt)) / sum(w)
    off = lambda wo: (lg["rpa"] + (wo - lg["woba"]) / WOBA_SCALE) / lg["rpa"]  # noqa: E731
    return {"batters": bt, "wobaVsSp": wo_sp, "wobaVsPen": wo_bp, "oSp": off(wo_sp), "oPen": off(wo_bp),
            "dSp": opp_sp["talent"] / lg["ra9"], "dPen": opp_pen_ra9 / lg["ra9"], "fracSp": opp_sp["ipStart"] / 9}


def lambdas(side_eff, rpg, park, inn_share):
    """Carreras esperadas: juego completo, primeras 5 y primera entrada."""
    e = side_eff
    full = rpg * park * (e["fracSp"] * e["oSp"] * e["dSp"] + (1 - e["fracSp"]) * e["oPen"] * e["dPen"])
    f5sp = min(1.0, e["fracSp"] * 9 / 5)
    f5 = rpg * inn_share["f5"] * park * (f5sp * e["oSp"] * e["dSp"] + (1 - f5sp) * e["oPen"] * e["dPen"])
    first = rpg * inn_share["first"] * park * e["oSp"] * e["dSp"]
    return full, f5, first


def outcome(la, lh, var_ratio, x_home):
    pa, ph = M.negbin_pmf(la, var_ratio), M.negbin_pmf(lh, var_ratio)
    _, tie, home = M.outcome_probs(pa, ph)          # (P(visita > local), P(empate), P(local > visita))
    return pa, ph, home + tie * x_home, tie


def var_ratio_f5(results):
    """Var/Media de las carreras de cada equipo en las primeras 5 entradas (medida en la liga)."""
    xs = [sum((x[i] or 0) for x in g["inn"][:5]) for g in results if len(g.get("inn") or []) >= 5 for i in (0, 1)]
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return v / m, m


def methods_pca(rows):
    """PCA de los tres métodos del framework (Log5, Elo, λ) sobre las predicciones guardadas."""
    X = [[r["methods"]["log5"], r["methods"]["elo"], r["methods"]["lambda"]] for r in rows
         if all(r.get("methods", {}).get(k) is not None for k in ("log5", "elo", "lambda"))]
    if len(X) < 8:
        return None
    P = pca(X, [1.0] * len(X))
    return {"n": len(X), "explained": P["explained"], "corr": P["corr"], "loadings": P["vecs"]}


# ------------------------------------------------------------------ conclusiones: dónde sirve PCA y lectura del partido

def _pct(x):
    return f"{100 * x:.0f}%"


def where_verdict(pS, bS, tS, meth):
    """Evalúa cada nivel con sus propios números: estabilidad de los componentes y valor predictivo fuera de muestra."""
    out = []
    for key, S, what, use in (
        ("pitchers", pS, "Pitchers", "estimar la calidad del abridor (carreras por 9 de habilidad) y leer su estilo: contacto que permite, dominio (whiff/K) y control"),
        ("batters", bS, "Bateadores", "estimar el wOBA de habilidad de cada bateador del lineup (poder, contacto/disciplina, velocidad) y su choque con el abridor"),
    ):
        k = S["k"]
        stab = S["stability"]
        stable = sum(1 for x in stab[:4] if x >= 0.9)
        out.append({"level": what, "key": key, "n": S["n"], "k": k, "pcr": S["cvR2"][k - 1], "ols": S["olsAll"], "base": S["olsBase"],
                    "explained3": sum(S["explained"][:3]), "stability": stab[:4], "stable": stable,
                    "verdict": f"{S['n']} jugadores: los primeros {stable} componentes son estables en bootstrap (≥ 0.90) y resumen {_pct(sum(S['explained'][:3]))} de la variación con 3 ejes. "
                               f"Regresión sobre {k} componentes: R² fuera de muestra {S['cvR2'][k - 1]:.2f} (todas las métricas {S['olsAll']:.2f}; solo las 3 clásicas {S['olsBase']:.2f}).",
                    "use": use})
    stab = tS["stability"]
    kt = tS["k"]
    out.append({"level": "Equipos", "key": "teams", "n": tS["n"], "k": kt, "pcr": tS["cvR2"][kt - 1], "ols": tS["olsAll"], "base": tS["olsBase"],
                "explained3": sum(tS["explained"][:3]), "stability": stab[:4], "stable": sum(1 for x in stab[:4] if x >= 0.9),
                "verdict": f"Solo {tS['n']} equipos para {len(tS['features'])} métricas: el componente 1 es estable ({stab[0]:.2f}) pero del 3 en adelante ya no ({', '.join(f'{x:.2f}' for x in stab[2:4])}). "
                           f"Para explicar el win% la regresión sobre componentes da R² {tS['cvR2'][kt - 1]:.2f}, contra {tS['olsBase']:.2f} con solo carreras anotadas y permitidas.",
                "use": "describir estilos (pitcheo vs ofensiva, contacto vs poder); para pronosticar alcanza con las carreras anotadas y permitidas"})
    if meth:
        out.append({"level": "Métodos del framework", "key": "methods", "n": meth["n"], "explained3": meth["explained"][0],
                    "verdict": f"En {meth['n']} predicciones guardadas, el primer componente de Log5, Elo y λ explica {_pct(meth['explained'][0])} de su variación: casi son una sola opinión.",
                    "use": "corregir el «consenso» del índice de confianza: tres métodos que coinciden no valen como tres confirmaciones independientes"})
    order = {"pitchers": 1, "batters": 2, "methods": 3, "teams": 4}
    for o in out:
        o["rank"] = order[o["key"]]
    return sorted(out, key=lambda o: o["rank"])


def reading(base, eig):
    t = base["teams"]
    a, h = t["away"]["abbr"], t["home"]["abbr"]
    g, sp, lu = eig["game"], eig["starters"], eig["lineups"]
    rows = []
    for s, opp in (("away", "home"), ("home", "away")):
        p = sp[s]
        if p.get("name"):
            rows.append({"k": f"Abridor de {t[s]['abbr']}: {p['name']}",
                         "v": f"carreras por 9 observadas {p['ra9Obs']:.2f} (ERA {p['era']:.2f}) en {p['bf']} bateadores; por componentes {p['fit']:.2f}; habilidad estimada {p['talent']:.2f} "
                              f"(liga {eig['league']['ra9']:.2f}). Se espera que cubra ~{p['ipStart']:.1f} entradas.",
                         "src": "PCA + PCR de pitchers"})
    for s in ("away", "home"):
        l = lu[s]
        rows.append({"k": f"Lineup de {t[s]['abbr']} ({l['status'].lower()})",
                     "v": f"wOBA de habilidad {l['wobaVsSp']:.3f} contra el abridor (con ventaja/desventaja de mano) y {l['wobaVsPen']:.3f} contra el bullpen; liga {eig['league']['woba']:.3f}.",
                     "src": "PCA + PCR de bateadores"})
    rows.append({"k": "Carreras esperadas", "v": f"{a} {g['lambda']['away']:.2f} – {g['lambda']['home']:.2f} {h}; primeras 5: {g['f5']['away']:.2f} – {g['f5']['home']:.2f}; parque ×{g['park']['mult']:.3f}.",
                 "src": "lineup × pitcheo rival × parque"})
    rows.append({"k": f"P({h} gana)", "v": f"{_pct(g['pHome'])} (intervalo 90% por bootstrap de jugadores {_pct(g['ci90'][0])}–{_pct(g['ci90'][1])}); sin PCA sería {_pct(eig['ablation']['pHome'])}.",
                 "src": "Binomial Negativa + bootstrap"})
    return rows
