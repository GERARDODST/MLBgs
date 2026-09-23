"""PRISMA: Posterior de caRreras con Regresión jerárquica, Integración por Simulación, Muestreo y Actualización.

Segundo modelo del Pro-Lab, centrado en probabilidad avanzada (secciones 5.7.4-5.7.9 del framework):

1. Modelo jerárquico bayesiano de ataque y defensa (GLM Poisson con exposición):
       log μ = log(entradas bateadas) + log(park) + α + h·local + ataque_ofensiva − defensa_rival
   con priors normales cuyo σ se estima por Bayes empírico (EM con aproximación de Laplace) y
   decaimiento temporal de la verosimilitud (vida media de 60 días; temporada anterior ×0.35).
2. Posterior por aproximación de Laplace: θ ~ N(θ̂, (−H)⁻¹); muestras con Cholesky.
3. Predictiva posterior: Poisson–lognormal–gamma con fragilidad compartida del partido (clima,
   umpire, contexto) y fragilidad individual, ambas estimadas por momentos con los residuos.
4. Incertidumbre de segundo orden: distribución de P(victoria) sobre la posterior → intervalo creíble y
   P(valor) = P(p_real > p_implícita del momio).
5. Probabilidad de victoria en vivo: matriz entrada × diferencia de carreras por programación dinámica
   sobre la distribución de carreras por media entrada.
6. Validación fuera de muestra (1-22 sep) contra Log5 y contra "siempre el local".
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from . import mathlib as M
from .markov import solve

LN2 = math.log(2)


# ============================================================ álgebra lineal mínima

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-12))
            else:
                L[i][j] = s / L[j][j]
    return L


def inverse_spd(A):
    n = len(A)
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ============================================================ datos

HALF_LIFE = None       # sin decaimiento: elegido por validación fuera de muestra (la forma reciente es ruido)
PREV_W = 1.0           # temporada anterior con peso completo (también por validación)
GRID = [(30, 0.35), (60, 0.35), (120, 0.35), (240, 0.35), (None, 0.35), (60, 1.0), (240, 1.0), (None, 1.0)]


def observations(results, prev, parks, today, half_life=HALF_LIFE, prev_w=PREV_W, until=None):
    """Dos observaciones por juego (cada equipo bateando), con exposición = medias entradas bateadas."""
    obs = []
    for season_w, games in ((1.0, results), (prev_w, prev)):
        for g in games:
            if until and g["date"] >= until:
                continue
            inn = g.get("inn") or []
            n_a = sum(1 for x in inn if x[0] is not None) or 9
            n_h = sum(1 for x in inn if x[1] is not None) or 8.5
            days = max(0, (today - _d(g["date"])).days)
            w = season_w * (math.exp(-LN2 * days / half_life) if half_life else 1.0)
            pk = math.log(parks.get(g.get("venue"), 1.0))
            obs.append((g["away"], g["home"], 0, g["ar"], n_a, w, pk, g["pk"]))
            obs.append((g["home"], g["away"], 1, g["hr"], n_h, w, pk, g["pk"]))
    return obs


def _d(s):
    import datetime as dt
    return dt.date.fromisoformat(s)


# ============================================================ ajuste MAP + Bayes empírico

class Fit:
    pass


def fit(obs, teams, iters=25, eb_rounds=6):
    idx = {t: i for i, t in enumerate(sorted(teams))}
    nT = len(idx)
    P = 2 + 2 * nT                       # α, h, ataque[nT], defensa[nT]
    theta = [0.0] * P
    tot_w = sum(o[5] for o in obs)
    theta[0] = math.log(sum(o[5] * o[3] for o in obs) / sum(o[5] * o[4] for o in obs))
    sa2 = sd2 = 0.01
    H = None
    for _ in range(eb_rounds):
        for _ in range(iters):
            g = [0.0] * P
            H = [[0.0] * P for _ in range(P)]
            for off, dfn, home, y, expo, w, pk, _pk in obs:
                ia, idf = 2 + idx[off], 2 + nT + idx[dfn]
                eta = math.log(expo) + pk + theta[0] + theta[1] * home + theta[ia] - theta[idf]
                mu = math.exp(eta)
                r = w * (y - mu)
                xs = ((0, 1.0), (1, float(home)), (ia, 1.0), (idf, -1.0))
                for k, xk in xs:
                    if xk:
                        g[k] += r * xk
                wm = w * mu
                for k, xk in xs:
                    if not xk:
                        continue
                    for l, xl in xs:
                        if xl:
                            H[k][l] -= wm * xk * xl
            for i in range(nT):
                g[2 + i] -= theta[2 + i] / sa2
                g[2 + nT + i] -= theta[2 + nT + i] / sd2
                H[2 + i][2 + i] -= 1 / sa2
                H[2 + nT + i][2 + nT + i] -= 1 / sd2
            H[0][0] -= 1e-6
            H[1][1] -= 1e-6
            negH = [[-v for v in row] for row in H]
            step = solve(negH, g)
            theta = [t + s for t, s in zip(theta, step)]
            if max(abs(s) for s in step) < 1e-7:
                break
        cov = inverse_spd([[-v for v in row] for row in H])
        att = [theta[2 + i] for i in range(nT)]
        dfn = [theta[2 + nT + i] for i in range(nT)]
        sa2 = max(1e-4, sum(a * a + cov[2 + i][2 + i] for i, a in enumerate(att)) / nT)
        sd2 = max(1e-4, sum(d * d + cov[2 + nT + i][2 + nT + i] for i, d in enumerate(dfn)) / nT)
    f = Fit()
    f.theta, f.cov, f.idx, f.nT, f.sa, f.sd = theta, cov, idx, nT, math.sqrt(sa2), math.sqrt(sd2)
    f.nobs, f.weight = len(obs), tot_w
    # dispersión: fragilidad compartida (covarianza de residuos del mismo juego) e individual
    mus = {}
    num_s = den = num_i = 0.0
    pear = pw = 0.0
    by_game = defaultdict(list)
    for off, dfn_t, home, y, expo, w, pk, gpk in obs:
        mu = math.exp(math.log(expo) + pk + theta[0] + theta[1] * home + theta[2 + idx[off]] - theta[2 + nT + idx[dfn_t]])
        by_game[gpk].append((y, mu, w))
        pear += w * (y - mu) ** 2 / mu
        pw += w
        num_i += w * ((y - mu) ** 2 - mu) / (mu * mu)
        den += w
    cov_num = cov_den = 0.0
    for rows in by_game.values():
        if len(rows) == 2:
            (ya, ma, w), (yh, mh, _) = rows
            cov_num += w * (ya - ma) * (yh - mh) / (ma * mh)
            cov_den += w
    f.vSharedRaw = cov_num / cov_den if cov_den else 0.0
    f.vShared = max(0.0, f.vSharedRaw)
    f.vTotal = max(0.02, num_i / den)
    f.vInd = max(0.01, f.vTotal - f.vShared)
    f.phi = pear / pw
    f.rho = f.vShared / f.vTotal if f.vTotal else 0.0
    return f


def team_params(f, t):
    i = f.idx[t]
    return f.theta[2 + i], f.theta[2 + f.nT + i]


def mu_game(theta, f, off, dfn, home, expo, pk, offset=0.0):
    ia, idf = 2 + f.idx[off], 2 + f.nT + f.idx[dfn]
    return math.exp(math.log(expo) + pk + theta[0] + theta[1] * home + theta[ia] - theta[idf] + offset)


def p_home_nb(mu_a, mu_h, v, x_home=0.5):
    """P(gana el local) con carreras NB independientes (var = μ + v μ²); empates → extra innings."""
    pa = M.negbin_pmf(mu_a, 1 + v * mu_a)
    ph = M.negbin_pmf(mu_h, 1 + v * mu_h)
    w, t, _ = M.outcome_probs(ph, pa)
    return w + t * x_home


# ============================================================ validación fuera de muestra

def _oos(results, prev, parks, teams, cutoff, half_life, prev_w):
    obs = observations(results, prev, parks, _d(cutoff), half_life=half_life, prev_w=prev_w, until=cutoff)
    f = fit(obs, teams, iters=15, eb_rounds=4)
    test = [g for g in results if g["date"] >= cutoff]
    ll = br = 0.0
    for g in test:
        pk = math.log(parks.get(g.get("venue"), 1.0))
        ma = mu_game(f.theta, f, g["away"], g["home"], 0, 9.05, pk)
        mh = mu_game(f.theta, f, g["home"], g["away"], 1, 8.6, pk)
        p = min(max(p_home_nb(ma, mh, f.vTotal), 1e-4), 1 - 1e-4)
        y = g["hr"] > g["ar"]
        ll -= math.log(p if y else 1 - p)
        br += (p - y) ** 2
    n = len(test) or 1
    return ll / n, br / n


def grid_search(results, prev, parks, teams, cutoff):
    return [{"halfLife": hl, "prevW": pw, **dict(zip(("logloss", "brier"), _oos(results, prev, parks, teams, cutoff, hl, pw)))}
            for hl, pw in GRID]


def validate(results, prev, parks, teams, today, cutoff, home_rate, log5_fn):
    obs = observations(results, prev, parks, _d(cutoff), until=cutoff)
    f = fit(obs, teams, iters=15, eb_rounds=4)
    test = [g for g in results if g["date"] >= cutoff]
    ll = {"prisma": 0.0, "log5": 0.0, "local": 0.0}
    br = {"prisma": 0.0, "log5": 0.0, "local": 0.0}
    tot_err = []
    for g in test:
        pk = math.log(parks.get(g.get("venue"), 1.0))
        ma = mu_game(f.theta, f, g["away"], g["home"], 0, 9.05, pk)
        mh = mu_game(f.theta, f, g["home"], g["away"], 1, 8.6, pk)
        ps = {"prisma": p_home_nb(ma, mh, f.vTotal), "log5": log5_fn(g["home"], g["away"]), "local": home_rate}
        y = 1 if g["hr"] > g["ar"] else 0
        for k, p in ps.items():
            p = min(max(p, 1e-4), 1 - 1e-4)
            ll[k] -= math.log(p if y else 1 - p)
            br[k] += (p - y) ** 2
        tot_err.append((ma + mh) - (g["ar"] + g["hr"]))
    n = len(test) or 1
    return {"cutoff": cutoff, "games": len(test), "logloss": {k: v / n for k, v in ll.items()},
            "brier": {k: v / n for k, v in br.items()},
            "totalMae": statistics.fmean(abs(e) for e in tot_err) if tot_err else None,
            "totalBias": statistics.fmean(tot_err) if tot_err else None}


# ============================================================ carreras por media entrada y probabilidad en vivo

def half_inning_dist(results, kmax=12):
    """Distribución empírica de carreras en una media entrada (entradas 1-9 jugadas)."""
    c = [0] * (kmax + 1)
    for g in results:
        for x in (g.get("inn") or [])[:9]:
            for v in x:
                if v is not None:
                    c[min(v, kmax)] += 1
    n = sum(c) or 1
    return [v / n for v in c]


def scaled_half(dist, target_mean):
    """Escala la distribución de una media entrada a otra media: P(0) = P0^m y la forma condicional a anotar se conserva."""
    p0 = dist[0]
    cond = [v / (1 - p0) for v in dist[1:]]
    cm = sum((k + 1) * v for k, v in enumerate(cond))
    lo, hi = 0.05, 5.0
    for _ in range(50):
        m = (lo + hi) / 2
        mean = (1 - p0 ** m) * cm
        if mean < target_mean:
            lo = m
        else:
            hi = m
    q0 = p0 ** ((lo + hi) / 2)
    return [q0] + [(1 - q0) * v for v in cond]


def conv(a, b, kmax=40):
    out = [0.0] * min(kmax + 1, len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x < 1e-12:
            continue
        for j, y in enumerate(b):
            if i + j < len(out):
                out[i + j] += x * y
    return out


def win_prob(state_inning, top, diff, half_a, half_h, x_home=0.5, innings=9):
    """P(gana el local) al inicio de la media entrada (entrada, top/bottom) con diferencia local−visita."""
    rem_a = innings - state_inning + (1 if top else 0)
    rem_h = innings - state_inning + 1
    ra = [1.0]
    for _ in range(max(0, rem_a)):
        ra = conv(ra, half_a)
    rh = [1.0]
    for _ in range(max(0, rem_h)):
        rh = conv(rh, half_h)
    # el local no batea la 9ª si va ganando: el cálculo por diferencia final lo absorbe (gana igual)
    p = 0.0
    for i, pa in enumerate(ra):
        for j, ph in enumerate(rh):
            d = diff + j - i
            p += pa * ph * (1.0 if d > 0 else x_home if d == 0 else 0.0)
    return p


def wp_matrix(half_a, half_h, x_home):
    rows = []
    for diff in range(-5, 6):
        rows.append([win_prob(i, True, diff, half_a, half_h, x_home) for i in range(1, 10)])
    return rows


# ============================================================ corrida del modelo para un partido

def run_game(bundle, ctx, game, base, n_draws=2000, n_sims=200000, seed=9, projected_offsets=None):
    """PRISMA sobre un partido. `base` = análisis del framework (model.analyze) para ajustes de abridor y lineup."""
    import datetime as dt
    rng = random.Random(seed)
    lg = ctx.lg
    today = dt.date.fromisoformat(game["date"])
    teams = {int(t) for t in bundle["teams"]}
    parks = dict(ctx.pf)
    obs = observations(bundle["results"], bundle.get("prevResults", []), parks, today)
    f = fit(obs, teams)
    a_id, h_id = game["away"], game["home"]
    pk = math.log(parks.get(game["venue"], 1.0))
    exp_a = statistics.fmean(sum(1 for x in (g.get("inn") or []) if x[0] is not None) or 9 for g in bundle["results"])
    exp_h = statistics.fmean(sum(1 for x in (g.get("inn") or []) if x[1] is not None) or 8.5 for g in bundle["results"])

    # --- ajustes con información del día (escala log, se muestran como actualización secuencial)
    s5 = base["sections"]["s5"]
    L = s5["lambda"]
    offsets = {}
    for side, opp in (("away", "home"), ("home", "away")):
        c = L[side]
        share_sp = sum(L[side]["starterFracs"]) / 9 if L[side].get("starterFracs") else 0.6
        team_rot = ((base["sections"]["s5"]["triangulation"]["elo"]["starterAdj"].get(opp) or {}).get("rotationFip")) or lg.era
        # el abridor rival vs la rotación de su equipo (la defensa del modelo ya es el promedio del equipo)
        sp_ratio = (c["starter"] * lg.ra9) / (team_rot * lg.ra_per_era) if team_rot else 1.0
        offsets[side] = {
            "abridor": share_sp * math.log(max(0.4, min(2.5, sp_ratio * c["form"] * c["split"]))),
            "mano": math.log(c["handRatio"]) * share_sp,
            "lineup": math.log(c["lineup"]),
            "fatiga": (1 - share_sp) * math.log(c["fatigue"]),
        }
    if projected_offsets:
        for side in offsets:
            offsets[side]["lineup"] = projected_offsets.get(side, offsets[side]["lineup"])

    def mus(theta, use=("abridor", "mano", "lineup", "fatiga")):
        oa = sum(v for k, v in offsets["away"].items() if k in use)
        oh = sum(v for k, v in offsets["home"].items() if k in use)
        return (mu_game(theta, f, a_id, h_id, 0, exp_a, pk, oa), mu_game(theta, f, h_id, a_id, 1, exp_h, pk, oh))

    x_home = 0.5 * lg.extra_home_win + 0.25
    # --- actualización secuencial en escala logit (prior de temporada → +abridores → +mano → +lineup → +fatiga)
    steps = []
    acc = ()
    for label, key in (("Solo temporada (ataque/defensa + local + parque)", None), ("+ abridores del día", "abridor"),
                       ("+ split por mano", "mano"), ("+ lineups", "lineup"), ("+ fatiga del bullpen", "fatiga")):
        if key:
            acc = acc + (key,)
        ma, mh = mus(f.theta, acc)
        p = p_home_nb(ma, mh, f.vTotal, x_home)
        steps.append({"label": label, "pHome": p, "logit": math.log(p / (1 - p)), "muAway": ma, "muHome": mh})

    # --- posterior: muestras de θ por Laplace
    Lc = cholesky(f.cov)
    P_ = len(f.theta)
    draws = []
    for _ in range(n_draws):
        z = [rng.gauss(0, 1) for _ in range(P_)]
        th = [f.theta[i] + sum(Lc[i][k] * z[k] for k in range(i + 1)) for i in range(P_)]
        ma, mh = mus(th)
        draws.append((ma, mh, p_home_nb(ma, mh, f.vTotal, x_home)))
    p_draws = sorted(d[2] for d in draws)
    p_mean = statistics.fmean(p_draws)

    def q(x):
        return p_draws[min(len(p_draws) - 1, int(x * len(p_draws)))]

    # --- predictiva posterior por simulación con fragilidades (compartida + individual)
    vs, vi = f.vShared, f.vInd
    ks, ki = (1 / vs if vs > 1e-4 else None), 1 / vi
    runs = {"away": defaultdict(int), "home": defaultdict(int)}
    total = defaultdict(int)
    margin = defaultdict(int)
    f5 = {"away": 0, "home": 0, "tie": 0}
    f5tot = defaultdict(int)
    share5 = {s: sum(lg.inning_runs[s][1:6]) / (sum(lg.inning_runs[s][1:10]) + lg.extra_runs[s]) for s in ("away", "home")}
    home_w = 0
    for _ in range(n_sims):
        ma, mh, _p = draws[rng.randrange(n_draws)]
        zs = rng.gammavariate(ks, vs) if ks else 1.0
        ra = _poisson(rng, ma * zs * rng.gammavariate(ki, vi))
        rh = _poisson(rng, mh * zs * rng.gammavariate(ki, vi))
        a5 = _binom(rng, ra, share5["away"])
        h5 = _binom(rng, rh, share5["home"])
        if ra == rh:
            if rng.random() < x_home:
                rh += 1
            else:
                ra += 1
        runs["away"][ra] += 1
        runs["home"][rh] += 1
        total[ra + rh] += 1
        margin[rh - ra] += 1
        home_w += rh > ra
        f5["home" if h5 > a5 else "away" if a5 > h5 else "tie"] += 1
        f5tot[a5 + h5] += 1

    def dist(d, hi):
        return [d.get(k, 0) / n_sims for k in range(hi + 1)]

    # --- probabilidad en vivo
    hd = half_inning_dist(bundle["results"])
    ma, mh = mus(f.theta)
    half_a = scaled_half(hd, ma / exp_a)
    half_h = scaled_half(hd, mh / exp_h)
    wpm = wp_matrix(half_a, half_h, x_home)

    # --- tabla de fuerzas
    inv = {i: t for t, i in f.idx.items()}
    strengths = []
    for i in range(f.nT):
        t = inv[i]
        att, dfn = f.theta[2 + i], f.theta[2 + f.nT + i]
        strengths.append({"team": (ctx.T.info.get(t) or {}).get("abbr"), "att": att, "def": dfn,
                          "attSd": math.sqrt(f.cov[2 + i][2 + i]), "defSd": math.sqrt(f.cov[2 + f.nT + i][2 + f.nT + i]),
                          "net": att + dfn})
    strengths.sort(key=lambda r: -r["net"])

    ab, hb = ctx.T.abbr(a_id), ctx.T.abbr(h_id)
    rl = {f"{hb} -1.5": sum(v for k, v in margin.items() if k >= 2) / n_sims,
          f"{ab} +1.5": 1 - sum(v for k, v in margin.items() if k >= 2) / n_sims,
          f"{ab} -1.5": sum(v for k, v in margin.items() if k <= -2) / n_sims,
          f"{hb} +1.5": 1 - sum(v for k, v in margin.items() if k <= -2) / n_sims}
    extra = {"n": n_sims, "label": "PRISMA bayesiano", "pHome": home_w / n_sims, "total": dist(total, 40),
             "runs_away": dist(runs["away"], 25), "runs_home": dist(runs["home"], 25), "f5total": dist(f5tot, 25),
             "f5": {k: v / n_sims for k, v in f5.items()}, "rl": rl}
    return {
        "fit": {"alpha": f.theta[0], "home": f.theta[1], "sigmaAtt": f.sa, "sigmaDef": f.sd, "nobs": f.nobs,
                "phi": f.phi, "vShared": vs, "vSharedRaw": f.vSharedRaw, "vInd": vi, "vTotal": f.vTotal, "rho": f.rho,
                "rhoRaw": f.vSharedRaw / f.vTotal if f.vTotal else 0.0,
                "expAway": exp_a, "expHome": exp_h, "halfLife": HALF_LIFE, "prevWeight": PREV_W},
        "teams": {"away": {"abbr": ab, "att": team_params(f, a_id)[0], "def": team_params(f, a_id)[1],
                           "attSd": math.sqrt(f.cov[2 + f.idx[a_id]][2 + f.idx[a_id]]),
                           "defSd": math.sqrt(f.cov[2 + f.nT + f.idx[a_id]][2 + f.nT + f.idx[a_id]])},
                  "home": {"abbr": hb, "att": team_params(f, h_id)[0], "def": team_params(f, h_id)[1],
                           "attSd": math.sqrt(f.cov[2 + f.idx[h_id]][2 + f.idx[h_id]]),
                           "defSd": math.sqrt(f.cov[2 + f.nT + f.idx[h_id]][2 + f.nT + f.idx[h_id]])}},
        "offsets": offsets, "steps": steps, "park": math.exp(pk), "xHome": x_home,
        "posterior": {"mean": p_mean, "sd": statistics.pstdev(p_draws), "q05": q(0.05), "q25": q(0.25), "q50": q(0.5),
                      "q75": q(0.75), "q95": q(0.95), "quantiles": [p_draws[int(i * (n_draws - 1) / 100)] for i in range(101)],
                      "hist": _hist(p_draws, 0.25, 0.85, 30), "draws": n_draws,
                      "muAway": statistics.fmean(d[0] for d in draws), "muHome": statistics.fmean(d[1] for d in draws)},
        "predictive": {"pHome": home_w / n_sims, "runs": {"away": dist(runs["away"], 20), "home": dist(runs["home"], 20)},
                       "total": dist(total, 30), "f5": {k: v / n_sims for k, v in f5.items()}, "sims": n_sims,
                       "meanTotal": sum(k * v for k, v in total.items()) / n_sims,
                       "sdTotal": math.sqrt(sum(k * k * v for k, v in total.items()) / n_sims - (sum(k * v for k, v in total.items()) / n_sims) ** 2)},
        "halfInning": {"league": hd, "away": half_a, "home": half_h},
        "wpMatrix": wpm, "strengths": strengths, "extra": extra,
    }


def _poisson(rng, lam):
    if lam <= 0:
        return 0
    if lam > 40:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _binom(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


def _hist(vals, lo, hi, bins):
    c = [0] * bins
    for v in vals:
        i = int((v - lo) / (hi - lo) * bins)
        if 0 <= i < bins:
            c[i] += 1
    n = len(vals) or 1
    return {"lo": lo, "hi": hi, "counts": [x / n for x in c]}
