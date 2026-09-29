"""OCTUBRE (Pro-Lab · postemporada): cómo le va a cada abridor contra ESE rival y cómo se desarrolla el juego.

Tres partes que se conectan (y una cuarta de prueba):

A. Encogimiento pitcher × rival (Bayes empírico Beta-Binomial). Con los game logs de todos los abridores de la
   liga se mide, para cada evento por bateador enfrentado (ponche, base por bolas, jonrón y hit en bola en juego),
   cuánto se aparta un pitcher de lo esperado contra un equipo en particular, más allá de su nivel y del nivel del
   equipo (razón de momios). La varianza entre parejas (método de momentos) da κ: el historial contra el rival pesa
   n/(n + κ). Si la varianza entre parejas no supera a la del azar, κ = ∞ y el historial no se usa.

B. Matchup por tipo de lanzamiento (log5 aditivo en escala wOBA con el arsenal de Baseball Savant). Para cada
   bateador del lineup rival: Σ_t uso_t · [L_t + (pitcher_t − L_t) + (bateador − L) + residuo del bateador contra t],
   cada término encogido con su propia muestra (κ medidos en la liga por método de momentos) + ajuste de mano.

C. Desarrollo del juego: salida del abridor (Weibull ajustada a sus aperturas y encogida a la liga) escalada por el
   gancho de postemporada MEDIDO (outs de los abridores en las postemporadas 2024-2026 contra su propio promedio de
   temporada), castigo por vuelta al lineup (la 3.ª vuelta es la peor y el gancho corto la recorta) y bullpen de
   octubre (los relevistas de confianza cargan más entradas; los de relleno casi no lanzan).

v2 (algoritmo 2026.09.29.5, post-mortem de PHI @ ATL juego 1, validado fuera de muestra con los game logs de 2026):
   · la «forma del día» (varianza de juego a juego, ANOVA dentro de cada pareja) se resta de la varianza entre
     parejas: sin eso, 1–2 aperturas buenas o malas contra un equipo parecían «efecto del rival» (κ de ponches 284 →
     ∞; el historial empeoraba la predicción de ponches fuera de muestra);
   · la tasa de temporada del pitcher y la del equipo se encogen con κ medidos en la liga (antes 30 fijo: le creía
     el 96% a su tasa de jonrones y el 94% a su BABIP);
   · el gancho se mide solo con abridores de verdad (sin relevistas usados de abridor) y con pendiente Theil-Sen;
   · los ponches del abridor se mezclan sobre la forma del día (Beta-Binomial acoplada a la profundidad).

D. Familiaridad (variable de prueba): ¿le va peor a un abridor la 2.ª, 3.ª o 4.ª vez que ve al mismo rival en la
   temporada? Regresión ponderada del residuo por apertura contra el número de enfrentamientos previos. Solo entra al
   modelo si el efecto es claro (|t| ≥ 2); si no, se reporta y se deja en cero.

Todo termina en carreras por entrada → Binomial Negativa → mercados → picks → la decisión única del partido.
"""
from __future__ import annotations

import math
import random

from . import mathlib as M

EVENTS = ("k", "bb", "hr", "babip")             # k, bb (+ pelotazo) y hr por bateador; babip = hits por bola en juego
EV_NAMES = {"k": "Ponches", "bb": "Bases por bolas", "hr": "Jonrones", "babip": "Hits en bola en juego"}
W_BB, W_HR = 0.70, 2.03                         # pesos lineales wOBA (los de EIGEN; BB y pelotazo juntos)
WOBA_SCALE = 1.23
SLOT_PA = [4.65, 4.55, 4.43, 4.33, 4.24, 4.13, 4.03, 3.93, 3.83]
PREV_WEIGHT = 0.5                               # la temporada anterior contra el rival pesa la mitad
SIGMA2_XWOBA = 0.15                             # varianza por turno del xwOBA: 9% BB (0.70) y 69% bolas en juego
                                                # (media 0.37, de 0.40) con la mezcla de eventos de la liga
TTO_WOBA = (-0.008, 0.002, 0.014, 0.024)        # 1.ª, 2.ª, 3.ª, 4.ª vuelta vs el promedio del abridor (se recalibra)
FAMILIARITY_T = 2.0


# ============================================================ utilidades

def outs_of(ip) -> int:
    """'5.2' → 17 outs."""
    try:
        whole, _, frac = str(ip or "0").partition(".")
        return int(whole) * 3 + int(frac or 0)
    except ValueError:
        return 0


def odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def from_odds(o: float) -> float:
    return o / (1 + o)


def odds_ratio(p_pitcher: float, p_team: float, p_league: float) -> float:
    """Tasa esperada de un evento combinando pitcher, equipo rival y liga (razón de momios, «log5»)."""
    return from_odds(odds(p_pitcher) * odds(p_team) / odds(p_league))


def counts(row: dict) -> dict:
    """Conteos de una apertura (o de un total): bf, k, bb, hr, bip (bolas en juego) y hits en bola en juego."""
    bf = row.get("battersFaced") or row.get("bf") or 0
    k = row.get("strikeOuts") if row.get("strikeOuts") is not None else row.get("k") or 0
    bb = (row.get("baseOnBalls") if row.get("baseOnBalls") is not None else row.get("bb") or 0) + (row.get("hitByPitch") or 0)
    hr = row.get("homeRuns") if row.get("homeRuns") is not None else row.get("hr") or 0
    h = row.get("hits") if row.get("hits") is not None else row.get("h") or 0
    bip = max(0, bf - k - bb - hr)
    return {"bf": bf, "k": k, "bb": bb, "hr": hr, "bip": bip, "hbip": max(0, h - hr)}


def rates(c: dict) -> dict:
    return {"k": c["k"] / c["bf"] if c["bf"] else None, "bb": c["bb"] / c["bf"] if c["bf"] else None,
            "hr": c["hr"] / c["bf"] if c["bf"] else None, "babip": c["hbip"] / c["bip"] if c["bip"] else None}


def trials(c: dict, ev: str) -> tuple[float, float]:
    """(éxitos, intentos) de un evento: por bateador enfrentado, o por bola en juego en el BABIP."""
    return (c["hbip"], c["bip"]) if ev == "babip" else (c[ev], c["bf"])


def add(a: dict, b: dict, w: float = 1.0) -> dict:
    """Suma de conteos (solo los campos numéricos)."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    return {k: (a.get(k) if num(a.get(k)) else 0) + w * (b.get(k) if num(b.get(k)) else 0)
            for k in set(a) | set(b) if num(a.get(k)) or num(b.get(k))}


def woba_of(r: dict, hit_w: float) -> float:
    """wOBA de un perfil de tasas: BB·W_BB + HR·W_HR + (1 − K − BB − HR)·BABIP·valor medio del hit sin HR."""
    bip = max(0.0, 1 - r["k"] - r["bb"] - r["hr"])
    return W_BB * r["bb"] + W_HR * r["hr"] + bip * r["babip"] * hit_w


# ============================================================ liga y equipos

def team_rates(bundle: dict) -> tuple[dict, dict, float]:
    """Tasas de bateo de cada equipo (por turno) y de la liga, y el valor medio de un hit sin jonrón."""
    out, tot = {}, {}
    d2 = d3 = h1 = 0
    for tid, t in (bundle.get("teamStats") or {}).items():
        h = t.get("hitting") or {}
        pa = h.get("plateAppearances") or 0
        if not pa:
            continue
        c = counts({"battersFaced": pa, "strikeOuts": h.get("strikeOuts"), "baseOnBalls": h.get("baseOnBalls"),
                    "hitByPitch": h.get("hitByPitch"), "homeRuns": h.get("homeRuns"), "hits": h.get("hits")})
        out[int(tid)] = {**rates(c), "rpa": (h.get("runs") or 0) / pa, "_c": c}
        tot = add(tot, {**c, "runs": h.get("runs") or 0})
        d2 += h.get("doubles") or 0
        d3 += h.get("triples") or 0
        h1 += (h.get("hits") or 0) - (h.get("doubles") or 0) - (h.get("triples") or 0) - (h.get("homeRuns") or 0)
    lg = {**rates(tot), "rpa": tot["runs"] / tot["bf"]} if tot.get("bf") else {"k": 0.22, "bb": 0.09, "hr": 0.03, "babip": 0.29, "rpa": 0.12}
    # v2: la tasa de cada equipo se encoge a la liga con κ medido entre equipos (método de momentos)
    for e in EVENTS:
        num = den = 0.0
        for t in out.values():
            y, n = trials(t["_c"], e)
            if n > 0:
                v = lg[e] * (1 - lg[e])
                num += n * ((y / n - lg[e]) ** 2 - v / n)
                den += n * v
        r0 = num / den if den else 0.0
        k_t = (1 / r0 - 1) if r0 > 0 else 1e9
        for t in out.values():
            y, n = trials(t["_c"], e)
            t[e] = (y + k_t * lg[e]) / (n + k_t)
        lg.setdefault("teamKappa", {})[e] = k_t
    for t in out.values():
        t.pop("_c", None)
    hit_w = (0.88 * h1 + 1.25 * d2 + 1.58 * d3) / max(1, h1 + d2 + d3) if h1 else 0.95
    return out, lg, hit_w


def start_rows(logs: dict) -> list[dict]:
    """Aperturas de la liga (game logs del snapshot) con sus conteos."""
    rows = []
    for pid, lst in (logs or {}).items():
        if not isinstance(lst, list):
            continue
        for r in lst:
            if not r.get("gamesStarted") or not r.get("battersFaced") or r.get("opp") is None:
                continue
            rows.append({"pid": int(pid), "opp": r["opp"], "date": r.get("date") or "", **counts(r),
                         "outs": outs_of(r.get("inningsPitched")), "pitches": r.get("numberOfPitches"),
                         "er": r.get("earnedRuns") or 0, "runs": r.get("runs") or 0})
    rows.sort(key=lambda r: (r["pid"], r["date"]))
    return rows


# ============================================================ A. encogimiento pitcher × rival

def pitcher_k0(k_pit: dict | None, e: str) -> float:
    """κ para encoger la tasa del pitcher en el evento e: el medido (v2) o 30 si no se midió; sin señal entre
    pitchers (κ = ∞) se usa la liga."""
    if k_pit is None:
        return 30.0
    return ((k_pit.get(e) or {}).get("kappa")) or 1e9


def pair_table(rows: list[dict], teams: dict, lg: dict, k_pit: dict | None = None) -> dict:
    """{(pitcher, rival): {n, y, mu, g, s2} por evento}: lo observado contra ese rival, lo esperado por la razón de
    momios con la tasa del pitcher SIN esas aperturas (deja uno fuera, encogida con κ del pitcher) y la del equipo
    rival, y cuántas aperturas son (g) con Σn² (s2) para quitar la forma del día."""
    tot, pair, starts = {}, {}, {}
    for r in rows:
        tot[r["pid"]] = add(tot.get(r["pid"], {}), r)
        key = (r["pid"], r["opp"])
        pair[key] = add(pair.get(key, {}), r)
        starts.setdefault(key, []).append(r)
    out = {}
    for (pid, opp), c in pair.items():
        rest = add(tot[pid], c, -1)
        ev = {}
        for e in EVENTS:
            y, n = trials(c, e)
            ry, rn = trials(rest, e)
            if n <= 0 or rn < 30 or opp not in teams:
                continue
            k0 = pitcher_k0(k_pit, e)
            p_pit = (ry + k0 * lg[e]) / (rn + k0)             # el pitcher sin este rival, encogido a la liga
            mu = odds_ratio(p_pit, teams[opp][e], lg[e])
            # varianza de μ por estimar la tasa del pitcher (método delta): no es efecto de la pareja
            vmu = (mu * (1 - mu)) ** 2 / (p_pit * (1 - p_pit) * (rn + k0))
            ns = [trials(r, e)[1] for r in starts[(pid, opp)]]
            ev[e] = {"n": n, "y": y, "mu": mu, "vmu": vmu, "g": sum(1 for x in ns if x > 0), "s2": sum(x * x for x in ns)}
        if ev and rest.get("bf", 0) >= 30 and opp in teams:     # carreras por bateador (para la familiaridad)
            ev["runs"] = {"mu": (rest.get("runs", 0) + 30 * lg["rpa"]) / (rest["bf"] + 30) * teams[opp]["rpa"] / lg["rpa"]}
        if ev:
            out[(pid, opp)] = ev
    return out


def kappa(pairs: dict, ev: str, boot: int = 200, seed: int = 29, rho_game: float = 0.0) -> dict:
    """κ del Beta-Binomial por método de momentos: ρ = Σn[(r−μ)² − μ(1−μ)/n − Var(μ̂) − ρ_día·μ(1−μ)·Σn²/n²] /
    Σn·μ(1−μ) = 1/(κ+1). El término ρ_día (forma del día, ver game_rho) quita la varianza de juego a juego: con 1–2
    aperturas contra un equipo, un buen o mal día no es un efecto del rival.

    Devuelve κ, ρ, su intervalo por bootstrap de parejas y el número de parejas y de intentos."""
    xs = [v[ev] for v in pairs.values() if ev in v and v[ev]["n"] >= 5]

    def rho(sample):
        num = den = 0.0
        for x in sample:
            n, y, mu = x["n"], x["y"], x["mu"]
            r = y / n
            day = rho_game * mu * (1 - mu) * x.get("s2", n * n) / (n * n)
            num += n * ((r - mu) ** 2 - mu * (1 - mu) / n - x.get("vmu", 0.0) - day)
            den += n * mu * (1 - mu)
        return num / den if den else 0.0

    if len(xs) < 20:
        return {"kappa": None, "rho": None, "ci": None, "pairs": len(xs), "n": sum(x["n"] for x in xs)}
    r0 = rho(xs)
    rng = random.Random(seed)
    bs = sorted(rho([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(boot))
    ci = [bs[int(0.05 * boot)], bs[int(0.95 * boot) - 1]]
    k = (1 / r0 - 1) if r0 > 0 else None                  # None = sin señal entre parejas: κ infinito
    return {"kappa": k, "rho": r0, "ci": ci, "pairs": len(xs), "n": sum(x["n"] for x in xs)}


def game_rho(rows: list[dict], ev: str) -> dict:
    """Forma del día: varianza de juego a juego de la tasa de un abridor, por ANOVA dentro de cada pareja
    pitcher-rival con ≥ 2 aperturas (su nivel y un posible efecto del rival se cancelan):
    E[Σ n_i (r_i − r̄)²] = v(g − 1) + ρ·v·(n − Σn_i²/n), v = r̄(1 − r̄)."""
    groups = {}
    for r in rows:
        y, n = trials(r, ev)
        if n >= 3:
            groups.setdefault((r["pid"], r["opp"]), []).append((y, n))
    num = den = 0.0
    used = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        n = sum(x[1] for x in g)
        rb = sum(x[0] for x in g) / n
        v = rb * (1 - rb)
        if v <= 0:
            continue
        num += sum(nn * (yy / nn - rb) ** 2 for yy, nn in g) - v * (len(g) - 1)
        den += v * (n - sum(nn * nn for _, nn in g) / n)
        used += 1
    return {"rho": max(0.0, num / den) if den else 0.0, "groups": used}


def pitcher_kappa(rows: list[dict], lg: dict, rho_game: dict | None = None) -> dict:
    """κ para encoger la tasa de temporada de cada abridor a la liga (Bayes empírico, método de momentos entre
    pitchers, sin la forma del día): cuántos bateadores «vale» la liga. HR y BABIP necesitan muchos más que K."""
    tot, sq = {}, {}
    for r in rows:
        tot[r["pid"]] = add(tot.get(r["pid"], {}), r)
        for e in EVENTS:
            sq.setdefault((r["pid"], e), 0.0)
            sq[(r["pid"], e)] += trials(r, e)[1] ** 2
    out = {}
    for e in EVENTS:
        rg = ((rho_game or {}).get(e) or {}).get("rho", 0.0)
        num = den = 0.0
        npit = 0
        for pid, c in tot.items():
            y, n = trials(c, e)
            if n < 50:
                continue
            mu = lg[e]
            v = mu * (1 - mu)
            num += n * ((y / n - mu) ** 2 - v / n - rg * v * sq[(pid, e)] / (n * n))
            den += n * v
            npit += 1
        r0 = num / den if den else 0.0
        out[e] = {"kappa": (1 / r0 - 1) if r0 > 0 else None, "rho": r0, "pitchers": npit}
    return out


def form_depth(rows: list[dict]) -> dict:
    """Cómo se mueve la profundidad con la forma del día: pendiente de los outs de una apertura contra su tasa de
    ponche (ambos contra el promedio del mismo pitcher). Positiva: el día que poncha más, dura más."""
    by = {}
    for r in rows:
        if r.get("bf", 0) >= 9 and r.get("outs"):
            by.setdefault(r["pid"], []).append(r)
    xs, ys = [], []
    for rs in by.values():
        if len(rs) < 6:
            continue
        mk = sum(r["k"] for r in rs) / sum(r["bf"] for r in rs)
        mo = sum(r["outs"] for r in rs) / len(rs)
        for r in rs:
            xs.append(r["k"] / r["bf"] - mk)
            ys.append(r["outs"] - mo)
    if len(xs) < 50:
        return {"b": 0.0, "corr": None, "n": len(xs)}
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return {"b": sxy / sxx if sxx else 0.0, "corr": sxy / math.sqrt(sxx * syy) if sxx and syy else None, "n": len(xs)}


def shrink_pair(y: float, n: float, mu: float, k: float | None) -> dict:
    """Posterior del Beta-Binomial centrado en lo esperado: (y + κμ)/(n + κ); peso del historial n/(n+κ)."""
    if k is None or n <= 0:
        return {"post": mu, "w": 0.0, "raw": (y / n) if n else None, "mu": mu, "n": n, "y": y}
    w = n / (n + k)
    post = (y + k * mu) / (n + k)
    sd = math.sqrt(post * (1 - post) / (n + k + 1))
    return {"post": post, "w": w, "raw": y / n, "mu": mu, "n": n, "y": y, "sd": sd, "mult": post / mu if mu else 1.0}


def starter_vs_rival(p: dict, opp: int, teams: dict, lg: dict, kap: dict, k_pit: dict | None = None) -> dict:
    """Historial del abridor del partido contra este rival (temporada + la anterior a la mitad), encogido con κ."""
    season = add({}, {})
    for r in p.get("log") or []:
        season = add(season, counts(r["stat"]))
    vs, prev_vs, starts = {}, {}, []
    for r in p.get("log") or []:
        if r.get("opp") == opp:
            vs = add(vs, counts(r["stat"]))
            starts.append({"date": r["date"], "outs": outs_of(r["stat"].get("inningsPitched")), "er": r["stat"].get("earnedRuns"),
                           **{k: v for k, v in counts(r["stat"]).items() if k in ("bf", "k", "bb", "hr")}})
    for r in p.get("prevLog") or []:
        if r.get("opp") == opp:
            prev_vs = add(prev_vs, counts(r["stat"]))
    comb = add(vs, prev_vs, PREV_WEIGHT)
    rest = add(season, vs, -1)
    out = {}
    for e in EVENTS:
        k_pair = (kap.get(e) or {}).get("kappa")
        k0 = pitcher_k0(k_pit, e)
        # con efecto de pareja medible, μ sin las aperturas contra este rival (no contar dos veces); sin él, toda la temporada
        ry, rn = trials(rest if k_pair else season, e)
        y, n = trials(comb, e) if comb else (0, 0)
        p_pit = (ry + k0 * lg[e]) / (rn + k0) if rn else lg[e]
        mu = odds_ratio(p_pit, (teams.get(opp) or lg)[e], lg[e])
        out[e] = {**shrink_pair(y, n, mu, k_pair), "season": trials(season, e)[0] / max(1, trials(season, e)[1]),
                  "pitcherShrink": rn / (rn + k0) if rn else 0.0}
    raw = rates(vs) if vs.get("bf") else None
    return {"events": out, "starts": starts, "bf": vs.get("bf", 0), "prevBf": prev_vs.get("bf", 0), "raw": raw,
            "er": sum(s["er"] or 0 for s in starts), "outs": sum(s["outs"] for s in starts)}


# ============================================================ D. familiaridad

def _day(d: str) -> float:
    try:
        y, m, dd = (int(x) for x in d[:10].split("-"))
        return (y - 2026) * 365 + (m - 3) * 30.4 + dd
    except ValueError:
        return 0.0


def _wls_partial(pts):
    """Pendiente de y sobre f controlando por la fecha (Frisch–Waugh–Lovell), ponderada; error estándar robusto."""
    def resid(v_of):
        sw = sum(w for *_, w in pts)
        dm = sum(w * d for _, d, _, w in pts) / sw
        vm = sum(w * v_of(p) for p in pts for w in [p[3]]) / sw
        sdd = sum(w * (d - dm) ** 2 for _, d, _, w in pts) or 1e-9
        b = sum(w * (d - dm) * (v_of(p) - vm) for p in pts for _, d, _, w in [p]) / sdd
        return [v_of(p) - vm - b * (p[1] - dm) for p in pts]
    fx = resid(lambda p: p[0])
    fy = resid(lambda p: p[2])
    w = [p[3] for p in pts]
    sxx = sum(wi * x * x for wi, x in zip(w, fx))
    if sxx <= 1e-9 * sum(w):                              # f y la fecha son lo mismo: no se puede separar
        return 0.0, float("inf")
    b = sum(wi * x * y for wi, x, y in zip(w, fx, fy)) / sxx
    se = math.sqrt(sum((wi * x * (y - b * x)) ** 2 for wi, x, y in zip(w, fx, fy))) / sxx
    return b, se


def familiarity(rows: list[dict], pairs_mu: dict) -> dict:
    """Residuo relativo por apertura [(y − n·μ)/(n·μ)] contra cuántas veces el pitcher ya había enfrentado a ese
    rival en la temporada (0, 1, 2, 3+), controlando por la fecha (la liga cambia a lo largo del año y las
    repeticiones se juntan al final). Mínimos cuadrados ponderados por n·μ; error estándar robusto.

    `used`: el efecto en carreras entra al modelo solo si |t| ≥ 2, encogido por (1 − 1/t²) (James-Stein de un solo
    estimado: con t = 2 se usa 75%)."""
    seen, out = {}, {}
    data = {e: [] for e in ("k", "bb", "hr", "runs")}
    for r in rows:                                     # ya ordenadas por pitcher y fecha
        key = (r["pid"], r["opp"])
        f = min(3, seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1
        mu = pairs_mu.get(key)
        if not mu or not r["bf"]:
            continue
        d = _day(r.get("date") or "")
        for e in ("k", "bb", "hr"):
            if e in mu:
                exp = r["bf"] * mu[e]["mu"]
                if exp:
                    data[e].append((f, d, (r[e] - exp) / exp, exp))
        if "runs" in mu:
            exp = r["bf"] * mu["runs"]["mu"]
            if exp:
                data["runs"].append((f, d, (r["runs"] - exp) / exp, exp))
    for e, pts in data.items():
        if len(pts) < 50:
            out[e] = None
            continue
        b, se = _wls_partial(pts)
        t = b / se if se else 0.0
        by = {}
        for x, _, y, w in pts:
            a = by.setdefault(x, [0.0, 0.0, 0])
            a[0] += w * y
            a[1] += w
            a[2] += 1
        out[e] = {"slope": b, "se": se, "t": t, "n": len(pts), "controls": "fecha",
                  "shrunk": b * max(0.0, 1 - 1 / (t * t)) if abs(t) >= FAMILIARITY_T else 0.0,
                  "byCount": {str(k): {"rel": v[0] / v[1], "starts": v[2]} for k, v in sorted(by.items())}}
    return out


# ============================================================ B. matchup por tipo de lanzamiento

def arsenal_league(ars: dict) -> tuple[dict, float]:
    """xwOBA de la liga por tipo de lanzamiento (ponderado por turnos) y el global."""
    acc, tot = {}, [0.0, 0.0]
    for rows in (ars.get("pitcher") or {}).values():
        for r in rows:
            if r.get("xwoba") is None or not r.get("pa"):
                continue
            a = acc.setdefault(r["type"], [0.0, 0.0])
            a[0] += r["pa"] * r["xwoba"]
            a[1] += r["pa"]
            tot[0] += r["pa"] * r["xwoba"]
            tot[1] += r["pa"]
    return {t: a[0] / a[1] for t, a in acc.items() if a[1] > 200}, (tot[0] / tot[1] if tot[1] else 0.315)


def arsenal_kappa(rows_by_player: dict, L: dict, center=None) -> dict:
    """κ para encoger el xwOBA por tipo de lanzamiento: τ² = var ponderada de (x − L_t [− centro del jugador]) − E[σ²/n];
    κ = σ²/τ². `center` (opcional) quita el nivel general de cada jugador para medir solo lo específico del pitch."""
    pts = []
    for pid, rows in rows_by_player.items():
        c = center(pid) if center else 0.0
        for r in rows:
            if r.get("xwoba") is None or (r.get("pa") or 0) < 20 or r["type"] not in L:
                continue
            pts.append((r["xwoba"] - L[r["type"]] - c, r["pa"]))
    if len(pts) < 30:
        return {"kappa": 200.0, "tau2": None, "n": len(pts)}
    sw = sum(n for _, n in pts)
    m = sum(n * d for d, n in pts) / sw
    var = sum(n * (d - m) ** 2 for d, n in pts) / sw
    noise = sum(n * (SIGMA2_XWOBA / n) for _, n in pts) / sw
    tau2 = var - noise
    return {"kappa": SIGMA2_XWOBA / tau2 if tau2 > 1e-6 else 5000.0, "tau2": tau2, "n": len(pts)}


def pitcher_mix(rows: list[dict], L: dict, k_p: float) -> list[dict]:
    """Arsenal del pitcher: uso normalizado y xwOBA por tipo encogido a la liga."""
    rows = [r for r in rows or [] if r.get("usage") and r["type"] in L]
    tot = sum(r["usage"] for r in rows) or 1.0
    out = []
    for r in rows:
        n = r.get("pa") or 0
        x = r.get("xwoba") if r.get("xwoba") is not None else L[r["type"]]
        out.append({"type": r["type"], "name": r.get("name") or r["type"], "usage": r["usage"] / tot, "pa": n,
                    "xwobaObs": r.get("xwoba"), "xwoba": L[r["type"]] + (x - L[r["type"]]) * n / (n + k_p),
                    "whiff": r.get("whiff"), "k": r.get("k")})
    return sorted(out, key=lambda r: -r["usage"])


def batter_level(pid, bundle: dict, Lw: float, k_bat: float = 300) -> tuple[float, float, str | None]:
    """Nivel general del bateador vs la liga: su xwOBA de todo el arsenal que enfrentó (Savant, Σ turnos por tipo
    de lanzamiento) encogido con k = 300 turnos. Sin datos: 15 puntos abajo de la liga (bateador de reemplazo)."""
    rows = [r for r in ((bundle.get("arsenal") or {}).get("batter") or {}).get(str(pid)) or []
            if r.get("xwoba") is not None and r.get("pa")]
    pa = sum(r["pa"] for r in rows)
    if not pa:
        return -0.015, 0, None
    xw = sum(r["pa"] * r["xwoba"] for r in rows) / pa
    return (xw - Lw) * pa / (pa + k_bat), pa, None


def matchup(bid, mix: list[dict], bundle: dict, L: dict, Lw: float, k_b: float, platoon: float) -> dict:
    """xwOBA esperado del bateador contra el abridor: Σ uso_t·[L_t + (pitcher_t − L_t) + nivel + residuo_t] + mano."""
    lvl, pa, _ = batter_level(bid, bundle, Lw)
    brow = {r["type"]: r for r in ((bundle.get("arsenal") or {}).get("batter") or {}).get(str(bid)) or []}
    tot = gen = 0.0
    parts = []
    for m in mix:
        b = brow.get(m["type"]) or {}
        n = b.get("pa") or 0
        res = 0.0
        if n and b.get("xwoba") is not None:
            res = ((b["xwoba"] - L[m["type"]]) - lvl) * n / (n + k_b)
        val = L[m["type"]] + (m["xwoba"] - L[m["type"]]) + lvl + res
        tot += m["usage"] * val
        gen += m["usage"] * (L[m["type"]] + (m["xwoba"] - L[m["type"]]) + lvl)
        parts.append({"type": m["type"], "usage": m["usage"], "val": val, "res": res, "pa": n})
    return {"id": bid, "pa": pa, "level": lvl, "xwoba": tot + platoon, "generic": gen + platoon, "platoon": platoon,
            "pitchEdge": tot - gen, "parts": parts}


def k_rate(k_pit: float, k_bat: float | None, k_lg: float) -> float:
    return odds_ratio(k_pit, k_bat if k_bat is not None else k_lg, k_lg)


# ============================================================ C. desarrollo del juego

def weibull_fit(xs: list[float]) -> tuple[float, float]:
    """(forma, escala) de una Weibull por momentos (aprox. de Justus para la forma a partir del CV)."""
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) or 0.1 * m
    shape = max(1.2, min(12.0, (sd / m) ** -1.086))
    scale = m / math.gamma(1 + 1 / shape)
    return shape, scale


def outs_dist(shape: float, scale: float, hook: float = 1.0, max_outs: int = 27) -> list[float]:
    """P(el abridor saca exactamente k outs), k = 0..27, con la escala multiplicada por el gancho."""
    sc = scale * hook
    cdf = lambda x: 1 - math.exp(-((max(x, 0) / sc) ** shape))  # noqa: E731
    p = [cdf(k + 0.5) - cdf(k - 0.5) for k in range(max_outs)]
    p.append(1 - cdf(max_outs - 0.5))
    s = sum(p)
    return [x / s for x in p]


def inning_fracs(dist: list[float]) -> list[float]:
    """Fracción esperada de cada entrada (1..9) que cubre el abridor."""
    out = [0.0]
    for i in range(1, 10):
        lo = 3 * (i - 1)
        out.append(sum(p * min(3, max(0, k - lo)) / 3 for k, p in enumerate(dist)))
    return out


def post_starts(post: dict, reg: dict) -> list[dict]:
    """Aperturas de postemporada (primer pitcher de cada equipo) con el promedio de outs por apertura del mismo
    pitcher en la temporada regular de ese año (≥ 8 aperturas; los openers no cuentan)."""
    rows = []
    for year, games in (post or {}).items():
        for pk, g in (games or {}).items():
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
                avg = outs_of(rs["inningsPitched"]) / gs
                # v2: solo abridores de verdad (un relevista usado de abridor infla IP/GS: 94 IP en 9 aperturas = 31 outs)
                bfgs = (rs.get("battersFaced") or 0) / gs
                gp = rs.get("gamesPlayed")
                if avg <= 0 or avg > 21 or bfgs > 28 or (gp and gs / gp < 0.8):
                    continue
                rows.append({"year": year, "id": st["id"], "outs": outs_of(st.get("ip")), "avg": avg,
                             "pitches": st.get("pitches"), "avgPitches": (rs.get("numberOfPitches") or 0) / gs or None})
    return rows


def hook_factor(post: dict, reg: dict, boot: int = 400, seed: int = 31) -> dict:
    """Gancho de postemporada, medido de dos formas:

    * cociente: outs en postemporada / su promedio de temporada (media geométrica con intervalo por bootstrap);
    * regresión: outs_post = α + β·(promedio de temporada). Si β ≈ 0 el gancho lo decide el partido (el mánager),
      no la profundidad habitual del abridor. β se acota a [0, 1] para usarlo.

    La salida del abridor en el modelo usa la regresión (media) y la dispersión medida en postemporada (forma de
    la Weibull por el coeficiente de variación de los residuos)."""
    rows = post_starts(post, reg)
    if len(rows) < 20:
        return {"hook": 0.90, "ci": [0.85, 0.95], "n": len(rows), "measured": False, "pitchRatio": None, "byYear": {},
                "alpha": None, "beta": None, "betaUsed": 1.0, "meanPost": None, "meanReg": None, "sdResid": None, "shape": None}
    ratios = [max(0.05, r["outs"] / r["avg"]) for r in rows]
    lg_ = [math.log(x) for x in ratios]
    h = math.exp(sum(lg_) / len(lg_))
    rng = random.Random(seed)
    bs = sorted(math.exp(sum(lg_[rng.randrange(len(lg_))] for _ in lg_) / len(lg_)) for _ in range(boot))
    xs, ys = [r["avg"] for r in rows], [r["outs"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    alpha = my - beta * mx
    res = [y - alpha - beta * x for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / (len(xs) - 2)
    se_b = math.sqrt(s2 / sxx) if sxx else float("inf")
    # v2: pendiente robusta (Theil-Sen: mediana de las pendientes por pares), la que se usa; OLS queda como referencia
    sl = sorted((y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(zip(xs, ys))
                for x2, y2 in list(zip(xs, ys))[i + 1:] if abs(x2 - x1) > 1e-9)
    beta_ts = sl[len(sl) // 2] if sl else 0.0
    pitches = [r["pitches"] / r["avgPitches"] for r in rows if r.get("pitches") and r.get("avgPitches")]
    by = {}
    for r in rows:
        a = by.setdefault(r["year"], [0, 0.0, 0.0])
        a[0] += 1
        a[1] += r["outs"]
        a[2] += r["avg"]
    cv = math.sqrt(s2) / my if my else 0.35
    return {"hook": h, "ci": [bs[int(0.05 * boot)], bs[int(0.95 * boot) - 1]], "n": len(rows), "measured": True,
            "pitchRatio": (sum(pitches) / len(pitches)) if pitches else None,
            "alpha": alpha, "beta": beta, "seBeta": se_b, "betaTS": beta_ts, "betaUsed": min(1.0, max(0.0, beta_ts)),
            "meanPost": my, "meanReg": mx,
            "seMean": math.sqrt(s2 / len(xs)), "sdResid": math.sqrt(s2), "shape": max(1.5, min(12.0, cv ** -1.086)),
            "byYear": {y: {"starts": a[0], "outsPost": a[1] / a[0], "outsReg": a[2] / a[0]} for y, a in sorted(by.items())}}


def post_outs(hook: dict, avg_reg: float) -> float:
    """Outs esperados del abridor en postemporada según la regresión medida (β acotado a [0, 1])."""
    if not hook.get("measured") or hook.get("meanPost") is None:
        return avg_reg * hook.get("hook", 0.9)
    return hook["meanPost"] + hook["betaUsed"] * (avg_reg - hook["meanReg"])


def tto_offsets(bf_by_inning: float, frac: list[float]) -> list[float]:
    """Castigo medio por vuelta al lineup (wOBA) en cada entrada del abridor, según los bateadores que lleva."""
    out = [0.0]
    faced = 0.0
    for i in range(1, 10):
        f = frac[i]
        start, end = faced, faced + f * bf_by_inning
        if end <= start:
            out.append(0.0)
            continue
        acc = 0.0
        x = start
        while x < end - 1e-9:
            j = min(3, int(x // 9))
            nxt = min(end, (j + 1) * 9 if j < 3 else end)
            acc += (nxt - x) * TTO_WOBA[j]
            x = nxt
        out.append(acc / (end - start))
        faced = end
    return out


def season_tto_mean(bf_start: float) -> float:
    """Castigo medio por vuelta de un abridor que enfrenta `bf_start` bateadores (su promedio de temporada)."""
    acc, x = 0.0, 0.0
    while x < bf_start - 1e-9:
        j = min(3, int(x // 9))
        nxt = min(bf_start, (j + 1) * 9 if j < 3 else bf_start)
        acc += (nxt - x) * TTO_WOBA[j]
        x = nxt
    return acc / bf_start if bf_start else 0.0


ROLE_W = {"Cerrador": 1.6, "Setup": 1.5, "Intermedio": 1.0, "Largo": 0.35, "Opener": 0.6}


def october_pen(relievers: list[dict], ra9_lg: float, k_ip: float = 40) -> dict:
    """Bullpen de octubre: RA9 esperado de cada relevista (FIP encogido) ponderado por su uso en postemporada
    (los de confianza cargan más: rol × probabilidad de uso con descanso completo). Devuelve también el de
    temporada regular (ponderado por entradas) para comparar."""
    rows = []
    for r in relievers or []:
        if r.get("starterLike"):
            continue
        fip = r.get("fipShrunk") or r.get("fip") or ra9_lg
        ip = r.get("ip") or 0
        ra9 = (ip * fip * 1.08 + k_ip * ra9_lg) / (ip + k_ip)    # FIP → RA9 (≈ +8% por carreras no limpias)
        w_oct = ROLE_W.get(r.get("role"), 0.8) * max(0.25, min(1.0, (r.get("pUse") or 0.3) / 0.45))
        rows.append({"id": r.get("id"), "name": r.get("name"), "role": r.get("role"), "ip": ip, "fip": fip, "ra9": ra9,
                     "wReg": ip, "wOct": w_oct})
    if not rows:
        return {"ra9": ra9_lg, "ra9Reg": ra9_lg, "rows": []}
    so, sr = sum(x["wOct"] for x in rows), sum(x["wReg"] for x in rows) or 1
    for x in rows:
        x["shareOct"], x["shareReg"] = x["wOct"] / so, x["wReg"] / sr
    rows.sort(key=lambda x: -x["shareOct"])
    return {"ra9": sum(x["shareOct"] * x["ra9"] for x in rows), "ra9Reg": sum(x["shareReg"] * x["ra9"] for x in rows), "rows": rows}


# ============================================================ carreras por entrada

def runs_by_inning(rpi: list[float], park: float, frac: list[float], sp_off: list[float], pen_off: float) -> list[float]:
    """λ por entrada = carreras de la liga en esa entrada × parque × [parte del abridor × ofensiva vs abridor +
    parte del bullpen × ofensiva vs bullpen]."""
    return [0.0] + [rpi[i] * park * (frac[i] * sp_off[i] + (1 - frac[i]) * pen_off) for i in range(1, 10)]


def off_mult(woba: float, lg_woba: float, rpa: float) -> float:
    return max(0.3, (rpa + (woba - lg_woba) / WOBA_SCALE) / rpa)


def k_dist(k_rates: list[float], bf_probs: list[float]) -> list[float]:
    """Distribución de ponches del abridor: bateador por bateador (Poisson-binomial) con la probabilidad de que
    siga en el juego para enfrentarlo."""
    dist = [1.0]
    for p, s in zip(k_rates, bf_probs):
        q = p * s
        nxt = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nxt[k] += v * (1 - q)
            nxt[k + 1] += v * q
        dist = nxt
    return dist


# nodos y pesos de Gauss-Hermite (7 puntos) para una Normal estándar
_GH = [(-3.750439717725742, 0.000548268855972), (-2.366759410734541, 0.030757123967586), (-1.154405394739968, 0.240123178605013),
       (0.0, 0.457142857142857), (1.154405394739968, 0.240123178605013), (2.366759410734541, 0.030757123967586),
       (3.750439717725742, 0.000548268855972)]


def k_dist_form(k_rates: list[float], shape: float, mean_outs: float, bf_per_out: float, rho: float, b_outs: float) -> list[float]:
    """Ponches del abridor mezclando sobre la forma del día (v2): con z ~ Normal, la tasa de ponche de cada bateador
    se mueve en la escala logit con varianza ρ/(p(1−p)) (Beta-Binomial) y la salida del abridor b·Δp outs (el día que
    poncha más, dura más). Sin forma (ρ = 0) es la Poisson-binomial de antes."""
    if not k_rates:
        return [1.0]
    pbar = sum(k_rates) / len(k_rates)
    sd = math.sqrt(rho / (pbar * (1 - pbar))) if rho > 0 else 0.0
    nodes = _GH if sd > 0 else [(0.0, 1.0)]
    mix = []
    for z, w in nodes:
        rates_z = [from_odds(odds(p) * math.exp(z * sd)) for p in k_rates]
        dp = sum(rates_z) / len(rates_z) - pbar
        m = max(3.0, mean_outs + b_outs * dp)
        dist = outs_dist(shape, m / math.gamma(1 + 1 / shape))
        # el bateador j (desde 0) llega si el abridor sigue: outs > (j + ½)/(bateadores por out); con int(j/bpo) se
        # contaban ~0.5 bateadores de más por salida (validado: 22.9 esperados contra 22.2 reales → 22.4)
        surv = [sum(pr for k, pr in enumerate(dist) if k > (j + 0.5) / bf_per_out) for j in range(len(rates_z))]
        kd = k_dist(rates_z, surv)
        if len(mix) < len(kd):
            mix += [0.0] * (len(kd) - len(mix))
        for k, v in enumerate(kd):
            mix[k] += w * v
    tot = sum(mix)
    return [v / tot for v in mix]


def reading(o: dict, a_ab: str, h_ab: str) -> list[str]:
    """Lectura en palabras: qué pesó en el modelo y qué no."""
    out = []
    kap = o["A"]["kappa"]
    ks = [f"{EV_NAMES[e].lower()} κ={kap[e]['kappa']:,.0f}" for e in EVENTS if kap.get(e) and kap[e].get("kappa")]
    none = [EV_NAMES[e].lower() for e in EVENTS if kap.get(e) and kap[e].get("pairs") and kap[e].get("kappa") is None]
    out.append("Historial contra el rival: en la liga, la parte propia de cada pareja pitcher-rival se mide así: "
               + (", ".join(ks) if ks else "sin señal medible") + (f"; sin señal en {', '.join(none)}" if none else "")
               + ". Un κ grande quiere decir que 30 o 40 bateadores contra un equipo casi no dicen nada nuevo.")
    rd = o["A"].get("rhoDay")
    if rd and rd.get("k"):
        out.append(f"Forma del día (v2): de una apertura a otra la tasa de ponche de un abridor se mueve ±"
                   f"{100 * math.sqrt(rd['k']['rho'] * 0.3 * 0.7):.1f} pp sin importar el rival (ρ = {rd['k']['rho']:.4f}, "
                   f"{rd['k']['groups']} parejas con ≥ 2 aperturas). Esa varianza ya no se confunde con «efecto del rival» y "
                   "entra a la distribución de ponches.")
    for s, ab in (("away", a_ab), ("home", h_ab)):
        st = o["A"]["starters"][s]
        if st:
            k = st["events"]["k"]
            prev = f" (+{round(st['prevBf'])} en 2025, a la mitad)" if st.get("prevBf") else ""
            out.append(f"{st['name']} ({ab}) contra su rival: {round(st['bf'])} bateadores en 2026{prev}; el ponche pasa de "
                       f"{100 * k['mu']:.1f}% esperado a {100 * k['post']:.1f}% (el historial pesa {100 * k['w']:.0f}%).")
    h = o["C"]["hook"]
    if h.get("measured"):
        bt = h.get("betaTS")
        dep = (f"y por cada out más de promedio en temporada duran {bt:+.2f} outs más en octubre (pendiente Theil-Sen, robusta; "
               f"OLS {h['beta']:+.2f} ± {h['seBeta']:.2f})" if bt is not None else
               f"y casi sin importar cuánto duran normalmente (β = {h['beta']:+.2f} ± {h['seBeta']:.2f})")
        out.append(f"Gancho de postemporada medido en {h['n']} aperturas de abridores de verdad (2024-2025): sacan {h['meanPost']:.1f} outs "
                   f"contra {h['meanReg']:.1f} en temporada ({100 * h['hook']:.0f}%), {dep}. Eso recorta la 3.ª vuelta al lineup, la "
                   "parte donde más castigan los bateadores, y pasa más entradas al bullpen de confianza.")
    else:
        out.append("Gancho de postemporada supuesto (no hubo suficientes aperturas para medirlo).")
    ab = o["ablation"]
    out.append(f"Qué mueve la predicción (P local): completo {100 * o['game']['pHome']:.1f}% · sin historial contra el rival "
               f"{100 * ab['noA']:.1f}% · sin matchup por lanzamiento {100 * ab['noB']:.1f}% · con gancho de temporada regular {100 * ab['noC']:.1f}%.")
    fam = o["D"]
    if fam.get("used"):
        out.append(f"Familiaridad: en la liga, cada vez que el rival ya vio al abridor en la temporada suben sus carreras "
                   f"{100 * fam['runs']['slope']:+.1f}% (t = {fam['runs']['t']:.1f}, controlando por la fecha); entra encogido a "
                   f"{100 * fam['runs']['shrunk']:+.1f}% por enfrentamiento previo.")
    else:
        out.append("Familiaridad (variable de prueba): el efecto por enfrentamiento previo no es claro en la liga (|t| < 2); se reporta y no se usa.")
    return out
