"""Fórmulas del framework v2 (sección 5.7) y utilidades de probabilidad.

Todo es Python estándar para que el pipeline corra en GitHub Actions sin dependencias.
"""
from __future__ import annotations

import math
from typing import Sequence

MAX_RUNS = 30  # soporte de las distribuciones de carreras


# ---------------------------------------------------------------- distribuciones

def poisson_pmf(lam: float, n: int = MAX_RUNS) -> list[float]:
    """P(X = k) = e^-λ λ^k / k!  (ecuación 12)."""
    lam = max(lam, 1e-9)
    out = [math.exp(-lam)]
    for k in range(1, n + 1):
        out.append(out[-1] * lam / k)
    return _normalize(out)


def negbin_pmf(mean: float, var_ratio: float, n: int = MAX_RUNS) -> list[float]:
    """Binomial Negativa con E[X] = mean y Var[X] = var_ratio · mean (ecuación 18).

    Parametrización del framework: E = r p / (1 - p), Var = r p / (1 - p)^2
    ⇒ Var / E = 1 / (1 - p) ⇒ p = 1 - 1/var_ratio, r = mean (1 - p) / p.
    Con var_ratio <= 1 se degenera a Poisson.
    """
    if var_ratio <= 1.0001:
        return poisson_pmf(mean, n)
    mean = max(mean, 1e-9)
    p = 1.0 - 1.0 / var_ratio
    r = mean * (1.0 - p) / p
    out = [(1.0 - p) ** r]
    for k in range(1, n + 1):
        out.append(out[-1] * (k + r - 1) / k * p)
    return _normalize(out)


def _normalize(pmf: list[float]) -> list[float]:
    s = sum(pmf)
    return [x / s for x in pmf]


def outcome_probs(pmf_a: Sequence[float], pmf_b: Sequence[float]) -> tuple[float, float, float]:
    """(P(A > B), P(A = B), P(A < B)) con carreras independientes (ecuaciones 13-14)."""
    win = tie = 0.0
    cdf_b = []
    acc = 0.0
    for x in pmf_b:
        acc += x
        cdf_b.append(acc)
    for i, pa in enumerate(pmf_a):
        if i > 0:
            win += pa * cdf_b[min(i - 1, len(cdf_b) - 1)]
        if i < len(pmf_b):
            tie += pa * pmf_b[i]
    return win, tie, max(0.0, 1.0 - win - tie)


def margin_probs(pmf_a: Sequence[float], pmf_b: Sequence[float]) -> dict[int, float]:
    """Distribución del margen A - B."""
    out: dict[int, float] = {}
    for i, pa in enumerate(pmf_a):
        if pa < 1e-12:
            continue
        for j, pb in enumerate(pmf_b):
            out[i - j] = out.get(i - j, 0.0) + pa * pb
    return out


def sum_pmf(pmf_a: Sequence[float], pmf_b: Sequence[float]) -> list[float]:
    """Convolución: distribución del total A + B."""
    out = [0.0] * (len(pmf_a) + len(pmf_b) - 1)
    for i, pa in enumerate(pmf_a):
        if pa < 1e-12:
            continue
        for j, pb in enumerate(pmf_b):
            out[i + j] += pa * pb
    return out


def over_under(pmf: Sequence[float], line: float) -> dict[str, float]:
    """P(Over), P(Under) y P(Push) para una línea (entera o .5)."""
    over = sum(p for k, p in enumerate(pmf) if k > line)
    under = sum(p for k, p in enumerate(pmf) if k < line)
    push = pmf[int(line)] if float(line).is_integer() and int(line) < len(pmf) else 0.0
    return {"over": over, "under": under, "push": push}


def mean_of(pmf: Sequence[float]) -> float:
    return sum(k * p for k, p in enumerate(pmf))


def var_of(pmf: Sequence[float]) -> float:
    m = mean_of(pmf)
    return sum((k - m) ** 2 * p for k, p in enumerate(pmf))


# ---------------------------------------------------------------- sabermetría

def era(er: float, ip: float) -> float | None:
    return er * 9.0 / ip if ip > 0 else None


def whip(h: float, bb: float, ip: float) -> float | None:
    return (h + bb) / ip if ip > 0 else None


def fip(hr: float, bb: float, hbp: float, k: float, ip: float, c_fip: float) -> float | None:
    """FIP = (13·HR + 3·(BB+HBP) − 2·K) / IP + C_liga  (ecuación 5)."""
    return (13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k) / ip + c_fip if ip > 0 else None


def fip_constant(lg_era: float, lg_hr: float, lg_bb: float, lg_hbp: float, lg_k: float, lg_ip: float) -> float:
    """Constante de liga: C = ERA_liga − (13·HR + 3·(BB+HBP) − 2·K)/IP de toda la liga."""
    return lg_era - (13.0 * lg_hr + 3.0 * (lg_bb + lg_hbp) - 2.0 * lg_k) / lg_ip


def ip_to_float(ip) -> float:
    """'123.2' (notación de béisbol: .1 = 1 out) → 123.667."""
    if ip is None:
        return 0.0
    s = str(ip)
    if "." in s:
        whole, frac = s.split(".", 1)
        return int(whole or 0) + int(frac[:1] or 0) / 3.0
    return float(s)


def pythagenpat_exponent(rs: float, ra: float, g: float) -> float:
    """n = ((RS + RA)/G)^0.287  (Pythagenpat, sección 5.7.1)."""
    if g <= 0 or rs + ra <= 0:
        return 1.83
    return ((rs + ra) / g) ** 0.287


def pythagorean(rs: float, ra: float, n: float = 1.83) -> float:
    """Win% esperado = RS^n / (RS^n + RA^n)  (ecuación 15)."""
    if rs <= 0 and ra <= 0:
        return 0.5
    return rs ** n / (rs ** n + ra ** n)


def log5(pa: float, pb: float) -> float:
    """P(A gana) = (pA − pA·pB) / (pA + pB − 2·pA·pB)  (ecuación 16)."""
    den = pa + pb - 2 * pa * pb
    return 0.5 if den <= 0 else (pa - pa * pb) / den


def shrink(observed: float, n: float, k: float, league: float) -> float:
    """θ̂ = n/(n+k)·x̄ + k/(n+k)·μ_liga  (ecuación 17)."""
    if n <= 0:
        return league
    return (n / (n + k)) * observed + (k / (n + k)) * league


def elo_expected(ra: float, rb: float) -> float:
    """E_A = 1 / (1 + 10^((R_B − R_A)/400))  (ecuación 20)."""
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def prob_to_elo_diff(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def with_home_edge(p: float, home_odds_ratio: float) -> float:
    """Aplica ventaja de local multiplicando los momios (odds) de p."""
    o = p / (1 - p) * home_odds_ratio
    return o / (1 + o)


# ---------------------------------------------------------------- mercado

def american_to_prob(m: float) -> float:
    """Probabilidad implícita (ecuaciones 26-27)."""
    return abs(m) / (abs(m) + 100.0) if m < 0 else 100.0 / (m + 100.0)


def american_to_decimal(m: float) -> float:
    return 1.0 + (100.0 / abs(m) if m < 0 else m / 100.0)


def fair_american(p: float) -> float | None:
    """Momio justo (ecuación 29)."""
    if p <= 0 or p >= 1:
        return None
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def kelly(p: float, american: float) -> float:
    """f* = (b·p − q)/b con b = momio neto decimal (ecuación 21)."""
    b = american_to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1 - p)) / b)


def zscore(x: float, values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return 0.0 if sd == 0 else (x - m) / sd
