"""Descarga documentos de la investigación y vuelca extractos enfocados al log de Actions (no se guarda en el repo)."""
import html
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research; MLBgs)"}
HEAD = 2200          # caracteres del inicio (resumen / introducción)
WIN = 700            # ventana alrededor de cada coincidencia
CAP = 7000           # tope de extractos por documento
KW = {
    "stern1994": ["baseball", "sigma", "variance", "Lindsey", "drift", "probability of winning", "P(", "normal"],
    "polson_stern2015": ["implied volatility", "point spread", "moneyline", "sigma", "Black", "formula", "Phi"],
    "swing_mrp2025": ["Markov reward", "transition", "count", "terminal", "foul", "called strike", "value of"],
    "ttop_brill2023": ["continuous", "discontinu", "times through", "batter sequence", "fatigue", "estimate", "conclusion"],
    "glass_lowry2008": ["quasigeometric", "runs per inning", "extra inning", "parameter", "probability"],
    "miller_weibull": ["Weibull", "gamma", "independent", "fit", "chi"],
    "green_zwiebel": ["hot hand", "standard deviation", "home run", "strikeout", "pitcher", "magnitude"],
    "higher_order_markov": ["order", "transition", "lineup", "result", "accuracy"],
    "winprob_difficulty": ["baseball", "overfit", "calibration", "sample size", "variance", "conclusion"],
    "count_effects": ["transition", "count", "1-0", "0-1", "runs value", "3-0"],
    "runs_expectancy": ["runs expectancy", "24", "state", "matrix"],
    "ursin_markov_thesis": ["transition", "absorbing", "fundamental", "win probability", "lineup", "validation"],
    "hmm_pitching": ["hidden", "state", "transition", "emission", "good", "bad", "Viterbi"],
    "runs_per_inning_nb": ["negative binomial", "per inning", "zero", "overdispersion", "parameters"],
    "sabr_15_pitches": ["pitches per inning", "average", "starter", "distribution"],
    "pull_pitcher_trends": ["pitch count", "remove", "times through", "runs", "hazard", "survival"],
    "fangraphs_we": ["Win Expectancy", "calculated", "Tango", "run environment"],
    "bref_wpa": ["Win Expectancy", "Run Expectancy", "Leverage", "calculated", "Markov"],
    "savant_csv_docs": ["description", "events", "balls", "strikes", "n_thruorder", "delta_run_exp", "delta_home_win_exp", "type"],
    "mlb_rules_2026": ["pitch timer", "automatic ball", "automatic strike", "runner on second", "three batters", "challenge", "extra inning", "disengage"],
}


def text_of(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
    if body[:5] == b"%PDF-" or "pdf" in ctype:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(body)
            f.flush()
            out = subprocess.run(["pdftotext", f.name, "-"], capture_output=True, timeout=120)
            return out.stdout.decode("utf-8", "replace")
    t = body.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style|nav|header|footer|svg)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(t)


def digest(key, t):
    t = re.sub(r"\s+", " ", t).strip()
    spans = [(0, min(len(t), HEAD))]
    for kw in KW.get(key, []):
        for m in list(re.finditer(re.escape(kw), t, flags=re.I))[:4]:
            spans.append((max(0, m.start() - WIN // 2), min(len(t), m.end() + WIN // 2)))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    out, used = [], 0
    for a, b in merged:
        if used >= CAP + HEAD:
            break
        out.append(t[a:b])
        used += b - a
    return " […] ".join(out), len(t)


keys = [k.strip() for k in open("scripts/research_keys.txt", encoding="utf-8").read().split()] if len(sys.argv) < 2 else sys.argv[1:]
for line in open("scripts/research_urls.txt", encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    key, url = line.split("|", 1)
    if keys and key not in keys:
        continue
    try:
        d, n = digest(key, text_of(url))
        print(f"@@@@@ DOC {key} | {n} chars total @@@@@")
        print("\n".join(textwrap.wrap(d, 1500)))
    except Exception as e:  # noqa: BLE001
        print(f"@@@@@ DOC {key} | ERROR {e!r} @@@@@")
