"""Sonda: ¿cómo publica Playdoit sus momios? Descarga la portada y la sección de deportes, busca la
plataforma del sportsbook (iframes, scripts, dominios de API) y guarda un resumen en dev/playdoit/.
Pocas peticiones, sin iniciar sesión y sin saltar protecciones: solo lo que ve cualquier visitante."""
import json
import os
import re
import sys
import time
import urllib.request

OUT = "dev/playdoit"
os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
MARKERS = ["altenar", "kambi", "betconstruct", "sportradar", "digitain", "sbtech", "betby", "openbet", "playtech",
           "gan.com", "coolbet", "pinnacle", "bet365", "sportsbook", "odds", "biahosted", "sb2frontend", "swarm",
           "springbuilder", "vivaro", "bragg", "oddsmatrix", "everymatrix", "betradar", "lsports", "txodds", "gamingtec",
           "isoftbet", "pragmatic", "sisal", "codere", "apuestas", "deportes", "wss://", "graphql"]
report = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pages": []}


def get(url, accept="text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept, "Accept-Language": "es-MX,es;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read()
            return r.status, dict(r.headers), body, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read() if hasattr(e, "read") else b"", url
    except Exception as e:  # noqa: BLE001
        return None, {}, repr(e).encode(), url


def scan(name, url):
    st, hd, body, final = get(url)
    text = body.decode("utf-8", "replace")
    page = {"name": name, "url": url, "final": final, "status": st, "bytes": len(body),
            "server": hd.get("Server") or hd.get("server"), "cf": bool(hd.get("CF-RAY") or hd.get("cf-ray")),
            "scripts": sorted(set(re.findall(r'<script[^>]+src=["\']([^"\']+)', text)))[:80],
            "iframes": sorted(set(re.findall(r'<iframe[^>]+src=["\']([^"\']+)', text)))[:20],
            "hosts": sorted(set(re.findall(r'https?://([a-z0-9.-]+\.[a-z]{2,})', text, re.I)))[:120],
            "markers": {m: len(re.findall(re.escape(m), text, re.I)) for m in MARKERS if re.search(re.escape(m), text, re.I)},
            "title": (re.search(r"<title>(.*?)</title>", text, re.S | re.I) or [None, None])[1]}
    with open(os.path.join(OUT, f"{name}.html"), "w", encoding="utf-8") as f:
        f.write(text[:400000])
    report["pages"].append(page)
    return page, text


home, text = scan("home", "https://www.playdoit.mx/")
for path in ("/deportes", "/sports", "/apuestas-deportivas", "/es/sports", "/deportes/beisbol"):
    scan("p" + re.sub(r"\W", "_", path), "https://www.playdoit.mx" + path)
    time.sleep(1.5)
# los primeros scripts propios pueden traer la URL del API del sportsbook
js_hits = []
for src in [s for p in report["pages"] for s in p["scripts"]][:25]:
    url = src if src.startswith("http") else "https://www.playdoit.mx" + (src if src.startswith("/") else "/" + src)
    st, hd, body, _ = get(url, "*/*")
    t = body.decode("utf-8", "replace")
    found = sorted(set(re.findall(r'https?://[a-z0-9.-]+\.[a-z]{2,}[^"\'\s)]{0,80}', t, re.I)))
    api = [u for u in found if re.search(r"api|odds|sport|feed|event|market|widget|frontend", u, re.I)]
    js_hits.append({"src": url, "status": st, "bytes": len(body), "api": api[:40],
                    "markers": {m: len(re.findall(re.escape(m), t, re.I)) for m in MARKERS if re.search(re.escape(m), t, re.I)}})
    time.sleep(0.8)
report["js"] = js_hits
with open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print(json.dumps({p["name"]: (p["status"], p["bytes"], p["server"], p["cf"], list(p["markers"])[:8]) for p in report["pages"]}, ensure_ascii=False), file=sys.stderr)
