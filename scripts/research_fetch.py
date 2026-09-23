"""Descarga documentos de la investigación y vuelca su texto al log de Actions (no se guarda en el repo)."""
import html
import re
import subprocess
import sys
import tempfile
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research; MLBgs)"}
KEYS = [k.lower() for k in sys.argv[1:]]


def text_of(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
    if body[:5] == b"%PDF-" or "pdf" in ctype:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(body)
            f.flush()
            out = subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, timeout=120)
            return out.stdout.decode("utf-8", "replace")
    t = body.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style|nav|header|footer|svg)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</h[1-6]>|</li>|</tr>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(t)


for line in open("scripts/research_urls.txt", encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    key, url = line.split("|", 1)
    if KEYS and key.lower() not in KEYS:
        continue
    try:
        t = text_of(url)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n\s*\n+", "\n", t).strip()
        print(f"\n@@@@@ DOC {key} | {url} | {len(t)} chars @@@@@", flush=True)
        print(t[:90000], flush=True)
        print(f"@@@@@ END {key} @@@@@", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n@@@@@ DOC {key} | {url} | ERROR {e!r} @@@@@", flush=True)
