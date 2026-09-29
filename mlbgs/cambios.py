"""Qué cambió desde la revisión anterior, para enfocarse en eso.

Cada corrida de update.yml (cada ~12–25 min en horario de juegos; ver mlbgs/cadencia.py) vuelve a leer la MLB
(abridores, lineups, umpire, clima, horario y movimientos del roster) y los momios. Este módulo compara cada
partido que todavía no empieza contra la foto de la corrida anterior y anota lo que cambió:

  abridor       anunciado · cambiado (alto: el modelo cambia y el momio visto antes ya no sirve)
  lineup        confirmado · cambiado (quién entra y quién sale) · solo el orden
  umpire        asignado · cambiado
  clima         viento (≥ 5 mph o cambia a favor/en contra), temperatura (≥ 10 °F), lluvia
  horario       cambia la hora (≥ 15 min) · retrasado o pospuesto
  movimiento    nuevo movimiento oficial del equipo (lista de lesionados, llamados, bajas)
  momio         el moneyline se mueve ≥ 3 pp de probabilidad o cambia la línea del total; mercado abierto
  verificacion  DraftKings (ESPN) no coincide con el consenso de las otras casas
  decision      cambia la decisión del partido (pick, stake o apostar/esperar/no apostar)

Lo que cambia se atiende en la misma corrida: un partido con cambio de abridor, lineup, horario o baja se
vuelve «caliente» y sus momios se revisan aunque no les toque por horario (ESPN, sin costo) y, si el cambio fue
de abridor, los mercados de The Odds API que ya se habían pedido se piden de nuevo (dentro de los topes).
El momio que se vio ANTES de un cambio de abridor no cuenta para el stake (`stale_after`).

data/cambios/<fecha>.json guarda, por partido, la foto de la última corrida (`snap`) y la lista de eventos
(`events`). La primera foto de un partido es la base (no genera eventos). Un dato que la API deja de mandar un
momento (lineup o abridor vacío) no cuenta como cambio: la foto conserva el último valor visto.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics

from . import mathlib as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "cambios")

IMPACT = {"alto": 3, "medio": 2, "bajo": 1}
HOT_KINDS = ("abridor", "lineup", "horario", "movimiento")   # piden revisar los momios en la misma corrida
ML_MOVE, ML_MOVE_HIGH = 0.03, 0.06        # movimiento del moneyline en probabilidad sin vig
KEEP_EVENTS = 400
TITLES = {"abridor": "Abridor", "lineup": "Lineup", "umpire": "Umpire", "clima": "Clima", "horario": "Horario",
          "movimiento": "Movimiento", "momio": "Momio", "verificacion": "Verificación", "decision": "Decisión"}


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(x) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")) if x else None
    except ValueError:
        return None


# ------------------------------------------------------------------ foto de un partido

def wind(text: str | None) -> dict:
    """'12 mph, Out To CF' → {"speed": 12, "dir": "out"}; dir: out · in · cruzado · calma."""
    m = re.match(r"\s*(\d+)\s*mph,?\s*(.*)", text or "")
    if not m:
        return {"speed": None, "dir": None}
    d = m.group(2).strip().lower()
    kind = "out" if d.startswith("out") else "in" if d.startswith("in") else "calma" if (
        d.startswith("calm") or int(m.group(1)) == 0) else "cruzado"
    return {"speed": int(m.group(1)), "dir": kind}


def closed_roof(g: dict) -> bool:
    cond = ((g.get("weather") or {}).get("condition") or "")
    return bool(re.search(r"dome|roof closed", cond, re.I)) or (g.get("roof") or "").lower() == "dome"


def game_snapshot(g: dict, bundle: dict) -> dict:
    pitchers = bundle.get("pitchers") or {}
    names = {}
    for side in ("away", "home"):
        names.update({str(k): v for k, v in ((g.get("lineupNames") or {}).get(side) or {}).items()})
    prob = {}
    for side in ("away", "home"):
        pid = (g.get("probable") or {}).get(side)
        prob[side] = {"id": pid, "name": (pitchers.get(str(pid)) or {}).get("name") or (str(pid) if pid else None)} if pid else None
    ump = next((o.get("name") for o in g.get("officials") or [] if o.get("type") == "Home Plate"), None)
    w = g.get("weather") or {}
    try:
        temp = float(w.get("temp")) if w.get("temp") not in (None, "") else None
    except (TypeError, ValueError):
        temp = None
    return {
        "time": g.get("time"), "detailed": g.get("detailed"),
        "away": g.get("away"), "home": g.get("home"),
        "prob": prob,
        "lu": {s: list((g.get("lineups") or {}).get(s) or []) for s in ("away", "home")},
        "names": {k: v for k, v in names.items()},
        "ump": ump,
        "wx": {"cond": w.get("condition"), "temp": temp, **wind(w.get("wind")), "closed": closed_roof(g)} if w else None,
    }


def abbr(bundle: dict, tid) -> str:
    t = (bundle.get("teams") or {}).get(str(tid)) or {}
    return t.get("abbr") or str(tid)


# ------------------------------------------------------------------ comparaciones

def _ev(now, g, kind, impact, text, side=None, before=None, after=None, action=None, teams=None) -> dict:
    ev = {"at": _iso(now), "pk": g["pk"], "date": g.get("date"), "kind": kind, "impact": impact, "text": text}
    if teams:
        ev["game"] = teams
    if side:
        ev["side"] = side
    if before is not None:
        ev["before"] = before
    if after is not None:
        ev["after"] = after
    if action:
        ev["action"] = action
    return ev


def compare_mlb(old: dict, new: dict, g: dict, now: dt.datetime, label: dict, teams: str) -> tuple[list[dict], dict]:
    """Eventos de un partido entre dos fotos; devuelve (eventos, foto a guardar)."""
    evs, keep = [], dict(new)
    mk = lambda *a, **k: _ev(now, g, *a, teams=teams, **k)  # noqa: E731
    # abridores: un vacío no borra al anunciado (la API a veces lo deja de mandar un momento)
    keep["prob"] = {}
    keep["probChangedAt"] = old.get("probChangedAt")
    for side in ("away", "home"):
        o, n = (old.get("prob") or {}).get(side), new["prob"].get(side)
        keep["prob"][side] = n or o
        if n and not o:
            evs.append(mk("abridor", "medio", f"Abridor anunciado ({label[side]}): {n['name']}", side=side,
                          after=n["name"], action="El modelo ya usa a este abridor; momios revisados en esta corrida."))
        elif n and o and n["id"] != o["id"]:
            keep["probChangedAt"] = _iso(now)
            evs.append(mk("abridor", "alto", f"Cambio de abridor ({label[side]}): {o['name']} → {n['name']}", side=side,
                          before=o["name"], after=n["name"],
                          action="Modelo recalculado y momios revisados en esta corrida; el momio visto antes del "
                                 "cambio ya no cuenta para el stake."))
    # lineups: solo cuando la MLB los publica (un lineup vacío no borra el anterior)
    keep["lu"] = {}
    names = dict(old.get("names") or {}, **(new.get("names") or {}))
    keep["names"] = names
    nm = lambda i: names.get(str(i)) or str(i)  # noqa: E731
    for side in ("away", "home"):
        o, n = (old.get("lu") or {}).get(side) or [], new["lu"].get(side) or []
        keep["lu"][side] = n or o
        if not n:
            continue
        if not o:
            evs.append(mk("lineup", "medio", f"Lineup confirmado ({label[side]})", side=side,
                          action="El modelo pasa del lineup proyectado al confirmado."))
            continue
        out_, in_ = [i for i in o if i not in n], [i for i in n if i not in o]
        if out_ or in_:
            parts = ([f"sale {', '.join(nm(i) for i in out_)}"] if out_ else []) + (
                [f"entra {', '.join(nm(i) for i in in_)}"] if in_ else [])
            evs.append(mk("lineup", "alto", f"Cambio en el lineup ({label[side]}): {'; '.join(parts)}", side=side,
                          before=[nm(i) for i in out_], after=[nm(i) for i in in_],
                          action="Modelo recalculado con el lineup nuevo; momios revisados en esta corrida."))
        elif o != n:
            evs.append(mk("lineup", "bajo", f"Cambia el orden al bate ({label[side]})", side=side))
    # umpire de home
    if new.get("ump") and new["ump"] != old.get("ump"):
        if old.get("ump"):
            evs.append(mk("umpire", "medio", f"Cambio de umpire de home: {old['ump']} → {new['ump']}",
                          before=old["ump"], after=new["ump"]))
        else:
            evs.append(mk("umpire", "bajo", f"Umpire de home asignado: {new['ump']}", after=new["ump"]))
    keep["ump"] = new.get("ump") or old.get("ump")
    # clima
    ow, nw = old.get("wx") or {}, new.get("wx") or {}
    keep["wx"] = nw or ow
    if ow and nw and not nw.get("closed"):
        sp_o, sp_n = ow.get("speed"), nw.get("speed")
        # a favor (out) o en contra (in) del bateo: cuenta si cambia la dirección o la fuerza (≥ 5 mph)
        dir_change = ow.get("dir") != nw.get("dir") and bool({ow.get("dir"), nw.get("dir")} & {"in", "out"})
        if sp_o is not None and sp_n is not None and (
                (dir_change and max(sp_o, sp_n) >= 5) or (abs(sp_n - sp_o) >= 5 and nw.get("dir") in ("out", "in"))):
            evs.append(mk("clima", "medio", f"Viento: {sp_o} mph {ow.get('dir') or ''} → {sp_n} mph {nw.get('dir') or ''}".replace("  ", " "),
                          action="El total proyectado ya usa el viento nuevo."))
        if ow.get("temp") is not None and nw.get("temp") is not None and abs(nw["temp"] - ow["temp"]) >= 10:
            evs.append(mk("clima", "bajo", f"Temperatura: {ow['temp']:.0f} °F → {nw['temp']:.0f} °F"))
        rain = lambda c: bool(re.search(r"rain|drizzle|storm|lluv", c or "", re.I))  # noqa: E731
        if rain(nw.get("cond")) and not rain(ow.get("cond")):
            evs.append(mk("clima", "medio", f"Lluvia en el pronóstico: {nw.get('cond')}",
                          action="Riesgo de retraso: vigilar el horario."))
    # horario y estado
    to, tn = _ts(old.get("time")), _ts(new.get("time"))
    if to and tn and abs((tn - to).total_seconds()) >= 15 * 60:
        evs.append(mk("horario", "medio", f"Cambia la hora del partido ({(tn - to).total_seconds() / 60:+.0f} min)",
                      before=old.get("time"), after=new.get("time")))
    bad = lambda s: bool(re.search(r"delay|postpon|suspend|cancel", s or "", re.I))  # noqa: E731
    if bad(new.get("detailed")) and new.get("detailed") != old.get("detailed"):
        evs.append(mk("horario", "alto", f"Estado del partido: {new.get('detailed')}",
                      before=old.get("detailed"), after=new.get("detailed")))
    return evs, keep


def tx_events(bundle: dict, g: dict, seen: set, now: dt.datetime, teams: str, label_of, days: int = 3) -> list[dict]:
    """Movimientos oficiales nuevos de los dos equipos (de los últimos `days` días)."""
    today = now.date()
    lu = {str(i) for s in ("away", "home") for i in ((g.get("lineups") or {}).get(s) or [])}
    probs = {str(p) for p in (g.get("probable") or {}).values() if p}
    out = []
    for t in bundle.get("transactions") or []:
        if t.get("team") not in (g.get("away"), g.get("home")) or str(t.get("id")) in seen:
            continue
        try:
            d = dt.date.fromisoformat(str(t.get("date"))[:10])
        except ValueError:
            continue
        if (today - d).days > days:
            continue
        text = t.get("text") or f"{t.get('type')}: {t.get('name')}"
        il = "injured list" in text.lower() and ("placed" in text.lower() or "transferred" in text.lower())
        key = str(t.get("person")) in lu | probs
        imp = "alto" if (il and key) else "medio" if il or key else "bajo"
        side = "away" if t.get("team") == g.get("away") else "home"
        out.append(_ev(now, g, "movimiento", imp, f"{label_of(side)}: {text}", side=side, teams=teams,
                       action="El modelo ya usa el roster nuevo." if imp != "bajo" else None))
        out[-1]["who"] = t.get("name") or str(t.get("person"))
    # muchos movimientos menores del mismo equipo (p. ej. el roster de postemporada) van en una sola línea
    grouped = []
    for side in ("away", "home"):
        low = [e for e in out if e.get("side") == side and e["impact"] == "bajo"]
        if len(low) > 2:
            names = [e["who"] for e in low]
            grouped.append(_ev(now, g, "movimiento", "bajo", f"{label_of(side)}: {len(low)} movimientos del roster "
                               f"({', '.join(names[:6])}{'…' if len(names) > 6 else ''})", side=side, teams=teams))
            out = [e for e in out if e not in low]
    for e in out:
        e.pop("who", None)
    return out + grouped


def no_vig_home(ml: dict) -> float | None:
    if not ml or ml.get("home") is None or ml.get("away") is None:
        return None
    ih, ia = 1 / M.american_to_decimal(ml["home"]), 1 / M.american_to_decimal(ml["away"])
    return ih / (ih + ia)


def odds_snapshot(ev: dict, anchor: str = "draftkings") -> dict | None:
    """Del evento de momios del bundle (forma de The Odds API): moneyline y probabilidad sin vig del local, y la
    línea del total. Se sigue a una sola casa (DraftKings, la de ESPN) para que sumar casas no parezca movimiento;
    si no está, la mediana de las casas. Las marcadas «previo al cambio» no cuentan."""
    rows = {}
    for bk in ev.get("bookmakers") or []:
        if bk.get("stale"):
            continue
        r = {}
        for m in bk.get("markets") or []:
            if m.get("key") == "h2h":
                r["ml"] = {("home" if o.get("name") == ev.get("home_team") else "away"): o.get("price")
                           for o in m.get("outcomes") or []}
            elif m.get("key") == "totals":
                pts = [o.get("point") for o in m.get("outcomes") or [] if o.get("point") is not None]
                if pts:
                    r["line"] = pts[0]
        if r.get("ml") or r.get("line") is not None:
            rows[bk.get("key")] = r
    if not rows:
        return None
    if anchor in rows and rows[anchor].get("ml"):
        r = rows[anchor]
        return {"pHome": round(no_vig_home(r["ml"]), 4) if no_vig_home(r["ml"]) is not None else None,
                "line": r.get("line"), "ml": r["ml"], "src": anchor}
    ps = [p for p in (no_vig_home(r.get("ml")) for r in rows.values()) if p is not None]
    lines = [r["line"] for r in rows.values() if r.get("line") is not None]
    return {"pHome": round(statistics.median(ps), 4) if ps else None, "line": statistics.mode(lines) if lines else None,
            "ml": next((r["ml"] for r in rows.values() if no_vig_home(r.get("ml")) is not None), None),
            "src": "mediana"}


def fmt_am(x) -> str:
    return f"{int(x):+d}" if isinstance(x, (int, float)) else "—"


# ------------------------------------------------------------------ registro por fecha

def describe(x: dict) -> str:
    """apostar ATL ML · stake 6 · esperar momio (ATL ML) · no apostar."""
    if x.get("status") == "apostar":
        return f"apostar {x.get('pick') or ''} · stake {x.get('level')}"
    if x.get("status") == "esperar":
        return f"esperar {x.get('waitFor') or 'dato'}" + (f" ({x['pick']})" if x.get("pick") else "")
    return x.get("status") or "sin decisión"


class Tracker:
    """Lleva la foto y los eventos de cada fecha. Se usa en la descarga (MLB y momios) y en la construcción de
    la página (decisiones)."""

    def __init__(self, base: str | None = None, now: dt.datetime | None = None):
        self.base, self.now = base or DIR, now or dt.datetime.now(dt.timezone.utc)
        self.days: dict[str, dict] = {}
        self.new: list[dict] = []
        self.hot: dict[int, list[str]] = {}

    def day(self, date: str) -> dict:
        if date not in self.days:
            path = os.path.join(self.base, f"{date}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    self.days[date] = json.load(f)
            else:
                self.days[date] = {"date": date, "snap": {}, "events": []}
        return self.days[date]

    def _add(self, date: str, evs: list[dict]) -> None:
        if not evs:
            return
        d = self.day(date)
        d["events"] = (d.get("events") or []) + evs
        d["events"] = d["events"][-KEEP_EVENTS:]
        self.new += evs
        for e in evs:
            if e["kind"] in HOT_KINDS and IMPACT[e["impact"]] >= 2:
                self.hot.setdefault(e["pk"], [])
                if e["kind"] not in self.hot[e["pk"]]:
                    self.hot[e["pk"]].append(e["kind"])

    def mlb(self, bundle: dict) -> dict[int, list[str]]:
        """Compara abridores, lineups, umpire, clima, horario y movimientos; devuelve los partidos «calientes»."""
        games = [g for g in bundle.get("upcoming") or [] if g.get("date")]
        if not games and (bundle.get("meta") or {}).get("errors", {}).get("upcoming"):
            return self.hot                        # la MLB no respondió: no se compara nada
        # un movimiento del equipo se anota en su próximo partido (no en hoy y mañana a la vez)
        first = {}
        for g in sorted(games, key=lambda x: x.get("time") or ""):
            for t in (g["away"], g["home"]):
                first.setdefault(t, g["pk"])
        for g in games:
            d = self.day(g["date"])
            pk = str(g["pk"])
            label = {"away": abbr(bundle, g["away"]), "home": abbr(bundle, g["home"])}
            teams = f"{label['away']} @ {label['home']}"
            new = game_snapshot(g, bundle)
            old = d["snap"].get(pk)
            if old is None or "time" not in old:     # primera foto del partido: es la base, no genera eventos
                new.update({k: v for k, v in (old or {}).items() if k in ("dec", "odds", "verify")})
                new["tx"] = sorted({str(t.get("id")) for t in bundle.get("transactions") or []
                                    if t.get("team") in (g["away"], g["home"])})
                new["probChangedAt"] = None
                new["since"] = _iso(self.now)
                d["snap"][pk] = new
                continue
            evs, keep = compare_mlb(old, new, g, self.now, label, teams)
            seen = set(old.get("tx") or [])
            if bundle.get("transactions") is not None and not (bundle.get("meta") or {}).get("errors", {}).get("transactions"):
                mine = [t for t in bundle.get("transactions") or [] if t.get("team") in (g["away"], g["home"])]
                fresh = [t for t in mine if str(t.get("id")) not in seen and first.get(t.get("team")) == g["pk"]]
                evs += tx_events({"transactions": fresh}, g, seen, self.now, teams, lambda s: label[s])
                seen |= {str(t.get("id")) for t in mine}      # el de mañana no lo repite después
            keep["tx"] = sorted(seen)
            for k in ("odds", "dec", "verify", "since"):
                if k in old:
                    keep[k] = old[k]
            d["snap"][pk] = keep
            self._add(g["date"], evs)
        return self.hot

    def stale_after(self) -> dict[int, str]:
        """{pk: hora del último cambio de abridor}: el momio visto antes de esa hora no cuenta para el stake."""
        out = {}
        for d in self.days.values():
            for pk, s in (d.get("snap") or {}).items():
                if s.get("probChangedAt"):
                    out[int(pk)] = s["probChangedAt"]
        return out

    def odds(self, bundle: dict) -> None:
        """Movimiento del moneyline y de la línea del total, y la verificación entre casas."""
        by_pk = {e.get("pk"): e for e in bundle.get("odds") or [] if e.get("pk") is not None}
        for g in bundle.get("upcoming") or []:
            ev = by_pk.get(g["pk"])
            d = self.day(g["date"])
            snap = d["snap"].get(str(g["pk"]))
            if not ev or snap is None or ev.get("closed"):
                continue
            label = {"away": abbr(bundle, g["away"]), "home": abbr(bundle, g["home"])}
            teams = f"{label['away']} @ {label['home']}"
            cur = odds_snapshot(ev)
            old = snap.get("odds")
            evs = []
            if cur and old is None and "since" in snap and snap["since"] != _iso(self.now):
                ml = cur.get("ml") or {}
                evs.append(_ev(self.now, g, "momio", "bajo",
                               f"Mercado abierto: {label['away']} {fmt_am(ml.get('away'))} / {label['home']} "
                               f"{fmt_am(ml.get('home'))}" + (f" · total {cur['line']}" if cur.get("line") else ""),
                               teams=teams))
            elif cur and old:
                if cur.get("pHome") is not None and old.get("pHome") is not None:
                    dp = cur["pHome"] - old["pHome"]
                    if abs(dp) >= ML_MOVE:
                        fav = label["home"] if dp > 0 else label["away"]
                        ml_o, ml_n = old.get("ml") or {}, cur.get("ml") or {}
                        evs.append(_ev(self.now, g, "momio", "alto" if abs(dp) >= ML_MOVE_HIGH else "medio",
                                       f"El moneyline se mueve hacia {fav} ({abs(dp) * 100:.1f} pp): "
                                       f"{label['away']} {fmt_am(ml_o.get('away'))}→{fmt_am(ml_n.get('away'))}, "
                                       f"{label['home']} {fmt_am(ml_o.get('home'))}→{fmt_am(ml_n.get('home'))}",
                                       teams=teams, action="Edge, índice de confianza y stake recalculados con el momio nuevo."))
                if cur.get("line") is not None and old.get("line") is not None and cur["line"] != old["line"]:
                    evs.append(_ev(self.now, g, "momio", "medio", f"La línea del total pasa de {old['line']} a {cur['line']}",
                                   teams=teams, before=old["line"], after=cur["line"],
                                   action="Los picks de total usan la línea nueva."))
            if cur:
                # la foto solo avanza cuando hay movimiento (si no, un goteo lento nunca se vería)
                if old is None or any(e["kind"] == "momio" for e in evs):
                    snap["odds"] = cur
            vf = ev.get("verify") or {}
            flag = {k: v.get("status") for k, v in vf.items() if isinstance(v, dict)}
            was = snap.get("verify") or {}
            for mk_, st in flag.items():
                if st == "difiere" and was.get(mk_) != "difiere":
                    v = vf[mk_]
                    evs.append(_ev(self.now, g, "verificacion", "medio", f"{v.get('text') or 'Las casas no coinciden'}",
                                   teams=teams, action="El stake usa la mediana de las casas, no una sola."))
            if flag:
                snap["verify"] = flag
            self._add(g["date"], evs)

    def decisions(self, analyses: list[dict]) -> None:
        """Cambios en la decisión de cada partido (después de correr el modelo)."""
        for a in analyses:
            dec = a.get("decision") or {}
            if not a.get("date"):
                continue
            d = self.day(a["date"])
            snap = d["snap"].setdefault(str(a["pk"]), {"since": _iso(self.now)})
            pick = (dec.get("pick") or {}).get("pick")
            cur = {"status": dec.get("status"), "pick": pick, "level": (dec.get("stake") or {}).get("level", 0),
                   "waitFor": dec.get("waitFor")}
            old = snap.get("dec")
            snap["dec"] = cur
            if not old or old == cur:
                continue
            teams = f"{a['teams']['away']['abbr']} @ {a['teams']['home']['abbr']}"
            big = (old["level"] > 0) != (cur["level"] > 0) or (cur["level"] > 0 and old["pick"] != cur["pick"])
            recent = [e["kind"] for e in d.get("events") or [] if e.get("pk") == a["pk"] and e["kind"] != "decision"
                      and (self.now - (_ts(e["at"]) or self.now)).total_seconds() <= 3600]
            why = f" · tras: {', '.join(TITLES[k].lower() for k in dict.fromkeys(recent))}" if recent else ""
            self._add(a["date"], [_ev(self.now, {"pk": a["pk"], "date": a["date"]}, "decision",
                                      "alto" if big else "medio", f"Decisión: {describe(old)} → {describe(cur)}{why}",
                                      teams=teams, before=describe(old), after=describe(cur))])

    def save(self) -> list[str]:
        os.makedirs(self.base, exist_ok=True)
        paths = []
        for date, d in self.days.items():
            if not d.get("snap") and not d.get("events"):
                continue
            path = os.path.join(self.base, f"{date}.json")
            body = json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    if f.read() == body:
                        continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(body)
            paths.append(path)
        return paths


def recent(days: list[dict], now: dt.datetime | None = None, hours: int = 36) -> dict:
    """Para la página: eventos recientes de las fechas de la Jornada y un resumen por partido."""
    now = now or dt.datetime.now(dt.timezone.utc)
    events, per = [], {}
    for d in days:
        for e in d.get("events") or []:
            t = _ts(e.get("at"))
            if t and (now - t).total_seconds() <= hours * 3600:
                events.append(e)
        for pk, s in (d.get("snap") or {}).items():
            if s.get("probChangedAt") or s.get("verify"):
                per[pk] = {k: s[k] for k in ("probChangedAt", "verify") if s.get(k)}
    events.sort(key=lambda e: (e["at"], IMPACT.get(e["impact"], 0)), reverse=True)
    for e in events:
        p = per.setdefault(str(e["pk"]), {})
        p["n"] = p.get("n", 0) + 1
        if not p.get("last"):
            p["last"] = e["at"]
        if IMPACT[e["impact"]] > IMPACT.get(p.get("top"), 0):
            p["top"], p["topText"], p["topKind"], p["topAt"] = e["impact"], e["text"], e["kind"], e["at"]
    return {"events": events[:200], "games": per}
