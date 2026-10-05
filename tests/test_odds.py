"""Momios automáticos (mlbgs/odds.py) con respuestas que tienen la forma documentada de odds-api.net."""
import copy
import datetime as dt
import gzip
import json
import os
import tempfile
import unittest
import urllib.parse

from mlbgs import model
from mlbgs import odds as O
from mlbgs.cadencia import first_run_after

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bundle_small.json.gz")
ESPN_FX = os.path.join(os.path.dirname(__file__), "fixtures", "espn_scoreboard.json")   # respuesta real de ESPN (26-sep), movida a estos partidos
UTC = dt.timezone.utc


def ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def line(bk, typ, side, odds, line=None, name=None, **kw):
    sel = f"{typ}:{side}" + (f":{line}" if line is not None else "")
    return {"id": f"{bk}::{typ}::{sel}", "event_id": "x", "bookmaker": bk, "bookmaker_name": name or bk.title(),
            "selection_key": sel, "market_group_id": f"{typ}::0" + (f"::{abs(float(line))}" if line is not None else ""),
            "market_key": typ, "type": typ, "period": 0, "period_str": "full time", "line": line, "side": side,
            "odds": odds, "is_available": True, **kw}


def snapshot(ml_home, ml_away, shift=0.0):
    """Tres casas: Codere y bet365 (operan en México) y Pinnacle (no). Incluye ruido que debe ignorarse."""
    items = []
    for bk, d in (("codere", 0.0), ("bet365", 0.04), ("pinnacle", 0.10)):
        items += [line(bk, "moneyline", "home", round(ml_home + d + shift, 2)),
                  line(bk, "moneyline", "away", round(ml_away + d - shift, 2)),
                  line(bk, "handicap", "home", 2.30, "-1.5"), line(bk, "handicap", "away", 1.62, "1.5"),
                  line(bk, "handicap", "home", 3.40, "-2.5"), line(bk, "handicap", "away", 1.30, "2.5"),
                  line(bk, "total", "over", 1.91, "8.5"), line(bk, "total", "under", 1.91, "8.5"),
                  line(bk, "total", "over", 2.30, "9.5"), line(bk, "total", "under", 1.60, "9.5")]
    x3 = {"market_group_id": "moneyline3way::0"}                                       # 1X2 (con empate): se descarta
    items += [line("codere", "moneyline", "home", 9.0, player_name="Alguien"),          # prop de jugador
              dict(line("bet365", "moneyline", "home", 1.60), **x3), dict(line("bet365", "moneyline", "draw", 12.0), **x3),
              dict(line("bet365", "moneyline", "away", 2.50), **x3),
              dict(line("codere", "total", "over", 5.0, "7.5"), is_available=False)]   # suspendido
    return items


class FakeNet:
    """odds-api.net simulado: registra cada llamada y responde según la ruta."""

    def __init__(self, bundle):
        up = {g["pk"]: g for g in bundle["upcoming"]}
        self.events = [
            {"event_id": "e-was-det", "sport": "baseball", "league": "MLB", "start_time": ts(up[824223]["time"]),
             "home_team": "Detroit Tigers", "away_team": "Washington Nationals", "bookmakers": {}},
            {"event_id": "e-tor-bal-2", "sport": "baseball", "league": "MLB", "start_time": ts(up[824784]["time"]),
             "home_team": "Baltimore Orioles", "away_team": "Toronto Blue Jays", "bookmakers": {}},
            {"event_id": "e-tor-bal-1", "sport": "baseball", "league": "MLB", "start_time": ts(up[824785]["time"]),
             "home_team": "Baltimore Orioles", "away_team": "Toronto Blue Jays", "bookmakers": {}},
        ]
        self.prices = {"e-was-det": (1.73, 2.20), "e-tor-bal-1": (2.05, 1.80), "e-tor-bal-2": (1.95, 1.90)}
        self.shift = 0.0
        self.calls = []

    def __call__(self, url, headers=None):
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        self.calls.append((u.path, q, dict(headers or {})))
        path = u.path.replace("/v1", "", 1)
        if path == "/bookmakers":
            return {"items": [{"bookmaker": "codere", "country_codes": ["MX", "ES"]},
                              {"bookmaker": "bet365", "country_codes": ["MX", "UK"]}]}
        if path == "/usage":
            return {"plan": "free", "api_credits_used": 12, "api_credits_limit": 1000, "exceeded": False}
        if path == "/events":
            return {"items": [e for e in self.events if int(q["start_from"]) <= e["start_time"] <= int(q["start_to"])],
                    "next_cursor": None, "count": 3}
        if path.endswith("/odds/snapshot"):
            ev = path.split("/")[2]
            h, a = self.prices[ev]
            return {"event_id": ev, "as_of_ts_ms": 0, "items": snapshot(h, a, self.shift), "next_cursor": None}
        raise AssertionError(f"ruta inesperada {path}")


class MomiosOddsApiNet(unittest.TestCase):
    def setUp(self):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            self.bundle = json.load(f)
        self.dir = tempfile.mkdtemp()
        self.api = FakeNet(self.bundle)
        self.env = {"ODDS_API_NET_KEY": "clave-de-prueba", "ODDS_ESPN": "0"}     # aquí solo odds-api.net
        self.logs = []

    def run_at(self, when, env=None):
        return O.update(self.bundle, now=dt.datetime.fromisoformat(when).replace(tzinfo=UTC), base=self.dir,
                        env=self.env if env is None else env, fetch=self.api, log=self.logs.append)

    def test_conversion_decimal_americano(self):
        self.assertEqual(O.dec_to_am(1.73), -137)
        self.assertEqual(O.dec_to_am(2.15), 115)
        self.assertEqual(O.dec_to_am(2.0), 100)
        self.assertIsNone(O.dec_to_am(1.0))

    def test_renglones_ignoran_ruido(self):
        rows = O.rows_from_net(snapshot(1.73, 2.20))
        self.assertEqual(sorted(rows), ["bet365", "codere", "pinnacle"])
        c = rows["codere"]
        self.assertEqual(c["ml"], {"home": -137, "away": 120})
        self.assertEqual(c["rl"]["home"], {"point": -1.5, "price": 130})     # ±1.5 aunque haya −2.5
        self.assertEqual(c["rl"]["away"], {"point": 1.5, "price": -161})
        self.assertEqual(c["total"]["over"]["point"], 8.5)                   # la línea más pareja
        self.assertEqual(O.flip(c)["ml"], {"away": -137, "home": 120})

    def test_primera_corrida_apertura_y_llave(self):
        st = self.run_at("2026-09-23T15:00:00")
        self.assertEqual(st["provider"], "odds-api.net")
        self.assertIsNone(st["error"])
        self.assertEqual(st["refreshed"], 3)
        paths = [c[0] for c in self.api.calls]
        self.assertEqual(paths.count("/v1/events"), 1)
        snaps = [c for c in self.api.calls if c[0].endswith("/odds/snapshot")]
        self.assertEqual(sum(1 for c in snaps if c[1].get("periods") == "0"), 3)     # uno por partido
        self.assertEqual(sum(1 for c in snaps if "periods" not in c[1]), 1)           # la muestra completa del día
        u = O.load_usage(self.dir)
        self.assertEqual(u["netUsage"]["api_credits_limit"], 1000)
        self.assertIn("pinnacle", u["netProbe"]["books"])
        self.assertTrue(any(k.startswith("moneyline|moneyline|0") for k in u["netProbe"]["combos"]))
        self.assertTrue(all(c[2].get("X-API-Key") == "clave-de-prueba" for c in self.api.calls))
        snap = next(c for c in self.api.calls if c[0].endswith("/odds/snapshot") and "periods" in c[1])[1]
        self.assertEqual(snap["types"], "moneyline,handicap,total")
        day = O.load_day("2026-09-23", self.dir)
        # doble cartelera: cada juego con su evento según la hora
        self.assertEqual(day["games"]["824785"]["event"], "e-tor-bal-1")
        self.assertEqual(day["games"]["824784"]["event"], "e-tor-bal-2")
        b = day["games"]["824223"]["books"]["codere"]
        self.assertTrue(b["mx"])
        self.assertFalse(day["games"]["824223"]["books"]["pinnacle"]["mx"])
        self.assertEqual(b["open"]["ml"], b["last"]["ml"])

    def test_cadencia_movimiento_y_cierre(self):
        self.run_at("2026-09-23T15:00:00")
        n = len(self.api.calls)
        self.assertEqual(self.run_at("2026-09-23T15:10:00")["calls"], 0)      # nada toca todavía
        self.assertEqual(len(self.api.calls), n)
        self.api.shift = 0.10                                                  # el mercado se mueve
        st = self.run_at("2026-09-23T16:20:00")
        # 17:10 (a 50 min) y 17:35 (a 75 min, última vez hace 80) sí; 22:35 (a 6 h 15 min, cada 3 h) no
        self.assertEqual(st["refreshed"], 2)
        g = O.load_day("2026-09-23", self.dir)["games"]
        was = g["824223"]["books"]["codere"]
        self.assertEqual(was["open"]["ml"]["home"], -137)                     # la apertura no cambia
        self.assertEqual(was["last"]["ml"]["home"], -120)                     # 1.83
        self.assertEqual(g["824784"]["books"]["codere"]["last"]["ml"], g["824784"]["books"]["codere"]["open"]["ml"])
        self.api.shift = 0.30
        self.run_at("2026-09-23T17:15:00")                                     # ya empezó 824223: cierre congelado
        was2 = O.load_day("2026-09-23", self.dir)["games"]["824223"]["books"]["codere"]
        self.assertEqual(was2["last"], was["last"])

    def test_topes_de_llamadas(self):
        old = dict(O.DAILY)
        try:
            O.DAILY["odds-api.net"] = 2
            st = self.run_at("2026-09-23T15:00:00")
            self.assertLessEqual(st["calls"], 2)
            self.assertEqual(self.run_at("2026-09-23T16:30:00")["calls"], 0)
            self.assertTrue(any("tope" in str(x) for x in self.logs + [st["error"]]))
        finally:
            O.DAILY.update(old)

    def test_sin_llave_no_llama(self):
        st = self.run_at("2026-09-23T15:00:00", env={"ODDS_ESPN": "0"})
        self.assertEqual((st["provider"], st["calls"]), (None, 0))
        self.assertEqual(self.api.calls, [])
        self.assertIsNone(O.to_bundle(self.bundle, base=self.dir))

    def test_al_modelo_precio_de_referencia(self):
        self.run_at("2026-09-23T15:00:00")
        b = copy.deepcopy(self.bundle)
        b["odds"] = O.to_bundle(b, base=self.dir, now=dt.datetime(2026, 9, 23, 15, 5, tzinfo=UTC))
        self.assertEqual(b["odds"][0]["provider"], "odds-api.net")
        ctx = model.Context(b)
        up = {g["pk"]: g for g in b["upcoming"]}
        o = model.odds_for_game(ctx, up[824223])
        self.assertEqual((o["nBooks"], o["nMx"]), (3, 2))
        # referencia = mediana de Codere (1.73) y bet365 (1.77) → 1.75 = −133; Pinnacle (1.83) solo es el «mejor»
        self.assertEqual(o["refMl"]["home"][0], -133)
        self.assertEqual(o["bestMl"]["home"], (-120, "Pinnacle"))
        self.assertEqual(o["rlPoint"], {"home": -1.5, "away": 1.5})
        self.assertEqual(o["totalLine"], 8.5)
        # doble cartelera: cada juego recibe sus momios
        self.assertNotEqual(model.odds_for_game(ctx, up[824785])["refMl"]["home"],
                            model.odds_for_game(ctx, up[824784])["refMl"]["home"])
        a = model.analyze(ctx, up[824223])
        s7 = a["sections"]["s7"]
        self.assertTrue(s7["hasOdds"])
        self.assertEqual(s7["provider"], "odds-api.net")
        self.assertEqual(s7["books"][0]["book"], "Bet365")                    # casas MX primero
        self.assertTrue(s7["books"][0]["mx"])
        self.assertEqual(s7["books"][0]["open"]["mlHome"], s7["books"][0]["mlHome"])
        ml = next(m for m in s7["fair"] if m["market"] == "Moneyline" and m["pick"] == a["teams"]["home"]["abbr"])
        self.assertEqual(ml["price"], -133)


class MomiosEspn(unittest.TestCase):
    """ESPN publica sin clave los momios de su casa socia (DraftKings) con apertura y actual."""

    def setUp(self):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            self.bundle = json.load(f)
        with open(ESPN_FX, encoding="utf-8") as f:
            self.sb = json.load(f)
        self.dir = tempfile.mkdtemp()
        self.urls = []

    def fetch(self, url, headers=None):
        self.urls.append(url)
        assert url.startswith(O.ESPN), url
        return self.sb

    def run_at(self, when, env=None):
        return O.update(self.bundle, now=dt.datetime.fromisoformat(when).replace(tzinfo=UTC), base=self.dir,
                        env={} if env is None else env, fetch=self.fetch, log=lambda *a: None)

    def test_parser_de_la_respuesta_real(self):
        evs = {e["id"]: e for e in O.espn_events(self.sb)}
        dk = evs["9001"]["rows"]["draftkings"]
        self.assertEqual(dk["title"], "DraftKings")
        self.assertEqual(dk["ml"], {"away": -112, "home": -108})
        self.assertEqual(dk["rl"], {"away": {"point": -1.5, "price": 144}, "home": {"point": 1.5, "price": -175}})
        self.assertEqual(dk["total"], {"over": {"point": 8.5, "price": -102}, "under": {"point": 8.5, "price": -118}})
        op = evs["9001"]["opens"]["draftkings"]
        self.assertEqual(op["ml"], {"away": -131, "home": 108})               # la apertura de la casa
        self.assertEqual(op["total"]["over"]["point"], 8.0)
        self.assertEqual(evs["9002"]["rows"]["draftkings"]["rl"], {})         # sin run line publicado
        self.assertEqual(evs["9003"]["rows"]["draftkings"]["ml"]["away"], 100)  # EVEN
        self.assertEqual(evs["9004"]["rows"], {})                             # ya empezó: sin momios

    def test_sin_clave_trae_momios_con_apertura_de_la_casa(self):
        st = self.run_at("2026-09-23T15:00:00")
        self.assertEqual((st["provider"], st["calls"], st["refreshed"], st["error"]), ("ESPN", 1, 3, None))
        self.assertIn("dates=20260923", self.urls[0])
        g = O.load_day("2026-09-23", self.dir)["games"]
        dk = g["824223"]["books"]["draftkings"]
        self.assertEqual(dk["open"]["ml"], {"away": -131, "home": 108})       # apertura de DraftKings, no la nuestra
        self.assertEqual(dk["last"]["ml"], {"away": -112, "home": -108})
        self.assertEqual(g["824223"]["providers"], ["espn"])
        # doble cartelera por hora: 17:35 → evento 9002, 22:35 → 9003
        self.assertEqual((g["824785"]["espn"], g["824784"]["espn"]), ("9002", "9003"))
        self.assertEqual(self.run_at("2026-09-23T15:10:00")["calls"], 0)      # nada toca todavía
        b = copy.deepcopy(self.bundle)
        b["odds"] = O.to_bundle(b, base=self.dir, now=dt.datetime(2026, 9, 23, 15, 5, tzinfo=UTC))
        self.assertEqual(b["odds"][0]["provider"], "ESPN")
        ctx = model.Context(b)
        up = {x["pk"]: x for x in b["upcoming"]}
        o = model.odds_for_game(ctx, up[824223])
        self.assertEqual(o["refMl"]["home"], (-108, "DraftKings"))            # una casa: ese es el de referencia
        a = model.analyze(ctx, up[824223])
        self.assertEqual(a["sections"]["s7"]["books"][0]["open"]["mlAway"], -131)

    def test_se_puede_apagar(self):
        self.assertEqual(self.run_at("2026-09-23T15:00:00", env={"ODDS_ESPN": "0"})["calls"], 0)
        self.assertEqual(self.urls, [])



def f5_event(eid, home, away, ml=(-125, 105), tot=(4.5, -110, -110)):
    """Respuesta de The Odds API (/events/{id}/odds) con F5: dos casas y una de 3 vías que se descarta."""
    def mk(key, title, ml_, tot_):
        return {"key": key, "title": title, "last_update": "2026-09-23T15:00:00Z", "markets": [
            {"key": "h2h_1st_5_innings", "outcomes": [{"name": home, "price": ml_[0]}, {"name": away, "price": ml_[1]}]},
            {"key": "totals_1st_5_innings", "outcomes": [{"name": "Over", "price": tot_[1], "point": tot_[0]},
                                                         {"name": "Under", "price": tot_[2], "point": tot_[0]},
                                                         {"name": "Over", "price": 150, "point": 5.5},
                                                         {"name": "Under", "price": -190, "point": 5.5}]}]}
    three = {"key": "betrivers", "title": "BetRivers", "markets": [
        {"key": "h2h_1st_5_innings", "outcomes": [{"name": home, "price": 140}, {"name": away, "price": 190},
                                                  {"name": "Draw", "price": 400}]}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": "2026-09-23T17:10:00Z",
            "bookmakers": [mk("draftkings", "DraftKings", ml, tot), mk("fanduel", "FanDuel", (ml[0] - 5, ml[1] + 5), tot), three]}


class MomiosF5TheOddsApi(unittest.TestCase):
    """The Odds API (plan gratis): solo el F5 de los partidos cuyo pick es F5, sin pasarse de los créditos."""

    def setUp(self):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            self.bundle = json.load(f)
        with open(ESPN_FX, encoding="utf-8") as f:
            self.sb = json.load(f)
        self.dir, self.pred = tempfile.mkdtemp(), tempfile.mkdtemp()
        with open(os.path.join(self.pred, "2026-09-23.json"), "w") as f:     # último análisis guardado
            json.dump([{"pk": 824223, "topPicks": [{"family": "Total", "pick": "Over 8.5"}, {"family": "F5", "pick": "DET F5"}]},
                       {"pk": 824785, "topPicks": [{"family": "F5 total", "pick": "F5 Under 4.5"}]},
                       {"pk": 824784, "topPicks": [{"family": "ML", "pick": "BAL"}]}], f)
        up = {g["pk"]: g for g in self.bundle["upcoming"]}
        self.events = [{"id": f"t{pk}", "home_team": h, "away_team": a, "commence_time": up[pk]["time"]}
                       for pk, a, h in ((824223, "Washington Nationals", "Detroit Tigers"),
                                        (824785, "Toronto Blue Jays", "Baltimore Orioles"),
                                        (824784, "Toronto Blue Jays", "Baltimore Orioles"))]
        self.calls, self.logs = [], []

    def fetch(self, url, headers=None):
        self.calls.append(url)
        if url.startswith(O.ESPN):
            return self.sb
        u = urllib.parse.urlparse(url)
        if u.path.endswith("/events"):
            return self.events
        eid = u.path.split("/")[-2]
        ev = next(e for e in self.events if e["id"] == eid)
        return f5_event(eid, ev["home_team"], ev["away_team"])

    def run_at(self, when):
        return O.update(self.bundle, now=dt.datetime.fromisoformat(when).replace(tzinfo=UTC), base=self.dir,
                        env={"ODDS_API_KEY": "k"}, fetch=self.fetch, log=self.logs.append, pred_dir=self.pred)

    def odds_calls(self):
        return [urllib.parse.urlparse(c) for c in self.calls if "/odds?" in c and c.startswith(O.THE)]

    def test_solo_lo_que_falta(self):
        st = self.run_at("2026-09-23T15:00:00")
        self.assertEqual(st["provider"], "ESPN + The Odds API")
        oc = self.odds_calls()
        self.assertEqual(len(oc), 2)                                           # 824784 no tiene pick F5
        mk = sorted(dict(urllib.parse.parse_qsl(u.query))["markets"] for u in oc)
        self.assertEqual(mk, ["h2h_1st_5_innings", "totals_1st_5_innings"])    # solo el mercado del pick
        self.assertEqual(st["credits"], 2)
        dk = O.load_day("2026-09-23", self.dir)["games"]["824223"]["books"]["draftkings"]
        self.assertEqual(dk["last"]["ml"], {"away": -112, "home": -108})       # el ML de ESPN no se borra
        self.assertEqual(dk["last"]["f5"]["ml"], {"home": -125, "away": 105})
        self.assertEqual(dk["last"]["f5"]["total"]["over"]["point"], 4.5)      # la línea pareja, no la 5.5
        books = O.load_day("2026-09-23", self.dir)["games"]["824223"]["books"]
        self.assertNotIn("betrivers", books)                                   # F5 de 3 vías: fuera
        # cadencia: a las 15:30 nada; a las 16:05 los dos ya están a ≤ 90 min: un refresco de cada uno, y ya no más
        self.run_at("2026-09-23T15:30:00")
        self.assertEqual(len(self.odds_calls()), 2)
        self.run_at("2026-09-23T16:05:00")
        self.assertEqual(len(self.odds_calls()), 4)
        self.run_at("2026-09-23T17:08:00")
        self.assertEqual(len(self.odds_calls()), 4)

    def test_topes_de_creditos(self):
        old = O.THE_DAY_CREDITS
        try:
            O.THE_DAY_CREDITS = 1
            st = self.run_at("2026-09-23T15:00:00")
            self.assertEqual(len(self.odds_calls()), 1)
            self.assertTrue(any("tope de créditos" in x for x in self.logs))
            self.assertEqual(st["credits"], 1)
        finally:
            O.THE_DAY_CREDITS = old

    def test_al_modelo_f5_con_momio_real(self):
        self.run_at("2026-09-23T15:00:00")
        b = copy.deepcopy(self.bundle)
        b["odds"] = O.to_bundle(b, base=self.dir, now=dt.datetime(2026, 9, 23, 15, 5, tzinfo=UTC))
        self.assertEqual(b["odds"][0]["provider"], "ESPN + The Odds API")
        ctx = model.Context(b)
        g = next(x for x in b["upcoming"] if x["pk"] == 824223)
        o = model.odds_for_game(ctx, g)
        self.assertEqual(o["nBooks"], 1)                                       # ML/total: solo DraftKings (ESPN)
        self.assertEqual(o["refF5"]["home"][0], -127)                          # mediana (en decimal) de −125 y −130
        self.assertEqual(o["f5TotalLine"], 4.5)
        a = model.analyze(ctx, g)
        det = a["teams"]["home"]["abbr"]
        f5 = next(m for m in a["sections"]["s7"]["fair"] if m["pick"] == f"{det} F5")
        self.assertEqual(f5["price"], -127)
        dk = next(x for x in a["sections"]["s7"]["books"] if x["book"] == "DraftKings")
        self.assertEqual((dk["f5Home"], dk["f5Away"]), (-125, 105))
        self.assertEqual(dk["mlHome"], -108)


class MomiosPropsTheOddsApi(unittest.TestCase):
    """Ponches, team total y primera entrada (NRFI/YRFI) de The Odds API: solo el mercado de los picks."""

    def setUp(self):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            self.bundle = json.load(f)
        self.dir, self.pred = tempfile.mkdtemp(), tempfile.mkdtemp()
        with open(os.path.join(self.pred, "2026-09-23.json"), "w") as f:
            json.dump([{"pk": 824223, "topPicks": [{"family": "K", "pick": "Framber Valdez Over 5.5 K"},
                                                   {"family": "Team total", "pick": "DET Over 4.5"}]},
                       {"pk": 824785, "topPicks": [{"family": "NRFI", "pick": "NRFI"}]}], f)
        up = {g["pk"]: g for g in self.bundle["upcoming"]}
        self.events = [{"id": f"t{pk}", "home_team": h, "away_team": a, "commence_time": up[pk]["time"]}
                       for pk, a, h in ((824223, "Washington Nationals", "Detroit Tigers"),
                                        (824785, "Toronto Blue Jays", "Baltimore Orioles"))]
        self.calls = []

    def fetch(self, url, headers=None):
        self.calls.append(url)
        u = urllib.parse.urlparse(url)
        if u.path.endswith("/events"):
            return self.events
        eid = u.path.split("/")[-2]
        ev = next(e for e in self.events if e["id"] == eid)
        mk = dict(urllib.parse.parse_qsl(u.query))["markets"].split(",")
        markets = []
        if "pitcher_strikeouts" in mk:
            markets.append({"key": "pitcher_strikeouts", "outcomes": [
                {"name": "Over", "description": "Framber Valdez", "price": -125, "point": 5.5},
                {"name": "Under", "description": "Framber Valdez", "price": -105, "point": 5.5},
                {"name": "Over", "description": "Richard Lovelady", "price": 110, "point": 3.5},
                {"name": "Under", "description": "Richard Lovelady", "price": -140, "point": 3.5}]})
        if "team_totals" in mk:
            markets.append({"key": "team_totals", "outcomes": [
                {"name": "Over", "description": "Detroit Tigers", "price": -110, "point": 4.5},
                {"name": "Under", "description": "Detroit Tigers", "price": -120, "point": 4.5},
                {"name": "Over", "description": "Washington Nationals", "price": 105, "point": 3.5},
                {"name": "Under", "description": "Washington Nationals", "price": -135, "point": 3.5}]})
        if "totals_1st_1_innings" in mk:
            markets.append({"key": "totals_1st_1_innings", "outcomes": [
                {"name": "Over", "price": 110, "point": 0.5}, {"name": "Under", "price": -140, "point": 0.5}]})
        return dict(ev, bookmakers=[{"key": "draftkings", "title": "DraftKings", "markets": markets}])

    def run_at(self, when):
        return O.update(self.bundle, now=dt.datetime.fromisoformat(when).replace(tzinfo=UTC), base=self.dir,
                        env={"ODDS_API_KEY": "k", "ODDS_ESPN": "0"}, fetch=self.fetch, log=lambda *a: None,
                        pred_dir=self.pred)

    def test_mercados_de_los_picks_y_al_modelo(self):
        st = self.run_at("2026-09-23T15:00:00")
        asked = sorted(dict(urllib.parse.parse_qsl(urllib.parse.urlparse(c).query))["markets"]
                       for c in self.calls if "/odds?" in c)
        self.assertEqual(asked, ["pitcher_strikeouts,team_totals", "totals_1st_1_innings"])
        self.assertEqual(st["credits"], 3)
        dk = O.load_day("2026-09-23", self.dir)["games"]["824223"]["books"]["draftkings"]["last"]
        self.assertEqual(dk["k"]["framber valdez"]["over"], {"point": 5.5, "price": -125})
        self.assertEqual(dk["tt"]["home"]["over"]["point"], 4.5)
        nr = O.load_day("2026-09-23", self.dir)["games"]["824785"]["books"]["draftkings"]["last"]["nrfi"]
        self.assertEqual(nr, {"nrfi": -140, "yrfi": 110})
        b = copy.deepcopy(self.bundle)
        b["odds"] = O.to_bundle(b, base=self.dir, now=dt.datetime(2026, 9, 23, 15, 5, tzinfo=UTC))
        ctx = model.Context(b)
        up = {g["pk"]: g for g in b["upcoming"]}
        a = model.analyze(ctx, up[824223])
        s7 = a["sections"]["s7"]
        self.assertEqual(s7["marketTT"]["home"], 4.5)
        self.assertEqual(s7["marketK"]["framber valdez"], 5.5)
        k = next(p for p in a["picks"] if p["family"] == "K" and p["pick"].startswith("Framber Valdez"))
        self.assertEqual(k["line"], 5.5)                                   # la línea del mercado
        self.assertTrue(k["priceIsReal"])
        self.assertIn(k["price"], (-125, -105))
        tt = next(p for p in a["picks"] if p["market"] == "Team total " + a["teams"]["home"]["abbr"])
        self.assertEqual((tt["line"], tt["priceIsReal"]), (4.5, True))
        a2 = model.analyze(ctx, up[824785])
        nrp = next(p for p in a2["picks"] if p["family"] == "NRFI")
        self.assertTrue(nrp["priceIsReal"])
        self.assertIn(nrp["price"], (-140, 110))


class CuandoSePideElMomio(unittest.TestCase):
    """«Esperar momio»: el mercado de la decisión se pide desde 12 h antes, con reintentos si nadie lo publicaba."""
    START = dt.datetime(2026, 9, 30, 2, 0, tzinfo=UTC)

    def at(self, h):                                   # h horas antes del partido
        return self.START - dt.timedelta(hours=h)

    def test_primera_vez_segun_la_prioridad(self):
        e = {"start": O._iso(self.START)}
        self.assertFalse(O.market_due(e, "totals_1st_5_innings", self.at(12.5), O.WINDOW[1]))
        self.assertTrue(O.market_due(e, "totals_1st_5_innings", self.at(11.9), O.WINDOW[1]))    # decisión: 12 h
        self.assertFalse(O.market_due(e, "totals_1st_5_innings", self.at(7), O.WINDOW[3]))      # otro pick: 6 h
        self.assertEqual(O.next_ask(e, "totals_1st_5_innings", self.at(20), O.WINDOW[1]), self.at(12))

    def test_refrescos_a_las_6_h_y_hora_y_media(self):
        e = {"start": O._iso(self.START), "theAt": {"totals_1st_5_innings": O._iso(self.at(11.9))},
             "theN": {"totals_1st_5_innings": 1}, "theGot": {"totals_1st_5_innings": 5},
             "books": {"fanduel": {"last": {"f5": {"total": {"over": {"point": 4.0, "price": -110}}}}}}}
        self.assertEqual(O.next_ask(e, "totals_1st_5_innings", self.at(11), 720), self.at(6))
        e["theAt"]["totals_1st_5_innings"] = O._iso(self.at(5.9))
        self.assertEqual(O.next_ask(e, "totals_1st_5_innings", self.at(5), 720), self.at(1.5))
        e["theAt"]["totals_1st_5_innings"] = O._iso(self.at(1.4))
        self.assertIsNone(O.next_ask(e, "totals_1st_5_innings", self.at(1), 720))            # ya no más
        e["theAt"]["totals_1st_5_innings"] = O._iso(self.at(2))                               # pedido a las 2 h:
        self.assertEqual(O.next_ask(e, "totals_1st_5_innings", self.at(1.9), 720), self.at(1))  # 60 min después

    def test_si_nadie_lo_publicaba_se_reintenta(self):
        e = {"start": O._iso(self.START), "theAt": {"team_totals": O._iso(self.at(12))}, "theEmpty": {"team_totals": 1}}
        self.assertEqual(O.next_ask(e, "team_totals", self.at(11), 720), self.at(10))
        e["theAt"]["team_totals"], e["theEmpty"]["team_totals"] = O._iso(self.at(8)), 3     # 3 vacíos seguidos
        self.assertEqual(O.next_ask(e, "team_totals", self.at(7.5), 720), self.at(6))       # solo los refrescos
        chg = O._iso(self.at(3))                                                             # cambió el abridor
        e["theAt"]["team_totals"] = O._iso(self.at(5))
        self.assertTrue(O.market_due(e, "team_totals", self.at(2.9), 720, chg))

    def test_la_pagina_sabe_cuando_llega(self):
        e = {"start": O._iso(self.START), "theAt": {"totals_1st_5_innings": O._iso(self.at(11))},
             "theGot": {"totals_1st_5_innings": 0}, "theEmpty": {"totals_1st_5_innings": 1}}
        v = O.ask_view(e, self.at(10.5))
        self.assertEqual(v["totals_1st_5_innings"], {"at": O._iso(self.at(11)), "got": 0, "empty": 1, "next": O._iso(self.at(9))})
        self.assertEqual(v["pitcher_strikeouts"]["next"], O._iso(self.at(10.5)))              # nunca pedido: ya toca
        # juego de las 18:00Z sin momios todavía: se pide 12 h antes (06:00Z; desde el 5-oct hay corrida a toda hora)
        e2 = {"start": "2026-09-30T18:00:00Z"}
        v2 = O.ask_view(e2, dt.datetime(2026, 9, 29, 22, 0, tzinfo=UTC))
        self.assertEqual(v2["totals_1st_5_innings"]["next"], "2026-09-30T06:00:00Z")


class EsperarMomioConTheOddsApi(unittest.TestCase):
    """La decisión espera el F5: se pide desde 12 h antes; si ninguna casa lo tiene se reintenta sin tocar la reserva."""

    def setUp(self):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            self.bundle = json.load(f)
        self.dir, self.pred = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.g = next(g for g in self.bundle["upcoming"] if g["pk"] == 824785)
        self.start = dt.datetime.fromisoformat(self.g["time"].replace("Z", "+00:00"))
        with open(os.path.join(self.pred, f"{self.g['date']}.json"), "w") as f:
            json.dump([{"pk": 824785, "decision": {"status": "esperar", "waitFor": "momio", "family": "F5 total"},
                        "topPicks": [{"family": "F5 total", "pick": "F5 Under 4.5"}]}], f)
        self.events = [{"id": "t824785", "home_team": "Baltimore Orioles", "away_team": "Toronto Blue Jays",
                        "commence_time": self.g["time"]}]
        self.calls, self.posted = [], False

    def fetch(self, url, headers=None):
        self.calls.append(url)
        if urllib.parse.urlparse(url).path.endswith("/events"):
            return self.events
        if not self.posted:                                   # las casas todavía no publican el F5
            return dict(self.events[0], bookmakers=[])
        return f5_event("t824785", "Baltimore Orioles", "Toronto Blue Jays")

    def run_before(self, h):
        return O.update(self.bundle, now=self.start - dt.timedelta(hours=h), base=self.dir,
                        env={"ODDS_API_KEY": "k", "ODDS_ESPN": "0"}, fetch=self.fetch, log=lambda *a: None, pred_dir=self.pred)

    def asked(self):
        return len([c for c in self.calls if "/odds?" in c])

    def test_se_pide_temprano_y_se_reintenta(self):
        self.run_before(11.8)
        self.assertEqual(self.asked(), 1)                                          # 12 h antes, no 6
        e = O.load_day(self.g["date"], self.dir)["games"]["824785"]
        self.assertEqual((e["theGot"]["totals_1st_5_innings"], e["theEmpty"]["totals_1st_5_innings"]), (0, 1))
        self.run_before(10.8)
        self.assertEqual(self.asked(), 1)                                          # el reintento es a las 2 h
        self.posted = True
        self.run_before(9.7)
        self.assertEqual(self.asked(), 2)
        e = O.load_day(self.g["date"], self.dir)["games"]["824785"]
        self.assertEqual(e["theEmpty"]["totals_1st_5_innings"], 0)
        self.assertGreater(e["theGot"]["totals_1st_5_innings"], 0)
        b = copy.deepcopy(self.bundle)
        now = self.start - dt.timedelta(hours=9.6)
        ask = O.ask_all(b, base=self.dir, now=now)["824785"]["totals_1st_5_innings"]
        # refresco a las 6 h (11:35Z): hay corrida a toda hora (relevo continuo y cron de respaldo cada hora)
        self.assertEqual(O._ts(ask["next"]), first_run_after(self.start - dt.timedelta(hours=6)))
        self.assertEqual(ask["next"], "2026-09-23T11:35:00Z")
        # llega al análisis (sección 7) aunque el partido no tenga casas en el feed
        b["oddsAsk"] = O.ask_all(b, base=self.dir, now=now)
        a = model.analyze(model.Context(b), next(x for x in b["upcoming"] if x["pk"] == 824785))
        self.assertEqual(a["sections"]["s7"]["ask"]["totals_1st_5_innings"]["next"], "2026-09-23T11:35:00Z")

    def test_alternativas_de_la_decision_en_la_misma_llamada(self):
        with open(os.path.join(self.pred, f"{self.g['date']}.json"), "w") as f:
            json.dump([{"pk": 824785, "decision": {"status": "esperar", "waitFor": "momio", "family": "F5 total",
                                                   "alts": ["F5", "Total", "RL"]}, "topPicks": []}], f)
        self.assertEqual(O.market_needs(self.g["date"], self.pred)[824785],
                         {"totals_1st_5_innings": 1, "h2h_1st_5_innings": O.ALT})
        self.posted = True
        self.run_before(11.8)
        asked = [dict(urllib.parse.parse_qsl(urllib.parse.urlparse(c).query))["markets"] for c in self.calls if "/odds?" in c]
        self.assertEqual(asked, ["totals_1st_5_innings,h2h_1st_5_innings"])

    def only_asked(self, url, headers=None):
        """Como la API real: solo los mercados que se pidieron."""
        ev = self.fetch(url, headers)
        if "/odds?" not in url:
            return ev
        mk = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))["markets"].split(",")
        return dict(ev, bookmakers=[dict(b, markets=[m for m in b["markets"] if m["key"] in mk]) for b in ev.get("bookmakers", [])])

    def test_el_f5_ganador_no_borra_el_f5_total(self):
        """Caso real (CHC @ SD, 29-sep): el F5 total a las 20:06 y el F5 ganador a las 20:14 en otra llamada."""
        self.posted = True
        upd = lambda h: O.update(self.bundle, now=self.start - dt.timedelta(hours=h), base=self.dir,  # noqa: E731
                                 env={"ODDS_API_KEY": "k", "ODDS_ESPN": "0"}, fetch=self.only_asked,
                                 log=lambda *a: None, pred_dir=self.pred)
        upd(5.9)
        with open(os.path.join(self.pred, f"{self.g['date']}.json"), "w") as f:     # la decisión pasa al F5 ganador
            json.dump([{"pk": 824785, "decision": {"status": "esperar", "waitFor": "momio", "family": "F5"}}], f)
        upd(5.7)
        e = O.load_day(self.g["date"], self.dir)["games"]["824785"]
        dk = e["books"]["draftkings"]["last"]["f5"]
        self.assertTrue(dk["ml"] and dk["total"])
        self.assertEqual(e["theGot"], {"totals_1st_5_innings": 2, "h2h_1st_5_innings": 2})   # la de 3 vías no cuenta
        # si un mercado que llegó ya no está guardado (como pasó con el error), se vuelve a pedir
        for b in e["books"].values():
            b["last"]["f5"]["total"] = {}
        now = self.start - dt.timedelta(hours=5.5)
        self.assertTrue(O.market_due(e, "totals_1st_5_innings", now, O.WINDOW[1]))
        self.assertFalse(O.market_due(e, "h2h_1st_5_innings", now, O.WINDOW[1]))

    def test_lo_temprano_no_toca_la_reserva(self):
        usage = {"calls": {(self.start - dt.timedelta(hours=11)).strftime("%Y-%m-%d"): {"the-odds-api": O.THE_DAY_CREDITS - O.RESERVE}}}
        O.save_usage(usage, self.dir)
        st = self.run_before(11)
        self.assertEqual(self.asked(), 0)
        self.assertEqual(st["theQueue"][0]["skipped"], ["totals_1st_5_innings"])
        st = self.run_before(5.5)                                                  # dentro de las 6 h sí la usa
        self.assertEqual(self.asked(), 1)


if __name__ == "__main__":
    unittest.main()
