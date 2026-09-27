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

    def test_respaldo_the_odds_api(self):
        evs = [{"id": "t1", "commence_time": "2026-09-23T17:10:00Z", "home_team": "Detroit Tigers",
                "away_team": "Washington Nationals",
                "bookmakers": [{"key": "draftkings", "title": "DraftKings", "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Detroit Tigers", "price": -140}, {"name": "Washington Nationals", "price": 120}]},
                    {"key": "spreads", "outcomes": [{"name": "Detroit Tigers", "price": 135, "point": -1.5},
                                                    {"name": "Washington Nationals", "price": -160, "point": 1.5}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 8.5}, {"name": "Under", "price": -110, "point": 8.5}]}]}]}]
        calls = []

        def fetch(url, headers=None):
            calls.append(url)
            return evs
        env = {"ODDS_API_KEY": "k", "ODDS_ESPN": "0"}
        now = dt.datetime(2026, 9, 23, 15, 0, tzinfo=UTC)
        st = O.update(self.bundle, now=now, base=self.dir, env=env, fetch=fetch, log=self.logs.append)
        self.assertEqual((st["provider"], st["calls"], st["refreshed"]), ("The Odds API", 1, 1))
        st = O.update(self.bundle, now=now + dt.timedelta(minutes=30), base=self.dir, env=env, fetch=fetch, log=self.logs.append)
        self.assertEqual(st["calls"], 0)                                      # una llamada cada 2 h como mucho
        row = O.load_day("2026-09-23", self.dir)["games"]["824223"]["books"]["draftkings"]["last"]
        self.assertEqual(row["ml"], {"home": -140, "away": 120})
        self.assertEqual(row["rl"]["home"], {"point": -1.5, "price": 135})



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


if __name__ == "__main__":
    unittest.main()
