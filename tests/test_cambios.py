"""Revisión periódica: qué cambió (mlbgs/cambios.py), momios por prioridad y verificación entre casas
(mlbgs/odds.py) y cada cuánto se revisa (mlbgs/cadencia.py)."""
import copy
import datetime as dt
import gzip
import json
import os
import tempfile
import unittest
import urllib.parse

from mlbgs import cadencia as CA
from mlbgs import cambios as C
from mlbgs import model
from mlbgs import odds as O

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bundle_small.json.gz")
ESPN_FX = os.path.join(os.path.dirname(__file__), "fixtures", "espn_scoreboard.json")
UTC = dt.timezone.utc


def at(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


def load():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


class CambiosDeLaMlb(unittest.TestCase):
    def setUp(self):
        self.b = load()
        self.dir = tempfile.mkdtemp()
        self.up = lambda b, pk: next(g for g in b["upcoming"] if g["pk"] == pk)  # noqa: E731

    def run_at(self, b, when):
        t = C.Tracker(base=self.dir, now=at(when))
        t.mlb(b)
        t.save()
        return t

    def test_la_primera_foto_es_la_base(self):
        t = self.run_at(self.b, "2026-09-23T14:00:00")
        self.assertEqual(t.new, [])
        self.assertEqual(t.hot, {})
        with open(os.path.join(self.dir, "2026-09-23.json"), encoding="utf-8") as f:
            snap = json.load(f)["snap"]
        self.assertEqual(snap["824223"]["ump"], "Mike Muchlinski")
        self.assertEqual(snap["824223"]["wx"]["dir"], "in")

    def test_cambio_de_abridor_es_alto_y_calienta_el_partido(self):
        self.run_at(self.b, "2026-09-23T14:00:00")
        b = copy.deepcopy(self.b)
        self.up(b, 824223)["probable"]["home"] = 999001
        b["pitchers"]["999001"] = {"id": 999001, "name": "Relevo Abridor"}
        t = self.run_at(b, "2026-09-23T14:20:00")
        ev = [e for e in t.new if e["kind"] == "abridor"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["impact"], "alto")
        self.assertIn("→ Relevo Abridor", ev[0]["text"])
        self.assertEqual(t.hot, {824223: ["abridor"]})
        self.assertEqual(t.stale_after(), {824223: "2026-09-23T14:20:00Z"})
        # la API deja de mandar al abridor un momento: no es un cambio y la foto lo conserva
        b2 = copy.deepcopy(b)
        self.up(b2, 824223)["probable"]["home"] = None
        t2 = self.run_at(b2, "2026-09-23T14:40:00")
        self.assertEqual([e for e in t2.new if e["kind"] == "abridor"], [])
        t3 = self.run_at(b, "2026-09-23T15:00:00")
        self.assertEqual(t3.new, [])

    def test_lineup_confirmado_y_cambiado(self):
        self.run_at(self.b, "2026-09-23T14:00:00")
        b = copy.deepcopy(self.b)
        g = self.up(b, 824784)
        g["lineups"] = {"away": list(range(100, 109)), "home": []}
        g["lineupNames"] = {"away": {str(i): f"Bateador {i}" for i in range(100, 109)}, "home": {}}
        t = self.run_at(b, "2026-09-23T15:00:00")
        self.assertEqual([(e["kind"], e["impact"]) for e in t.new], [("lineup", "medio")])
        b2 = copy.deepcopy(b)
        g2 = self.up(b2, 824784)
        g2["lineups"]["away"][3] = 200
        g2["lineupNames"]["away"]["200"] = "Suplente Nuevo"
        t2 = self.run_at(b2, "2026-09-23T15:20:00")
        ev = t2.new[0]
        self.assertEqual((ev["kind"], ev["impact"]), ("lineup", "alto"))
        self.assertIn("sale Bateador 103", ev["text"])
        self.assertIn("entra Suplente Nuevo", ev["text"])
        self.assertEqual(t2.hot, {824784: ["lineup"]})
        b3 = copy.deepcopy(b2)
        a = self.up(b3, 824784)["lineups"]["away"]
        a[0], a[1] = a[1], a[0]
        t3 = self.run_at(b3, "2026-09-23T15:40:00")
        self.assertEqual([(e["kind"], e["impact"]) for e in t3.new], [("lineup", "bajo")])
        self.assertEqual(t3.hot, {})                          # solo el orden: no pide momios

    def test_umpire_clima_y_horario(self):
        self.run_at(self.b, "2026-09-23T14:00:00")
        b = copy.deepcopy(self.b)
        g = self.up(b, 824785)
        g["officials"][0]["name"] = "Otro Umpire"
        g["weather"] = {"condition": "Rain", "temp": "75", "wind": "15 mph, Out To CF"}
        g["time"] = "2026-09-23T18:35:00Z"
        g["detailed"] = "Delayed Start: Rain"
        t = self.run_at(b, "2026-09-23T15:00:00")
        kinds = {(e["kind"], e["impact"]) for e in t.new}
        self.assertIn(("umpire", "medio"), kinds)
        self.assertIn(("clima", "medio"), kinds)             # viento en contra → a favor, y lluvia
        self.assertIn(("clima", "bajo"), kinds)              # 62 → 75 °F
        self.assertIn(("horario", "medio"), kinds)           # +60 min
        self.assertIn(("horario", "alto"), kinds)            # retrasado
        self.assertEqual(t.hot, {824785: ["horario"]})

    def test_movimiento_del_roster_en_el_proximo_partido(self):
        self.run_at(self.b, "2026-09-23T14:00:00")
        b = copy.deepcopy(self.b)
        g = self.up(b, 824785)
        player = g["lineups"]["away"][0]
        b["transactions"] = [{"id": 77, "date": "2026-09-23", "type": "Status Change", "code": "SC", "team": 141,
                              "person": player, "name": "Titular", "text": "Toronto Blue Jays placed SS Titular on the "
                              "10-day injured list. Left hamstring strain."}]
        t = self.run_at(b, "2026-09-23T15:00:00")
        mv = [e for e in t.new if e["kind"] == "movimiento"]
        self.assertEqual(len(mv), 1)                          # TOR juega dos veces hoy: se anota en el primero
        self.assertEqual((mv[0]["pk"], mv[0]["impact"]), (824785, "alto"))
        t2 = self.run_at(b, "2026-09-23T15:20:00")
        self.assertEqual(t2.new, [])                          # ya visto

    def test_muchos_movimientos_menores_van_juntos(self):
        self.run_at(self.b, "2026-09-23T14:00:00")
        b = copy.deepcopy(self.b)
        b["transactions"] = [{"id": 900 + i, "date": "2026-09-23", "type": "Status Change", "code": "SC", "team": 110,
                              "person": 5000 + i, "name": f"Jugador {i}", "text": f"RHP Jugador {i} roster status changed."}
                             for i in range(9)]
        t = self.run_at(b, "2026-09-23T15:00:00")
        mv = [e for e in t.new if e["kind"] == "movimiento"]
        self.assertEqual(len(mv), 1)
        self.assertIn("9 movimientos del roster", mv[0]["text"])
        self.assertNotIn("who", mv[0])

    def test_decision_que_cambia(self):
        a = {"pk": 824223, "date": "2026-09-23", "teams": {"away": {"abbr": "WSH"}, "home": {"abbr": "DET"}},
             "decision": {"status": "apostar", "pick": {"pick": "DET ML"}, "stake": {"level": 5}}}
        t = C.Tracker(base=self.dir, now=at("2026-09-23T14:00:00"))
        t.decisions([a])
        self.assertEqual(t.new, [])
        a2 = copy.deepcopy(a)
        a2["decision"] = {"status": "esperar", "waitFor": "momio", "pick": {"pick": "DET ML"}, "stake": {"level": 0}}
        t.now = at("2026-09-23T14:20:00")
        t.decisions([a2])
        self.assertEqual((t.new[0]["kind"], t.new[0]["impact"]), ("decision", "alto"))
        self.assertIn("apostar DET ML · stake 5 → esperar momio", t.new[0]["text"])


class MomiosPorPrioridadYVerificacion(unittest.TestCase):
    def setUp(self):
        self.b = load()
        with open(ESPN_FX, encoding="utf-8") as f:
            self.sb = json.load(f)
        self.dir, self.pred = tempfile.mkdtemp(), tempfile.mkdtemp()
        up = {g["pk"]: g for g in self.b["upcoming"]}
        self.events = [{"id": f"t{pk}", "home_team": h, "away_team": a, "commence_time": up[pk]["time"]}
                       for pk, a, h in ((824223, "Washington Nationals", "Detroit Tigers"),
                                        (824785, "Toronto Blue Jays", "Baltimore Orioles"),
                                        (824784, "Toronto Blue Jays", "Baltimore Orioles"))]
        self.calls = []
        self.logs = []

    def preds(self, rows):
        with open(os.path.join(self.pred, "2026-09-23.json"), "w") as f:
            json.dump(rows, f)

    def fetch(self, url, headers=None):
        self.calls.append(url)
        if url.startswith(O.ESPN):
            return self.sb
        u = urllib.parse.urlparse(url)
        if u.path.endswith("/events"):
            return self.events
        eid = u.path.split("/")[-2]
        ev = next(e for e in self.events if e["id"] == eid)
        h, a = ev["home_team"], ev["away_team"]
        mk = dict(urllib.parse.parse_qsl(u.query))["markets"].split(",")

        def book(key, title, ml):
            ms = []
            if "h2h" in mk:
                ms.append({"key": "h2h", "outcomes": [{"name": h, "price": ml[0]}, {"name": a, "price": ml[1]}]})
            if "h2h_1st_5_innings" in mk:
                ms.append({"key": "h2h_1st_5_innings", "outcomes": [{"name": h, "price": -120}, {"name": a, "price": 100}]})
            return {"key": key, "title": title, "markets": ms}
        return {"id": eid, "home_team": h, "away_team": a,
                "bookmakers": [book("fanduel", "FanDuel", (-150, 130)), book("betmgm", "BetMGM", (-155, 135)),
                               book("caesars", "Caesars", (-145, 125))]}

    def run_at(self, when, **kw):
        return O.update(self.b, now=at(when), base=self.dir, env={"ODDS_API_KEY": "k"}, fetch=self.fetch,
                        log=self.logs.append, pred_dir=self.pred, **kw)

    def odds_calls(self):
        return [dict(urllib.parse.parse_qsl(urllib.parse.urlparse(c).query))["markets"]
                for c in self.calls if "/odds?" in c and c.startswith(O.THE)]

    def test_prioridades(self):
        self.preds([
            {"pk": 824223, "decision": {"status": "apostar", "family": "ML", "pick": "DET ML", "level": 4},
             "topPicks": [{"family": "ML"}, {"family": "K"}]},
            {"pk": 824785, "decision": {"status": "esperar", "waitFor": "momio", "family": "F5", "pick": "BAL F5"},
             "topPicks": [{"family": "F5"}]},
            {"pk": 824784, "decision": {"status": "no apostar", "family": "Total"}, "topPicks": [{"family": "Total"}]}])
        need = O.market_needs("2026-09-23", self.pred)
        self.assertEqual(need[824223], {"h2h": 2, "pitcher_strikeouts": 3})
        self.assertEqual(need[824785], {"h2h_1st_5_innings": 1})
        self.assertNotIn(824784, need)                         # sin stake ni espera: no gasta créditos
        # a las 15:00 (824223 a 2 h 10 min): primero el F5 que decide, luego la verificación y el K
        self.run_at("2026-09-23T15:00:00")
        self.assertEqual(self.odds_calls(), ["h2h_1st_5_innings", "h2h,pitcher_strikeouts"])

    def test_la_reserva_queda_para_los_cambios(self):
        self.preds([{"pk": 824223, "decision": {"status": "apostar", "family": "ML", "pick": "DET ML", "level": 4},
                     "topPicks": [{"family": "ML"}]}])
        old = O.THE_DAY_CREDITS
        try:
            O.THE_DAY_CREDITS = O.RESERVE          # solo queda la reserva: la verificación (prioridad 2) espera
            st = self.run_at("2026-09-23T15:00:00")
            self.assertEqual(self.odds_calls(), [])
            self.assertEqual(st["theQueue"][0]["skipped"], ["h2h"])
        finally:
            O.THE_DAY_CREDITS = old

    def test_cambio_de_abridor_repide_y_el_momio_previo_no_cuenta(self):
        self.preds([{"pk": 824785, "decision": {"status": "esperar", "waitFor": "momio", "family": "F5"},
                     "topPicks": [{"family": "F5"}]}])
        self.run_at("2026-09-23T15:00:00")
        self.assertEqual(self.odds_calls(), ["h2h_1st_5_innings"])
        self.run_at("2026-09-23T15:20:00")
        self.assertEqual(len(self.odds_calls()), 1)            # no le toca todavía
        chg = {824785: "2026-09-23T15:30:00Z"}
        # a las 15:40 ya se sabe del cambio: el F5 pedido antes se pide otra vez (prioridad 0)
        self.run_at("2026-09-23T15:40:00", hot={824785: ["abridor"]}, stale_after=chg)
        self.assertEqual(len(self.odds_calls()), 2)
        self.run_at("2026-09-23T15:45:00", stale_after=chg)
        self.assertEqual(len(self.odds_calls()), 2)            # una vez por cambio
        # el momio de una casa que no se actualizó después del cambio sale marcado y el modelo no lo usa
        day = O.load_day("2026-09-23", self.dir)
        day["games"]["824785"]["books"]["draftkings"]["last"]["at"] = "2026-09-23T15:10:00Z"
        O.save_day(day, self.dir)
        b = copy.deepcopy(self.b)
        b["odds"] = O.to_bundle(b, base=self.dir, now=at("2026-09-23T15:50:00"), stale_after=chg)
        ev = next(e for e in b["odds"] if e["pk"] == 824785)
        dk = next(x for x in ev["bookmakers"] if x["key"] == "draftkings")
        self.assertTrue(dk["stale"])
        o = model.odds_for_game(model.Context(b), next(g for g in b["upcoming"] if g["pk"] == 824785))
        self.assertEqual([s["book"] for s in o["stale"]], ["DraftKings"])
        self.assertFalse(any(x["book"] == "DraftKings" for x in o["books"]))
        self.assertIsNone(o["refMl"]["home"])                  # sin momio vigente de ML: esperar momio

    def test_hot_revisa_espn_aunque_no_le_toque(self):
        self.preds([])
        self.run_at("2026-09-23T14:00:00")
        n = sum(1 for c in self.calls if c.startswith(O.ESPN))
        self.run_at("2026-09-23T14:10:00")
        self.assertEqual(sum(1 for c in self.calls if c.startswith(O.ESPN)), n)          # por horario, nada
        self.run_at("2026-09-23T14:20:00", hot={824784: ["lineup"]})
        self.assertEqual(sum(1 for c in self.calls if c.startswith(O.ESPN)), n + 1)      # con cambio, sí

    def test_verificacion_entre_casas(self):
        row = lambda h, a, ln=8.5, o=-110, u=-110: {"ml": {"home": h, "away": a},  # noqa: E731
                                                     "total": {"over": {"point": ln, "price": o}, "under": {"point": ln, "price": u}}}
        ok = O.verify({"draftkings": row(-150, 130), "fanduel": row(-148, 128), "betmgm": row(-155, 135)}, "TOR", "BAL")
        self.assertEqual((ok["ml"]["status"], ok["total"]["status"]), ("ok", "ok"))
        bad = O.verify({"draftkings": row(-190, 160), "fanduel": row(-148, 128), "betmgm": row(-150, 130, 9.0)})
        self.assertEqual(bad["ml"]["status"], "difiere")
        self.assertGreaterEqual(abs(bad["ml"]["diff"]), 3)
        solo = O.verify({"draftkings": row(-150, 130, 9.0), "fanduel": row(-150, 130, 8.5), "betmgm": row(-150, 130, 8.5)})
        self.assertEqual(solo["total"]["status"], "difiere")
        self.assertIn("DraftKings pone 9.0", solo["total"]["text"])
        one = O.verify({"draftkings": row(-150, 130)})
        self.assertEqual(one["ml"]["status"], "una fuente")
        # run line (lo que se pide para verificar un pick de RL): mismo hándicap que DraftKings
        rl = lambda h, a: {"rl": {"home": {"point": -1.5, "price": h}, "away": {"point": 1.5, "price": a}}}  # noqa: E731
        v = O.verify({"draftkings": rl(118, -143), "fanduel": rl(122, -146), "betmgm": rl(118, -145),
                      "betrivers": rl(110, -134)}, "PHI", "ATL")
        self.assertEqual(v["rl"]["status"], "ok")
        self.assertEqual(v["rl"]["n"], 4)
        self.assertIn("Run line ATL -1.5", v["rl"]["text"])


class VerificacionEntraAlStake(unittest.TestCase):
    def test_casas_que_solo_traen_run_line_cuentan_para_la_referencia(self):
        b = load()
        g = next(x for x in b["upcoming"] if x["pk"] == 824223)
        home, away = "Detroit Tigers", "Washington Nationals"

        def book(key, rl_home, rl_away, ml=None):
            ms = [{"key": "spreads", "outcomes": [{"name": home, "price": rl_home, "point": -1.5},
                                                   {"name": away, "price": rl_away, "point": 1.5}]}]
            if ml:
                ms.append({"key": "h2h", "outcomes": [{"name": home, "price": ml[0]}, {"name": away, "price": ml[1]}]})
            return {"key": key, "title": key.title(), "markets": ms}
        b["odds"] = [{"pk": 824223, "home_team": home, "away_team": away, "commence_time": g["time"], "bookmakers": [
            book("draftkings", 150, -180, ml=(-120, 100)), book("fanduel", 140, -165), book("betmgm", 140, -165)]}]
        o = model.odds_for_game(model.Context(b), g)
        self.assertEqual(o["nBooks"], 3)
        self.assertEqual(o["refRl"]["away"][0], -165)                  # mediana de las 3 casas, no solo DraftKings
        self.assertIn("mediana de 3", o["refRl"]["away"][1])


class MovimientoDeMomios(unittest.TestCase):
    def test_moneyline_que_se_mueve(self):
        b = load()
        d = tempfile.mkdtemp()
        g = next(x for x in b["upcoming"] if x["pk"] == 824223)

        def odds(h, a):
            return [{"pk": 824223, "home_team": "Detroit Tigers", "away_team": "Washington Nationals", "bookmakers": [
                {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [{"name": "Detroit Tigers", "price": h},
                                                                           {"name": "Washington Nationals", "price": a}]}]}]}]
        t = C.Tracker(base=d, now=at("2026-09-23T13:00:00"))
        t.mlb(b)
        b["odds"] = odds(-120, 100)
        t.odds(b)
        self.assertEqual(t.new, [])
        t.now = at("2026-09-23T13:20:00")
        b["odds"] = odds(-125, 105)                            # < 3 pp: nada
        t.odds(b)
        self.assertEqual(t.new, [])
        t.now = at("2026-09-23T13:40:00")
        b["odds"] = odds(-145, 125)
        t.odds(b)
        self.assertEqual([(e["kind"], e["impact"]) for e in t.new], [("momio", "medio")])
        self.assertIn("hacia DET", t.new[0]["text"])
        self.assertEqual(g["pk"], t.new[0]["pk"])


class Estado(unittest.TestCase):
    def test_huella_de_cambios(self):
        """La rutina solo republica si cambia algo que se ve; los relojes no cuentan."""
        from mlbgs import build as BU
        p = {"meta": {"generatedAt": "2026-10-05T10:00:00+00:00"},
             "games": [{"pk": 1, "time": "2026-10-05T21:00:00Z", "decision": {"status": "esperar", "waitFor": "momio",
                                                                             "pick": {"pick": "NRFI"}, "stake": {"level": 0}},
                        "summary": {"lineupsConfirmed": {"away": False, "home": False}}, "picks": [{"pick": "NRFI", "price": -120, "ic": 45.3}]}],
             "live": {"games": []}, "historial": {"tickets": []}, "prolabs": [], "cambios": {"events": []}}
        e = BU.estado(p)
        self.assertEqual((e["games"], e["nextGame"], e["live"]), (1, "2026-10-05T21:00:00Z", 0))
        q = json.loads(json.dumps(p))
        q["meta"]["generatedAt"] = "2026-10-05T10:30:00+00:00"
        self.assertEqual(BU.estado(q)["digest"], e["digest"])                  # solo pasó el tiempo
        q["games"][0]["summary"]["lineupsConfirmed"] = {"away": True, "home": True}
        self.assertNotEqual(BU.estado(q)["digest"], e["digest"])               # salió el lineup
        q = json.loads(json.dumps(p))
        q["games"][0]["picks"][0]["price"] = -145
        self.assertNotEqual(BU.estado(q)["digest"], e["digest"])               # se movió el momio


class Cadencia(unittest.TestCase):
    def test_mas_seguido_cerca_del_juego(self):
        now = at("2026-09-29T12:00:00")
        self.assertEqual(CA.plan([at("2026-09-29T14:00:00")], now)["sleep"], 420)
        self.assertEqual(CA.plan([at("2026-09-29T17:00:00")], now)["sleep"], 780)
        self.assertEqual(CA.plan([at("2026-09-29T21:00:00")], now)["sleep"], 1500)       # ≤ 12 h
        self.assertEqual(CA.plan([at("2026-09-30T12:00:00")], now)["sleep"], 3300)       # ≤ 36 h: cada hora
        self.assertEqual(CA.plan([at("2026-10-02T12:00:00")], now)["sleep"], 0)          # nada en 36 h: se detiene
        self.assertEqual(CA.plan([], now)["sleep"], 0)
        # un partido en juego (empezó hace 2 h) con el siguiente lejos: cada ~15 min
        self.assertEqual(CA.plan([at("2026-09-29T10:00:00"), at("2026-09-30T12:00:00")], now)["sleep"], 780)
        self.assertEqual(CA.plan([at("2026-09-29T11:45:00")], now)["sleep"], 420)   # en calentamiento o retrasado


if __name__ == "__main__":
    unittest.main()
