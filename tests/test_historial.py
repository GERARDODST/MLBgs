"""Base de datos de tickets: registro, congelamiento, calificación, anulación y escritura estable."""
import json
import os
import tempfile
import unittest

from mlbgs import historial as HI


def analysis(p_home=0.40, pk=1, date="2026-09-24", time="2026-09-24T23:05:00Z"):
    return {
        "pk": pk, "date": date, "time": time, "venue": {"name": "Citizens Bank Park"},
        "teams": {"away": {"abbr": "MIL", "name": "Milwaukee Brewers"}, "home": {"abbr": "PHI", "name": "Philadelphia Phillies"}},
        "summary": {"pHome": p_home, "proj": {"away": 5.0, "home": 4.0, "total": 9.0}, "nrfi": 0.5, "light": "Gris",
                    "confidence": "media", "probables": {"away": {"name": "Pitcher A"}, "home": {"name": "Pitcher B"}},
                    "lineupsConfirmed": {"away": True, "home": False}},
        "picks": [
            {"family": "RL", "market": "Run Line", "pick": "MIL +1.5", "p": 0.75, "line": None, "ic": 80.0, "level": "Alta",
             "methods": {"λ + Binomial Negativa (5.3 · 5.7.5)": 0.75}, "how": "Distribución del margen", "fair": -300.0},
            {"family": "K", "market": "Prop de ponches", "pick": "Pitcher B Over 5.5 K", "p": 0.55, "line": 5.5, "ic": 50.0, "level": "Baja"},
            {"family": "Total", "market": "Total completo", "pick": "Over 8.5", "p": 0.52, "line": 8.5, "ic": 40.0, "level": "Baja"},
        ],
    }


def sb(state, ar=None, hr=None, pk=1, date="2026-09-24", detailed=None, box=None):
    row = {"pk": pk, "date": date, "state": state, "detailed": detailed or state, "ar": ar, "hr": hr,
           "inn": [[1, 0], [0, 1], [2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] if ar is not None else []}
    if box:
        row["box"] = box
    return row


class Historial(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "historial")
        self.pred = os.path.join(self.tmp.name, "predictions")
        os.makedirs(self.pred)

    def tearDown(self):
        self.tmp.cleanup()

    def run_update(self, analyses, scoreboard, now):
        bundle = {"results": [], "boxscores": [], "scoreboard": scoreboard}
        return HI.update(bundle, analyses, [], now, dir_=self.dir, pred_dir=self.pred)

    def test_ticket_guarda_modelo_analisis_y_picks(self):
        t = self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["status"], "abierto")
        self.assertEqual(t["model"], "framework")
        self.assertIn("λ + Binomial Negativa", t["analyses"])
        self.assertEqual([p["principal"] for p in t["picks"]], [True, True, False])
        self.assertEqual(t["picks"][0]["methods"], ["λ + Binomial Negativa (5.3 · 5.7.5)"])
        self.assertEqual(t["pred"]["lineups"], "Proyectados")
        with open(os.path.join(self.dir, "2026-09-24.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["id"], "2026-09-24-1-framework")

    def test_se_actualiza_antes_y_se_congela_al_empezar(self):
        self.run_update([analysis(0.40)], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        t = self.run_update([analysis(0.45)], [sb("Preview")], "2026-09-24T19:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["pred"]["pHome"], 0.45)
        self.assertEqual(t["registeredAt"], "2026-09-24T19:00:00+00:00")
        t = self.run_update([], [sb("Live", 1, 0)], "2026-09-24T23:20:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["status"], "cerrado")
        # un análisis tardío (datos atrasados) ya no puede reescribir un ticket cerrado
        t = self.run_update([analysis(0.90)], [sb("Live", 1, 0)], "2026-09-24T23:40:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["pred"]["pHome"], 0.45)

    def test_sin_cambios_no_reescribe(self):
        self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        path = os.path.join(self.dir, "2026-09-24.json")
        before = os.path.getmtime(path)
        os.utime(path, (before - 100, before - 100))
        t = self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:20:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(os.path.getmtime(path), before - 100)
        self.assertEqual(t["registeredAt"], "2026-09-24T18:00:00+00:00")

    def test_calificacion_con_resultado_oficial(self):
        self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        t = self.run_update([], [sb("Final", 4, 1)], "2026-09-25T02:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["status"], "calificado")
        self.assertEqual((t["result"]["ar"], t["result"]["hr"]), (4, 1))
        res = {p["pick"]: p["res"] for p in t["picks"]}
        self.assertEqual(res["MIL +1.5"], "ganado")
        self.assertEqual(res["Over 8.5"], "perdido")
        self.assertEqual(res["Pitcher B Over 5.5 K"], "sin dato")   # sin box score todavía
        box = {"away": {"pit": [{"name": "Pitcher A", "k": 3}]}, "home": {"pit": [{"name": "Pitcher B", "k": 8}]}}
        t = self.run_update([], [sb("Final", 4, 1, box=box)], "2026-09-25T02:20:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual({p["pick"]: p["res"] for p in t["picks"]}["Pitcher B Over 5.5 K"], "ganado")
        self.assertEqual(t["result"]["gradedAt"], "2026-09-25T02:00:00+00:00")

    def test_push_en_linea_exacta(self):
        a = analysis()
        a["picks"][2] = {**a["picks"][2], "pick": "Over 5.0", "line": 5.0}
        self.run_update([a], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        t = self.run_update([], [sb("Final", 4, 1)], "2026-09-25T02:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual({p["pick"]: p["res"] for p in t["picks"]}["Over 5.0"], "push")

    def test_pospuesto_anula_el_ticket(self):
        self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        t = self.run_update([], [sb("Final", detailed="Postponed")], "2026-09-24T22:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["status"], "anulado")
        self.assertTrue(all(p["res"] == "anulado" for p in t["picks"]))

    def test_reprogramado_a_otra_fecha(self):
        self.run_update([analysis()], [sb("Preview")], "2026-09-24T18:00:00+00:00")
        t = self.run_update([], [sb("Preview", date="2026-09-25")], "2026-09-24T22:00:00+00:00")["2026-09-24-1-framework"]
        self.assertEqual(t["status"], "anulado")

    def test_respaldo_desde_el_seguimiento(self):
        with open(os.path.join(self.pred, "2026-09-23.json"), "w", encoding="utf-8") as f:
            json.dump([{"pk": 7, "date": "2026-09-23", "time": "2026-09-23T23:05:00Z", "generatedAt": "2026-09-23T22:00:00+00:00",
                        "away": "MIL", "home": "PHI", "pHome": 0.4, "projAway": 5, "projHome": 4, "total": 9, "nrfi": 0.5,
                        "light": "Gris", "confidence": "media", "probables": {"away": "A", "home": "B"},
                        "topPicks": [{"family": "F5", "market": "F5 Moneyline", "pick": "MIL F5", "p": 0.6, "line": None, "ic": 70, "level": "Alta"}]}], f)
        t = self.run_update([], [sb("Final", 4, 1, pk=7, date="2026-09-23")], "2026-09-24T02:00:00+00:00")["2026-09-23-7-framework"]
        self.assertEqual(t["source"], "seguimiento")
        self.assertEqual(t["picks"][0]["res"], "ganado")

    def test_vista_de_pagina_recorta_explicaciones_viejas(self):
        tickets = self.run_update([analysis(), analysis(pk=2, date="2026-08-01", time="2026-08-01T23:05:00Z")], [], "2026-07-31T18:00:00+00:00")
        view = HI.page_view(tickets, "x", full_days=1)
        old = next(t for t in view["tickets"] if t["pk"] == 2)
        new = next(t for t in view["tickets"] if t["pk"] == 1)
        self.assertNotIn("how", old["picks"][0])
        self.assertIn("how", new["picks"][0])
        self.assertIn("kronos", view["models"])


if __name__ == "__main__":
    unittest.main()
