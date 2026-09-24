"""Decisión única por partido sobre un recorte real del bundle: una opción, stake 1–10 y todo el análisis conectado."""
import copy
import gzip
import json
import os
import unittest

from mlbgs import decision as DE
from mlbgs import model
from mlbgs import stake as ST

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bundle_small.json.gz")


class Decision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            bundle = json.load(f)
        ctx = model.Context(bundle)
        cls.games = [model.analyze(ctx, g) for g in bundle["upcoming"]]
        ST.attach(cls.games, [], [])
        DE.attach(cls.games, [], {})

    def test_una_sola_opcion_por_partido(self):
        for g in self.games:
            d = g["decision"]
            self.assertIn(d["status"], ("apostar", "esperar", "no apostar"))
            self.assertIn(d["pick"]["pick"], [p["pick"] for p in g["picks"]])
            if d["status"] == "apostar":
                self.assertTrue(1 <= d["stake"]["level"] <= 10)
                self.assertEqual(d["stake"]["amount"], ST.AMOUNTS[d["stake"]["level"]])
                self.assertIsNotNone(d["stake"]["minPrice"])
            else:
                self.assertEqual(d["stake"]["level"], 0)

    def test_elige_el_de_mas_confianza_que_pasa_las_reglas(self):
        for g in self.games:
            d = g["decision"]
            if d["status"] != "apostar":
                continue
            ok = [p for p in g["picks"] if not p["stake"]["block"] and p["stake"]["ladder"]]
            self.assertEqual(d["pick"]["ic"], max(p["ic"] for p in ok))

    def test_checklist_conecta_todo_el_analisis(self):
        keys = {"Contexto y motivación", "Forma e historial", "Abridores", "Bullpen y fatiga", "Modelo de carreras",
                "Guion del partido", "Contradicciones", "Parque, clima y umpire", "Datos obligatorios", "Cuotas"}
        for g in self.games:
            got = {c["k"] for c in g["decision"]["checklist"]}
            self.assertTrue(keys <= got, keys - got)
            self.assertTrue(all(c["tone"] in ("pro", "contra", "neutral") for c in g["decision"]["checklist"]))

    def test_esperar_si_solo_falta_un_dato(self):
        g = copy.deepcopy(self.games[0])
        for p in g["picks"]:
            p["blockedWithOdds"] = True
            p["stake"] = ST.plan(p)
        d = DE.decide(g, g["picks"])
        self.assertEqual(d["status"], "esperar")
        self.assertEqual(d["stake"]["level"], 0)

    def test_no_apostar_si_nada_pasa(self):
        g = copy.deepcopy(self.games[0])
        for p in g["picks"]:
            p["guion"] = "No"
            p["stake"] = ST.plan(p)
        d = DE.decide(g, g["picks"])
        self.assertEqual(d["status"], "no apostar")


if __name__ == "__main__":
    unittest.main()
