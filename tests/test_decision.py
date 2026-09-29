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
            ok = [p for p in g["picks"] if not p["stake"]["block"] and p["stake"]["ladder"] and p["stake"]["level"]]
            self.assertEqual(d["pick"]["ic"], max(p["ic"] for p in ok))

    def test_con_momio_real_elige_el_que_llega_a_stake(self):
        g = self.games[0]
        a, b = (copy.deepcopy(p) for p in g["picks"][:2])
        a["priceIsReal"] = b["priceIsReal"] = True
        lad = [{"level": 1, "stake": 500, "from": -150}]
        a.update(ic=80.0, stake={"block": None, "ladder": lad, "level": 0, "price": -105,
                                 "why": "el casino lo ve 10 pp o más distinto que el modelo: verificar lesiones, descansos y lineup"})
        b.update(ic=70.0, stake={"block": None, "ladder": lad, "level": 4, "price": -120})
        d = DE.decide(g, [a, b])
        self.assertEqual((d["status"], d["pick"]["pick"], d["stake"]["level"]), ("apostar", b["pick"], 4))
        self.assertIn("-120", d["why"])
        b["stake"].update(level=0, why="edge menor a 3 pp con ese momio")
        d = DE.decide(g, [a, b])
        # el de más confianza cae en el filtro contra el mercado: se verifica ESE pick (algoritmo 2026.09.29.3)
        self.assertEqual((d["status"], d["waitFor"], d["pick"]["pick"]), ("esperar", "verificar", a["pick"]))
        self.assertIn("momio de referencia (-105)", d["why"])
        self.assertIn("se verifica", d["why"])
        self.assertEqual(d["stake"]["level"], 0)
        # aunque haya otro candidato sin momio, no se salta a esperar el momio de ese otro mercado
        c = copy.deepcopy(b)
        c.update(pick="Otro mercado", ic=60.0, priceIsReal=False, stake={"block": None, "ladder": lad, "level": 0})
        d = DE.decide(g, [a, b, c])
        self.assertEqual((d["status"], d["waitFor"], d["pick"]["pick"]), ("esperar", "verificar", a["pick"]))

    def test_checklist_conecta_todo_el_analisis(self):
        keys = {"Contexto y motivación", "Forma e historial", "Abridores", "Bullpen y fatiga", "Modelo de carreras",
                "Guion del partido", "Contradicciones", "Parque, clima y umpire", "Datos obligatorios", "Cuotas"}
        for g in self.games:
            got = {c["k"] for c in g["decision"]["checklist"]}
            self.assertTrue(keys <= got, keys - got)
            self.assertTrue(all(c["tone"] in ("pro", "contra", "neutral") for c in g["decision"]["checklist"]))

    def test_sin_momio_real_se_espera_el_momio(self):
        # corrección del 28-sep: nunca hay stake sin precio real; la decisión dice desde qué momio conviene
        g = self.games[0]
        a = copy.deepcopy(g["picks"][0])
        a.update(ic=80.0, priceIsReal=False, stake={"block": None, "ladder": [{"level": 1, "stake": 500, "from": -169},
                                                                            {"level": 10, "stake": 1500, "from": -126}], "level": 0})
        d = DE.decide(g, [a])
        self.assertEqual((d["status"], d["waitFor"], d["stake"]["level"]), ("esperar", "momio", 0))
        self.assertIn("-169", d["why"])
        self.assertIn("falta el momio", d["why"])

    def test_esperar_si_solo_falta_un_dato(self):
        g = copy.deepcopy(self.games[0])
        for p in g["picks"]:
            p["blockedWithOdds"] = True
            p["stake"] = ST.plan(p)
        d = DE.decide(g, g["picks"])
        self.assertEqual(d["status"], "esperar")
        self.assertEqual(d["waitFor"], "dato")
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
