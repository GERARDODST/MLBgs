"""Prueba de humo del modelo sobre un recorte real del bundle (MLB Stats API + Savant, 23-sep-2026)."""
import gzip
import json
import os
import unittest

from mlbgs import build, model

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bundle_small.json.gz")


class ModeloCompleto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            cls.bundle = json.load(f)
        cls.ctx = model.Context(cls.bundle)
        cls.games = [model.analyze(cls.ctx, g) for g in cls.bundle["upcoming"]]

    def test_diez_secciones(self):
        for g in self.games:
            self.assertEqual(sorted(g["sections"]), sorted(f"s{i}" for i in range(1, 11)))

    def test_probabilidades_validas(self):
        for g in self.games:
            s = g["summary"]
            self.assertTrue(0.2 < s["pHome"] < 0.8)
            self.assertAlmostEqual(s["pHome"] + s["pAway"], 1.0)
            self.assertTrue(5.0 < s["proj"]["total"] < 13.0)
            self.assertTrue(0.3 < s["nrfi"] < 0.75)
            for d in g["sections"]["s8"]["decisions"]:
                if d.get("p") is not None:
                    self.assertTrue(0 <= d["p"] <= 1)

    def test_sin_momios_nunca_verde(self):
        # sin cuotas no hay edge calculable → el framework no permite Verde
        for g in self.games:
            for d in g["sections"]["s8"]["decisions"]:
                self.assertNotEqual(d["light"], "Verde")

    def test_gate_bloquea_abridor_por_anunciar(self):
        tbd = [g for g in self.games if g["summary"]["probables"]["away"]["missing"]]
        self.assertTrue(tbd)
        for g in tbd:
            self.assertTrue(g["sections"]["s9"]["gate"]["blocks"]["f5"])

    def test_ajustes_en_base_no_se_duplican(self):
        for g in self.games:
            for r in g["sections"]["s5"]["adjustments"]:
                if r["inBase"]:
                    self.assertFalse(r["applied"])
            ups = [r for r in g["sections"]["s5"]["adjustments"] if r["applied"] and r["factor"] > 1]
            self.assertNotEqual(len(ups), 1, "nunca subir el total por un solo factor")

    def test_pagina_se_construye(self):
        payload = build.build(self.bundle, save=False)
        self.assertEqual(len(payload["games"]), len(self.bundle["upcoming"]))
        json.dumps(payload)  # serializable


if __name__ == "__main__":
    unittest.main()
