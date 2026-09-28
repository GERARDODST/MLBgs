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


class Correcciones20260928b(unittest.TestCase):
    """Algoritmo 2026.09.28.2 (solo partidos futuros): F5 con Binomial Negativa, equipo sin nada en juego y alertas 8.3/7.5."""

    @classmethod
    def setUpClass(cls):
        with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
            cls.bundle = json.load(f)
        cls.ctx = model.Context(cls.bundle)
        cls.games = [model.analyze(cls.ctx, g) for g in cls.bundle["upcoming"]]

    def test_f5_y_f3_sobredispersos(self):
        lg = self.ctx.lg
        self.assertGreater(lg.var_ratio_f5, 1.2)
        self.assertGreater(lg.var_ratio_f3, 1.2)
        from mlbgs import mathlib as M
        # con la misma λ, la Binomial Negativa da un favorito F5 menos extremo que Poisson
        nb = M.outcome_probs(M.negbin_pmf(3.4, lg.var_ratio_f5), M.negbin_pmf(2.0, lg.var_ratio_f5))
        po = M.outcome_probs(M.poisson_pmf(3.4), M.poisson_pmf(2.0))
        self.assertLess(nb[0] / (nb[0] + nb[2]), po[0] / (po[0] + po[2]))
        for g in self.games:
            for p in g["picks"]:
                if p["family"] in ("F5", "F5 total"):
                    self.assertIn("λ + Binomial Negativa F5 (5.4)", p["methods"])

    def test_equipo_sin_nada_en_juego(self):
        row = {"team": "ATL", "status": "Clasificado", "leverage": {"po": {"delta": 0.0}, "div": {"delta": 0.0}, "bye": {"delta": 0.002}}}
        self.assertIn("ya clasificó", model.idle_team(row, {"type": "R"}))
        self.assertIn("eliminado", model.idle_team({"team": "CIN", "status": "Eliminado", "leverage": {}}, {"type": "R"}))
        self.assertIsNone(model.idle_team({**row, "leverage": {"bye": {"delta": 0.08}}}, {"type": "R"}))   # pelea la siembra
        self.assertIsNone(model.idle_team({"team": "X", "status": "En la pelea", "leverage": {}}, {"type": "R"}))
        self.assertIsNone(model.idle_team(row, {"type": "F"}))            # postemporada: todo en juego

    def test_alerta_8_3_marca_los_picks_que_dependen_del_colapso(self):
        from mlbgs import picks as P
        g = self.games[0]
        g["sections"]["s8"]["collapse"] = {"home": "Pitcher Local"}
        a = g["teams"]["away"]["abbr"]
        for p in P.build(g):
            dep = (p["family"] in ("ML", "RL", "F5") and p["pick"].split(" ")[0] == a) or \
                  (p["family"] in ("Total", "F5 total") and p["pick"].startswith("Over")) or p["pick"] == "YRFI"
            self.assertEqual(bool(p["alerts"]), dep, p["pick"])
