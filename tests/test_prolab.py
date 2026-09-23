"""Pruebas de DIAMANTE-24 (cadena de Markov + Monte Carlo), del índice de confianza y del cotejo de datos."""
import os
import unittest

from mlbgs import markov as MK
from mlbgs import picks as P
from mlbgs import validate as V

LEAGUE = {"K": 0.221, "BB": 0.100, "1B": 0.142, "2B": 0.041, "3B": 0.004, "HR": 0.030, "OUT": 0.462}


class CadenaDeMarkov(unittest.TestCase):
    def test_transiciones_suman_uno(self):
        for st in MK.STATES:
            for ev in MK.EVENTS:
                self.assertAlmostEqual(sum(p for p, *_ in MK.transitions(st, ev)), 1.0, places=9)

    def test_casa_llena_con_base_por_bolas_anota(self):
        self.assertEqual(MK.transitions((7, 0), "BB"), [(1.0, 7, 0, 1)])
        self.assertEqual(MK.transitions((0, 1), "HR"), [(1.0, 0, 0, 1)])

    def test_re24_tiene_la_forma_empirica(self):
        re = MK.re24(MK.with_residual(LEAGUE, 0.011), 0.45)["table"]
        self.assertTrue(0.40 < re[0][0] < 0.60)            # bases vacías, 0 outs
        self.assertTrue(re[7][0] > re[3][0] > re[1][0] > re[0][0])   # más corredores, más carreras
        for b in range(8):
            self.assertTrue(re[b][0] > re[b][1] > re[b][2])  # más outs, menos carreras

    def test_calibracion_reproduce_el_objetivo(self):
        e = MK.calibrate_residual(LEAGUE, 0.50, 0.45)
        self.assertAlmostEqual(MK.re24(MK.with_residual(LEAGUE, e), 0.45)["RE"][0], 0.50, places=3)
        self.assertTrue(0 < e < 0.05)

    def test_matriz_de_rotacion_del_lineup(self):
        chain = MK.lineup_chain([MK.with_residual(LEAGUE, 0.011)] * 9, 0.45, iters=40)
        for row in chain["T"]:
            self.assertAlmostEqual(sum(row), 1.0, places=3)
        self.assertAlmostEqual(chain["R"][0], chain["R"][4], places=6)   # lineup idéntico: todos iguales

    def test_log5_multinomial_neutro(self):
        self.assertEqual({k: round(v, 12) for k, v in MK.combine(LEAGUE, LEAGUE, LEAGUE).items()},
                         {k: round(v / sum(LEAGUE.values()), 12) for k, v in LEAGUE.items()})


class IndiceDeConfianza(unittest.TestCase):
    def test_formula(self):
        r = P.score_pick(0.60, {"a": 0.60, "b": 0.58}, -110, None, False, True, 1.0, 0, [])
        self.assertAlmostEqual(r["parts"]["fuerza"], (0.60 - 110 / 210) / 0.12)
        self.assertAlmostEqual(r["parts"]["consenso"], 1 - 0.02 / 0.12)
        self.assertEqual(r["level"], "Alta" if r["ic"] >= 70 else "Media" if r["ic"] >= 55 else r["level"])

    def test_gate_y_contradicciones_bajan_el_indice(self):
        base = P.score_pick(0.60, {"a": 0.6}, -110, None, False, True, 1.0, 0, [])["ic"]
        self.assertLess(P.score_pick(0.60, {"a": 0.6}, -110, None, True, True, 1.0, 0, [])["ic"], base)
        self.assertLess(P.score_pick(0.60, {"a": 0.6}, -110, None, False, True, 1.0, 2, ["x", "y"])["ic"], base)

    def test_calificacion_de_picks(self):
        inn = [[1, 0], [0, 0], [0, 2], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]
        self.assertTrue(P.grade({"pick": "DET", "family": "ML"}, "WSH", "DET", 1, 3, inn))
        self.assertTrue(P.grade({"pick": "WSH +1.5", "family": "RL"}, "WSH", "DET", 1, 2, inn))
        self.assertFalse(P.grade({"pick": "Over 8.5", "family": "Total", "line": 8.5}, "WSH", "DET", 1, 3, inn))
        self.assertFalse(P.grade({"pick": "NRFI", "family": "NRFI"}, "WSH", "DET", 1, 3, inn))
        self.assertTrue(P.grade({"pick": "V Under 5.5 K", "family": "K", "line": 5.5}, "WSH", "DET", 1, 3, inn, {"V": 4}))


class Cotejo(unittest.TestCase):
    def test_detecta_duplicados_y_standings(self):
        g = {"pk": 1, "date": "2026-09-01", "away": 1, "home": 2, "ar": 3, "hr": 2, "inn": [[3, 2]]}
        bundle = {"teams": {"1": {"abbr": "A"}, "2": {"abbr": "B"}}, "results": [g, dict(g)],
                  "standings": {"1": {"w": 1, "l": 0, "rs": 3, "ra": 2}, "2": {"w": 0, "l": 1, "rs": 2, "ra": 3}},
                  "teamStats": {}, "upcoming": []}
        out = {c["id"]: c for c in V.validate(bundle)["checks"]}
        self.assertEqual(out["dup"]["status"], "Falla")
        self.assertEqual(out["standings"]["status"], "Alerta")


@unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(__file__), "..", "prolab", "snapshot_824223_pre.json.gz")),
                     "sin datos congelados del Pro-Lab")
class ProLab(unittest.TestCase):
    def test_corrida_corta(self):
        from mlbgs import prolab
        out = prolab.run(824223, n_sims=1500, n_morning=500)
        self.assertEqual(out["model"], "DIAMANTE-24")
        self.assertTrue(0.3 < out["mc"]["pHome"] < 0.7)
        self.assertAlmostEqual(sum(out["mc"]["total"]), 1.0, places=6)
        self.assertLess(out["frozenAt"], out["firstPitch"].replace("Z", "+00:00"))  # datos previos al partido
        self.assertEqual(len(out["lineups"]["away"]), 9)
        self.assertTrue(out["picks"][0]["ic"] >= out["picks"][-1]["ic"])


if __name__ == "__main__":
    unittest.main()
