"""Pruebas de KRONOS: cadena absorbente de la cuenta y análisis de procesos."""
import unittest

from mlbgs import kronos as KR


def table(cat):
    return {c: {k: (1.0 if k == cat else 0.0) for k in KR.CATS} for c in KR.CKEY}


class CadenaDeLaCuenta(unittest.TestCase):
    def test_solo_bolas_es_base_por_bolas_en_4(self):
        ab = KR.absorb(table("B"))
        self.assertAlmostEqual(ab["BB"], 1.0, places=9)
        self.assertAlmostEqual(ab["pitches"][0], 4.0, places=9)

    def test_solo_strikes_es_ponche_en_3(self):
        ab = KR.absorb(table("CS"))
        self.assertAlmostEqual(ab["K"], 1.0, places=9)
        self.assertAlmostEqual(ab["pitches"][0], 3.0, places=9)

    def test_foul_con_dos_strikes_no_suma(self):
        self.assertEqual(KR.next_state(1, 2, "F"), (1, 2))
        self.assertEqual(KR.next_state(1, 2, "FB"), ("K",))
        self.assertEqual(KR.next_state(3, 1, "B"), ("BB",))

    def test_probabilidades_absorbentes_suman_uno(self):
        P = {c: {"B": 0.35, "CS": 0.15, "SS": 0.12, "F": 0.17, "FB": 0.005, "HBP": 0.005, "X": 0.2} for c in KR.CKEY}
        ab = KR.absorb(P)
        self.assertAlmostEqual(ab["K"] + ab["BB"] + ab["HBP"] + ab["X"], 1.0, places=9)
        self.assertAlmostEqual(sum(KR.pa_length_dist(P, kmax=40)), 1.0, places=6)

    def test_log5_por_cuenta_con_jugadores_promedio_da_la_liga(self):
        lg = {"B": 0.4, "CS": 0.2, "SS": 0.1, "F": 0.15, "FB": 0.0, "HBP": 0.01, "X": 0.14}
        out = KR.combine(lg, lg, lg)
        for k, v in lg.items():
            self.assertAlmostEqual(out[k], v / sum(lg.values()), places=9)


class ProcesosDeLaLiga(unittest.TestCase):
    def test_cuasigeometrica_recupera_d(self):
        a, d = 0.72, 0.45
        games = []
        probs = [a] + [(1 - a) * (1 - d) * d ** (n - 1) for n in range(1, 12)]
        counts = [round(p * 90000) for p in probs]
        inn = [[k, None] for k, c in enumerate(counts) for _ in range(c)]
        for i in range(0, len(inn), 9):
            games.append({"inn": inn[i:i + 9]})
        qg = KR.quasigeometric(games)
        self.assertAlmostEqual(qg["a"], a, places=2)
        self.assertAlmostEqual(qg["d"], d, delta=0.02)

    def test_sigma_implicita(self):
        # P = Φ(μ/σ): con μ = 0.5 y σ = 4, P ≈ 0.5498
        self.assertAlmostEqual(KR.implied_sigma(0.549738, 0.5), 4.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
