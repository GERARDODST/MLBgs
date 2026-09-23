import math
import unittest

from mlbgs import mathlib as M


class Distribuciones(unittest.TestCase):
    def test_poisson_suma_uno_y_media(self):
        pmf = M.poisson_pmf(4.5)
        self.assertAlmostEqual(sum(pmf), 1.0, places=9)
        self.assertAlmostEqual(M.mean_of(pmf), 4.5, places=4)
        self.assertAlmostEqual(pmf[0], math.exp(-4.5), places=6)

    def test_binomial_negativa_media_y_sobredispersion(self):
        pmf = M.negbin_pmf(4.5, 2.0)
        self.assertAlmostEqual(sum(pmf), 1.0, places=9)
        self.assertAlmostEqual(M.mean_of(pmf), 4.5, places=2)
        self.assertAlmostEqual(M.var_of(pmf) / M.mean_of(pmf), 2.0, places=1)
        # sobredispersión: más blanqueadas que Poisson (sección 5.7.5)
        self.assertGreater(pmf[0], M.poisson_pmf(4.5)[0])

    def test_negbin_sin_sobredispersion_es_poisson(self):
        self.assertEqual(M.negbin_pmf(3.0, 1.0), M.poisson_pmf(3.0))

    def test_resultados_simetricos(self):
        p = M.poisson_pmf(4.0)
        w, t, l = M.outcome_probs(p, p)
        self.assertAlmostEqual(w, l, places=9)
        self.assertAlmostEqual(w + t + l, 1.0, places=9)

    def test_over_under_linea_entera_tiene_push(self):
        pmf = M.sum_pmf(M.poisson_pmf(4.2), M.poisson_pmf(4.3))
        ou = M.over_under(pmf, 8.0)
        self.assertGreater(ou["push"], 0.05)
        self.assertAlmostEqual(ou["over"] + ou["under"] + ou["push"], 1.0, places=9)
        self.assertEqual(M.over_under(pmf, 8.5)["push"], 0.0)


class Sabermetria(unittest.TestCase):
    def test_innings_notacion_beisbol(self):
        self.assertAlmostEqual(M.ip_to_float("123.2"), 123 + 2 / 3)
        self.assertAlmostEqual(M.ip_to_float("5.1"), 5 + 1 / 3)
        self.assertEqual(M.ip_to_float(None), 0.0)

    def test_fip_con_constante(self):
        # (13·10 + 3·(40+5) − 2·150)/150 + 3.10
        self.assertAlmostEqual(M.fip(10, 40, 5, 150, 150, 3.10), (130 + 135 - 300) / 150 + 3.10)

    def test_constante_fip_reproduce_era_de_liga(self):
        c = M.fip_constant(4.2, 5000, 14000, 1800, 40000, 43000)
        self.assertAlmostEqual(M.fip(5000, 14000, 1800, 40000, 43000, c), 4.2)

    def test_pitagoras(self):
        self.assertAlmostEqual(M.pythagorean(700, 700), 0.5)
        self.assertGreater(M.pythagorean(750, 650), 0.55)
        self.assertAlmostEqual(M.pythagenpat_exponent(729, 729, 162), 9 ** 0.287)

    def test_log5(self):
        self.assertAlmostEqual(M.log5(0.6, 0.6), 0.5)
        self.assertAlmostEqual(M.log5(0.6, 0.4), (0.6 - 0.24) / (1.0 - 0.48))

    def test_shrinkage(self):
        self.assertAlmostEqual(M.shrink(0.30, 70, 70, 0.22), 0.26)
        self.assertEqual(M.shrink(0.30, 0, 70, 0.22), 0.22)

    def test_elo_y_ventaja_local(self):
        self.assertAlmostEqual(M.elo_expected(1500, 1500), 0.5)
        self.assertAlmostEqual(M.elo_expected(1500 + M.prob_to_elo_diff(0.6), 1500), 0.6, places=9)
        self.assertAlmostEqual(M.with_home_edge(0.5, 1.0), 0.5)


class Mercado(unittest.TestCase):
    def test_probabilidad_implicita(self):
        self.assertAlmostEqual(M.american_to_prob(-150), 0.6)
        self.assertAlmostEqual(M.american_to_prob(150), 0.4)

    def test_momio_justo_ida_y_vuelta(self):
        for p in (0.35, 0.5, 0.62):
            self.assertAlmostEqual(M.american_to_prob(M.fair_american(p)), p)

    def test_kelly(self):
        # momio +100 (b = 1) y p = 0.55 → f* = 0.10
        self.assertAlmostEqual(M.kelly(0.55, 100), 0.10)
        self.assertEqual(M.kelly(0.40, 100), 0.0)


if __name__ == "__main__":
    unittest.main()
