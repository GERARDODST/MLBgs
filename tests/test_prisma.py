"""Pruebas de PRISMA (modelo bayesiano jerárquico) y de la probabilidad de victoria en vivo."""
import math
import random
import unittest

from mlbgs import picks as P
from mlbgs import prisma as PR


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


class AlgebraLineal(unittest.TestCase):
    A = [[4.0, 1.2, 0.4], [1.2, 3.0, 0.5], [0.4, 0.5, 2.0]]

    def test_cholesky_reconstruye_la_matriz(self):
        L = PR.cholesky(self.A)
        LT = [list(r) for r in zip(*L)]
        for i, row in enumerate(matmul(L, LT)):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, self.A[i][j], places=9)

    def test_inversa_por_identidad(self):
        inv = PR.inverse_spd(self.A)
        for i, row in enumerate(matmul(self.A, inv)):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, 1.0 if i == j else 0.0, places=9)


class AjusteJerarquico(unittest.TestCase):
    def test_recupera_el_orden_de_fuerzas(self):
        rng = random.Random(3)
        teams = list(range(1, 9))
        att = {t: 0.25 * (t - 4.5) / 3.5 for t in teams}
        dfn = {t: -0.20 * (t - 4.5) / 3.5 for t in teams}
        obs, pk = [], 0
        for _ in range(40):
            for a in teams:
                for h in teams:
                    if a == h:
                        continue
                    pk += 1
                    for off, d_, home in ((a, h, 0), (h, a, 1)):
                        mu = 4.4 * math.exp(att[off] - dfn[d_] + 0.04 * home)
                        # Poisson por inversión
                        y, p, u = 0, math.exp(-mu), rng.random()
                        c = p
                        while u > c:
                            y += 1
                            p *= mu / y
                            c += p
                        obs.append((off, d_, home, y, 9, 1.0, 0.0, pk))
        f = PR.fit(obs, set(teams), iters=15, eb_rounds=3)
        est = [PR.team_params(f, t)[0] for t in teams]
        self.assertEqual(sorted(range(8), key=lambda i: est[i]), list(range(8)))
        self.assertTrue(0.05 < f.sa < 0.4)
        self.assertAlmostEqual(math.exp(f.theta[0]) * 9, 4.4, delta=0.4)


class ProbabilidadEnVivo(unittest.TestCase):
    half = [0.72, 0.15, 0.07, 0.035, 0.015, 0.006, 0.003, 0.001]

    def test_escala_conserva_la_media(self):
        d = PR.scaled_half(self.half, 0.62)
        self.assertAlmostEqual(sum(d), 1.0, places=9)
        self.assertAlmostEqual(sum(k * v for k, v in enumerate(d)), 0.62, places=3)

    def test_juego_parejo_al_inicio(self):
        s = sum(self.half)
        h = [v / s for v in self.half]
        self.assertAlmostEqual(PR.win_prob(1, True, 0, h, h, 0.5), 0.5, delta=0.02)

    def test_monotona_en_la_ventaja(self):
        h = PR.scaled_half(self.half, 0.5)
        ps = [PR.win_prob(6, True, d, h, h, 0.5) for d in range(-3, 4)]
        self.assertEqual(ps, sorted(ps))

    def test_local_arriba_en_la_baja_de_la_novena_ya_gano(self):
        h = PR.scaled_half(self.half, 0.5)
        self.assertAlmostEqual(PR.win_prob(9, False, 1, h, h, 0.5), 1.0, places=9)


class CalificacionDePicks(unittest.TestCase):
    inn = [[0, 1], [0, 0], [2, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 2], [0, None]]

    def test_f5_empatado_es_push(self):
        p = {"pick": "DET F5", "family": "F5", "line": None}
        self.assertIsNone(P.grade(p, "WSH", "DET", 3, 4, self.inn))   # F5 2-2

    def test_total_y_nrfi(self):
        self.assertTrue(P.grade({"pick": "Under 8.5", "family": "Total", "line": 8.5}, "WSH", "DET", 3, 4, self.inn))
        self.assertTrue(P.grade({"pick": "YRFI", "family": "NRFI", "line": None}, "WSH", "DET", 3, 4, self.inn))

    def test_ponches_del_abridor(self):
        p = {"pick": "Framber Valdez Under 5.5 K", "family": "K", "line": 5.5}
        self.assertFalse(P.grade(p, "WSH", "DET", 3, 4, self.inn, {"Framber Valdez": 7}))
        self.assertIsNone(P.grade(p, "WSH", "DET", 3, 4, self.inn, {}))


if __name__ == "__main__":
    unittest.main()
