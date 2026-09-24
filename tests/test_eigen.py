"""EIGEN (PCA): álgebra en Python puro, regresión sobre componentes y el ajuste sobre un bundle real congelado."""
import gzip
import json
import math
import os
import random
import unittest

from mlbgs import eigen as EG

ROOT = os.path.dirname(os.path.dirname(__file__))
FROZEN = os.path.join(ROOT, "prolab", "bundle_823411_pre.json.gz")


def latent_data(n=300, seed=3):
    """Cinco métricas que dependen de dos factores ocultos (como barrel/hard-hit/EV y whiff/K)."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
        X.append([f1 + 0.2 * rng.gauss(0, 1), f1 + 0.2 * rng.gauss(0, 1), f1 + 0.2 * rng.gauss(0, 1),
                  f2 + 0.2 * rng.gauss(0, 1), f2 + 0.2 * rng.gauss(0, 1)])
        y.append(2.0 * f1 - 1.0 * f2 + 0.3 * rng.gauss(0, 1))
    return X, y


class Algebra(unittest.TestCase):
    def test_jacobi_valores_y_vectores_propios(self):
        A = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]]
        vals, vecs = EG.jacobi(A)
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertAlmostEqual(sum(vals), 8.0, places=9)                 # traza
        for j, lam in enumerate(vals):
            v = [vecs[i][j] for i in range(3)]
            Av = [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(Av[i], lam * v[i], places=8)       # A·v = λ·v
            self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=9)
        dot = sum(vecs[i][0] * vecs[i][1] for i in range(3))
        self.assertAlmostEqual(dot, 0.0, places=9)                        # ortogonales

    def test_pca_encuentra_los_factores(self):
        X, _ = latent_data()
        P = EG.pca(X, [1.0] * len(X))
        self.assertAlmostEqual(sum(P["explained"]), 1.0, places=9)
        self.assertGreater(P["explained"][0] + P["explained"][1], 0.9)   # dos factores ocultos
        self.assertLess(P["explained"][2], 0.05)
        # el componente 1 carga en las tres primeras métricas (o en las dos últimas), no mezcla
        pc1 = [abs(P["vecs"][i][0]) for i in range(5)]
        grupo = pc1[:3] if pc1[0] > pc1[3] else pc1[3:]
        otro = pc1[3:] if pc1[0] > pc1[3] else pc1[:3]
        self.assertGreater(min(grupo), 0.5)
        self.assertLess(max(otro), 0.2)

    def test_pcr_elige_pocos_componentes_y_predice(self):
        X, y = latent_data()
        w = [1.0] * len(X)
        k, r2 = EG.choose_k(X, y, w, 5)
        self.assertEqual(k, 2)
        self.assertGreater(r2[k - 1], 0.9)
        self.assertGreater(r2[1], r2[0] + 0.05)          # el segundo factor también cuenta

    def test_estabilidad_alta_con_estructura_clara(self):
        X, _ = latent_data(n=200)
        st = EG.stability(X, [1.0] * len(X), 2, B=15)
        self.assertGreater(min(st), 0.95)

    def test_marcador_suma_uno_y_favorece_al_que_anota_mas(self):
        pa, ph, p_home, tie = EG.outcome(3.6, 5.6, 2.27, 0.53)
        self.assertAlmostEqual(sum(pa), 1.0, places=4)
        self.assertAlmostEqual(sum(ph), 1.0, places=4)
        self.assertGreater(p_home, 0.6)
        self.assertTrue(0 < tie < 0.15)
        self.assertLess(EG.outcome(5.6, 3.6, 2.27, 0.53)[2], 0.4)


@unittest.skipUnless(os.path.exists(FROZEN), "sin bundle congelado")
class BundleReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(FROZEN, "rt", encoding="utf-8") as f:
            cls.bundle = json.load(f)
        cls.F = EG.fit_all(cls.bundle, B=6)

    def test_pca_de_jugadores_y_equipos(self):
        F = self.F
        self.assertGreater(len(F["pitchers"]["rows"]), 200)
        self.assertGreater(len(F["batters"]["rows"]), 200)
        self.assertEqual(len(F["teams"]["rows"]), 30)
        # bateadores: el primer eje (poder) explica la mayor parte y predice wOBA fuera de muestra
        bS = F["batters"]["summary"]
        self.assertGreater(bS["explained"][0], 0.4)
        self.assertGreater(bS["cvR2"][F["batters"]["k"] - 1], 0.5)

    def test_habilidad_entre_lo_observado_y_los_componentes(self):
        row = max(self.F["pitchers"]["rows"], key=lambda r: r["bf"])
        t = EG.pitcher_talent(row["id"], self.bundle, self.F, self.F["lg"])
        lo, hi = sorted((t["ra9Obs"], t["fit"]))
        self.assertTrue(lo - 1e-9 <= t["talent"] <= hi + 1e-9)
        self.assertTrue(4.0 <= t["ipStart"] <= 6.5)
        desconocido = EG.pitcher_talent(1, self.bundle, self.F, self.F["lg"])
        self.assertGreater(desconocido["talent"], self.F["lg"]["ra9"])     # sin datos: peor que la liga

    def test_donde_conviene_pca(self):
        F = self.F
        meth = {"n": 40, "explained": [0.9, 0.07, 0.03], "corr": [[1, 0.9, 0.8], [0.9, 1, 0.85], [0.8, 0.85, 1]], "loadings": []}
        where = EG.where_verdict(F["pitchers"]["summary"], F["batters"]["summary"], F["teams"]["summary"], meth)
        self.assertEqual([w["key"] for w in sorted(where, key=lambda w: w["rank"])], ["pitchers", "batters", "methods", "teams"])
        self.assertTrue(all(w["verdict"] and w["use"] for w in where))
        self.assertTrue(math.isfinite(F["lg"]["woba"]) and 0.28 < F["lg"]["woba"] < 0.34)


if __name__ == "__main__":
    unittest.main()
