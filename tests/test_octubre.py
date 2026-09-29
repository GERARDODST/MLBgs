"""OCTUBRE (Pro-Lab, postemporada): las piezas matemáticas con datos sintéticos de resultado conocido."""
import math
import random
import unittest

from mlbgs import octubre as OC


def binom(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


class EncogimientoPitcherRival(unittest.TestCase):
    def pairs(self, kappa_true, n_pairs=600, seed=3):
        """Parejas con tasa real Beta(μκ, (1−μ)κ) alrededor de μ y n bateadores cada una."""
        rng = random.Random(seed)
        out = {}
        for i in range(n_pairs):
            mu = rng.uniform(0.18, 0.28)
            p = rng.betavariate(mu * kappa_true, (1 - mu) * kappa_true) if kappa_true else mu
            n = rng.randint(40, 120)
            out[(i, 0)] = {"k": {"n": n, "y": binom(rng, n, p), "mu": mu}}
        return out

    def test_kappa_recupera_el_valor_real(self):
        est = OC.kappa(self.pairs(150), "k")
        self.assertIsNotNone(est["kappa"])
        self.assertGreater(est["kappa"], 80)
        self.assertLess(est["kappa"], 300)
        self.assertLess(est["ci"][0], est["rho"])
        self.assertGreater(est["ci"][1], est["rho"])

    def test_el_ruido_de_lo_esperado_no_cuenta_como_efecto_de_pareja(self):
        """Si μ se estima con ruido (la tasa del pitcher sale de pocas aperturas), esa varianza no es de la pareja."""
        rng = random.Random(4)
        pairs = {}
        for i in range(800):
            mu_true = rng.uniform(0.18, 0.28)
            vmu = 0.0006
            mu_hat = min(0.4, max(0.05, mu_true + rng.gauss(0, vmu ** 0.5)))
            n = 80
            pairs[(i, 0)] = {"k": {"n": n, "y": binom(rng, n, mu_true), "mu": mu_hat, "vmu": vmu}}
        est = OC.kappa(pairs, "k")
        naive = OC.kappa({k: {"k": {kk: vv for kk, vv in v["k"].items() if kk != "vmu"}} for k, v in pairs.items()}, "k")
        self.assertTrue(est["kappa"] is None or est["kappa"] > 1000)
        self.assertIsNotNone(naive["kappa"])              # sin corregir, el ruido parece efecto de pareja
        self.assertLess(naive["kappa"], 1000)

    def test_sin_efecto_de_pareja_el_historial_no_pesa(self):
        est = OC.kappa(self.pairs(None), "k")          # solo azar binomial
        self.assertTrue(est["kappa"] is None or est["kappa"] > 1000)
        post = OC.shrink_pair(30, 80, 0.22, est["kappa"])
        self.assertLess(abs(post["post"] - 0.22), 0.02)

    def test_posterior_beta_binomial(self):
        s = OC.shrink_pair(y=30, n=100, mu=0.20, k=100)
        self.assertAlmostEqual(s["w"], 0.5)
        self.assertAlmostEqual(s["post"], 0.25)          # (30 + 100·0.20) / 200
        self.assertAlmostEqual(s["mult"], 1.25)

    def test_razon_de_momios(self):
        self.assertAlmostEqual(OC.odds_ratio(0.22, 0.22, 0.22), 0.22)
        self.assertGreater(OC.odds_ratio(0.30, 0.25, 0.22), 0.30)   # pitcher de ponches contra equipo que se poncha


class DesarrolloDelJuego(unittest.TestCase):
    def test_gancho_de_postemporada(self):
        post, reg = {}, {}
        rng = random.Random(5)
        for i in range(60):
            pid = 1000 + i
            reg[f"2025:{pid}"] = {"gamesStarted": 30, "inningsPitched": "165.0", "numberOfPitches": 2700}   # 16.5 outs
            outs = max(3, round(16.5 * 0.85 + rng.gauss(0, 2)))
            post[str(i)] = {"home": {"team": 1, "pitchers": [{"id": pid, "ip": f"{outs // 3}.{outs % 3}", "pitches": 80}]}}
        h = OC.hook_factor({"2025": post}, reg)
        self.assertTrue(h["measured"])
        self.assertAlmostEqual(h["hook"], 0.85, delta=0.04)
        self.assertLess(h["ci"][0], h["hook"])
        self.assertAlmostEqual(h["pitchRatio"], 80 / 90, places=3)

    def test_gancho_como_regresion(self):
        """Si en octubre todos sacan ~14 outs sin importar su profundidad, β ≈ 0 y el modelo usa la media."""
        post, reg = {}, {}
        rng = random.Random(8)
        for i in range(80):
            pid = 2000 + i
            avg = rng.uniform(14, 19)                     # outs por apertura en temporada
            reg[f"2024:{pid}"] = {"gamesStarted": 30, "inningsPitched": f"{int(avg * 30 // 3)}.{int(avg * 30) % 3}"}
            outs = max(3, round(14 + rng.gauss(0, 2)))
            post[str(i)] = {"away": {"team": 1, "pitchers": [{"id": pid, "ip": f"{outs // 3}.{outs % 3}"}]}}
        h = OC.hook_factor({"2024": post}, reg)
        self.assertLess(abs(h["beta"]), 3 * h["seBeta"])
        self.assertAlmostEqual(h["meanPost"], 14, delta=0.6)
        self.assertLessEqual(h["betaUsed"], 0.5)
        self.assertAlmostEqual(OC.post_outs(h, 19.0) - OC.post_outs(h, 15.0), 4 * h["betaUsed"])
        self.assertGreater(h["shape"], 1.5)

    def test_sin_datos_suficientes_el_gancho_es_supuesto(self):
        self.assertFalse(OC.hook_factor({}, {})["measured"])

    def test_salida_del_abridor_y_fracciones(self):
        shape, scale = OC.weibull_fit([15, 18, 16, 17, 14, 19, 18, 16])
        d = OC.outs_dist(shape, scale)
        self.assertAlmostEqual(sum(d), 1.0)
        mean = sum(k * p for k, p in enumerate(d))
        self.assertAlmostEqual(mean, 16.6, delta=0.8)
        f = OC.inning_fracs(d)
        self.assertGreater(f[1], 0.97)                    # la 1.ª entrada casi siempre es del abridor
        self.assertLess(f[8], 0.1)
        self.assertAlmostEqual(sum(f[1:]) * 3, mean, delta=0.3)
        corto = OC.inning_fracs(OC.outs_dist(shape, scale, hook=0.85))
        self.assertLess(sum(corto[1:]), sum(f[1:]))

    def test_vueltas_al_lineup(self):
        frac = [0.0] + [1.0] * 6 + [0.0] * 3               # 6 entradas completas, ~4.3 bateadores por entrada
        tto = OC.tto_offsets(4.3, frac)
        self.assertLess(tto[1], 0)                         # 1.ª vuelta: mejor que su promedio
        self.assertGreater(tto[6], tto[3])                 # la 3.ª vuelta castiga más
        self.assertLess(OC.season_tto_mean(18), OC.season_tto_mean(27))   # gancho corto = menos castigo

    def test_bullpen_de_octubre(self):
        rel = [{"name": "Cerrador", "role": "Cerrador", "fipShrunk": 3.0, "ip": 60, "pUse": 0.45},
               {"name": "Setup", "role": "Setup", "fipShrunk": 3.2, "ip": 65, "pUse": 0.45},
               {"name": "Relleno", "role": "Largo", "fipShrunk": 5.0, "ip": 90, "pUse": 0.2}]
        pen = OC.october_pen(rel, 4.2)
        self.assertLess(pen["ra9"], pen["ra9Reg"])         # en octubre lanzan más los de confianza
        self.assertAlmostEqual(sum(x["shareOct"] for x in pen["rows"]), 1.0)

    def test_ponches_poisson_binomial(self):
        d = OC.k_dist([0.25] * 20, [1.0] * 20)
        self.assertAlmostEqual(sum(k * p for k, p in enumerate(d)), 5.0)
        d2 = OC.k_dist([0.25] * 20, [1.0] * 10 + [0.0] * 10)
        self.assertAlmostEqual(sum(k * p for k, p in enumerate(d2)), 2.5)


class MatchupYFamiliaridad(unittest.TestCase):
    def test_matchup_por_lanzamiento(self):
        L = {"FF": 0.340, "SL": 0.270}
        mix = OC.pitcher_mix([{"type": "FF", "usage": 60, "pa": 400, "xwoba": 0.300},
                              {"type": "SL", "usage": 40, "pa": 200, "xwoba": 0.250}], L, 200)
        self.assertAlmostEqual(sum(m["usage"] for m in mix), 1.0)
        self.assertLess(mix[0]["xwoba"], 0.340)            # encogido pero mejor que la liga
        bundle = {"savant": {"expected": {"7": {"pa": 600, "xwoba": 0.320}}},
                  "arsenal": {"batter": {"7": [{"type": "SL", "pa": 150, "xwoba": 0.200}]}}}
        m = OC.matchup(7, mix, bundle, L, 0.315, 200, 0.0)
        self.assertLess(m["pitchEdge"], 0)                 # le cuesta el slider: el matchup lo baja
        self.assertAlmostEqual(m["generic"] + m["pitchEdge"], m["xwoba"])

    def test_nivel_del_bateador_con_su_arsenal(self):
        bundle = {"arsenal": {"batter": {"9": [{"type": "FF", "pa": 300, "xwoba": 0.380}, {"type": "SL", "pa": 100, "xwoba": 0.300}]}}}
        lvl, pa, _ = OC.batter_level(9, bundle, 0.315)
        self.assertEqual(pa, 400)
        self.assertAlmostEqual(lvl, (0.360 - 0.315) * 400 / 700)
        self.assertEqual(OC.batter_level(10, bundle, 0.315)[0], -0.015)

    def test_familiaridad_detecta_un_efecto_real(self):
        rng = random.Random(11)
        rows, mu = [], {}
        for pid in range(120):
            for opp in range(4):
                key = (pid, opp)
                mu[key] = {"k": {"mu": 0.22}, "bb": {"mu": 0.08}, "hr": {"mu": 0.03}, "runs": {"mu": 0.12}}
                days = sorted(rng.sample(range(1, 180), 3))  # fechas al azar: la familiaridad no es la fecha
                for j, d in enumerate(days):
                    lam = 25 * 0.12 * (1 + 0.15 * j)       # 15% más carreras por cada vez que ya lo vieron
                    runs = sum(1 for _ in range(25) if rng.random() < lam / 25)
                    m, dd = divmod(d, 30)
                    rows.append({"pid": pid, "opp": opp, "date": f"2026-{m + 4:02d}-{dd + 1:02d}", "bf": 25, "k": 5, "bb": 2,
                                 "hr": 1, "runs": runs})
        rows.sort(key=lambda r: (r["pid"], r["date"]))
        fam = OC.familiarity(rows, mu)
        self.assertGreater(fam["runs"]["slope"], 0.08)
        self.assertGreater(fam["runs"]["t"], 2)
        self.assertGreater(fam["runs"]["shrunk"], 0)
        self.assertLess(fam["runs"]["shrunk"], fam["runs"]["slope"])      # entra encogido
        # si las repeticiones caen siempre en la misma fecha que su número, no se puede separar: no se inventa un efecto
        seen, same = {}, []
        for r in rows:
            k = (r["pid"], r["opp"])
            seen[k] = seen.get(k, 0) + 1
            same.append(dict(r, date=f"2026-0{3 + seen[k]}-01"))
        same.sort(key=lambda r: (r["pid"], r["date"]))
        self.assertEqual(OC.familiarity(same, mu)["runs"]["shrunk"], 0.0)


class OctubreV2(unittest.TestCase):
    """v2 (post-mortem PHI @ ATL juego 1): forma del día, κ sin ella, encogimiento medido, gancho limpio, ponches con forma."""

    def starts(self, rho_day, pair_sd=0.0, n_pit=150, seed=21):
        """Aperturas sintéticas: cada pitcher tiene su tasa, cada día una forma (varianza rho_day·p(1−p)) y, si
        pair_sd > 0, un efecto propio contra cada rival."""
        rng = random.Random(seed)
        rows = []
        for pid in range(n_pit):
            base = rng.uniform(0.18, 0.30)
            eff = {opp: rng.gauss(0, pair_sd) for opp in range(6)}
            for j in range(24):
                opp = j % 6
                p = min(0.6, max(0.05, base + eff[opp] + rng.gauss(0, math.sqrt(rho_day * base * (1 - base)))))
                bf = rng.randint(18, 27)
                k = binom(rng, bf, p)
                rows.append({"pid": pid, "opp": opp, "date": f"2026-{4 + j // 6:02d}-{1 + j % 28:02d}",
                             **OC.counts({"bf": bf, "k": k, "bb": 2, "hr": 1, "h": 5}), "runs": 2, "outs": 18})
        return rows

    def test_forma_del_dia_se_recupera(self):
        est = OC.game_rho(self.starts(0.01), "k")
        self.assertAlmostEqual(est["rho"], 0.01, delta=0.004)
        self.assertLess(OC.game_rho(self.starts(0.0), "k")["rho"], 0.002)

    def test_la_forma_del_dia_no_es_efecto_del_rival(self):
        """Sin efecto de pareja, la forma del día hacía creer que sí lo había (κ finito); quitándola, κ = ∞."""
        rows = self.starts(0.012)
        lg = {"k": 0.24, "bb": 0.08, "hr": 0.04, "babip": 0.3, "rpa": 0.1}
        teams = {o: dict(lg) for o in range(6)}
        pairs = OC.pair_table(rows, teams, lg, None)
        naive = OC.kappa(pairs, "k")
        fixed = OC.kappa(pairs, "k", rho_game=OC.game_rho(rows, "k")["rho"])
        self.assertIsNotNone(naive["kappa"])
        self.assertTrue(fixed["kappa"] is None or fixed["kappa"] > 5 * naive["kappa"])
        # con un efecto real de pareja, se sigue detectando
        real = self.starts(0.012, pair_sd=0.05, seed=5)
        pr = OC.pair_table(real, teams, lg, None)
        k_real = OC.kappa(pr, "k", rho_game=OC.game_rho(real, "k")["rho"])["kappa"]
        self.assertIsNotNone(k_real)
        self.assertLess(k_real, 300)

    def test_encogimiento_del_pitcher_medido(self):
        rows = self.starts(0.005)
        lg = {"k": 0.24, "bb": 0.08, "hr": 0.04, "babip": 0.3, "rpa": 0.1}
        kp = OC.pitcher_kappa(rows, lg, {"k": OC.game_rho(rows, "k")})
        # tasas uniformes en [0.18, 0.30]: varianza 0.001 → κ ≈ p(1−p)/σ² − 1 ≈ 180
        self.assertGreater(kp["k"]["kappa"], 100)
        self.assertLess(kp["k"]["kappa"], 320)
        self.assertEqual(OC.pitcher_k0(None, "k"), 30.0)
        self.assertEqual(OC.pitcher_k0({"k": {"kappa": None}}, "k"), 1e9)     # sin señal: la liga

    def test_gancho_sin_relevistas_de_abridor(self):
        post, reg = {}, {}
        rng = random.Random(3)
        for i in range(60):
            pid = 3000 + i
            avg = rng.uniform(15, 19)
            reg[f"2025:{pid}"] = {"gamesStarted": 30, "gamesPlayed": 30, "battersFaced": int(30 * avg * 1.4),
                                  "inningsPitched": f"{int(avg * 30 // 3)}.{int(avg * 30) % 3}"}
            outs = max(3, round(4 + 0.6 * avg + rng.gauss(0, 2)))
            post[str(i)] = {"home": {"team": 1, "pitchers": [{"id": pid, "ip": f"{outs // 3}.{outs % 3}"}]}}
        # un relevista usado de abridor: 94 IP en 9 aperturas (31 outs por apertura), 0 outs en octubre
        reg["2025:9"] = {"gamesStarted": 9, "gamesPlayed": 40, "battersFaced": 400, "inningsPitched": "94.1"}
        post["x"] = {"away": {"team": 2, "pitchers": [{"id": 9, "ip": "0.0"}]}}
        h = OC.hook_factor({"2025": post}, reg)
        self.assertEqual(h["n"], 60)                                  # el relevista no entra
        self.assertAlmostEqual(h["betaTS"], 0.6, delta=0.25)
        self.assertAlmostEqual(h["betaUsed"], min(1.0, max(0.0, h["betaTS"])))

    def test_ponches_con_forma_del_dia(self):
        rates = [0.28] * 36
        shape, mean = 3.0, 16.0
        base = OC.k_dist_form(rates, shape, mean, 1.4, 0.0, 0.0)
        form = OC.k_dist_form(rates, shape, mean, 1.4, 0.008, 5.4)
        m0 = sum(k * p for k, p in enumerate(base))
        m1 = sum(k * p for k, p in enumerate(form))
        v0 = sum((k - m0) ** 2 * p for k, p in enumerate(base))
        v1 = sum((k - m1) ** 2 * p for k, p in enumerate(form))
        self.assertAlmostEqual(sum(form), 1.0)
        self.assertAlmostEqual(m1, m0, delta=0.15)                    # la media casi no cambia
        self.assertGreater(v1, v0)                                    # la cola sí: más varianza (Beta-Binomial)
        self.assertGreater(sum(form[9:]), sum(base[9:]))
        # bateadores esperados ≈ outs × bateadores por out (sin el medio bateador de más del redondeo)
        d = OC.outs_dist(shape, mean / math.gamma(1 + 1 / shape))
        ebf = sum(sum(p for k, p in enumerate(d) if k > (j + 0.5) / 1.4) for j in range(40))
        self.assertAlmostEqual(ebf, 1.4 * sum(k * p for k, p in enumerate(d)), delta=0.35)


if __name__ == "__main__":
    unittest.main()
