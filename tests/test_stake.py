"""Stake 1–10 por confianza: montos, reglas del semáforo, escalera de momios y factores de ajuste."""
import unittest

from mlbgs import stake as ST


def pick(p=0.60, ic=65.0, blocked=False, guion="Sí", contradiction="Baja", methods=None, **parts):
    base = {"fuerza": 0.5, "consenso": 0.8, "datos": 1.0, "estabilidad": 0.7, "contradicciones": 1.0}
    base.update(parts)
    return {"pick": "X", "p": p, "ic": ic, "parts": base, "blocked": blocked, "blockedWithOdds": blocked,
            "datosWithOdds": base["datos"], "guion": guion, "contradiction": contradiction,
            "methods": methods or {"Log5 (5.7.2)": p}}


def dec(am):
    return 1 + am / 100 if am > 0 else 1 + 100 / -am


class Stake(unittest.TestCase):
    def test_escala_de_montos(self):
        self.assertEqual(ST.AMOUNTS[1], 500)
        self.assertEqual(ST.AMOUNTS[5], 1000)
        self.assertEqual(ST.AMOUNTS[10], 1500)
        self.assertEqual(list(ST.AMOUNTS.values()), sorted(ST.AMOUNTS.values()))

    def test_nivel_por_confianza(self):
        self.assertEqual(ST.level_from_ic(54.9), 0)
        self.assertEqual(ST.level_from_ic(55), 1)
        self.assertEqual(ST.level_from_ic(70), 6)      # 1 + 9·15/30 = 5.5 → 6
        self.assertEqual(ST.level_from_ic(85), 10)
        self.assertEqual(ST.level_from_ic(99), 10)
        self.assertLess(ST.level_from_ic(80, a=0.7), ST.level_from_ic(80))

    def test_sube_con_mejor_momio(self):
        pk = pick()
        levels = [ST.stake_at(pk, dec(am))["level"] for am in (-135, -120, -105, 110)]
        self.assertEqual(levels, sorted(levels))
        self.assertGreater(levels[-1], levels[0])

    def test_reglas_del_semaforo(self):
        self.assertEqual(ST.stake_at(pick(), dec(-160))["level"], 0)              # edge < 3 pp
        self.assertEqual(ST.stake_at(pick(blocked=True), dec(120))["level"], 0)   # falta un dato obligatorio
        self.assertEqual(ST.stake_at(pick(guion="No"), dec(120))["level"], 0)
        self.assertEqual(ST.stake_at(pick(contradiction="Alta"), dec(120))["level"], 0)
        self.assertEqual(ST.stake_at(pick(ic=30, fuerza=0.9), dec(-120))["level"], 0)  # IC < 55 con ese momio
        self.assertGreater(ST.stake_at(pick(), dec(120))["level"], 0)

    def test_acuerdo_con_el_modelo_del_pro_lab(self):
        name = ST.LAB_METHODS["kronos"]
        igual = pick(methods={"Log5 (5.7.2)": 0.60, name: 0.60})
        peor = pick(methods={"Log5 (5.7.2)": 0.60, name: 0.48})
        mejor = pick(methods={"Log5 (5.7.2)": 0.60, name: 0.66})
        self.assertEqual(ST.agreement(igual, "kronos"), 1.0)
        self.assertAlmostEqual(ST.agreement(peor, "kronos"), 0.70)
        self.assertEqual(ST.agreement(mejor, "kronos"), 1.0)
        self.assertEqual(ST.agreement(igual, None), 1.0)

    def test_historial_encogido_y_acotado(self):
        tickets = [{"model": "framework", "picks": [{"principal": True, "p": 0.65, "res": "perdido"}] * 10}]
        h = ST.track(tickets)["framework"]["h"]
        self.assertLess(h, 1.0)
        self.assertGreaterEqual(h, 0.85)
        tickets = [{"model": "kronos", "picks": [{"principal": True, "p": 0.5, "res": "ganado"}] * 200}]
        self.assertEqual(ST.track(tickets)["kronos"]["h"], 1.05)

    def test_escalera_coherente(self):
        pk = pick()
        plan = ST.plan(pk)
        self.assertTrue(plan["ladder"])
        self.assertEqual(len(plan["steps"]), 10)
        prev = None
        for step in plan["ladder"]:
            self.assertGreaterEqual(ST.stake_at(pk, dec(step["from"]))["level"], step["level"])
            worse = step["from"] - 1 if step["from"] != 101 else -101
            self.assertLess(ST.stake_at(pk, dec(worse))["level"], step["level"])
            if prev:
                self.assertGreater(dec(step["from"]), dec(prev["from"]))
            prev = step

    def test_stake_mostrado_sale_del_ic_del_pick(self):
        self.assertEqual(ST.plan(pick(ic=66))["level"], ST.level_from_ic(66))
        self.assertEqual(ST.plan(pick(ic=50))["level"], 0)

    def test_sin_escalera_si_falta_un_dato(self):
        plan = ST.plan(pick(blocked=True))
        self.assertEqual(plan["ladder"], [])
        self.assertEqual(plan["level"], 0)
        self.assertIn("dato obligatorio", plan["block"])


if __name__ == "__main__":
    unittest.main()
