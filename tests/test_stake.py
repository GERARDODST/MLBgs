"""STAKE-K: montos entre $500 y $1,500, reglas del semáforo, escalera de momios y factores de confianza."""
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
    def test_monto_dentro_del_rango_y_redondeado(self):
        pk = pick()
        w = ST.weight(pk)
        for am in (-150, -130, -110, 100, 120, 150, 200):
            s = ST.stake_at(pk, w, dec(am))["stake"]
            self.assertTrue(s == 0 or ST.MIN <= s <= ST.MAX, (am, s))
            self.assertEqual(s % ST.STEP, 0)

    def test_sube_con_la_ventaja(self):
        pk = pick()
        w = ST.weight(pk)
        stakes = [ST.stake_at(pk, w, dec(am))["stake"] for am in (-135, -125, -115, -105)]
        self.assertEqual(stakes, sorted(stakes))
        self.assertGreater(stakes[-1], stakes[0])

    def test_reglas_del_semaforo(self):
        w = 0.8
        self.assertEqual(ST.stake_at(pick(), w, dec(-160))["stake"], 0)              # edge < 3 pp
        self.assertEqual(ST.stake_at(pick(blocked=True), w, dec(120))["stake"], 0)   # falta un dato obligatorio
        self.assertEqual(ST.stake_at(pick(guion="No"), w, dec(120))["stake"], 0)
        self.assertEqual(ST.stake_at(pick(contradiction="Alta"), w, dec(120))["stake"], 0)
        self.assertEqual(ST.stake_at(pick(ic=30, fuerza=0.9), w, dec(-120))["stake"], 0)  # IC < 55 con ese momio
        self.assertGreater(ST.stake_at(pick(), w, dec(120))["stake"], 0)

    def test_ventaja_minima_no_se_infla_al_minimo(self):
        # ¼ Kelly por debajo de $300: no se apuesta en lugar de subirlo a $500
        pk = pick(p=0.55, ic=70)
        s = ST.stake_at(pk, 0.3, dec(-108))
        self.assertEqual(s["stake"], 0)
        self.assertLess(s["raw"], ST.FLOOR)

    def test_confianza_baja_reduce_el_stake(self):
        alta = pick(consenso=1.0, estabilidad=1.0)
        baja = pick(consenso=0.2, estabilidad=0.2, datos=0.8)
        d = dec(-110)
        self.assertGreater(ST.weight(alta), ST.weight(baja))
        self.assertGreaterEqual(ST.stake_at(alta, ST.weight(alta), d)["stake"], ST.stake_at(baja, ST.weight(baja), d)["stake"])

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
        w = plan["w"]
        prev = None
        for step in plan["ladder"]:
            self.assertGreaterEqual(ST.stake_at(pk, w, dec(step["from"]))["stake"], step["stake"])
            worse = step["from"] - 1 if step["from"] != 101 else -101
            self.assertLess(ST.stake_at(pk, w, dec(worse))["stake"], step["stake"])
            if prev:
                self.assertGreater(dec(step["from"]), dec(prev["from"]))
            prev = step

    def test_sin_escalera_si_falta_un_dato(self):
        plan = ST.plan(pick(blocked=True))
        self.assertEqual(plan["ladder"], [])
        self.assertIn("dato obligatorio", plan["block"])


if __name__ == "__main__":
    unittest.main()
