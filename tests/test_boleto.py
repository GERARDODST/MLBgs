"""Boletos tipo casino: momio decimal, stake por la escalera, pago, cobro, neto y push."""
import unittest

from mlbgs import boleto as BO
from mlbgs import stake as ST

STEPS = [-173, -173, -173, -173, -166, -158, -151, -144, -137, -131]


def ticket(res="ganado", level=9, status="calificado", model="eigen", pk=1, pick="ATL", decision="apostar"):
    return {"pk": pk, "date": "2026-09-24", "time": "2026-09-24T23:15:00Z", "model": model, "away": "CIN", "home": "ATL",
            "status": status, "decision": {"status": decision, "pick": pick, "market": "Moneyline", "level": level, "minPrice": -173},
            "picks": [{"pick": pick, "market": "Moneyline", "p": 0.665, "res": res, "stake": {"steps": STEPS}}]}


class Boleto(unittest.TestCase):
    def test_momio_decimal(self):
        self.assertEqual(BO.parse_odds("-137"), {"am": -137, "dec": 1.73})
        self.assertEqual(BO.parse_odds("+120")["dec"], 2.2)
        self.assertEqual(BO.parse_odds("1.91")["dec"], 1.91)
        self.assertEqual(BO.parse_odds("−150")["am"], -150)
        self.assertIsNone(BO.parse_odds("abc"))
        self.assertIsNone(BO.parse_odds("50"))

    def test_referencia_es_el_momio_minimo_del_stake(self):
        s = BO.slip(ticket())
        self.assertEqual(s["source"], "referencia")
        self.assertEqual(s["am"], -137)                      # stake 9 vale desde −137
        self.assertEqual(s["amount"], ST.AMOUNTS[9])
        self.assertAlmostEqual(s["payout"], 1400 * 1.73)
        self.assertAlmostEqual(s["neto"], 1400 * 0.73)

    def test_perdido_push_y_anulado(self):
        self.assertEqual(BO.slip(ticket("perdido"))["neto"], -1400)
        self.assertEqual(BO.slip(ticket("perdido"))["cobro"], 0)
        push = BO.slip(ticket("push"))
        self.assertEqual((push["cobro"], push["neto"]), (1400, 0))
        anulado = BO.slip(ticket("anulado", status="anulado"))
        self.assertEqual((anulado["res"], anulado["neto"]), ("anulado", 0))
        self.assertEqual(BO.slip(ticket(None, status="cerrado"))["res"], "pendiente")

    def test_tu_momio_mueve_el_stake_por_la_escalera(self):
        mejor = BO.slip(ticket(), {"odds": "-120"})
        self.assertEqual((mejor["source"], mejor["level"], mejor["amount"]), ("tuyo", 10, 1500))
        peor = BO.slip(ticket(), {"odds": "-150"})
        self.assertEqual(peor["level"], 7)
        propio = BO.slip(ticket(), {"odds": "1.80", "amount": 900})
        self.assertEqual((propio["dec"], propio["amount"]), (1.8, 900))
        self.assertAlmostEqual(propio["neto"], 900 * 0.8)

    def test_un_boleto_por_partido_y_cartera(self):
        ts = [ticket(model="framework", level=8), ticket(model="eigen"), ticket("perdido", pk=2, model="framework"),
              ticket(pk=3, model="framework", decision="no apostar", level=0)]
        bs = BO.slips(ts, {"2": {"odds": "-150", "played": False}})
        self.assertEqual([(b["pk"], b["model"]) for b in bs], [(1, "eigen"), (2, "framework")])   # manda el Pro-Lab
        L = BO.ledger(bs)
        self.assertEqual((L["n"], L["win"], L["lose"]), (1, 1, 0))                               # el no jugado no cuenta
        self.assertAlmostEqual(L["net"], 1400 * 0.73)
        self.assertAlmostEqual(L["roi"], 0.73)


if __name__ == "__main__":
    unittest.main()
