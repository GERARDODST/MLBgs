"""Versión del algoritmo que produce cada ticket (los datos van aparte, en el expediente del partido).

Regla: una corrección a los modelos solo aplica a los partidos que todavía no empiezan. Cada ticket guarda
la versión con la que se hizo y queda bloqueado al primer lanzamiento; nunca se recalcula con una versión
posterior. Al cambiar las reglas de los modelos, agregar una entrada a CAMBIOS (arriba la más reciente).
"""
from __future__ import annotations

import glob
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMBIOS = [
    {"version": "2026.09.30", "fecha": "2026-09-30", "cambios": [
        "Pro-Lab: el modelo congela su probabilidad, no el momio. Cada corrida sus picks toman el momio actual del mismo "
        "pick (misma línea) y la decisión del partido y la de su ticket siguen al momio hasta el primer lanzamiento; antes "
        "un momio que llegaba después de congelar (NRFI, ponches) nunca entraba y la decisión se quedaba en «esperar momio».",
    ]},
    {"version": "2026.09.29.5", "fecha": "2026-09-29", "cambios": [
        "OCTUBRE v2 (post-mortem de PHI @ ATL juego 1, validado fuera de muestra con los juegos de abridores de 2026): la "
        "forma del día ya no se confunde con un efecto del rival (el historial contra el rival empeoraba los ponches y ya "
        "no pesa); las tasas del abridor y del equipo se encogen con κ medidos (antes 30 fijo); el gancho de octubre usa "
        "solo abridores de verdad con pendiente robusta; los ponches del abridor mezclan la forma del día y cuentan bien "
        "los bateadores que enfrenta.",
    ]},
    {"version": "2026.09.29.4", "fecha": "2026-09-29", "cambios": [
        "«Esperar momio»: el mercado de la decisión que ESPN no trae (F5, ponches, team total, NRFI) se pide a The Odds "
        "API desde 12 h antes del partido (antes 6 h), se refresca a las 6 h y a la hora y media, y si ninguna casa lo "
        "había publicado se vuelve a pedir cada 2 h. La decisión dice de dónde y a qué hora llega el momio.",
        "Si la decisión espera momio, en la misma llamada se piden los mercados de los candidatos sin momio que ESPN no trae: "
        "si el precio no alcanza, la siguiente opción ya tiene el suyo (antes tomaba una corrida por mercado).",
        "Corrección: pedir el F5 ganador en otra llamada borraba el F5 total guardado (pasó con CHC @ SD el 29-sep); "
        "ya no, y un mercado que llegó pero ya no está guardado se vuelve a pedir.",
    ]},
    {"version": "2026.09.29.3", "fecha": "2026-09-29", "cambios": [
        "Si el pick de más confianza cae en el filtro contra el mercado (edge ≥ 10 pp a su momio real), la decisión es "
        "«esperar · verificar» ese pick (lesiones, descansos, lineup) en lugar de saltar a otro mercado sin momio.",
        "El modelo del Pro-Lab entra completo al expediente de Claude; una conclusión de Claude hecha sin el modelo "
        "del Pro-Lab actual deja de mandar hasta volver a analizar.",
    ]},
    {"version": "2026.09.29.2", "fecha": "2026-09-29", "cambios": [
        "Pro-Lab OCTUBRE (postemporada): historial del abridor contra el rival con Bayes empírico (κ medido en la liga), "
        "matchup por tipo de lanzamiento, salida del abridor con el gancho medido en postemporadas 2024-2025, bullpen de "
        "octubre y familiaridad como variable de prueba. Primer partido: PHI @ ATL, juego 2 del comodín.",
    ]},
    {"version": "2026.09.29", "fecha": "2026-09-29", "cambios": [
        "Revisión periódica: cada corrida anota qué cambió (abridor, lineup, umpire, clima, horario, bajas, momio y "
        "decisión) y revisa en ese momento los momios del partido que cambió.",
        "El momio visto antes de un cambio de abridor ya no cuenta para el stake (hasta que la casa publique otro).",
        "The Odds API por prioridad (cambio de abridor, mercado de la decisión, verificación contra otras casas, el otro "
        "pick) con créditos reservados para cambios; DraftKings se verifica contra la mediana de las demás casas.",
    ]},
    {"version": "2026.09.28.2", "fecha": "2026-09-28", "cambios": [
        "Alertas de la auditoría vinculantes: si el pick depende de que un abridor en mala racha siga mal (8.3) el "
        "stake × 0.8; favorito caro, −170 o peor (7.5), × 0.8, y con total proyectado menor a 8 no hay stake.",
        "Equipo sin nada en juego (eliminado, o clasificado sin siembra en juego, en temporada regular): su bullpen "
        "se regresa 50% a la media de la liga en el modelo de carreras y el uso esperado de su cerrador y setups baja 40%.",
        "F5 y F3 con Binomial Negativa y la sobredispersión medida en esas entradas (antes Poisson, que subestimaba "
        "las entradas grandes y daba probabilidades F5 demasiado altas).",
        "El lineup muestra el OBP/SLG regresado que usa el modelo junto al de temporada (con pocos turnos, el crudo engaña).",
    ]},
    {"version": "2026.09.28", "fecha": "2026-09-28", "cambios": [
        "Sin momio real (feed o el tuyo) el índice de confianza ya no suma «fuerza» contra un momio de "
        "referencia de −110 y la decisión es «esperar momio»: nunca hay stake sin un precio real.",
        "Tickets bloqueados al primer lanzamiento, con expediente del partido (análisis completo, datos "
        "crudos y momios vistos) y la versión del algoritmo que los hizo.",
    ]},
    {"version": "2026.09.27", "fecha": "2026-09-27", "cambios": [
        "Momios automáticos (ESPN/DraftKings y The Odds API); calentamiento no cuenta como juego empezado.",
    ]},
]
VERSION = CAMBIOS[0]["version"]
# lo que cuenta como «algoritmo»: el código del modelo y la página (que repite las cuentas de stake)
CODE_GLOBS = ("mlbgs/*.py", "site/template.html")


def code_hash(root: str = ROOT) -> str:
    """Huella del código del algoritmo (no cambia con los datos ni con las corridas, solo con el código)."""
    h = hashlib.sha256()
    for pat in CODE_GLOBS:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            h.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:12]


def info(root: str = ROOT) -> dict:
    return {"version": VERSION, "code": code_hash(root)}
