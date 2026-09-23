# MLBgs

Análisis matemático y estadístico de los **próximos partidos de MLB** con datos oficiales de la liga,
siguiendo el **Framework MLB Picks v2** (las 10 secciones: contexto, historial, abridores, bullpen,
modelo matemático, Over/Under, cuotas, auditoría, fuentes y algoritmo maestro). Es el equivalente
para béisbol de [DATAGSPORTS](https://github.com/GERARDODST/DATAGSPORTS) (NFL), pero usando el
framework original de MLB.

## Qué hace

Varias veces al día un workflow de GitHub Actions:

1. **Descarga los datos oficiales** (FASE 0): calendario de hoy y mañana con abridores probables,
   lineups, umpires y clima; resultados con linescore de esta temporada y la anterior; standings;
   stats de abridores (temporada, splits casa/visita, game logs); bullpen y ofensiva por equipo;
   box scores de los últimos 7 días (uso de relevistas); xERA, Barrel% y park factors de Statcast.
2. **Corre el modelo** partido por partido (`mlbgs/model.py`).
3. **Guarda la predicción** de cada juego en `data/predictions/AAAA-MM-DD.json` y, cuando el juego
   termina, la compara contra el resultado oficial (pestaña *Seguimiento*: acierto, Brier, log-loss,
   error del total).
4. **Publica la página** `site/index.html` (estática, con los datos embebidos) en GitHub Pages.

| Fuente | Qué se obtiene |
| --- | --- |
| [MLB Stats API](https://statsapi.mlb.com) | Calendario, abridores probables, lineups, umpire, clima, resultados por entrada, standings, stats de jugadores y equipos, box scores |
| [Baseball Savant](https://baseballsavant.mlb.com) | xERA, xwOBA, Barrel%, Hard Hit%, park factors (índice 3 años) |
| [MLB Stats API](https://statsapi.mlb.com) — rosters y transacciones | Roster activo con splits, lista de lesionados, movimientos oficiales (noticias) |
| [Baseball Savant](https://baseballsavant.mlb.com) — arsenal | Run value por tipo de pitcheo de abridores y bateadores |
| [The Odds API](https://the-odds-api.com) (opcional) | Momios de varias casas: moneyline, run line y total |

## El modelo, sección por sección

- **Regresión a la media (5.7.4):** K% 70 BF, BB% 170 BF, HR ≈1150 BF, ERA 1000 BF; la temporada
  anterior pesa la mitad. ERA estimada del abridor = 0.45·FIP + 0.35·xERA + 0.20·ERA (todas regresadas).
  FIP con la constante de liga calculada con los totales de la temporada.
- **Carreras esperadas (5.3, 6.7):** λ por media entrada = carreras promedio de la liga en esa entrada
  × ofensiva (neutral de parque) × split vs la mano del abridor × lineup confirmado × park factor ×
  (fracción de la entrada que cubre el abridor × abridor + resto × bullpen con fatiga).
- **Ajustes de la tabla 5.3:** se evalúan todos; solo se aplican los que traen información nueva
  (viento, HR/9 alto + parque + contacto). Los que ya están en λ base (K-BB%, K% rival, bullpens,
  parque) se muestran pero no se duplican, y nunca se sube el total por un solo factor.
- **Distribuciones (5.4, 5.7.5, 6.8):** Binomial Negativa para el juego completo con la
  sobredispersión medida (Var/Media ≈ 2.3 en 2026), Poisson para F3/F5, NRFI calibrado con la
  frecuencia real de primeras entradas en blanco.
- **Triangulación (FASE 2):** Log5 con Pitágoras regresado + ventaja de local medida, Elo (K = 4,
  +24 local, ajuste por abridor del día) y el modelo λ. Divergencia ≤5 pp = confianza alta, >10 pp = baja.
- **Valor (sección 7):** probabilidad implícita, edge, momio justo, momio mínimo aceptable (edge ≥ 3 pp)
  y Kelly fraccional (¼ y ⅛).
- **Auditoría (sección 8):** los 17 filtros, dependencia de supuestos, correlación entre picks y la
  tabla Modelo + Guion + Cuota + Contradicción → semáforo.
- **Gate de completitud (9.3):** si falta un campo obligatorio (abridor, lineup, umpire, momios…), el
  mercado afectado nunca puede ser Verde.

## Picks e índice de confianza

Cada partido publica sus picks (moneyline, run line, total, F5, team totals, NRFI/YRFI y ponches de
los abridores) ordenados por un **Índice de Confianza (IC, 0-100)** que se explica pick por pick:

```
IC = 100 · (0.35·Fuerza + 0.25·Consenso + 0.15·Datos + 0.15·Estabilidad + 0.10·Sin contradicciones)
```

- **Fuerza:** probabilidad del modelo sobre el punto de equilibrio del momio (real o de referencia).
- **Consenso:** acuerdo entre métodos independientes del framework (Log5, Elo, λ-Binomial Negativa,
  Poisson y, en el Pro-Lab, el Monte Carlo de DIAMANTE-24).
- **Datos:** gate de completitud (9.3) y lineups confirmados. **Estabilidad:** tamaño de muestra tras
  la regresión a la media. **Contradicciones:** filtros de 5.6, 6.11 y 7.7.

Niveles: ≥70 Alta · 55-69 Media · 40-54 Baja · <40 Muy baja. Los dos picks de mayor IC aparecen en la
tarjeta del partido y arriba de su página; al final están todos. La pestaña *Seguimiento* mide el
acierto real por nivel de confianza.

**Momios de casas mexicanas (Draftea, Playdoit, Caliente, Codere, Strendus…):** no publican una API
pública, así que en cada pick hay un campo para escribir el momio de tu casa (americano −150/+130 o
decimal 1.91). La página recalcula al instante la probabilidad implícita, el edge, el IC, el semáforo
y el stake de ¼ de Kelly. Se guarda solo en tu navegador.

## Pro-Lab: modelo DIAMANTE-24

Laboratorio para probar un modelo nuevo contra un partido real con los datos **congelados antes del
primer lanzamiento** (sin fuga de información). Primera prueba: Nationals @ Tigers, 23-sep-2026
(datos congelados 18 minutos antes, con lineups y bullpen oficiales).

1. **Probabilidad de cada turno al bate** (K, BB, 1B, 2B, 3B, HR, OUT) bateador contra pitcher con
   log5 multinomial sobre tasas regresadas, splits por mano, parque, viento y veces en el orden.
2. **Cadena de Markov de las 24 situaciones base-out:** matriz de transición Q, matriz fundamental
   N = (I − Q)⁻¹, RE24 = N·r y P(anotar). Calibrada con un evento residual (errores, robos, wild
   pitches: 1.1% de los turnos) para reproducir las carreras reales de la liga; reproduce la RE24
   empírica de MLB.
3. **Cadena por lineup (216 estados):** carreras esperadas según quién abre la entrada y matriz T de
   rotación del lineup.
4. **Monte Carlo de 50,000 partidos:** abridor con límite de bateadores (detecta openers), bullpen
   real por rol, disponibilidad y carga típica, corredor fantasma en extra innings.
5. **Actualización bayesiana:** pasada con lineups proyectados contra pasada con los oficiales.

```bash
python -m mlbgs.prolab --pk 824223 --sims 50000   # usa los datos congelados de prolab/
```

Cuando el partido termina, cada actualización califica los picks del Pro-Lab con el resultado oficial.

## Cotejo de datos

Cada corrida verifica las fuentes entre sí (pestaña *Cotejo*): standings contra resultados, suma por
entrada contra marcador, temporada del abridor contra sus game logs, ERA de la MLB Stats API contra
Baseball Savant, abridores + relevistas = total, lineups contra roster activo, cobertura de box scores.
Así se encontraron y corrigieron un juego duplicado en el calendario oficial (inflaba el récord de
ATL y SF) y un límite de 50 filas en las stats por equipo que dejaba a 15 equipos sin bullpen.

## Activar la página y los momios

1. **GitHub Pages:** *Settings → Pages → Build and deployment → Source: GitHub Actions*. A partir de
   ahí cada corrida del workflow en `main` publica la página en `https://gerardodst.github.io/MLBgs/`.
2. **Momios (opcional):** crea una clave gratis en [the-odds-api.com](https://the-odds-api.com) y
   agrégala como secreto `ODDS_API_KEY` en *Settings → Secrets and variables → Actions*. El plan
   gratuito (500 créditos/mes) alcanza para las 5 corridas diarias. Sin momios el framework marca los
   mercados en Gris o Rojo (sin edge calculable → no bet) y la página muestra el momio justo y el
   mínimo aceptable para comparar con tu casa.

## Correr en local

Solo requiere Python 3.10+ (sin dependencias externas).

```bash
python -m mlbgs.fetch --out .cache/raw_bundle.json.gz      # descarga (≈30 s)
python -m mlbgs.build --bundle .cache/raw_bundle.json.gz   # modelo + site/index.html
python -m unittest discover -s tests                        # pruebas
```

`python -m mlbgs.fetch --date 2026-09-23` fija la fecha base; `--days 3` amplía la ventana.

## Estructura

| Ruta | Qué contiene |
| --- | --- |
| `mlbgs/fetch.py` | Ingesta (MLB Stats API, Baseball Savant, The Odds API) → bundle de datos |
| `mlbgs/features.py` | Liga, Pitágoras, Elo, perfiles de abridores con shrinkage, bullpen y fatiga |
| `mlbgs/model.py` | Análisis de cada partido: secciones 1-10 del framework |
| `mlbgs/context.py` | Lineup proyectado, bullpen completo, importancia (simulación de playoffs), noticias, arsenal |
| `mlbgs/picks.py` | Picks con índice de confianza y calificación contra el resultado |
| `mlbgs/markov.py` | DIAMANTE-24: turnos al bate, cadena de Markov (RE24) y Monte Carlo |
| `mlbgs/prolab.py` | Pro-Lab: DIAMANTE-24 sobre un partido con datos congelados |
| `mlbgs/validate.py` | Cotejo cruzado de las fuentes |
| `prolab/` | Datos congelados antes del partido de prueba y su resultado |
| `mlbgs/mathlib.py` | Fórmulas: Poisson, Binomial Negativa, FIP, Log5, Elo, Kelly, momios |
| `mlbgs/build.py` | Construye la página y el seguimiento de predicciones |
| `site/template.html` | Interfaz (español, tema claro/oscuro) |
| `data/predictions/` | Predicción previa de cada partido, para medir el modelo |
| `.github/workflows/update.yml` | Actualización programada y despliegue a GitHub Pages |

## Limitaciones conocidas

- **SIERA y xFIP de FanGraphs** no están conectados: se usa FIP (como indica 3.4) y un xFIP aproximado.
- **Tendencias de umpires** (UmpScorecards) no están conectadas: el ajuste de zona no se evalúa.
- **Props de bateadores** y momios de F5/NRFI no vienen en el plan básico de The Odds API.
- Los ajustes de la tabla 5.3 usan los factores del framework; el siguiente paso es medirlos con
  datos (como se hizo en DATAGSPORTS) usando el historial de `data/predictions/`.

Este framework es una herramienta de análisis estadístico: no garantiza resultados y apostar implica
riesgo financiero real.
