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
