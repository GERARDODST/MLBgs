# MLBgs: reglas de trabajo

MLBgs es la versión MLB de DATAGSPORTS: picks de la MLB con datos oficiales (MLB Stats API + Baseball
Savant), el Framework MLB Picks v2 y modelos de prueba del Pro-Lab. La página se publica como artefacto
en claude.ai (`site/index.html`) y los datos se actualizan con GitHub Actions (`update.yml`).
El usuario escribe en español: responder y documentar en español.

## Protocolo de decisión de cada partido (obligatorio)

Todo análisis de un partido termina en **una sola opción**: un pick con su stake, «esperar» o «no apostar».
Nunca se da un pick sin haber conectado antes todas las partes:

1. **Framework completo (secciones 1-10)**: contexto y motivación (importancia en la tabla, serie,
   noticias y lesionados), forma e historial, abridores (regresión a la media), bullpen y fatiga, modelo
   de carreras y triangulación (Log5 · Elo · λ-Binomial Negativa), totales y desarrollo por entradas,
   parque/clima/umpire, lineups, mercado, contradicciones y gate de datos obligatorios.
2. **Modelo de apoyo del Pro-Lab** cuando el partido lo tiene (DIAMANTE-24, PRISMA, KRONOS u otro):
   sus probabilidades entran a los picks y su acuerdo con el framework mueve el stake.
3. **Cuotas**: el índice de confianza (IC) se recalcula con el momio real; sin momio, cada pick dice
   desde qué momio conviene (escalera de stake).
4. **Decisión** (`mlbgs/decision.py`): el pick con más confianza que pasa guion, contradicciones y
   datos obligatorios; «esperar» si el mejor solo espera un dato; «no apostar» si ninguno alcanza.
   Va con su lista de cómo se conecta cada parte (a favor / en contra / contexto).
5. **Conclusión de Claude**: la página manda el expediente completo del partido a Claude (capacidad
   `sample` del artefacto, al pulsar «Analizar con Claude») y muestra su decisión final con el porqué.
   Claude elige solo entre los picks candidatos y nunca sube el stake por encima de lo que permiten
   las reglas. Cuando el análisis lo haga Claude en una sesión (Pro-Lab, check-ins), seguir el mismo
   orden y cerrar con una sola opción y su stake.

## Stake 1–10 por confianza (`mlbgs/stake.py`)

- stake 1 = $500 · stake 5 = $1,000 · stake 10 = $1,500 (de 1 a 5 sube $125 por nivel; de 5 a 10, $100).
- Nivel = redondeo((1 + 9·(IC − 55)/30) · A · H), entre 1 y 10; IC < 55 → no apostar.
  A = acuerdo con el modelo del Pro-Lab (0.70 si lo ve ≥ 10 pp peor); H = historial del modelo
  (acierto real vs esperado, encogido n/(n+60), entre 0.85 y 1.05).
- Solo con semáforo Verde a ese momio: edge ≥ 3 pp, IC ≥ 55, sin dato obligatorio faltante, guion que
  acompaña y contradicción no alta. Tope sugerido de $7,500 por día (la cartera avisa; no recorta).
- La página (`site/template.html`) repite estas cuentas en JavaScript: si se cambia una, cambiar la otra.

## Datos y registro

- `data/historial/` es la base de tickets (partido × modelo): picks, tipos de análisis, escalera de
  stake, decisión única y resultado oficial. Un ticket se congela al primer lanzamiento.
- Las predicciones del Pro-Lab se registran ANTES del primer lanzamiento y no se cambian después.
- El Historial de la página solo muestra partidos terminados; hoy y mañana viven en la Jornada.

## Flujo de trabajo

- Rama de trabajo: `claude/eloquent-ride-7quneh`. El usuario autorizó abrir y unir los PR a `main`.
- Antes de cada PR: `python -m unittest discover -s tests`, construir la página con
  `python -m mlbgs.build --bundle <bundle> --no-save` y revisarla en Chromium (escritorio y móvil).
- El contenedor no llega a statsapi.mlb.com: el bundle fresco se genera con `dev-bundle.yml` (se
  dispara al cambiar su línea de comentario en la rama) y no se sube a `main`
  (`git rm --cached dev/raw_bundle.json.gz` antes del PR).
- Al publicar el artefacto, conservar sus capacidades (`downloads`, `sample`).
