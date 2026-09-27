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
2. **Modelo de apoyo del Pro-Lab** cuando el partido lo tiene (DIAMANTE-24, PRISMA, KRONOS, EIGEN u otro):
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
- Filtro contra el mercado: con edge ≥ 10 pp no hay stake (se verifica: lesión, descanso, lineup). La
  escalera guarda `maxPrice`, el momio más alto que todavía pasa el filtro.
- Momios automáticos como SofaScore (`mlbgs/odds.py`): ESPN sin clave (su marcador público publica los de
  su casa socia, DraftKings, con apertura y actual; una llamada por fecha; `ODDS_ESPN=0` lo apaga);
  The Odds API (`ODDS_API_KEY`, plan gratis de 500 créditos/mes, sin tarjeta) SOLO para lo que ESPN no trae:
  el mercado de los dos picks del último análisis (`data/predictions/`) de cada partido — F5 (`h2h_1st_5_innings`,
  `totals_1st_5_innings`), ponches (`pitcher_strikeouts`), team total (`team_totals`) y primera entrada NRFI/YRFI
  (`totals_1st_1_innings`) —, cada mercado una vez a ≤ 6 h y un refresco a ≤ 90 min; 1 crédito por mercado y
  partido; topes `ODDS_API_DAY_CREDITS` (24) y `ODDS_API_MONTH_CREDITS` (470). Con línea del mercado, el pick
  usa esa línea (total, F5 total, team total, ponches). odds-api.net
  (`ODDS_API_NET_KEY`) queda como opción de pago. Nunca raspando casas. `data/odds/<fecha>.json`
  guarda por partido y casa la apertura (`open`) y el último momio (`last`); al primer lanzamiento se
  congela (cierre). Cadencia: > 6 h cada 3 h, 1–6 h cada hora, < 1 h cada corrida; topes por corrida y día.
  El edge/IC/stake/decisión usan el **momio de referencia** = mediana de las casas MX (marca de la API
  `country_code=MX`) o de todas si ninguna; el mejor precio solo se muestra. La decisión elige, por IC, el
  primer candidato que llega a stake a su momio real. El boleto usa ese momio («mercado») si el usuario
  no registró el suyo. Sin feed ni momio del usuario, el «justo» y el «mín.» salen del modelo: nunca
  presentarlos como el momio del casino; el stake es un rango según la escalera y, con el momio que
  escribe el usuario, la decisión se recalcula entre todos los candidatos.
  Playdoit, Caliente y Team México no están en ningún feed: su momio lo captura el usuario y manda.
- Jornada, «Por jugar»: cada uno de los dos picks de la tarjeta muestra su **momio actual** (el capturado
  de su casa si existe; si no, el mejor del feed) con la casa abajo en chico y en su color, y a un lado el
  **momio recomendado** (inicio de la escalera de stake: desde ahí conviene) con el veredicto (tómalo ·
  no · verificar · esperar). El plan del día, las tarjetas de picks principales, la tabla de todos los picks y
  el panel de decisión muestran lo mismo, sin casillas para escribir momios (la captura queda solo en el
  tablero de la sección 7, modo «Capturar», y en los boletos). La tarjeta de la Jornada siempre muestra la línea
  del mercado (ganador y total de la casa con ML) aunque el moneyline no sea pick. El tablero de la sección 7
  pone primero las casas del feed con más mercados (DraftKings primero).
- Momios de Playdoit: su sitio bloquea el acceso automático (Cloudflare, 403 «Acceso bloqueado» desde
  GitHub Actions); no evadirlo. La Jornada tiene «Momios de Playdoit»: el usuario sube una captura o pega
  el texto y Claude (capacidad `sample` con imágenes) los lee y los aplica solo a partidos que no han
  empezado. Se guardan en la base `db`, colección `momios` (doc = fecha: games por gamePk con ml, rl,
  total y f5); leerla con `ArtifactData` para usar los momios reales en un análisis.
- La página (`site/template.html`) repite estas cuentas en JavaScript: si se cambia una, cambiar la otra.

## Datos y registro

- `data/historial/` es la base de tickets (partido × modelo): picks, tipos de análisis, escalera de
  stake, decisión única y resultado oficial. Un ticket se congela al primer lanzamiento.
- Las predicciones del Pro-Lab se registran ANTES del primer lanzamiento y no se cambian después.
- El Historial de la página solo muestra partidos terminados; hoy y mañana viven en la Jornada.
- **Boletos** (`mlbgs/boleto.py`, repetido en la página): un boleto por partido con la decisión única
  (manda el ticket del Pro-Lab). Momio decimal a 2 cifras (−137 → 1.73), pago = stake × decimal;
  ganado: neto = pago − stake; perdido: −stake; push o anulado: se devuelve el stake. Momio: el que
  registró el usuario; si no, el capturado antes del juego; si no, el del mercado (feed al congelarse
  el ticket); si no, el de referencia (el más bajo con el que valía ese stake). Lo que registra el usuario vive en la base `db` del artefacto, colección
  `boletos` (doc = gamePk: odds, amount, played); leerla con `ArtifactData` para sacar las cuentas reales.

## Flujo de trabajo

- Rama de trabajo: `claude/eloquent-ride-7quneh`. El usuario autorizó abrir y unir los PR a `main`.
- Antes de cada PR: `python -m unittest discover -s tests`, construir la página con
  `python -m mlbgs.build --bundle <bundle> --no-save` y revisarla en Chromium (escritorio y móvil).
- El contenedor no llega a statsapi.mlb.com: el bundle fresco se genera con `dev-bundle.yml` (se
  dispara al cambiar su línea de comentario en la rama) y no se sube a `main`
  (`git rm --cached dev/raw_bundle.json.gz` antes del PR).
- Al publicar el artefacto, conservar sus capacidades (`downloads`, `sample`, `db`).
