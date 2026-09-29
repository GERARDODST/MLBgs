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
2. **Modelo de apoyo del Pro-Lab** cuando el partido lo tiene (DIAMANTE-24, PRISMA, KRONOS, EIGEN, OCTUBRE u otro):
   sus probabilidades entran a los picks y su acuerdo con el framework mueve el stake.
3. **Cuotas**: el índice de confianza (IC) se recalcula con el momio real; sin momio, cada pick dice
   desde qué momio conviene (escalera de stake).
4. **Decisión** (`mlbgs/decision.py`): el pick con más confianza que pasa guion, contradicciones y
   datos obligatorios; «esperar» si el mejor solo espera un dato; «no apostar» si ninguno alcanza.
   Va con su lista de cómo se conecta cada parte (a favor / en contra / contexto).
5. **Conclusión de Claude**: la página manda el expediente completo del partido a Claude (capacidad
   `sample` del artefacto, al pulsar «Analizar con Claude») y muestra su decisión final con el porqué.
   Claude elige solo entre los picks candidatos y nunca sube el stake por encima de lo que permiten
   las reglas. El expediente incluye el modelo del Pro-Lab completo (`labBrief` en la página: en OCTUBRE,
   historial contra el rival, pitch contra bateador, gancho y bullpen de octubre, familiaridad y ablación); una
   conclusión de Claude hecha sin el modelo del Pro-Lab actual (otro `frozenAt`) deja de mandar y la página pide
   «volver a analizar». Cuando el análisis lo haga Claude en una sesión (Pro-Lab, check-ins), seguir el mismo
   orden y cerrar con una sola opción y su stake.

## Stake 1–10 por confianza (`mlbgs/stake.py`)

- stake 1 = $500 · stake 5 = $1,000 · stake 10 = $1,500 (de 1 a 5 sube $125 por nivel; de 5 a 10, $100).
- Nivel = redondeo((1 + 9·(IC − 55)/30) · A · H · R), entre 1 y 10; IC < 55 → no apostar.
  A = acuerdo con el modelo del Pro-Lab (0.70 si lo ve ≥ 10 pp peor); H = historial del modelo
  (acierto real vs esperado, encogido n/(n+60), entre 0.85 y 1.05); R = alertas de la auditoría
  (algoritmo 2026.09.28.2): × 0.8 si el pick depende de que un abridor en mala racha siga mal (8.3),
  × 0.8 si es favorito caro a −170 o peor (7.5), y favorito caro con total proyectado < 8 = sin stake.
- Modelo (algoritmo 2026.09.28.2): F5 y F3 con Binomial Negativa y la sobredispersión medida en esas
  entradas (no Poisson). Equipo sin nada en juego (eliminado, o clasificado sin siembra en juego, solo
  temporada regular): su bullpen se regresa 50% a la media de la liga y el uso esperado de su cerrador y
  setups baja 40%. El lineup ya regresa el OBP/SLG de cada bateador a la liga por turnos (k = 300/320).
- Solo con semáforo Verde a ese momio: edge ≥ 3 pp, IC ≥ 55, sin dato obligatorio faltante, guion que
  acompaña y contradicción no alta. Tope sugerido de $7,500 por día (la cartera avisa; no recorta).
- Filtro contra el mercado: con edge ≥ 10 pp no hay stake (se verifica: lesión, descanso, lineup). La
  escalera guarda `maxPrice`, el momio más alto que todavía pasa el filtro. Si el pick de más confianza cae en el
  filtro a su momio real, la decisión es «esperar · verificar» ESE pick (`waitFor: verificar`, algoritmo
  2026.09.29.3); no se salta a otro mercado sin momio.
- Momios automáticos como SofaScore (`mlbgs/odds.py`): ESPN sin clave (su marcador público publica los de
  su casa socia, DraftKings, con apertura y actual; una llamada por fecha; `ODDS_ESPN=0` lo apaga);
  The Odds API (`ODDS_API_KEY`, plan gratis de 500 créditos/mes, sin tarjeta) SOLO para lo que ESPN no trae:
  el mercado de los dos picks del último análisis (`data/predictions/`) de cada partido — F5 (`h2h_1st_5_innings`,
  `totals_1st_5_innings`), ponches (`pitcher_strikeouts`), team total (`team_totals`) y primera entrada NRFI/YRFI
  (`totals_1st_1_innings`) —, el de la decisión desde 12 h antes (el otro pick a ≤ 6 h), refrescos al entrar a las 6 h
  y a ≤ 90 min, y si ninguna casa lo publicaba otro intento cada 2 h (hasta 2; `theGot`/`theEmpty`; `odds.next_ask`);
  lo pedido antes de las 6 h y los reintentos no tocan la reserva. «Esperar momio» lleva `ask` (de dónde y cuándo
  llega: ESPN en cada corrida o The Odds API a tal hora) y la página lo muestra con ⏱; 1 crédito por mercado y
  partido; topes `ODDS_API_DAY_CREDITS` (24) y `ODDS_API_MONTH_CREDITS` (470). Con línea del mercado, el pick
  usa esa línea (total, F5 total, team total, ponches). odds-api.net
  (`ODDS_API_NET_KEY`) queda como opción de pago. Nunca raspando casas. `data/odds/<fecha>.json`
  guarda por partido y casa la apertura (`open`) y el último momio (`last`); al primer lanzamiento se
  congela (cierre). Cadencia: > 6 h cada 3 h, 1–6 h cada hora, < 1 h cada corrida; topes por corrida y día.
  El edge/IC/stake/decisión usan el **momio de referencia** = mediana de las casas MX (marca de la API
  `country_code=MX`) o de todas si ninguna; el mejor precio solo se muestra. La decisión elige, por IC, el
  primer candidato que llega a stake a su momio real. El boleto usa ese momio («mercado») si el usuario
  no registró el suyo. Sin feed ni momio del usuario, el «justo» y el «mín.» salen del modelo: nunca
  presentarlos como el momio del casino. **Sin momio real no hay stake** (algoritmo 2026.09.28): el IC no
  suma «fuerza» contra el momio de referencia de −110 y la decisión es «esperar momio» (`waitFor: momio`)
  con la escalera (desde qué momio conviene); con el momio que escribe el usuario, la decisión se recalcula
  entre todos los candidatos.
  Playdoit, Caliente y Team México no están en ningún feed: su momio lo captura el usuario y manda.
- Jornada, «Por jugar»: cada uno de los dos picks de la tarjeta muestra su **momio actual** (el capturado
  de su casa si existe; si no, el de referencia del feed como el stake: mediana de las casas MX o de todas, o la única casa; el mejor precio solo en el tablero) con la casa abajo en chico y en su color, y a un lado el
  **momio recomendado** (inicio de la escalera de stake: desde ahí conviene) con el veredicto (tómalo ·
  no · verificar · esperar). El plan del día, las tarjetas de picks principales, la tabla de todos los picks y
  el panel de decisión muestran lo mismo, sin casillas para escribir momios (la captura queda solo en el
  tablero de la sección 7, modo «Capturar», y en los boletos). La tarjeta de la Jornada siempre muestra la línea
  del mercado (ganador y total de la casa con ML) aunque el moneyline no sea pick. El tablero de la sección 7
  pone primero las casas del feed con más mercados (DraftKings primero).
- **Revisión periódica** (`mlbgs/cambios.py`, `mlbgs/cadencia.py`): cada corrida compara abridores, lineups,
  umpire, clima, horario, movimientos, momios y la decisión de cada partido que no ha empezado contra la corrida
  anterior (`data/cambios/<fecha>.json`: `snap` por partido y `events`); la primera foto es la base y un dato que
  la API deja de mandar no cuenta como cambio. Un partido con cambio de abridor, lineup, horario o baja revisa sus
  momios de ESPN en la misma corrida (cada llamada a ESPN actualiza toda la fecha). The Odds API por prioridad:
  0 mercado cuyo abridor cambió después de pedirlo, 1 mercado de la decisión que ESPN no trae, 1.5 si la decisión
  espera momio, los de todos los candidatos sin momio que ESPN no trae (misma llamada; `alts` en `data/predictions`), 2 verificación del
  mercado de la decisión (ML/RL/total, `h2h`/`spreads`/`totals`) si hay stake o se espera momio (≤ 3 h), 3 el otro
  pick; 2 y 3 no tocan los últimos `ODDS_API_RESERVE` (4) créditos del día. `verify`: DraftKings vs mediana de las
  otras casas (difiere ≥ 3 pp o línea distinta). Algoritmo 2026.09.29: el momio visto antes de un cambio de
  abridor (`stale`) no cuenta para el stake hasta que la casa publique otro. El relevo de `update.yml` espera
  7 min con un partido a ≤ 2.5 h, 13 min a ≤ 6 h y 25 min más lejos. `data/predictions` guarda la `decision`.
  La página: «Última hora» en la Jornada, aviso en la tarjeta, «Qué cambió» y verificación en el partido.
- Momios de Playdoit: su sitio bloquea el acceso automático (Cloudflare, 403 «Acceso bloqueado» desde
  GitHub Actions); no evadirlo. La Jornada tiene «Momios de Playdoit»: el usuario sube una captura o pega
  el texto y Claude (capacidad `sample` con imágenes) los lee y los aplica solo a partidos que no han
  empezado. Se guardan en la base `db`, colección `momios` (doc = fecha: games por gamePk con ml, rl,
  total y f5); leerla con `ArtifactData` para usar los momios reales en un análisis.
- La página (`site/template.html`) repite estas cuentas en JavaScript: si se cambia una, cambiar la otra.

## Pro-Lab OCTUBRE (postemporada, `mlbgs/octubre.py` + `prolab.run_octubre`)

- Snapshot con `PROLAB_MODEL=octubre` en `prolab/request.env`: game logs de todos los abridores (≥ 5 aperturas),
  historial de carrera contra el rival (`vsTeam`), postemporadas de los dos años anteriores y de la actual (abridores
  y su temporada regular). `python -m mlbgs.prolab --pk <pk> --model octubre --label <label>`.
- A: Beta-Binomial por evento (K, BB+HBP, HR por bateador; BABIP por bola en juego) centrado en la razón de momios
  pitcher × equipo × liga; κ por método de momentos quitando el azar binomial y la varianza de estimar μ (método delta).
  El historial contra el rival suma la temporada anterior a la mitad.
- B: xwOBA por tipo de lanzamiento (Savant) encogido con κ medidos en la liga (σ² por turno = 0.15); el nivel del
  bateador sale de su arsenal enfrentado (k = 300). C: outs en postemporada = α + β·promedio de temporada (β acotado
  a [0, 1]), Weibull con la dispersión medida; castigo por vuelta al lineup; bullpen de octubre por rol y uso.
  D: familiaridad controlada por fecha; solo carreras y solo con |t| ≥ 2, encogida por (1 − 1/t²).
- Los κ, el gancho y la familiaridad se recalculan con cada snapshot: no fijarlos a mano.

## Datos y registro

- `data/historial/` es la base de tickets (partido × modelo): picks, tipos de análisis, escalera de
  stake, decisión única y resultado oficial. Un ticket se congela al primer lanzamiento.
- **Tickets inmutables (regla del usuario):** al primer lanzamiento cada ticket se BLOQUEA (`lock`: huella
  sha256 de todo lo registrado, versión del algoritmo y expediente). Después solo se agregan el resultado
  oficial y la calificación de cada pick; el resultado, una vez calificado, tampoco se reescribe. Las
  correcciones a los modelos aplican SOLO a partidos que no han empezado: nunca recalcular, completar ni
  «arreglar» un ticket bloqueado. `historial.update` se detiene con error si algo lo intenta y la prueba
  `TicketsDelRepositorio` revisa todas las huellas.
- **Datos separados del algoritmo:** `data/expedientes/<fecha>/<pk>.json.gz` (análisis completo, datos crudos
  de la API recortados al partido, momios vistos, huellas de los tickets y versión del algoritmo) se escribe
  una sola vez al bloquearse; mientras el partido no empieza, la versión pendiente viaja entre corridas en la
  caché de Actions (`.cache/expedientes`). `mlbgs/version.py` guarda la versión (`VERSION`, `CAMBIOS`) y la
  huella del código: al cambiar reglas de un modelo, agregar una entrada a `CAMBIOS`.
- **Tickets de antes de la regla (legado, 28-sep):** manda el ticket original; lo que se les agregó después
  del juego va aparte en `data/correcciones/<id>.json` (versión corregida, cambios, estadísticas, notas y si
  es circunstancial o de fondo) y la página lo muestra en una ventana emergente. Para anotar algo nuevo sobre
  un ticket bloqueado, crear una nota ahí; nunca editar el ticket.
- **Momio real aparte:** `data/mercado/<fecha>.json` (ESPN/DraftKings apertura y cierre, hasta el 26-sep;
  desde el 27-sep sale de `data/odds`). `mercado.evaluar` lo compara con la escalera congelada del ticket
  («con el cierre real, ¿entraba?») y la Cartera muestra un balance paralelo «con cuota real». No cambia
  tickets ni boletos.
- Estado del partido: la MLB marca `Live` desde el calentamiento (`Warmup`/`Pre-Game`, 20–30 min antes). Eso
  sigue siendo «por jugar» en la descarga (análisis y momios), en los tickets y en la página («calentamiento ·
  en N min»); «en juego» es solo desde el primer lanzamiento.
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
- Cada corrida de producción deja la página construida en la rama `pagina` (un solo commit que se reemplaza).
  Una rutina de Claude (Routine, sesión nueva en cada disparo) la republica en el artefacto cada ~2 h en horario
  de juegos: `git fetch origin pagina` → publicar `index.html` en el URL del artefacto sin tocar capacidades.
