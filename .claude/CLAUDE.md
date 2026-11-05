# TP3 Análisis Predictivo (ITBA): competencia Kaggle AP_2026Q2

## Contexto

- Competencia: https://www.kaggle.com/competitions/ap-2026-q-2/
- Tarea: clasificación binaria de items de MercadoLibre, predecir `condition` (`new` o `used`).
- Métrica: Accuracy. Leaderboard público con ~70% del test, el privado (30%) define el ranking final.
- Cierre: 15/11/2026 12:59:59 hora Argentina.
- Submission: CSV con columnas `ID` y `condition`. `ID` es el número de fila del archivo de
  test empezando en 1 (no el id de MercadoLibre) y `condition` es `new` o `used`. Formato
  confirmado con una submission aceptada. Hay 5 envíos por día y 2 cuentan para el final.
- Datos: `train_data.jsonlines` (70.000 filas, con `condition`) y `test_data.jsonlines` (30.000 filas, sin `condition`).

## Forma de trabajo

- La lógica va en módulos de Python reutilizables (carga de datos, features, modelo, etc.). Los
  scripts `run_*.py` y los notebooks de `notebooks/` son livianos y los usan, sin duplicar lógica.
- Los notebooks están numerados y se leen en orden, del dato crudo a la submission. No entrenan
  el modelo final: leen el registro. Lo que se quiera registrar se corre con los scripts.
- Los outputs generados (gráficos, tablas, submissions, modelos) van a carpetas de salida,
  separados del código.

## Estructura

- Estructura plana: los módulos van en la raíz del proyecto, sin un paquete anidado.
- `config.py`: rutas y constantes. `carga.py`: aplana los jsonlines y cachea parquet en
  `data/processed/`. `features.py`: features sin target (item + perfiles de vendedor y
  categoría calculados sobre train + test).
- `eda/`: secciones del reporte de EDA. `run_eda.py` genera `outputs/eda/REPORTE_EDA.md`.
- `diccionario.py`: significado de cada variable. `run_diccionario.py` genera
  `docs/DICCIONARIO.md` y avisa si hay columnas sin documentar. Toda variable o feature
  nueva se agrega ahí.
- `target_encoding.py`: features que usan el target (tasa de usados por vendedor y
  categoría, vecinos), calculadas con folds anidados. `modelo.py`: elige las columnas, une
  features y target encoding, y entrena un solo LightGBM con los hiperparámetros finales (D50, D51).
- `estadistica.py`: los tests. `decisiones.py`: la lista de decisiones y su prueba.
  `run_decisiones.py` los corre y genera `docs/DECISIONES.md`.
- `notebooks/`: el recorrido completo, numerado. `LEEME.md`: el orden de lectura y qué módulos
  usa cada notebook.
- Toda decisión lleva un comentario de una línea en el código, con su código `Dnn` si tiene test.
  Toda decisión nueva se agrega a `decisiones.py`.
- `errores.py` y `run_analizar_errores.py`: reporte de en qué items falla el modelo.
- `registro.py`: registro de corridas (ver sección propia).
- `run_verificar_leakage.py`: cinco controles de leakage sobre features, target encoding
  y modelo completo.
- Datos crudos en `data/raw/`. `data/`, `outputs/` y `.venv/` no se versionan.
- Entorno: `.venv` con `requirements.txt`. Correr con `.venv/Scripts/python <script>.py`.

## Registro de corridas

- Toda corrida de validación y toda submission se anota en `registro/runs.jsonl` con sus
  parámetros exactos y su resultado.
  Lo hace `registro.py`; nunca correr un modelo por fuera de los scripts que registran.
- `runs.jsonl` no se edita ni se reescribe, solo se agregan líneas. `registro/RUNS.md` es
  el resumen legible y se regenera solo. Los dos se versionan.
- Tipos: `validacion` (protocolo estándar de 5 folds y 2 semillas, comparables entre sí),
  `control` (semillas nuevas con `run_decisiones.py --semillas`, para confirmar que una
  mejora no fue suerte de los folds estándar) y `submission` (siempre atada a una corrida de
  validación). Las filas `busqueda` son históricas: el script que las generó ya no está.
- Scripts: `run_validar_modelo.py`, `run_submission.py`. Después de subir una submission a
  Kaggle, anotar el puntaje con `run_submission.py --anotar`.
- Para elegir modelo se comparan solo corridas de tipo `validacion`.
- Los modelos de un ensamble no se eligen mirando su resultado sobre los datos de evaluación
  (recomendación de la cátedra): se decide la composición de antemano.

## Leakage

- Las features de `features.py` no pueden usar el target ni nada que no exista en test.
- Los agregados (vendedor, categoría, palabra) se calculan sobre train + test juntos.
- Después de tocar `features.py`, `carga.py`, `target_encoding.py` o `modelo.py`, correr
  `run_verificar_leakage.py`. Tiene que dar OK en los cinco controles antes de commitear.
- Las features que usan el target van en `target_encoding.py`, nunca en `features.py`, y
  siempre con folds (no leave-one-out): validación y test se codifican con las etiquetas
  de entrenamiento, y el entrenamiento por sub-folds.

## Commits

- Commits chicos, de un solo tema cada uno.
- Mensaje corto en español, en minúscula, sin demasiadas siglas ni tecnicismos.
- Solo el título, sin descripción.
- Sin ningún tipo de autoría de Claude (nada de `Co-Authored-By` ni similares).

## Estilo de escritura

- Todo en español: código comentado, explicaciones, documentación y respuestas. Se pueden usar
  términos en inglés cuando sea lo natural (feature, target, leakage, etc.).
- NUNCA usar middle dots ni em dashes, en ningún lado (texto, código, gráficos, docs).
- Uso escueto y moderado de dos puntos, barras verticales y separadores de markdown (`---`).
  No abusar de estos recursos.
