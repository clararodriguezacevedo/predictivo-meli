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

- Nada de Jupyter notebooks hasta el momento de la entrega. Todo en scripts de Python.
- Código organizado en módulos reutilizables (carga de datos, features, modelos, etc.) y scripts
  de entrada livianos que los usan. Evitar duplicar lógica entre scripts.
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
- `evaluacion.py`: preparación de la matriz de features. `run_evaluar_features.py` es la
  ablación: cuánto pierde el modelo completo al quitar cada grupo de features o encoder.
- `target_encoding.py`: features que usan el target (tasa de usados por vendedor y
  categoría, modelo de texto, vecinos), calculadas con folds anidados. `clasificadores.py`:
  LightGBM, HistGradientBoosting, CatBoost, XGBoost y su promedio. `modelo.py`: une features,
  target encoding y clasificador. `pseudo_etiquetas.py`: suma al entrenamiento las
  predicciones seguras de items sin etiqueta.
- `errores.py` y `run_analizar_errores.py`: reporte de en qué items falla el modelo.
- `registro.py`: registro de corridas (ver sección propia).
- `run_verificar_leakage.py`: cinco controles de leakage sobre features, target encoding
  y modelo completo.
- Datos crudos en `data/raw/`. `data/`, `outputs/` y `.venv/` no se versionan.
- Entorno: `.venv` con `requirements.txt`. Correr con `.venv/Scripts/python <script>.py`.

## Registro de corridas

- Toda corrida de validación, cada prueba de una búsqueda de hiperparámetros y toda
  submission se anota en `registro/runs.jsonl` con sus parámetros exactos y su resultado.
  Lo hace `registro.py`; nunca correr un modelo por fuera de los scripts que registran.
- `runs.jsonl` no se edita ni se reescribe, solo se agregan líneas. `registro/RUNS.md` es
  el resumen legible y se regenera solo. Los dos se versionan.
- Tipos: `validacion` (protocolo estándar de 5 folds y 2 semillas, comparables entre sí),
  `busqueda` (protocolo liviano de `run_optimizar.py`, no comparable con las anteriores),
  `control` (semillas nuevas con `run_evaluar_features.py --semillas`, para confirmar que una
  mejora no fue suerte de los folds estándar) y `submission` (siempre atada a una corrida de
  validación).
- Scripts: `run_validar_modelo.py`, `run_optimizar.py`, `run_submission.py`. Después de
  subir una submission a Kaggle, anotar el puntaje con `run_submission.py --anotar`.
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
