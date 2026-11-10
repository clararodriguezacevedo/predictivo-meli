# TP3 Análisis Predictivo (ITBA), competencia Kaggle AP_2026Q2

## Contexto

- Competencia en https://www.kaggle.com/competitions/ap-2026-q-2/
- La tarea es clasificación binaria de items de MercadoLibre, predecir `condition` (`new` o `used`).
- La métrica es accuracy. Leaderboard público con ~70% del test, el privado (30%) define el ranking final.
- Cierra el 15/11/2026 a las 12:59:59, hora Argentina.
- La submission es un CSV con columnas `ID` y `condition`. `ID` es el número de fila del archivo de
  test empezando en 1 (no el id de MercadoLibre) y `condition` es `new` o `used`. Formato
  confirmado con una submission aceptada. Hay 5 envíos por día y 2 cuentan para el final.
- Los datos son `train_data.jsonlines` (70.000 filas, con `condition`) y `test_data.jsonlines` (30.000 filas, sin `condition`).

## Forma de trabajo

- La lógica del modelo va en pocos módulos de Python, divididos en secciones que se leen en orden.
  `correr.py` y los notebooks de `notebooks/` los usan, sin duplicar lógica. La excepción es el
  análisis de los notebooks 02 y 07, cuyo código va directo en el notebook. Una función que se usa
  en un solo notebook, y en ningún módulo ni tarea, va definida en ese notebook. En todos
  los notebooks, el comentario del análisis va en celdas de markdown y el código solo calcula y muestra.
- Los notebooks están numerados y se leen en orden, del dato crudo a la submission. No entrenan
  el modelo final, leen el registro. Lo que se quiera registrar se corre con `correr.py`.
- Los outputs generados (gráficos, tablas, submissions, modelos) van a carpetas de salida,
  separados del código.

## Estructura

- La estructura es plana, los módulos van en la raíz del proyecto, sin un paquete anidado.
- `datos.py` tiene todo lo que no usa el target, en tres secciones. Rutas y constantes, la carga
  que aplana los jsonlines y cachea parquet en `data/processed/`, y las features sin target (item
  + perfiles de vendedor y categoría calculados sobre train + test).
- El análisis exploratorio y el de errores viven enteros en sus notebooks (02 y 07), sin módulos aparte.
- `diccionario.py` tiene el significado y el grupo de cada variable. `docs/DICCIONARIO.md` es un
  documento fijo con esos significados y estadísticas de los datos. Toda variable o feature nueva
  se agrega en los dos, a mano.
- `modelo.py` tiene tres secciones. El target encoding (tasa de usados por vendedor y categoría,
  vecinos), calculado con folds anidados; el registro de corridas; y el modelo, que elige las
  columnas, une todo y entrena un solo LightGBM con los hiperparámetros finales (D50, D51).
- `pruebas.py` tiene los tests estadísticos, la lista de decisiones con su prueba, los cinco
  controles de leakage y la función que corre las pruebas. `python correr.py decisiones` genera
  `docs/DECISIONES.md`, con cada decisión medida con los folds de siempre y con folds nuevos.
- `notebooks/` tiene el recorrido completo, numerado. `LEEME.md` explica el orden de lectura y qué módulos
  usa cada notebook.
- Toda decisión lleva un comentario de una línea en el código, con su código `Dnn` si tiene test.
  Toda decisión nueva se agrega a la lista de decisiones de `pruebas.py`.
- `correr.py` reúne las tareas que entrenan o registran algo (`leakage`, `validar`, `decisiones`,
  `errores`, `submission`, `anotar`). `errores` guarda las predicciones que analiza
  `notebooks/07_errores.ipynb`.
- Datos crudos en `data/raw/`. `data/`, `outputs/` y `.venv/` no se versionan.
- El entorno es `.venv` con `requirements.txt`. Correr con `.venv/Scripts/python correr.py <orden>`.

## Registro de corridas

- Toda corrida de validación y toda submission se anota en `registro/runs.jsonl` con sus
  parámetros exactos y su resultado.
  Lo hace la sección de registro de `modelo.py`. Nunca correr un modelo por fuera de `correr.py`.
- `runs.jsonl` no se edita ni se reescribe, solo se agregan líneas. `registro/RUNS.md` es
  el resumen legible y se regenera solo. Los dos se versionan.
- Los tipos son `validacion` (protocolo estándar de 5 folds y 2 semillas, comparables entre sí),
  `control` (semillas nuevas, que corre `python correr.py decisiones`, para confirmar que una
  mejora no fue suerte de los folds estándar) y `submission` (siempre atada a una corrida de
  validación). Las filas `busqueda` son históricas, porque el script que las generó ya no está.
- Se valida con `python correr.py validar` y se genera la submission con `python correr.py
  submission`. Después de subirla a Kaggle, anotar el puntaje con `python correr.py anotar`.
- Para elegir modelo se comparan solo corridas de tipo `validacion`.
- Los modelos de un ensamble no se eligen mirando su resultado sobre los datos de evaluación
  (recomendación de la cátedra), sino que se decide la composición de antemano.

## Leakage

- Las features de `datos.py` no pueden usar el target ni nada que no exista en test.
- Los agregados (vendedor, categoría, palabra) se calculan sobre train + test juntos.
- Después de tocar `datos.py` o `modelo.py`, correr `python correr.py leakage`. Tiene que dar OK en
  los cinco controles antes de commitear.
- Las features que usan el target van en la sección de target encoding de `modelo.py`, nunca en
  `datos.py`, y
  siempre con folds (no leave-one-out). Validación y test se codifican con las etiquetas
  de entrenamiento, y el entrenamiento por sub-folds.

## Commits

- Commits chicos, de un solo tema cada uno.
- Mensaje corto en español, en minúscula, sin demasiadas siglas ni tecnicismos.
- Solo el título, sin descripción.
- Sin ningún tipo de autoría de Claude (nada de `Co-Authored-By` ni similares).

## Estilo de escritura

- Todo va en español, incluidos código comentado, explicaciones, documentación y respuestas. Se pueden usar
  términos en inglés cuando sea lo natural (feature, target, leakage, etc.).
- NUNCA usar middle dots ni em dashes, en ningún lado (texto, código, gráficos, docs).
- No usar dos puntos en el texto (comentarios, docs, markdown). Usar comas o cambiar la frase.
  Uso escueto de barras verticales y separadores de markdown (`---`).
