# TP3 Análisis Predictivo (ITBA): competencia Kaggle AP_2026Q2

## Contexto

- Competencia: https://www.kaggle.com/competitions/ap-2026-q-2/
- Tarea: clasificación binaria de items de MercadoLibre, predecir `condition` (`new` o `used`).
- Métrica: Accuracy. Leaderboard público con ~70% del test, el privado (30%) define el ranking final.
- Cierre: 15/11/2026 12:59:59 hora Argentina.
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
- `evaluacion.py`: validación cruzada rápida con LightGBM. `run_evaluar_features.py` mide
  el accuracy y cuánto aporta cada grupo de features.
- `run_verificar_leakage.py`: controla que ninguna feature dependa del target.
- Datos crudos en `data/raw/`. `data/`, `outputs/` y `.venv/` no se versionan.
- Entorno: `.venv` con `requirements.txt`. Correr con `.venv/Scripts/python <script>.py`.

## Leakage

- Las features de `features.py` no pueden usar el target ni nada que no exista en test.
- Los agregados (vendedor, categoría, palabra) se calculan sobre train + test juntos.
- Después de tocar `features.py` o `carga.py`, correr `run_verificar_leakage.py`. Tiene
  que dar OK en los tres controles antes de commitear.
- Las features que usan el target (target encoding, modelos de texto supervisados) solo
  son válidas calculadas out-of-fold y van en la etapa de modelado, no en `features.py`.

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
