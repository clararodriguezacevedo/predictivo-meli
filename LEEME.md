# Clasificador de items nuevos y usados de MercadoLibre

Para entender el trabajo, leer los notebooks en orden. Para ver cómo está hecho cada paso, leer los módulos que cada notebook importa.

## Los notebooks

| Notebook | Qué cuenta | Módulos que usa |
|---|---|---|
| `notebooks/01_datos.ipynb` | Del JSON crudo a una tabla | `config.py`, `carga.py` |
| `notebooks/02_exploracion.ipynb` | Qué separa a nuevos de usados | `eda/` |
| `notebooks/03_features.ipynb` | Variables que no usan el target | `features.py`, `diccionario.py` |
| `notebooks/04_target_encoding.ipynb` | Variables que usan el target, sin filtrarlo | `target_encoding.py` |
| `notebooks/05_modelo_y_validacion.ipynb` | El modelo y cómo se mide | `modelo.py`, `registro.py` |
| `notebooks/06_decisiones.ipynb` | El test detrás de cada decisión | `estadistica.py`, `decisiones.py` |
| `notebooks/07_errores.ipynb` | En qué items falla | `errores.py` |
| `notebooks/08_submission.ipynb` | Del modelo al archivo para Kaggle | `run_submission.py` |

Los notebooks no entrenan el modelo final: leen los resultados del registro, para abrirse y correrse rápido. Lo que tarda horas está en los scripts.

## Los scripts

| Script | Para qué |
|---|---|
| `run_verificar_leakage.py` | Comprueba que el modelo no filtre el target |
| `run_validar_modelo.py` | Valida el modelo final y lo registra |
| `run_decisiones.py` | Corre los tests y genera `docs/DECISIONES.md` |
| `run_submission.py` | Genera el archivo para Kaggle y anota su puntaje |
| `run_analizar_errores.py` | Genera el reporte de errores |
| `run_eda.py` | Genera el reporte de exploración |
| `run_diccionario.py` | Genera `docs/DICCIONARIO.md` |

Los comentarios del código citan el código de la decisión, por ejemplo `D25`. Cada corrida queda en `registro/runs.jsonl`, y `registro/RUNS.md` es su resumen.
