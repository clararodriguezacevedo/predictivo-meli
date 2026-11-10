# Clasificador de items nuevos y usados de MercadoLibre

## Replicar el modelo final

Con los datos crudos en `data/raw/` y las dependencias de `requirements.txt` instaladas, se corren dos órdenes.

```
python correr.py entrenar --validacion 20261110-001656-validacion-lightgbm
python correr.py predecir
```

`entrenar` entrena el modelo final con todo train y lo guarda en `outputs/modelo/lightgbm.joblib`, atado a la validación que lo respalda. `predecir` lo carga, lo aplica al test y escribe el archivo en `outputs/submissions/`, idéntico al enviado a Kaggle. La lógica de los dos está en `entrenar_final` y `predecir_test`, en la sección del modelo de `modelo.py`. La presentación está en `presentacion/`.

## El recorrido

Para entender el trabajo, leer los notebooks en orden. Para ver cómo está hecho cada paso, leer el módulo que cada notebook importa. Cada módulo está dividido en secciones que se leen de arriba abajo.

## Los archivos

| Archivo | Qué tiene |
|---|---|
| `datos.py` | Todo lo que no usa el target. Rutas, carga de los JSON y features |
| `modelo.py` | Target encoding, LightGBM, validación, registro de corridas y el baseline |
| `pruebas.py` | El t corregido, los controles de leakage y la validación adversaria |
| `diccionario.py` | Qué significa cada variable y a qué grupo pertenece |
| `correr.py` | Las tareas que entrenan o registran algo, cada una con su orden |

## Los notebooks

| Notebook | Qué cuenta | Usa |
|---|---|---|
| `notebooks/01_datos.ipynb` | Del JSON crudo a una tabla | `datos.py` |
| `notebooks/02_exploracion.ipynb` | Qué separa a nuevos de usados | `datos.py` (el análisis está en el notebook) |
| `notebooks/03_features.ipynb` | Variables que no usan el target | `datos.py`, `diccionario.py` |
| `notebooks/04_target_encoding.ipynb` | Variables que usan el target, sin filtrarlo | `modelo.py` |
| `notebooks/05_modelo_y_validacion.ipynb` | El modelo y cómo se mide | `modelo.py` |
| `notebooks/06_decisiones.ipynb` | El test detrás de cada decisión, y genera `docs/DECISIONES.md` | `pruebas.py` (las decisiones están en el notebook) |
| `notebooks/07_errores.ipynb` | En qué items falla | `modelo.py` (el análisis está en el notebook) |
| `notebooks/08_submission.ipynb` | Del modelo al archivo para Kaggle | `modelo.py` |

Los notebooks no entrenan el modelo final. Leen los resultados del registro, así se abren y corren rápido. Lo que tarda se corre con `correr.py`.

## Las tareas

| Orden | Para qué |
|---|---|
| `python correr.py leakage` | Comprueba que el modelo no filtre el target |
| `python correr.py validar --nota "..."` | Valida el modelo final y lo registra |
| `python correr.py baseline` | Valida el baseline (regresión logística con seis variables) con los mismos folds |
| `python correr.py errores` | Calcula las predicciones que analiza el notebook 07 |
| `python correr.py entrenar --validacion ID` | Entrena el modelo final con todo train y lo guarda |
| `python correr.py predecir` | Aplica el modelo guardado al test, escribe el archivo para Kaggle y lo registra |
| `python correr.py anotar ID --puntaje 0.95` | Anota el puntaje de Kaggle de una submission |

Los comentarios del código citan el código de la decisión, por ejemplo `D25`. Cada corrida queda en `registro/runs.jsonl`, y `registro/RUNS.md` es su resumen. `docs/DICCIONARIO.md` es un documento fijo con el significado de cada variable.
