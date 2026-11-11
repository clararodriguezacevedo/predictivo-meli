# Clasificador de items nuevos y usados de MercadoLibre

## Replicar el modelo final

El proyecto se probó con Python 3.12.10. Con versiones más viejas, las librerías de `requirements.txt` pueden no instalarse.

Primero, crear el entorno e instalar las dependencias.

```
python -m venv .venv
.venv\Scripts\activate          (en Windows)
source .venv/bin/activate       (en Linux o Mac)
pip install -r requirements.txt
```

Después, bajar los datos de la competencia desde https://www.kaggle.com/competitions/ap-2026-q-2/data y dejar los dos archivos en `data/raw/`, con estos nombres.

```
data/raw/train_data.jsonlines
data/raw/test_data.jsonlines
```

Por último, entrenar el modelo final y aplicarlo al test.

```
python correr.py entrenar
python correr.py predecir
```

`entrenar` entrena el modelo final con todo train y lo guarda en `outputs/modelo/lightgbm.joblib`, atado a la validación que lo respalda (`20261110-001656-validacion-lightgbm`). Si el código cambió desde esa validación, se niega a entrenar y avisa que hay que validar de nuevo. `predecir` lo carga, lo aplica al test y escribe el archivo en `outputs/submissions/`, idéntico al enviado a Kaggle. Ese archivo, que sacó 0,94790 en el leaderboard público, está en `submission_enviada.csv` para compararlo, por ejemplo con `cmp` en Linux o Mac y con `fc /b` en Windows. La lógica de los dos está en `entrenar_final` y `predecir_test`, en la sección del modelo de `modelo.py`.

## Los archivos

Para entender el trabajo, leer los notebooks en orden. Para ver cómo está hecho cada paso, leer el módulo que cada notebook importa

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

## Scripts

Todo lo que entrena o anota algo en el registro se corre desde `correr.py`, con una orden distinta para cada tarea. Para replicar el modelo final alcanza con `entrenar` y `predecir`. Las demás sirven para medir y revisar el modelo mientras se lo desarrolla.

`python correr.py validar --nota "texto"` mide qué tan bien predice el modelo. Divide train en 5 partes, entrena con 4 y mide el accuracy en la quinta, rota hasta medir las 5 y repite todo con otra división al azar. Anota en el registro las 10 mediciones y su promedio, con la nota o id que se le pase para reconocer la corrida. Se corre después de cualquier cambio al modelo y tarda unos 15 minutos.

`python correr.py baseline` mide el modelo de referencia, una regresión logística con las columnas del JSON tal como quedan después de aplanarlo, sin ninguna variable construida por nosotros. Solo se sacan el texto libre, los identificadores y la categoría, que no tienen una forma directa de entrar en un modelo lineal. Usa las mismas divisiones que `validar`. Así se sabe cuánto mejora el modelo final sobre algo simple. Tarda unos segundos y anota el resultado en el registro.

`python correr.py leakage` corre cinco controles de que el modelo no hace trampa usando el target. Comprueba, por ejemplo, que las variables no cambien si se mezclan las etiquetas, o que con etiquetas al azar el modelo no acierte más que prediciendo siempre la clase más frecuente. Tiene que dar OK en los cinco antes de commitear cambios en `datos.py` o `modelo.py`. Tarda unos minutos y no anota nada.

`python correr.py errores` predice cada item de train con un modelo que no lo vio y guarda esas predicciones en `outputs/errores/`. Las usa el notebook 07 para analizar en qué items falla el modelo. Tarda unos minutos.

`python correr.py entrenar` entrena el modelo final con todo train y lo guarda en `outputs/modelo/`. Antes comprueba que el código sea el mismo que se validó, como se explica más arriba.

`python correr.py predecir` aplica el modelo guardado al test y escribe en `outputs/submissions/` el archivo para subir a Kaggle. También lo anota en el registro.

`python correr.py anotar ID --puntaje 0.95` anota en el registro el puntaje que dio Kaggle para una submission, usando el id que imprimió `predecir`. Así queda la relación entre la validación y el leaderboard.

Los comentarios del código citan el código de la decisión, por ejemplo `D25`. Cada corrida queda en `registro/runs.jsonl`, y `registro/RUNS.md` es su resumen. `docs/DICCIONARIO.md` es un documento fijo con el significado de cada variable.
