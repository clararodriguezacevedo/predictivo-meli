"""Pseudo-etiquetado: aprovechar items sin etiqueta para entrenar con más datos.

La curva de aprendizaje muestra que el modelo sigue mejorando con más datos, también
en items de vendedores que nunca vio. Los items a predecir (el test) no tienen
etiqueta, pero en la mayoría el modelo está muy seguro. Se hace en dos etapas:

1. Un primer modelo, entrenado solo con las filas etiquetadas, predice todo el
   conjunto sin etiquetas (el "pool").
2. Las predicciones muy seguras del pool se suman al entrenamiento como si fueran
   etiquetas, y un segundo modelo vuelve a predecir.

La predicción final es el promedio de las dos etapas.

Cuidados para que no sea leakage ni autoengaño:

- Nunca se usan etiquetas reales del pool: la función ni siquiera las recibe.
- Cada fila del pool se predice en la segunda etapa con un modelo que NO usó su
  propia pseudo-etiqueta. Para eso el pool se divide en partes y cada parte se
  predice con las pseudo-etiquetas de las otras. Si no, el segundo modelo solo
  repetiría lo que dijo el primero.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import clasificadores
import modelo
import registro
import target_encoding

# Confianza mínima de la primera etapa para usar una predicción como pseudo-etiqueta
UMBRAL = 0.95
N_PARTES = 3


def _ajustar_y_predecir(X_lab, lab, y_lab, X_pred, pred, tipo, parametros, semilla):
    A, B = modelo.armar_matrices(X_lab, lab, y_lab, X_pred, pred, semilla)
    clasificador = clasificadores.crear_clasificador(tipo, parametros, semilla)
    return clasificador.fit(A, y_lab).predict_proba(B)[:, 1]


def predecir(X_lab, lab, y, X_pool, pool, tipo, parametros, semilla):
    """Probabilidad de usado para cada fila del pool, con pseudo-etiquetado.

    `X_lab`/`lab`/`y` son las filas etiquetadas; `X_pool`/`pool`, las filas sin
    etiqueta. Las matrices `X_*` tienen que venir de `modelo.preparar_juntos` para
    que las categóricas compartan niveles.
    """
    X_lab, lab, y = (o.reset_index(drop=True) for o in (X_lab, lab, y))
    X_pool, pool = X_pool.reset_index(drop=True), pool.reset_index(drop=True)

    primera = _ajustar_y_predecir(X_lab, lab, y, X_pool, pool, tipo, parametros, semilla)
    segura = np.maximum(primera, 1 - primera) >= UMBRAL
    pseudo = pd.Series((primera > 0.5).astype(int))

    segunda = np.empty(len(pool))
    partes = KFold(N_PARTES, shuffle=True, random_state=semilla).split(pool)
    for i_otras, i_parte in partes:
        usar = i_otras[segura[i_otras]]
        segunda[i_parte] = _ajustar_y_predecir(
            pd.concat([X_lab, X_pool.iloc[usar]], ignore_index=True),
            pd.concat([lab, pool.iloc[usar]], ignore_index=True),
            pd.concat([y, pseudo.iloc[usar]], ignore_index=True),
            X_pool.iloc[i_parte], pool.iloc[i_parte], tipo, parametros, semilla,
        )
    return (primera + segunda) / 2


def descripcion(activo, filas_extra_en_el_pool=0):
    """Configuración del pseudo-etiquetado, para dejarla anotada en el registro."""
    if not activo:
        return {"activo": False}
    return {"activo": True, "umbral": UMBRAL, "n_partes": N_PARTES,
            "filas_de_test_en_el_pool_de_validacion": filas_extra_en_el_pool}


# En la validación, el pool sin etiquetas es la parte de validación más esta cantidad
# de filas del test real. Así el pool pesa respecto del entrenamiento lo mismo que
# en la predicción final (30.000 filas de test contra 70.000 de train).
FILAS_DE_TEST_EN_VALIDACION = 10000


def validar(train, y, test, tipo, parametros, protocolo=None, registrar=True, nota=""):
    """Validación cruzada del modelo con pseudo-etiquetado: (accuracy, error estándar).

    En cada fold la parte de validación hace de test: entra al pool sin su etiqueta,
    junto con filas del test real. Usa los mismos folds que `modelo.validar`, así
    que el resultado es comparable.
    """
    protocolo = protocolo or modelo.PROTOCOLO_VALIDACION
    X_train, X_test = modelo.preparar_juntos(train, test)
    inicio = time.time()
    aciertos = []
    for semilla in protocolo["semillas"]:
        rng = np.random.default_rng(semilla)
        particion = StratifiedKFold(protocolo["n_folds"], shuffle=True, random_state=semilla)
        for i_train, i_val in particion.split(X_train, y):
            i_test = rng.choice(len(test), FILAS_DE_TEST_EN_VALIDACION, replace=False)
            proba = predecir(
                X_train.iloc[i_train], train.iloc[i_train], y.iloc[i_train],
                pd.concat([X_train.iloc[i_val], X_test.iloc[i_test]]),
                pd.concat([train.iloc[i_val], test.iloc[i_test]]),
                tipo, parametros, semilla,
            )[:len(i_val)]
            aciertos.append(float(((proba > 0.5).astype(int) == y.iloc[i_val].to_numpy()).mean()))
            print(f"fold {len(aciertos)}: {aciertos[-1]:.4f}", flush=True)
    accuracy, error = modelo.resumir(aciertos)
    if registrar:
        columnas = list(X_train.columns) + [
            c for e in target_encoding.crear_encoders()
            for c in e.fit(train.head(200), y.head(200)).transform(train.head(5)).columns
        ]
        registro.registrar(
            "validacion", tipo, parametros, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio), nota=nota,
            protocolo=protocolo, target_encoding=target_encoding.descripcion(),
            features=registro.describir_features(columnas),
            pseudo_etiquetas=descripcion(True, FILAS_DE_TEST_EN_VALIDACION),
        )
    return accuracy, error
