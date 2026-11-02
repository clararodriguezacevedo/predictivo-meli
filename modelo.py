"""Modelo completo: features sin target + target encoding con folds + boosting.

Los clasificadores disponibles están en `clasificadores.py`. El armado de los datos de
cada fold (`armar_folds`) está separado del entrenamiento (`validar_en_folds`) porque el target encoding es lo más
caro y no depende de los hiperparámetros: en una búsqueda se arma una sola vez y se
reutiliza para todas las pruebas.

`armar_matrices` es la única función que construye las features con target, tanto
para validar como para predecir el test. Así la validación ejercita exactamente el
mismo camino que la predicción final.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import evaluacion
from clasificadores import PARAMETROS_POR_DEFECTO, crear_clasificador
import registro
import target_encoding

# Protocolo estándar: todas las corridas de tipo "validacion" lo usan, por eso son
# comparables entre sí
PROTOCOLO_VALIDACION = {"n_folds": 5, "semillas": [0, 1]}
# Protocolo de la búsqueda de hiperparámetros: más liviano y con otra semilla, para
# no elegir hiperparámetros mirando los mismos folds con los que después se valida
PROTOCOLO_BUSQUEDA = {"n_folds": 4, "semillas": [100]}

def armar_matrices(X_train, train, y, X_pred, a_predecir, semilla):
    """Devuelve las matrices finales de entrenamiento y de predicción.

    `X_train` y `X_pred` son las features sin target ya preparadas
    (`evaluacion.preparar`); `train` y `a_predecir` son los DataFrames originales,
    de donde el target encoding toma vendedor, categoría y texto. Las etiquetas de
    `a_predecir` no intervienen.
    """
    te_train, te_pred = target_encoding.codificar(train, y, a_predecir, semilla)
    return pd.concat([X_train, te_train], axis=1), pd.concat([X_pred, te_pred], axis=1)


def armar_folds(train, y, protocolo):
    """Arma los datos de cada fold: lista de (X_train, y_train, X_val, y_val, semilla).

    En cada fold, la parte de validación no interviene en ningún cálculo que use
    etiquetas.
    """
    X = evaluacion.preparar(train)
    folds = []
    for semilla in protocolo["semillas"]:
        particion = StratifiedKFold(protocolo["n_folds"], shuffle=True, random_state=semilla)
        for i_train, i_val in particion.split(X, y):
            X_tr, X_val = armar_matrices(
                X.iloc[i_train], train.iloc[i_train], y.iloc[i_train],
                X.iloc[i_val], train.iloc[i_val], semilla,
            )
            folds.append((X_tr, y.iloc[i_train], X_val, y.iloc[i_val], semilla))
    return folds


def validar_en_folds(folds, tipo, parametros):
    """Entrena y evalúa en folds ya armados. Devuelve el accuracy de cada fold."""
    aciertos = []
    for X_tr, y_tr, X_val, y_val, semilla in folds:
        clasificador = crear_clasificador(tipo, parametros, semilla).fit(X_tr, y_tr)
        aciertos.append(float((clasificador.predict(X_val) == y_val).mean()))
    return aciertos


def resumir(aciertos):
    return float(np.mean(aciertos)), float(np.std(aciertos) / np.sqrt(len(aciertos)))


def descripcion_comun(folds, protocolo):
    """Detalles de la corrida que van al registro, además del modelo y sus parámetros."""
    return {
        "protocolo": protocolo,
        "target_encoding": target_encoding.descripcion(),
        "features": registro.describir_features(folds[0][0].columns),
    }


def validar(train, y, tipo="lightgbm", parametros=None, protocolo=PROTOCOLO_VALIDACION,
            registrar=True, nota=""):
    """Validación cruzada del modelo completo. Devuelve (accuracy, error estándar).

    Con el protocolo estándar la corrida se registra como "validacion".
    """
    parametros = parametros or PARAMETROS_POR_DEFECTO[tipo]
    inicio = time.time()
    folds = armar_folds(train, y, protocolo)
    aciertos = validar_en_folds(folds, tipo, parametros)
    accuracy, error = resumir(aciertos)
    if registrar:
        registro.registrar(
            "validacion", tipo, parametros, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            nota=nota, **descripcion_comun(folds, protocolo),
        )
    return accuracy, error


def predecir_en_folds(folds, tipo, parametros):
    """Predicción de validación de cada fila, hecha por un modelo que no la vio.

    Devuelve un DataFrame con la probabilidad de usado, la etiqueta real y las
    features de target encoding que recibió la fila al ser validada. Sirve para
    analizar errores. Si el protocolo tiene varias semillas, cada fila aparece una
    vez por semilla.
    """
    partes = []
    for X_tr, y_tr, X_val, y_val, semilla in folds:
        clasificador = crear_clasificador(tipo, parametros, semilla).fit(X_tr, y_tr)
        columnas_te = [c for c in X_val.columns if c.startswith("te_")]
        parte = X_val[columnas_te].copy()
        parte["proba_usado"] = clasificador.predict_proba(X_val)[:, 1]
        parte["y"] = y_val
        parte["semilla"] = semilla
        partes.append(parte)
    return pd.concat(partes)


def preparar_juntos(train, test):
    """Matrices de features sin target de train y test, con los mismos niveles en las categóricas.

    Hace falta para predecir el test.
    """
    niveles = {
        c: sorted(set(train[c].dropna()) | set(test[c].dropna())) for c in evaluacion.CATEGORICAS
    }
    return evaluacion.preparar(train, niveles), evaluacion.preparar(test, niveles)
