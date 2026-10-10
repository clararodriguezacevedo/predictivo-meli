"""Modelo completo: features sin target + target encoding con folds + LightGBM.

`entrenar_y_predecir` es la única puerta de entrada, tanto para validar como para
predecir el test. Así la validación cruzada ejercita exactamente el mismo camino
que la predicción final, incluido el target encoding.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import evaluacion
import target_encoding


def entrenar_y_predecir(X_train, train, y, X_pred, a_predecir, semilla=0,
                        usar_target_encoding=True):
    """Entrena con `train` y devuelve la probabilidad de usado para `a_predecir`.

    `X_train` y `X_pred` son las features sin target ya preparadas
    (`evaluacion.preparar`); `train` y `a_predecir` son los DataFrames originales,
    de donde el target encoding toma vendedor, categoría y texto.
    """
    if usar_target_encoding:
        te_train, te_pred = target_encoding.codificar(train, y, a_predecir, semilla)
        X_train = pd.concat([X_train, te_train], axis=1)
        X_pred = pd.concat([X_pred, te_pred], axis=1)
    modelo = lgb.LGBMClassifier(**evaluacion.PARAMETROS, random_state=semilla)
    modelo.fit(X_train, y)
    return modelo.predict_proba(X_pred)[:, 1]


def validar(train, y, n_folds=5, semillas=(0, 1), usar_target_encoding=True):
    """Accuracy de validación cruzada del modelo completo: (media, error estándar).

    En cada fold, la parte de validación no interviene en ningún cálculo que use
    etiquetas: ni en el target encoding ni en el entrenamiento.
    """
    X = evaluacion.preparar(train)
    aciertos = []
    for semilla in semillas:
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=semilla)
        for i_train, i_val in folds.split(X, y):
            proba = entrenar_y_predecir(
                X.iloc[i_train], train.iloc[i_train], y.iloc[i_train],
                X.iloc[i_val], train.iloc[i_val], semilla, usar_target_encoding,
            )
            aciertos.append(((proba > 0.5).astype(int) == y.iloc[i_val]).mean())
    return np.mean(aciertos), np.std(aciertos) / np.sqrt(len(aciertos))
