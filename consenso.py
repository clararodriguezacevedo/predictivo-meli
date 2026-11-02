"""Consenso del vendedor: qué opina un modelo sobre los otros items del mismo vendedor.

Los vendedores suelen vender solo nuevos o solo usados. Para un vendedor que no
tiene items con etiqueta (por ejemplo, uno que solo aparece en test) el target
encoding no aporta nada, pero sus otras publicaciones igual dicen mucho: si un
modelo cree que 9 de sus 10 items son usados, el décimo probablemente también.

Se hace en dos etapas:

1. Un modelo de primera etapa, entrenado solo con features sin target, predice
   todos los items (los de entrenamiento y los que hay que predecir).
2. Para cada item se promedian esas predicciones sobre los *otros* items de su
   vendedor. Ese promedio entra como feature al modelo final.

Para no filtrar el target hay que cuidar quién hace cada predicción de la primera
etapa. Las features de perfil identifican al vendedor casi como una huella
digital, así que un modelo que vio otros items del mismo vendedor "recuerda" sus
etiquetas. Por eso los folds de la primera etapa se arman agrupando por vendedor
y cada item lo predice un modelo que nunca vio a su vendedor:

- Items de entrenamiento: el modelo del fold en que su vendedor quedó afuera.
- Items a predecir cuyo vendedor está en entrenamiento: ese mismo modelo.
- Items a predecir de vendedores nuevos: el promedio de todos los modelos.

Así el consenso de un item no depende de su etiqueta ni de la de ningún item de su
vendedor, y significa lo mismo en entrenamiento, validación y test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import clasificadores

N_FOLDS = 5
TIPO_PRIMERA_ETAPA = "lightgbm"


def folds_por_vendedor(train):
    """Asigna cada fila de entrenamiento a un fold según su vendedor. No mira etiquetas."""
    particion = GroupKFold(N_FOLDS).split(train, groups=train["seller_id"])
    fold = np.empty(len(train), dtype=int)
    for k, (_, i_fold) in enumerate(particion):
        fold[i_fold] = k
    return fold


def _predicciones_sin_conocer_al_vendedor(X_train, train, y, X_pred, a_predecir, semilla):
    fold = folds_por_vendedor(train)
    fold_del_vendedor = pd.Series(fold, index=train["seller_id"].to_numpy())
    fold_del_vendedor = fold_del_vendedor[~fold_del_vendedor.index.duplicated()]
    fold_pred = fold_del_vendedor.reindex(a_predecir["seller_id"].to_numpy()).to_numpy()

    parametros = clasificadores.PARAMETROS_POR_DEFECTO[TIPO_PRIMERA_ETAPA]
    proba_train = np.empty(len(train))
    proba_pred_por_fold = np.empty((len(a_predecir), N_FOLDS))
    for k in range(N_FOLDS):
        afuera = fold == k
        clasificador = clasificadores.crear_clasificador(TIPO_PRIMERA_ETAPA, parametros, semilla)
        clasificador.fit(X_train[~afuera], y[~afuera])
        proba_train[afuera] = clasificador.predict_proba(X_train[afuera])[:, 1]
        proba_pred_por_fold[:, k] = clasificador.predict_proba(X_pred)[:, 1]

    vendedor_nuevo = np.isnan(fold_pred)
    proba_pred = proba_pred_por_fold.mean(axis=1)
    conocidos = np.where(~vendedor_nuevo)[0]
    proba_pred[conocidos] = proba_pred_por_fold[conocidos, fold_pred[conocidos].astype(int)]
    return proba_train, proba_pred


def codificar(X_train, train, y, X_pred, a_predecir, semilla):
    """Features de consenso para las filas de entrenamiento y las filas a predecir."""
    proba_train, proba_pred = _predicciones_sin_conocer_al_vendedor(
        X_train, train, y, X_pred, a_predecir, semilla
    )
    todo = pd.DataFrame({
        "seller_id": np.concatenate([train["seller_id"].to_numpy(),
                                     a_predecir["seller_id"].to_numpy()]),
        "proba": np.concatenate([proba_train, proba_pred]),
    })
    g = todo.groupby("seller_id")["proba"]
    suma, n = g.transform("sum"), g.transform("size")
    otros = n - 1
    # Media de los otros items del vendedor: se excluye la predicción del propio item
    media = ((suma - todo["proba"]) / otros).where(otros > 0)
    consenso = pd.DataFrame({
        "consenso_vendedor": media.to_numpy(),
        "consenso_vendedor_n": otros.to_numpy(),
        # Qué tan lejos está la predicción propia de la del resto del vendedor
        "consenso_vendedor_diferencia": (todo["proba"] - media).to_numpy(),
    })
    c_train = consenso.iloc[:len(train)].set_index(train.index)
    c_pred = consenso.iloc[len(train):].set_index(a_predecir.index)
    return c_train, c_pred


def descripcion(activo):
    """Configuración del consenso, para dejarla anotada en el registro."""
    if not activo:
        return {"activo": False}
    return {
        "activo": True, "n_folds": N_FOLDS, "primera_etapa": TIPO_PRIMERA_ETAPA,
        "parametros_primera_etapa": clasificadores.PARAMETROS_POR_DEFECTO[TIPO_PRIMERA_ETAPA],
    }
