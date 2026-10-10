"""Evaluación rápida de un conjunto de features con validación cruzada.

Sirve para medir si una feature nueva aporta, no para generar submissions. Usa un
LightGBM con hiperparámetros fijos y validación cruzada estratificada: el EDA
mostró que train y test vienen de la misma distribución, así que el accuracy de
validación es representativo del leaderboard.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

# Columnas que no entran al modelo: identificadores, texto libre, fechas crudas y
# el target. Su información ya está resumida en features derivadas.
NO_SON_FEATURES = [
    config.TARGET, "id", "seller_id", "category_id", "parent_item_id",
    "official_store_id", "catalog_product_id", "title", "warranty", "video_id",
    "attr_ids", "seller_city", "date_created", "last_updated", "start_time", "stop_time",
]
CATEGORICAS = [
    "listing_type_id", "buying_mode", "ship_mode", "status", "sub_status", "tags",
    "ship_tags", "currency_id", "seller_state", "attr_estado", "attr_antiguedad",
]
PARAMETROS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=63, colsample_bytree=0.5,
    subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8,
)


def columnas_modelo(df):
    return [c for c in df.columns if c not in NO_SON_FEATURES]


def preparar(df, categorias=None):
    """Deja el DataFrame listo para LightGBM (categóricas y objetos tipados)."""
    X = df[columnas_modelo(df)].copy()
    for col in CATEGORICAS:
        niveles = categorias[col] if categorias else None
        X[col] = pd.Categorical(X[col], categories=niveles)
    X["attr_unico_duenio"] = X["attr_unico_duenio"].astype(float)
    return X


def y_binario(train):
    return (train[config.TARGET] == config.CLASE_POSITIVA).astype(int)


def validar(X, y, n_folds=5, semillas=(0, 1)):
    """Devuelve (accuracy medio, error estándar) con CV estratificada repetida."""
    aciertos = []
    for semilla in semillas:
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=semilla)
        for i_train, i_val in folds.split(X, y):
            modelo = lgb.LGBMClassifier(**PARAMETROS, random_state=semilla)
            modelo.fit(X.iloc[i_train], y.iloc[i_train])
            aciertos.append((modelo.predict(X.iloc[i_val]) == y.iloc[i_val]).mean())
    return np.mean(aciertos), np.std(aciertos) / np.sqrt(len(aciertos))


def importancias(X, y):
    """Importancia por ganancia, normalizada a que sume 1."""
    modelo = lgb.LGBMClassifier(**PARAMETROS, importance_type="gain", random_state=config.SEED)
    modelo.fit(X, y)
    imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp / imp.sum()
