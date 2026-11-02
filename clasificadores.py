"""Clasificadores disponibles y sus hiperparámetros por defecto.

Todos reciben la misma matriz de features (con columnas categóricas de pandas,
booleanas y nulos). Los que no la aceptan tal cual se envuelven con el
preprocesamiento que necesitan, así se pueden comparar sobre los mismos datos.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

HILOS = 8

PARAMETROS_POR_DEFECTO = {
    "lightgbm": dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63, colsample_bytree=0.5,
        subsample=0.8, subsample_freq=1,
    ),
    "xgboost": dict(
        n_estimators=400, learning_rate=0.05, max_depth=8, colsample_bytree=0.5,
        subsample=0.8,
    ),
    "catboost": dict(iterations=1000, learning_rate=0.05, depth=8),
    "hist_gradient_boosting": dict(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, max_features=0.5,
    ),
    "random_forest": dict(n_estimators=500, max_features=0.3, min_samples_leaf=2),
    "extra_trees": dict(n_estimators=500, max_features=0.5, min_samples_leaf=2),
    "regresion_logistica": dict(C=1.0, max_iter=3000),
}


def _categoricas(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


class ConCodigos:
    """Envuelve modelos de árboles de scikit-learn que no aceptan categóricas ni nulos.

    Reemplaza cada categoría por su código entero y los nulos por -1. Un árbol puede
    separar ese valor del resto, así que "no tiene dato" sigue siendo información.
    """

    def __init__(self, modelo):
        self.modelo = modelo

    @staticmethod
    def _convertir(X):
        X = X.copy()
        for col in _categoricas(X):
            X[col] = X[col].cat.codes
        return X.astype(float).fillna(-1)

    def fit(self, X, y):
        self.modelo.fit(self._convertir(X), y)
        return self

    def predict_proba(self, X):
        return self.modelo.predict_proba(self._convertir(X))

    def predict(self, X):
        return self.modelo.predict(self._convertir(X))


class ConTextoEnCategoricas:
    """Envuelve CatBoost, que espera las categóricas como texto y sin nulos."""

    def __init__(self, modelo):
        self.modelo = modelo

    @staticmethod
    def _convertir(X):
        X = X.copy()
        for col in _categoricas(X):
            X[col] = X[col].astype(object).where(X[col].notna(), "(nulo)").astype(str)
        booleanas = X.select_dtypes("bool").columns
        X[booleanas] = X[booleanas].astype(int)
        return X

    def fit(self, X, y):
        self.modelo.fit(self._convertir(X), y, cat_features=_categoricas(X))
        return self

    def predict_proba(self, X):
        return self.modelo.predict_proba(self._convertir(X))

    def predict(self, X):
        return self.modelo.predict(self._convertir(X)).astype(int)


class Ensamble:
    """Promedio de las probabilidades de varios modelos.

    `parametros` es un diccionario {tipo de modelo: sus hiperparámetros}.
    """

    def __init__(self, parametros, semilla):
        self.modelos = [crear_clasificador(tipo, p, semilla) for tipo, p in parametros.items()]

    def fit(self, X, y):
        for m in self.modelos:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _regresion_logistica(parametros, semilla):
    """Modelo lineal: one-hot para categóricas, imputación y escalado para el resto."""
    numericas = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        # Lleva cada variable a una distribución normal: las cantidades y precios
        # tienen colas muy largas y un escalado común no alcanza
        QuantileTransformer(output_distribution="normal", random_state=semilla),
    )
    preprocesado = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_include="category")),
        ("numericas", numericas, make_column_selector(dtype_exclude="category")),
    ])
    return make_pipeline(preprocesado, LogisticRegression(**parametros, random_state=semilla))


def crear_clasificador(tipo, parametros, semilla):
    if tipo == "lightgbm":
        return lgb.LGBMClassifier(**parametros, random_state=semilla, n_jobs=HILOS, verbose=-1)
    if tipo == "xgboost":
        return xgb.XGBClassifier(
            **parametros, random_state=semilla, n_jobs=HILOS, tree_method="hist",
            enable_categorical=True,
        )
    if tipo == "catboost":
        return ConTextoEnCategoricas(CatBoostClassifier(
            **parametros, random_seed=semilla, thread_count=HILOS, verbose=0,
            allow_writing_files=False,
            # Las categóricas tienen pocos niveles: one-hot es mucho más rápido que la
            # codificación por estadísticas del target que CatBoost usa por defecto
            one_hot_max_size=255,
        ))
    if tipo == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **parametros, random_state=semilla, categorical_features="from_dtype",
            early_stopping=False,
        )
    if tipo == "random_forest":
        return ConCodigos(RandomForestClassifier(**parametros, random_state=semilla, n_jobs=HILOS))
    if tipo == "extra_trees":
        return ConCodigos(ExtraTreesClassifier(**parametros, random_state=semilla, n_jobs=HILOS))
    if tipo == "regresion_logistica":
        return _regresion_logistica(parametros, semilla)
    if tipo == "ensamble":
        return Ensamble(parametros, semilla)
    raise ValueError(f"Modelo desconocido: {tipo}")
