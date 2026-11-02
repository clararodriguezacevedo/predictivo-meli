"""Preparación de la matriz de features para los modelos.

Elige qué columnas entran al modelo y tipa las categóricas. La validación cruzada
del modelo completo está en `modelo.py`.
"""

import pandas as pd

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
