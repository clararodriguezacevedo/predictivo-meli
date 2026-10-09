"""Feature engineering compartido entre el EDA y los modelos.

Ninguna de estas funciones usa el target, así que se pueden calcular sobre train y
test concatenados sin generar leakage. Los perfiles de vendedor y categoría se
calculan justamente así, para que los vendedores que solo aparecen en test también
tengan perfil.
"""

import numpy as np
import pandas as pd

import config


def agregar_features_item(df):
    """Features derivadas de un único item (no miran otras filas)."""
    df = df.copy()

    # Stock: un vendedor que repone stock deja available > initial - sold
    df["iq_es_1"] = df["initial_quantity"] == 1
    df["vendio_algo"] = df["sold_quantity"] > 0
    df["stock_desbalance"] = (
        df["available_quantity"] - (df["initial_quantity"] - df["sold_quantity"])
    )

    # Precio: los comercios usan precios "psicológicos", los particulares redondean
    df["log_price"] = np.log1p(df["price"])
    df["price_con_decimales"] = df["price"] % 1 != 0
    df["price_termina_en_9"] = df["price"].astype(np.int64) % 10 == 9
    df["price_multiplo_100"] = df["price"] % 100 == 0

    # Fechas
    df["dias_hasta_update"] = (
        (df["last_updated"] - df["date_created"]).dt.total_seconds() / 86400
    )
    df["duracion_dias"] = (df["stop_time"] - df["start_time"]).dt.days
    # Las fechas vienen en UTC, se pasan a hora argentina (UTC-3)
    creacion_ar = df["date_created"] - pd.Timedelta(hours=3)
    df["hora_creacion"] = creacion_ar.dt.hour
    df["dia_semana"] = creacion_ar.dt.dayofweek

    # Texto
    df["title_len"] = df["title"].str.len()
    df["title_frac_mayus"] = df["title"].str.count(r"[A-Z]") / df["title_len"].clip(lower=1)
    df["warranty_len"] = df["warranty"].fillna("").str.len()
    df["tiene_warranty"] = df["warranty"].notna()

    # Presencia de campos opcionales
    df["tiene_video"] = df["video_id"].notna()
    df["es_tienda_oficial"] = df["official_store_id"].notna()
    df["tiene_parent"] = df["parent_item_id"].notna()
    return df


def _perfil(df, clave, prefijo):
    g = df.groupby(clave)
    perfil = pd.DataFrame({
        "n_items": g.size(),
        "frac_free": g["listing_type_id"].agg(lambda s: (s == "free").mean()),
        "frac_iq_1": g["iq_es_1"].mean(),
        "frac_vendio": g["vendio_algo"].mean(),
        "frac_warranty": g["tiene_warranty"].mean(),
        "frac_price_decimales": g["price_con_decimales"].mean(),
        "mediana_log_price": g["log_price"].median(),
        "n_categorias": g["category_id"].nunique(),
    })
    return perfil.add_prefix(prefijo)


def agregar_perfiles(train, test):
    """Agrega perfiles agregados por vendedor y categoría, calculados sobre train + test.

    Espera DataFrames ya pasados por `agregar_features_item`.
    """
    todo = pd.concat([train.drop(columns=[config.TARGET], errors="ignore"), test])
    perfil_vendedor = _perfil(todo, "seller_id", "seller_")
    perfil_vendedor["seller_ratio_categorias"] = (
        perfil_vendedor["seller_n_categorias"] / perfil_vendedor["seller_n_items"]
    )
    perfil_categoria = _perfil(todo, "category_id", "cat_")[
        ["cat_n_items", "cat_frac_free", "cat_frac_iq_1", "cat_mediana_log_price"]
    ]

    def _unir(df):
        df = df.join(perfil_vendedor, on="seller_id").join(perfil_categoria, on="category_id")
        df["log_price_rel_categoria"] = df["log_price"] - df["cat_mediana_log_price"]
        return df

    return _unir(train), _unir(test)


def construir(train, test):
    """Pipeline completo de features para train y test."""
    return agregar_perfiles(agregar_features_item(train), agregar_features_item(test))
