"""Feature engineering compartido entre el EDA y los modelos.

Ninguna de estas funciones usa el target, así que se pueden calcular sobre train y
test concatenados sin generar leakage. Los perfiles de vendedor y categoría se
calculan justamente así, para que los vendedores que solo aparecen en test también
tengan perfil.
"""

import unicodedata

import numpy as np
import pandas as pd

import config

# Año de referencia del dataset (casi todo se publicó en 2015)
ANIO_DATOS = 2015

# Patrones sobre el título normalizado (minúsculas, sin tildes). "como nuevo",
# "igual a nuevo" y "casi nuevo" se separan de "nuevo" porque significan lo contrario.
PATRONES_TITULO = {
    "t_nuevo": r"(?<!como )(?<!igual a )(?<!casi )\bnuev[oa]s?\b",
    "t_usado": r"\busad[oa]s?\b|segunda mano",
    "t_estado_de_uso": r"(?:como|igual a|casi) nuev|poco uso|impecable|\bestado\b",
    "t_0km_a_estrenar": r"\b0 ?km\b|estrenar",
    "t_sellado_en_caja": r"sellad|blister|en (?:su )?caja",
    "t_mayorista": r"por mayor|x mayor|mayorista|\bx ?\d+ ?(?:u|un|unid|unidades)\b",
    "t_antiguo_coleccion": r"antig|vintage|retro|coleccion|de epoca",
    "t_original": r"\boriginal",
    "t_anio": r"\b(?:19[5-9]\d|200\d|201[0-5])\b",
}

# Patrones sobre el texto de garantía normalizado
PATRONES_GARANTIA = {
    "w_fabrica": r"fabric|oficial|importador",
    "w_plazo": r"\d+\s*(?:mes|ano|dia)",
    "w_sin_garantia": r"sin garant|no tiene|no posee|no ofrece|^no\b",
    "w_solo_si": r"^\s*si\s*\.?\s*$",
    "w_vendedor_o_estado": r"vendedor|reputac|calificac|estado|funciona|probad",
}


def normalizar_texto(serie):
    """Minúsculas y sin tildes, para que los patrones no dependan de la escritura."""
    def _normalizar(texto):
        sin_tildes = unicodedata.normalize("NFKD", texto.lower())
        return sin_tildes.encode("ascii", "ignore").decode()
    return serie.fillna("").map(_normalizar)


def agregar_features_texto(df):
    """Banderas por palabras clave en título y garantía."""
    df = df.copy()
    titulo = normalizar_texto(df["title"])
    for nombre, patron in PATRONES_TITULO.items():
        df[nombre] = titulo.str.contains(patron, regex=True)
    garantia = normalizar_texto(df["warranty"])
    for nombre, patron in PATRONES_GARANTIA.items():
        df[nombre] = garantia.str.contains(patron, regex=True)
    return df


def agregar_features_atributos(df):
    """Features de la ficha técnica de vehículos e inmuebles."""
    df = df.copy()
    df["attr_es_0km"] = df["attr_kilometros"] == 0
    df["attr_anios_de_uso"] = ANIO_DATOS - df["attr_anio"]
    df["attr_a_estrenar"] = df["attr_antiguedad"] == "A estrenar"
    return df


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
    def _por_item(df):
        return agregar_features_atributos(agregar_features_texto(agregar_features_item(df)))
    return agregar_perfiles(_por_item(train), _por_item(test))
