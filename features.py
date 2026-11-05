"""Features sin target: del item y perfiles calculados sobre train + test. Códigos en docs/DECISIONES.md."""

import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import config

# Casi todo se publicó en 2015
ANIO_DATOS = 2015

# Proporciones de cámara (4:3 y 16:9, horizontales o verticales), a 2 decimales
ASPECTOS_DE_CAMARA = [1.33, 1.34, 0.75, 0.74, 1.78, 1.79, 0.56]

# Sobre el título normalizado; "como nuevo" se separa de "nuevo" porque significa lo contrario (D20)
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

# Sobre el texto de garantía normalizado (D20)
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
    """Ficha técnica de vehículos e inmuebles (D18)."""
    df = df.copy()
    df["attr_es_0km"] = df["attr_kilometros"] == 0
    df["attr_anios_de_uso"] = ANIO_DATOS - df["attr_anio"]
    df["attr_a_estrenar"] = df["attr_antiguedad"] == "A estrenar"
    return df


def agregar_features_item(df):
    """Features de un solo item, sin mirar otras filas (D19)."""
    df = df.copy()

    # Stock: reponer deja available > initial - sold
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
    # Unos segundos de más en la duración son mucho más comunes en usados (D25)
    df["duracion_resto_seg"] = (df["stop_time"] - df["start_time"]).dt.total_seconds() % 86400
    df["dias_update_a_fin"] = (df["stop_time"] - df["last_updated"]).dt.total_seconds() / 86400  # D26
    df["seg_creacion_a_inicio"] = (df["start_time"] - df["date_created"]).dt.total_seconds()  # D26
    df["dia_creacion"] = (df["date_created"] - pd.Timestamp(f"{ANIO_DATOS}-01-01")).dt.days  # D26
    # Las fechas vienen en UTC, se pasan a hora argentina
    creacion_ar = df["date_created"] - pd.Timedelta(hours=3)
    df["hora_creacion"] = creacion_ar.dt.hour
    df["dia_semana"] = creacion_ar.dt.dayofweek
    df["hora_update"] = (df["last_updated"] - pd.Timedelta(hours=3)).dt.hour  # D26

    # Texto
    df["title_len"] = df["title"].str.len()
    df["title_n_digitos"] = df["title"].str.count(r"\d")
    df["title_frac_mayus"] = df["title"].str.count(r"[A-Z]") / df["title_len"].clip(lower=1)
    df["warranty_len"] = df["warranty"].fillna("").str.len()
    df["tiene_warranty"] = df["warranty"].notna()  # La ausencia informa (D03)

    # Fotos: las de catálogo suelen ser cuadradas, las de celular 4:3 o 16:9 (D24)
    df["pic0_aspecto"] = df["pic0_ancho"] / df["pic0_alto"]
    df["pic_tiene_cuadrada"] = df["pic_frac_cuadradas"] > 0
    df["pic_aspecto_camara"] = df["pic0_aspecto"].round(2).isin(ASPECTOS_DE_CAMARA)
    df["pic_lado_max"] = df[["pic_max_ancho", "pic_max_alto"]].max(axis=1)

    # Campos opcionales: la ausencia informa (D03)
    df["tiene_video"] = df["video_id"].notna()
    df["es_tienda_oficial"] = df["official_store_id"].notna()
    df["tiene_parent"] = df["parent_item_id"].notna()
    return df


# Describen cómo publica el vendedor, no qué condición declaró
INDICADORES = {
    "free": lambda df: df["listing_type_id"] == "free",
    "bronze": lambda df: df["listing_type_id"] == "bronze",
    "iq_1": lambda df: df["iq_es_1"],
    "vendio": lambda df: df["vendio_algo"],
    "warranty": lambda df: df["tiene_warranty"],
    "price_decimales": lambda df: df["price_con_decimales"],
    "repuso_stock": lambda df: df["stock_desbalance"] > 0,
    "foto_cuadrada": lambda df: df["pic_tiene_cuadrada"],
    "pago_tarjeta": lambda df: df["pay_MLAOT"],
    "t_nuevo": lambda df: df["t_nuevo"],
    "t_estado_de_uso": lambda df: df["t_estado_de_uso"],
    "t_antiguo_coleccion": lambda df: df["t_antiguo_coleccion"],
    "w_plazo": lambda df: df["w_plazo"],
}
INDICADORES_CATEGORIA = [
    "free", "iq_1", "vendio", "warranty", "price_decimales", "repuso_stock",
    "foto_cuadrada", "t_nuevo", "t_estado_de_uso", "t_antiguo_coleccion", "w_plazo",
]
INDICADORES_PALABRA = ["free", "iq_1", "price_decimales", "repuso_stock", "foto_cuadrada"]
# Con menos títulos, el perfil de una palabra es puro ruido
MIN_TITULOS_POR_PALABRA = 5


def _indicadores(df):
    return pd.DataFrame({n: f(df).astype(float) for n, f in INDICADORES.items()}, index=df.index)


def _perfil_vendedor(todo, indicadores):
    """Cómo publica cada vendedor (D21)."""
    g = todo.groupby("seller_id")
    perfil = indicadores.groupby(todo["seller_id"]).mean().add_prefix("seller_frac_")
    perfil["seller_n_items"] = g.size()
    perfil["seller_mediana_log_price"] = g["log_price"].median()
    perfil["seller_std_log_price"] = g["log_price"].std()
    perfil["seller_n_categorias"] = g["category_id"].nunique()
    perfil["seller_ratio_categorias"] = perfil["seller_n_categorias"] / perfil["seller_n_items"]
    perfil["seller_max_initial_quantity"] = g["initial_quantity"].max()
    perfil["seller_media_n_pictures"] = g["n_pictures"].mean()
    perfil["seller_n_garantias_distintas"] = g["warranty"].nunique()
    perfil["seller_dias_activos"] = g["date_created"].agg(lambda s: s.dt.date.nunique())
    perfil["seller_span_dias"] = (g["date_created"].max() - g["date_created"].min()).dt.days
    return perfil


def _perfil_categoria(todo, indicadores):
    """Cómo se publica en cada categoría (D22)."""
    g = todo.groupby("category_id")
    perfil = (indicadores[INDICADORES_CATEGORIA].groupby(todo["category_id"]).mean()
              .add_prefix("cat_frac_"))
    perfil["cat_n_items"] = g.size()
    perfil["cat_mediana_log_price"] = g["log_price"].median()
    perfil["cat_n_sellers"] = g["seller_id"].nunique()
    perfil["cat_ratio_sellers"] = perfil["cat_n_sellers"] / perfil["cat_n_items"]
    return perfil


def _perfil_palabras(todo, indicadores):
    """Qué tipo de producto es, según cómo se publican los títulos con las mismas palabras (D23)."""
    vectorizador = CountVectorizer(
        binary=True, min_df=MIN_TITULOS_POR_PALABRA, token_pattern=r"[a-z0-9]+"
    )
    X = vectorizador.fit_transform(normalizar_texto(todo["title"])).tocoo()
    titulos_por_palabra = np.bincount(X.col, minlength=X.shape[1])
    palabras_por_titulo = np.bincount(X.row, minlength=X.shape[0])

    perfil = pd.DataFrame(index=todo.index)
    for nombre in INDICADORES_PALABRA:
        valor = indicadores[nombre].to_numpy()
        suma_por_palabra = np.bincount(X.col, weights=valor[X.row], minlength=X.shape[1])
        # Se excluye al propio item, para que no se describa a sí mismo
        media_sin_el_item = (suma_por_palabra[X.col] - valor[X.row]) / (
            titulos_por_palabra[X.col] - 1
        )
        M = sp.csr_matrix((media_sin_el_item, (X.row, X.col)), shape=X.shape)
        con_palabras = palabras_por_titulo > 0
        media = np.full(X.shape[0], np.nan)
        media[con_palabras] = (
            np.asarray(M.sum(axis=1)).ravel()[con_palabras] / palabras_por_titulo[con_palabras]
        )
        maximo = np.where(con_palabras, M.max(axis=1).toarray().ravel(), np.nan)
        perfil[f"pal_frac_{nombre}_media"] = media
        perfil[f"pal_frac_{nombre}_max"] = maximo
    return perfil


def _contexto_por_item(todo):
    """Features de cada item que dependen de las otras publicaciones."""
    ctx = pd.DataFrame(index=todo.index)
    titulo = normalizar_texto(todo["title"])
    ctx["titulo_repetido"] = titulo.map(titulo.value_counts())

    # Ráfagas: un comercio carga muchas publicaciones juntas
    vendedor, creado = todo["seller_id"], todo["date_created"]
    ctx["seller_items_mismo_dia"] = (
        todo.groupby([vendedor, creado.dt.date])["id"].transform("size")
    )
    ctx["seller_items_mismo_minuto"] = (
        todo.groupby([vendedor, creado.dt.floor("min")])["id"].transform("size")
    )
    orden = todo.sort_values("date_created")
    ctx["seller_seg_desde_previo"] = (
        orden.groupby("seller_id")["date_created"].diff().dt.total_seconds()
    )
    return ctx


def agregar_perfiles(train, test):
    """Perfiles sobre train + test: así también los tienen los vendedores que solo están en test."""
    # El target se descarta antes de calcular: run_verificar_leakage.py lo comprueba
    todo = pd.concat(
        [train.drop(columns=[config.TARGET], errors="ignore"), test], ignore_index=True
    )
    indicadores = _indicadores(todo)
    perfil_vendedor = _perfil_vendedor(todo, indicadores)
    perfil_categoria = _perfil_categoria(todo, indicadores)
    contexto = pd.concat(
        [_contexto_por_item(todo), _perfil_palabras(todo, indicadores)], axis=1
    ).set_index(todo["id"])

    def _unir(df):
        df = (df.join(perfil_vendedor, on="seller_id")
                .join(perfil_categoria, on="category_id")
                .join(contexto, on="id"))
        df["log_price_rel_categoria"] = df["log_price"] - df["cat_mediana_log_price"]
        return df

    return _unir(train), _unir(test)


def construir(train, test):
    """Pipeline completo de features para train y test."""
    def _por_item(df):
        return agregar_features_atributos(agregar_features_texto(agregar_features_item(df)))
    return agregar_perfiles(_por_item(train), _por_item(test))
