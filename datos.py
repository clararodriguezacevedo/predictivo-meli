"""Todo lo que no usa el target, desde las rutas y la carga de los JSON hasta las features.

Se lee en orden. Primero las rutas y constantes, después cómo se aplana cada item del JSON y al final
las features, que salen del propio item o de perfiles calculados sobre train + test.
"""

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


# ----------------------------------------------------------------------------
# Rutas y constantes
# ----------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent

DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
OUTPUTS = RAIZ / "outputs"
DOCS = RAIZ / "docs"

TRAIN_RAW = DATA_RAW / "train_data.jsonlines"
TEST_RAW = DATA_RAW / "test_data.jsonlines"

TARGET = "condition"
CLASE_POSITIVA = "used"
SEED = 42

# ----------------------------------------------------------------------------
# Carga, del JSON crudo a una fila por item
# ----------------------------------------------------------------------------

# Las seis primeras son constantes o vacías (D02); las URL no aportaron como variables (D70, D80)
COLUMNAS_DESCARTADAS = [
    "site_id", "listing_source", "international_delivery_mode", "coverage_areas",
    "differential_pricing", "subtitle", "permalink", "thumbnail", "secure_thumbnail",
]

# Medios de pago por fuera de MercadoPago (id y descripción del JSON)
MEDIOS_DE_PAGO = {
    "MLAMO": "Efectivo",
    "MLATB": "Transferencia bancaria",
    "MLAOT": "Tarjeta de crédito",
    "MLAWC": "Acordar con el comprador",
    "MLAWT": "Giro postal",
    "MLAMP": "MercadoPago",
    "MLAVS": "Visa",
    "MLAMC": "MasterCard",
    "MLAVE": "Visa Electron",
    "MLAMS": "Mastercard Maestro",
    "MLACD": "Contra reembolso",
    "MLAAM": "American Express",
    "MLADC": "Diners",
    "MLABC": "Cheque certificado",
}

# Los ids de foto nuevos terminan en mes y año, como "111301-MLA20304563887_052015"
_RE_FOTO_FECHA = re.compile(r"_(\d{2})(\d{4})$")


def _nombre(d, clave):
    valor = (d or {}).get(clave) or {}
    return valor.get("name") or None


def _tamanio(foto, campo="max_size"):
    """(ancho, alto) de una foto, o (nan, nan) si no está."""
    try:
        ancho, alto = foto.get(campo, "").split("x")
        return int(ancho), int(alto)
    except ValueError:
        return np.nan, np.nan


def _aplanar_fotos(fotos):
    tamanios = [_tamanio(f) for f in fotos]
    anchos = [a for a, _ in tamanios if not np.isnan(a)]
    altos = [h for _, h in tamanios if not np.isnan(h)]
    fechas = [_RE_FOTO_FECHA.search(f.get("id", "")) for f in fotos]
    fechas = [int(m.group(2)) * 12 + int(m.group(1)) - 1 for m in fechas if m]
    # size tiene lado mayor de 500 px como máximo, pero conserva la proporción original
    mostrados = [t for t in (_tamanio(f, "size") for f in fotos) if not np.isnan(t[0])]
    return {
        "n_pictures": len(fotos),
        "pic0_ancho": mostrados[0][0] if mostrados else np.nan,
        "pic0_alto": mostrados[0][1] if mostrados else np.nan,
        "pic_frac_cuadradas": (
            sum(a == h for a, h in mostrados) / len(mostrados) if mostrados else np.nan
        ),
        "pic_n_tamanios": len(set(mostrados)),
        "pic_max_ancho": max(anchos) if anchos else np.nan,
        "pic_max_alto": max(altos) if altos else np.nan,
        "pic_min_lado": min(min(a, h) for a, h in zip(anchos, altos)) if anchos else np.nan,
        "pic_frac_id_fechado": len(fechas) / len(fotos) if fotos else np.nan,
        "pic_mes_mas_viejo": min(fechas) if fechas else np.nan,  # En meses absolutos
    }


def _aplanar_variaciones(variaciones):
    precios = [v.get("price") for v in variaciones if v.get("price") is not None]
    return {
        "n_variations": len(variaciones),
        "var_sold_quantity": sum(v.get("sold_quantity") or 0 for v in variaciones),
        "var_available_quantity": sum(v.get("available_quantity") or 0 for v in variaciones),
        "var_n_precios_distintos": len(set(precios)),
    }


def _a_numero(texto):
    try:
        return float(str(texto).replace(".", "").replace(",", "."))
    except ValueError:
        return np.nan


def _aplanar_atributos(atributos):
    """Valores de la ficha técnica que hablan de la condición del item."""
    # Se buscan por nombre porque los ids cambian entre categorías
    valores = {a.get("name"): (a.get("value_name") or "").strip() for a in atributos}
    unico_duenio = valores.get("Único dueño")
    return {
        "n_attributes": len(atributos),
        "attr_kilometros": _a_numero(valores["Kilómetros"]) if "Kilómetros" in valores else np.nan,
        "attr_anio": _a_numero(valores["Año"]) if "Año" in valores else np.nan,
        "attr_antiguedad": valores.get("Antigüedad") or None,  # "A estrenar", ..., "Más de 40 años"
        "attr_estado": valores.get("Estado del inmueble") or valores.get("Estado") or None,
        "attr_unico_duenio": {"Sí": True, "No": False}.get(unico_duenio),
        "attr_numero_pieza": "Número pieza" in valores or "Número de pieza" in valores,
    }


def aplanar_item(r):
    """Un item crudo (dict) a un dict plano de una sola fila."""
    fila = {
        k: v for k, v in r.items()
        if not isinstance(v, (list, dict)) and k not in COLUMNAS_DESCARTADAS
    }

    fila["seller_state"] = _nombre(r.get("seller_address"), "state")

    envio = r.get("shipping") or {}
    fila["ship_local_pick_up"] = envio.get("local_pick_up")
    fila["ship_free_shipping"] = envio.get("free_shipping")
    fila["ship_mode"] = envio.get("mode")
    fila["ship_has_dimensions"] = envio.get("dimensions") is not None
    fila["ship_has_free_methods"] = bool(envio.get("free_methods"))
    fila["ship_tags"] = "|".join(envio.get("tags") or [])

    medios = {m["id"] for m in r.get("non_mercado_pago_payment_methods") or []}
    fila["n_payment_methods"] = len(medios)
    for medio in MEDIOS_DE_PAGO:
        fila[f"pay_{medio}"] = medio in medios

    fila["tags"] = "|".join(sorted(r.get("tags") or []))
    fila["sub_status"] = "|".join(r.get("sub_status") or [])
    fila["n_deal_ids"] = len(r.get("deal_ids") or [])
    fila["n_descriptions"] = len(r.get("descriptions") or [])

    fila.update(_aplanar_atributos(r.get("attributes") or []))
    fila.update(_aplanar_fotos(r.get("pictures") or []))
    fila.update(_aplanar_variaciones(r.get("variations") or []))
    return fila


def _leer_jsonlines(ruta):
    with open(ruta, encoding="utf8") as f:
        return pd.DataFrame(aplanar_item(json.loads(linea)) for linea in f)


def _tipar(df):
    for col in ["date_created", "last_updated"]:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    for col in ["start_time", "stop_time"]:
        df[col] = pd.to_datetime(df[col], unit="ms")
    for col in ["warranty", "video_id", "parent_item_id"]:
        df[col] = df[col].replace("", None)  # Vienen vacíos en lugar de nulos
    return df


def cargar(split="train", usar_cache=True):
    """DataFrame aplanado de "train" o "test". Se cachea porque parsear el JSON es lento."""
    rutas = {"train": TRAIN_RAW, "test": TEST_RAW}
    cache = DATA_PROCESSED / f"{split}.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)

    df = _tipar(_leer_jsonlines(rutas[split]))
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def cargar_train_test(usar_cache=True):
    return cargar("train", usar_cache), cargar("test", usar_cache)

# ----------------------------------------------------------------------------
# Features sin target (códigos en docs/DECISIONES.md)
# ----------------------------------------------------------------------------

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

    # Reponer stock deja available > initial - sold
    df["iq_es_1"] = df["initial_quantity"] == 1
    df["vendio_algo"] = df["sold_quantity"] > 0
    df["stock_desbalance"] = (
        df["available_quantity"] - (df["initial_quantity"] - df["sold_quantity"])
    )

    # Los comercios usan precios "psicológicos" y los particulares redondean
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

    # Las fotos de catálogo suelen ser cuadradas, las de celular 4:3 o 16:9 (D24)
    df["pic0_aspecto"] = df["pic0_ancho"] / df["pic0_alto"]
    df["pic_tiene_cuadrada"] = df["pic_frac_cuadradas"] > 0
    df["pic_aspecto_camara"] = df["pic0_aspecto"].round(2).isin(ASPECTOS_DE_CAMARA)
    df["pic_lado_max"] = df[["pic_max_ancho", "pic_max_alto"]].max(axis=1)

    # En los campos opcionales, la ausencia informa (D03)
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

    # Ráfagas, cuando un comercio carga muchas publicaciones juntas
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
    """Perfiles sobre train + test, para que también los tengan los vendedores que solo están en test."""
    # El target se descarta antes de calcular, y run_verificar_leakage.py lo comprueba
    todo = pd.concat(
        [train.drop(columns=[TARGET], errors="ignore"), test], ignore_index=True
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
