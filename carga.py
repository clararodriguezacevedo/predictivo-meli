"""Carga de los jsonlines crudos y aplanado a un DataFrame tabular.

El aplanado solo extrae información de los campos anidados (listas y dicts) sin
hacer feature engineering: eso vive en `features`. El resultado se cachea en
parquet para no re-parsear los JSON en cada corrida.
"""

import json
import re

import numpy as np
import pandas as pd

import config

# Columnas que en el train son constantes o 100% nulas: no aportan nada
COLUMNAS_DESCARTADAS = [
    "site_id", "listing_source", "international_delivery_mode", "coverage_areas",
    "differential_pricing", "subtitle", "permalink", "thumbnail", "secure_thumbnail",
]

# Medios de pago "no MercadoPago" que aparecen en los datos (id y descripción del JSON)
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

# Ids de foto nuevos: "111301-MLA20304563887_052015" (termina en MMYYYY)
_RE_FOTO_FECHA = re.compile(r"_(\d{2})(\d{4})$")


def _nombre(d, clave):
    valor = (d or {}).get(clave) or {}
    return valor.get("name") or None


def _tamanio_max(foto):
    """Devuelve (ancho, alto) del max_size de una foto, o (nan, nan)."""
    try:
        ancho, alto = foto.get("max_size", "").split("x")
        return int(ancho), int(alto)
    except ValueError:
        return np.nan, np.nan


def _aplanar_fotos(fotos):
    tamanios = [_tamanio_max(f) for f in fotos]
    anchos = [a for a, _ in tamanios if not np.isnan(a)]
    altos = [h for _, h in tamanios if not np.isnan(h)]
    fechas = [_RE_FOTO_FECHA.search(f.get("id", "")) for f in fotos]
    fechas = [int(m.group(2)) * 12 + int(m.group(1)) - 1 for m in fechas if m]
    return {
        "n_pictures": len(fotos),
        "pic_max_ancho": max(anchos) if anchos else np.nan,
        "pic_max_alto": max(altos) if altos else np.nan,
        "pic_min_lado": min(min(a, h) for a, h in zip(anchos, altos)) if anchos else np.nan,
        # Fracción de fotos con id en formato nuevo (incluye mes y año de subida)
        "pic_frac_id_fechado": len(fechas) / len(fotos) if fotos else np.nan,
        # Mes de la foto más vieja, como meses absolutos (anio * 12 + mes)
        "pic_mes_mas_viejo": min(fechas) if fechas else np.nan,
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
    """Extrae los valores de la ficha técnica que hablan de la condición del item.

    Solo existen en vehículos, inmuebles y repuestos. Se buscan por nombre porque
    los ids cambian entre categorías.
    """
    valores = {a.get("name"): (a.get("value_name") or "").strip() for a in atributos}
    unico_duenio = valores.get("Único dueño")
    return {
        "n_attributes": len(atributos),
        "attr_ids": "|".join(sorted({a.get("id", "") for a in atributos})),
        "attr_kilometros": _a_numero(valores["Kilómetros"]) if "Kilómetros" in valores else np.nan,
        "attr_anio": _a_numero(valores["Año"]) if "Año" in valores else np.nan,
        # Inmuebles: "A estrenar", "Hasta 5 años", ..., "Más de 40 años"
        "attr_antiguedad": valores.get("Antigüedad") or None,
        # Inmuebles: "Excelente", "Muy bueno", "Bueno", "A refaccionar"
        "attr_estado": valores.get("Estado del inmueble") or valores.get("Estado") or None,
        "attr_unico_duenio": {"Sí": True, "No": False}.get(unico_duenio),
        # Repuestos con número de pieza del fabricante
        "attr_numero_pieza": "Número pieza" in valores or "Número de pieza" in valores,
    }


def aplanar_item(r):
    """Convierte un item crudo (dict) en un dict plano de una sola fila."""
    fila = {
        k: v for k, v in r.items()
        if not isinstance(v, (list, dict)) and k not in COLUMNAS_DESCARTADAS
    }

    direccion = r.get("seller_address") or {}
    fila["seller_state"] = _nombre(direccion, "state")
    fila["seller_city"] = _nombre(direccion, "city")

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
    # Algunos campos de texto vienen vacíos en lugar de nulos
    for col in ["warranty", "video_id", "parent_item_id"]:
        df[col] = df[col].replace("", None)
    return df


def cargar(split="train", usar_cache=True):
    """Devuelve el DataFrame aplanado de `split` ("train" o "test")."""
    rutas = {"train": config.TRAIN_RAW, "test": config.TEST_RAW}
    cache = config.DATA_PROCESSED / f"{split}.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)

    df = _tipar(_leer_jsonlines(rutas[split]))
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def cargar_train_test(usar_cache=True):
    return cargar("train", usar_cache), cargar("test", usar_cache)
