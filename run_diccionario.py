"""Genera docs/DICCIONARIO.md a partir de `diccionario.py` y de los datos.

Para cada variable muestra su significado junto con estadísticas reales (tipo,
nulos, cantidad de valores distintos) y avisa si hay columnas sin documentar.

Uso: python run_diccionario.py
"""

import pandas as pd

import config
import features
from carga import cargar_train_test
from diccionario import FUENTES, GRUPOS, SIN_ESTADISTICAS, VARIABLES


def _tipo(serie):
    if pd.api.types.is_bool_dtype(serie):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    if pd.api.types.is_numeric_dtype(serie):
        return "numérica"
    return "texto"


def _resumen(serie):
    """Rango para numéricas, valores más frecuentes para el resto."""
    tipo = _tipo(serie)
    if tipo in ("numérica", "fecha") and serie.nunique() > 10:
        minimo, mediana, maximo = serie.min(), serie.median(), serie.max()
        if tipo == "fecha":
            return f"{minimo:%Y-%m-%d} a {maximo:%Y-%m-%d}"
        return f"{minimo:.4g} / {mediana:.4g} / {maximo:.4g}"
    frecuentes = serie.astype(str).replace("", "(vacío)").value_counts().head(3)
    return ", ".join(f"`{v}`" for v in frecuentes.index)


def _celda(texto):
    return texto.replace("|", "\\|").replace("\n", " ")


def generar(datos):
    lineas = [
        "# Diccionario de variables",
        "",
        "Generado por `run_diccionario.py` a partir de `diccionario.py`. Las estadísticas "
        "se calculan sobre train + test juntos (100.000 items). La columna *Resumen* "
        "muestra mínimo / mediana / máximo para numéricas y los valores más frecuentes "
        "para el resto.",
        "",
        "| Origen | Significado |",
        "|---|---|",
        "| JSON | campo de primer nivel del JSON original, sin cambios |",
        "| aplanado | extraído de un campo anidado (listas o diccionarios) en `carga.py` |",
        "| feature | calculada en `features.py`, sin usar el target |",
        "| descartada | campo del JSON que no se carga por ser constante o redundante |",
        "| con target | usa `condition`, se calcula con folds en `target_encoding.py` |",
        "",
    ]
    for grupo in GRUPOS:
        variables = [v for v in VARIABLES if v.grupo == grupo]
        lineas += [f"## {grupo}", "",
                   "| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |",
                   "|---|---|---|---|---|---|---|"]
        for v in variables:
            if v.nombre in datos and v.origen not in SIN_ESTADISTICAS:
                s = datos[v.nombre]
                stats = [_tipo(s), f"{s.isna().mean():.1%}", f"{s.nunique():,}", _resumen(s)]
            else:
                stats = ["", "", "", ""]
            fila = [f"`{v.nombre}`", v.origen, *stats, v.significado]
            lineas.append("| " + " | ".join(_celda(c) for c in fila) + " |")
        lineas.append("")

        for v in variables:
            if v.valores:
                lineas += [f"| Valor de `{v.nombre}` | Significado |", "|---|---|"]
                lineas += [f"| `{valor}` | {_celda(desc)} |" for valor, desc in v.valores.items()]
                lineas.append("")

    usadas = sorted({f for v in VARIABLES for f in v.fuentes})
    lineas += ["## Fuentes", "",
               "Documentación oficial de la API de MercadoLibre (consultada en octubre de 2026). "
               "Describe la API actual; el dataset es de 2015, así que algunos valores (por "
               "ejemplo los tipos de publicación) cambiaron de nombre o dejaron de existir.", ""]
    lineas += [f"- {FUENTES[f]}" for f in usadas]
    return "\n".join(lineas) + "\n"


def main():
    train, test = features.construir(*cargar_train_test())
    datos = pd.concat([train, test], ignore_index=True)

    documentadas = {v.nombre for v in VARIABLES}
    sin_documentar = sorted(set(datos.columns) - documentadas)
    if sin_documentar:
        print(f"ATENCIÓN, columnas sin documentar: {sin_documentar}")
    faltantes = sorted(v.nombre for v in VARIABLES
                       if v.origen not in SIN_ESTADISTICAS and v.nombre not in datos)
    if faltantes:
        print(f"ATENCIÓN, variables documentadas que no están en los datos: {faltantes}")

    config.DOCS.mkdir(exist_ok=True)
    ruta = config.DOCS / "DICCIONARIO.md"
    ruta.write_text(generar(datos), encoding="utf8")
    print(f"{len(VARIABLES)} variables documentadas. Diccionario guardado en {ruta}")


if __name__ == "__main__":
    main()
