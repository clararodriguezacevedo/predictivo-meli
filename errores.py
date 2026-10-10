"""Análisis de errores del modelo sobre las predicciones de validación.

Recibe el train con sus features, la probabilidad de usado que el modelo le dio a
cada item al validarlo (sin haberlo visto) y las features de target encoding de ese
momento. Genera un reporte que muestra en qué tipo de items se concentran los
errores.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.graficos import COLOR_NEUTRO, COLOR_USADO, estilo
from eda.reporte import Reporte

# Un vendedor es "casi puro" si tiene al menos estos items y esta fracción de una clase
MIN_ITEMS_VENDEDOR = 5
PUREZA = 0.8
# Se considera que el modelo está muy seguro por encima de esta probabilidad
MUY_SEGURO = 0.9


def _desglose(datos, grupos):
    """Items, tasa de error y porcentaje del total de errores por grupo."""
    g = datos.groupby(grupos, dropna=False, observed=True).agg(
        items=("error", "size"), errores=("error", "sum"), tasa_de_error=("error", "mean"),
    )
    g["% de los items"] = 100 * g["items"] / len(datos)
    g["% de los errores"] = 100 * g["errores"] / datos["error"].sum()
    g.index = g.index.astype(str)
    return g[["items", "% de los items", "tasa_de_error", "errores", "% de los errores"]]


def _preparar(datos):
    d = datos.copy()
    d["pred"] = (d["proba_usado"] > 0.5).astype(int)
    d["error"] = d["pred"] != d["y"]
    d["confianza"] = np.maximum(d["proba_usado"], 1 - d["proba_usado"])
    d["vendedor_conocido"] = pd.cut(
        d["te_seller_id_n"], [-1, 0, 2, 5, 20, np.inf],
        labels=["ninguno", "1 o 2", "3 a 5", "6 a 20", "más de 20"],
    )
    d["tamano_vendedor"] = pd.cut(
        d["seller_n_items"], [0, 1, 2, 5, 20, np.inf],
        labels=["1 (único item)", "2", "3 a 5", "6 a 20", "más de 20"],
    )
    por_vendedor = d.groupby("seller_id")["y"].agg(["mean", "size"])
    tasa, n = d["seller_id"].map(por_vendedor["mean"]), d["seller_id"].map(por_vendedor["size"])
    grande = n >= MIN_ITEMS_VENDEDOR
    es_minoria = ((tasa >= PUREZA) & (d["y"] == 0)) | ((tasa <= 1 - PUREZA) & (d["y"] == 1))
    d["rol_en_el_vendedor"] = np.select(
        [~grande, es_minoria, tasa.between(1 - PUREZA, PUREZA, inclusive="neither")],
        ["vendedor con pocos items", "minoría de un vendedor casi puro", "vendedor mixto"],
        default="mayoría de un vendedor casi puro",
    )
    return d


def generar_reporte(datos, carpeta):
    estilo()
    d = _preparar(datos)
    rep = Reporte(carpeta, "Análisis de errores del modelo")
    errores = d[d["error"]]
    falsos_usados = int(((d["pred"] == 1) & (d["y"] == 0)).sum())
    falsos_nuevos = int(((d["pred"] == 0) & (d["y"] == 1)).sum())

    rep.seccion("Resumen")
    rep.texto(f"""
Sobre {len(d):,} items de train, cada uno predicho por un modelo que no lo vio, el
accuracy es {1 - d["error"].mean():.4f}: {len(errores):,} errores. De esos,
{falsos_usados:,} son nuevos predichos como usados y {falsos_nuevos:,} usados
predichos como nuevos.
""")

    rep.seccion("Cuánto se conoce al vendedor")
    conocido = _desglose(d, "vendedor_conocido")
    tamano = _desglose(d, "tamano_vendedor")
    rep.texto("Según cuántos otros items con etiqueta tenía el vendedor al momento de validar:")
    rep.tabla(conocido)
    rep.texto("Según cuántos items tiene el vendedor en total (train + test):")
    rep.tabla(tamano)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
    for tabla, ax, titulo in [(conocido, ejes[0], "otros items con etiqueta del vendedor"),
                              (tamano, ejes[1], "items del vendedor (train + test)")]:
        ax.bar(tabla.index, tabla["tasa_de_error"], color=COLOR_USADO, alpha=0.85)
        ax.axhline(d["error"].mean(), color=COLOR_NEUTRO, ls="--", label="tasa de error global")
        ax.set(title=titulo, ylabel="tasa de error")
        ax.legend()
    rep.figura(fig, "error_por_vendedor")

    sin = conocido.loc["ninguno"]
    unico = tamano.loc["1 (único item)"]
    rep.texto(f"""
**El modelo falla sobre todo donde no conoce al vendedor.** Los items cuyo vendedor
no tiene ningún otro item con etiqueta son el {sin["% de los items"]:.0f}% de los datos
y concentran el {sin["% de los errores"]:.0f}% de los errores, con una tasa de error de
{sin["tasa_de_error"]:.1%}. La mayoría son vendedores de un único item
({int(unico["items"]):,} items, {unico["tasa_de_error"]:.1%} de error): para ellos no
hay otras publicaciones en las que apoyarse, ni en train ni en test, y solo queda lo
que dice la propia publicación.
""")

    rep.seccion("Confianza del modelo")
    d["tramo_de_confianza"] = pd.cut(d["confianza"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                     include_lowest=True)
    confianza = _desglose(d, "tramo_de_confianza")
    rep.tabla(confianza)
    seguros = d[d["confianza"] > MUY_SEGURO]
    rep.texto(f"""
El modelo está razonablemente calibrado: cuando duda se equivoca casi la mitad de las veces y
cuando está muy seguro (más de {MUY_SEGURO:.0%}) falla el {seguros["error"].mean():.1%}.
Pero como casi todos los items caen en ese tramo, esos fallos son
{seguros["error"].sum():,}, el {seguros["error"].sum() / len(errores):.0%} de todos los
errores.
""")

    rep.seccion("Errores con el modelo muy seguro")
    ejemplos = errores.sort_values("confianza", ascending=False).head(20)
    rep.tabla(pd.DataFrame({
        "título": ejemplos["title"].str.slice(0, 55),
        "etiqueta": ejemplos["condition"],
        "proba usado": ejemplos["proba_usado"],
        "stock inicial": ejemplos["initial_quantity"],
        "tipo": ejemplos["listing_type_id"],
        "garantía": ejemplos["warranty"].fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 30),
    }), index=False)
    rep.texto("""
Muchos de estos casos parecen etiquetas dudosas más que errores del modelo:
repuestos y consumibles con cientos de unidades de stock marcados como usados,
"libros digitales" con stock múltiple marcados como usados, o autos de varios años
marcados como nuevos. La condición la declara el vendedor y no siempre la carga
bien. Esto pone un techo al accuracy que ningún modelo puede superar.
""")

    rep.seccion("Rol del item dentro de su vendedor")
    rol = _desglose(d, "rol_en_el_vendedor")
    rep.tabla(rol)
    minoria = rol.loc["minoría de un vendedor casi puro"]
    rep.texto(f"""
Un vendedor es "casi puro" si tiene {MIN_ITEMS_VENDEDOR} o más items con etiqueta y al
menos {PUREZA:.0%} son de la misma clase. Los items que son la excepción de su
vendedor (el único usado de un comercio, o el único nuevo de un particular) fallan
el {minoria["tasa_de_error"]:.0%} de las veces: la señal más fuerte del modelo apunta
justo en la dirección contraria.
""")

    rep.seccion("Tipo de publicación y stock")
    rep.tabla(_desglose(d, "listing_type_id"))
    rep.tabla(_desglose(d.assign(stock_inicial=np.where(d["iq_es_1"], "1 unidad", "más de 1")),
                        "stock_inicial"))
    rep.texto("""
Con más de una unidad casi no hay errores. El problema está en las publicaciones
`bronze` de una sola unidad, que pueden ser tanto de un comercio como de un
particular.
""")

    rep.seccion("Categorías con más errores")
    categorias = d.groupby("category_id").agg(
        items=("error", "size"), errores=("error", "sum"), tasa_de_error=("error", "mean"),
        tasa_de_usados=("y", "mean"),
    ).sort_values("errores", ascending=False).head(12)
    categorias["ejemplo"] = [d.loc[d["category_id"] == c, "title"].iloc[0][:45]
                             for c in categorias.index]
    rep.tabla(categorias)
    return rep.guardar("REPORTE_ERRORES.md")
