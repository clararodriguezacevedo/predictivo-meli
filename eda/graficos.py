"""Helpers de gráficos reutilizables para el EDA.

Casi todos los gráficos muestran la tasa de items usados (media del target binario
`y`) contra alguna variable, con una línea de referencia en la tasa global.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLOR_USADO = "#c0392b"
COLOR_NUEVO = "#2471a3"
COLOR_NEUTRO = "#7f8c8d"


def estilo():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 11})


def _linea_base(ax, tasa_global, horizontal=False):
    dibujar = ax.axvline if horizontal else ax.axhline
    dibujar(tasa_global, color=COLOR_NEUTRO, ls="--", lw=1, label=f"global ({tasa_global:.2f})")


def tasa_por_categoria(df, col, ax, top=12, min_n=30, titulo=None):
    """Barras horizontales con la tasa de usados por categoría y el n de cada una."""
    g = df.groupby(col, dropna=False, observed=True)["y"].agg(["mean", "size"])
    g = g[g["size"] >= min_n].sort_values("size", ascending=False).head(top)
    g = g.sort_values("mean")
    etiquetas = [str(i) if str(i) != "" else "(vacío)" for i in g.index]
    colores = [COLOR_USADO if m > df["y"].mean() else COLOR_NUEVO for m in g["mean"]]
    ax.barh(etiquetas, g["mean"], color=colores, alpha=0.85)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(min(m + 0.01, 0.9), i, f"n={n:,}", va="center", fontsize=8)
    _linea_base(ax, df["y"].mean(), horizontal=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("tasa de usados")
    ax.set_title(titulo or col)
    return g


def tasa_por_bins(df, col, ax, bins=10, titulo=None, log=False):
    """Tasa de usados por cuantiles de una variable numérica, con el n de cada bin."""
    serie = df[col]
    cortes = pd.qcut(serie, bins, duplicates="drop")
    g = df.groupby(cortes, observed=True)["y"].agg(["mean", "size"])
    centros = df.groupby(cortes, observed=True)[col].median()
    ax.plot(centros, g["mean"], marker="o", color=COLOR_USADO)
    _linea_base(ax, df["y"].mean())
    if log:
        ax.set_xscale("symlog")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{col} (mediana del bin)")
    ax.set_ylabel("tasa de usados")
    ax.set_title(titulo or col)
    return g


def densidad_por_clase(df, col, ax, titulo=None, clip=None):
    """Distribución de una variable separada por clase."""
    datos = df[[col, "condition"]].copy()
    if clip is not None:
        datos[col] = datos[col].clip(*clip)
    sns.kdeplot(
        data=datos, x=col, hue="condition", ax=ax, common_norm=False, fill=True,
        palette={"new": COLOR_NUEVO, "used": COLOR_USADO}, alpha=0.3, warn_singular=False,
    )
    ax.set_title(titulo or col)


def heatmap_tasa(df, filas, columnas, ax, titulo=None, min_n=20):
    """Heatmap de la tasa de usados para el cruce de dos variables categóricas."""
    tabla = df.pivot_table(index=filas, columns=columnas, values="y", aggfunc=["mean", "size"])
    tasa = tabla["mean"].where(tabla["size"] >= min_n)
    anot = tasa.map(lambda v: "" if np.isnan(v) else f"{v:.2f}")
    anot = anot + tabla["size"].map(lambda n: "" if np.isnan(n) else f"\nn={int(n):,}")
    sns.heatmap(tasa, annot=anot, fmt="", cmap="RdBu_r", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "tasa de usados"}, annot_kws={"fontsize": 8})
    ax.set_title(titulo or f"{filas} x {columnas}")
    return tabla
