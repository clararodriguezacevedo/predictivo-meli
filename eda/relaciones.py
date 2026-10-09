"""Relaciones entre variables: vendedores, categorías y rankings de importancia."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import config
from eda import graficos as gr

NUMERICAS = [
    "initial_quantity", "sold_quantity", "available_quantity", "stock_desbalance",
    "log_price", "log_price_rel_categoria", "n_pictures", "n_attributes", "n_variations",
    "n_payment_methods", "warranty_len", "title_len", "title_frac_mayus",
    "dias_hasta_update", "duracion_dias", "hora_creacion", "pic_max_ancho",
    "seller_n_items", "seller_frac_free", "seller_frac_iq_1", "seller_frac_vendio",
    "seller_frac_warranty", "seller_frac_price_decimales", "seller_ratio_categorias",
    "cat_n_items", "cat_frac_free", "cat_frac_iq_1",
]
BOOLEANAS = [
    "iq_es_1", "vendio_algo", "automatic_relist", "accepts_mercadopago",
    "ship_free_shipping", "ship_local_pick_up", "price_con_decimales",
    "price_termina_en_9", "price_multiplo_100", "tiene_warranty", "tiene_video",
    "es_tienda_oficial",
]


def accuracy_leave_one_out(df, clave):
    """Accuracy de predecir la clase mayoritaria de los *otros* items del grupo.

    Solo evalúa items cuyo grupo tiene al menos otro item en train. Es una medida
    honesta de cuánta información aporta conocer el grupo.
    """
    g = df.groupby(clave)["y"]
    suma, n = g.transform("sum"), g.transform("size")
    otros = n - 1
    tasa_otros = (suma - df["y"]) / otros.replace(0, np.nan)
    evaluables = otros > 0
    pred = (tasa_otros[evaluables] > 0.5).astype(int)
    return (pred == df.loc[evaluables, "y"]).mean(), evaluables.mean()


def _pureza(df, clave, min_items=2):
    g = df.groupby(clave)["y"].agg(["mean", "size"])
    g = g[g["size"] >= min_items]
    puros = g[(g["mean"] == 0) | (g["mean"] == 1)]["size"].sum()
    return puros / g["size"].sum(), g


def vendedores(ctx):
    tr, te, rep = ctx.train, ctx.test, ctx.reporte
    rep.seccion("10. Vendedores: la relación oculta más fuerte")
    pureza, g = _pureza(tr, "seller_id")
    acc_loo, cobertura = accuracy_leave_one_out(tr, "seller_id")

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    ejes[0].hist(g["mean"], bins=20, weights=g["size"], color=gr.COLOR_NEUTRO)
    ejes[0].set(title="Tasa de usados del vendedor (pesada por items)",
                xlabel="tasa de usados del vendedor", ylabel="items")
    gr.tasa_por_categoria(
        tr.assign(seller_n=pd.cut(tr["seller_n_items"], [0, 1, 2, 5, 10, 50, 200, 1e5])),
        "seller_n", ejes[1], titulo="items del vendedor (train + test)",
    )
    rep.figura(fig, "vendedores_pureza")
    rep.texto(f"""
Hay {tr["seller_id"].nunique():,} vendedores en train. Entre los items de vendedores
con más de un item, el {pureza:.1%} pertenece a vendedores **puros** (todo nuevo o
todo usado): la distribución es una U casi perfecta. Un vendedor es un comercio o un
particular, rara vez ambas cosas.

Si para cada item se predice la clase mayoritaria de los *otros* items del mismo
vendedor, el accuracy es **{acc_loo:.3f}** sobre el {cobertura:.1%} de train que
tiene vendedor repetido. Y {te["seller_id"].isin(tr["seller_id"]).mean():.1%} de los
items de test tienen vendedor en train. Un target encoding out-of-fold por vendedor
va a ser probablemente la feature más importante.

El tamaño del vendedor no es monótono: los de 2 a 10 items son mayormente nuevos,
los de 50 a 200 vuelven a ser usados (revendedores de libros, discos, coleccionables)
y los más grandes son comercios.
""")

    rep.seccion("Perfil del vendedor (sin usar el target)", nivel=3)
    m = tr["seller_n_items"] >= 3
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
    cortes = [-0.01, 0, 0.25, 0.5, 0.75, 0.99, 1]
    for col, ax in zip(["seller_frac_free", "seller_frac_iq_1", "seller_frac_price_decimales"], ejes):
        gr.tasa_por_categoria(tr[m].assign(**{col: pd.cut(tr.loc[m, col], cortes)}), col, ax, top=6)
    rep.figura(fig, "vendedores_perfil", "Solo vendedores con 3 o más items (train + test)")

    no_free = tr[tr["listing_type_id"] != "free"]
    tasa_contaminado = no_free.loc[no_free["seller_frac_free"] > 0, "y"].mean()
    tasa_limpio = no_free.loc[no_free["seller_frac_free"] == 0, "y"].mean()
    rep.texto(f"""
Agregando por vendedor las variables del item (sobre train + test, sin mirar el
target) se obtiene un perfil del vendedor que generaliza a vendedores no vistos.

**Relación oculta**: lo que hace un vendedor en *sus otras* publicaciones dice más
que el item mismo. Si se toman solo los items que **no** son `free`, los de
vendedores que alguna vez publicaron gratis son {tasa_contaminado:.1%} usados, contra
{tasa_limpio:.1%} de los vendedores que nunca lo hicieron. Lo mismo con la fracción
de items con stock 1 y la fracción de precios con centavos.
""")


def categorias(ctx):
    tr, te, rep = ctx.train, ctx.test, ctx.reporte
    rep.seccion("11. Categorías")
    pureza, _ = _pureza(tr, "category_id", min_items=20)
    acc_loo, cobertura = accuracy_leave_one_out(tr, "category_id")
    top = (tr.groupby("category_id")["y"].agg(tasa_usados="mean", n="size")
           .sort_values("n", ascending=False).head(12))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tr.groupby("category_id")["y"].mean(), bins=20, color=gr.COLOR_NEUTRO,
            weights=tr.groupby("category_id").size())
    ax.set(title="Tasa de usados por categoría (pesada por items)", xlabel="tasa de usados", ylabel="items")
    rep.figura(fig, "categorias")
    rep.texto(f"""
Hay {tr["category_id"].nunique():,} categorías (hojas del árbol de MercadoLibre, sin
nombre). Predecir la clase mayoritaria de los otros items de la misma categoría da
accuracy **{acc_loo:.3f}** sobre el {cobertura:.1%} de train evaluable. Entre las
categorías con 20 o más items, el {pureza:.1%} de los items está en categorías
puras. {te["category_id"].isin(tr["category_id"]).mean():.1%} de los items de test
tienen categoría vista en train. Categorías más grandes:
""")
    rep.tabla(top)


def ranking_global(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("12. Ranking de variables y correlaciones")
    X = tr[NUMERICAS + BOOLEANAS].astype(float).fillna(-1)

    mi = mutual_info_classif(X, tr["y"], random_state=config.SEED)
    arbol = {
        c: cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=config.SEED),
                           X[[c]], tr["y"], cv=3).mean()
        for c in X.columns
    }
    ranking = pd.DataFrame({"mutual_info": mi, "acc_arbol_1_variable": pd.Series(arbol)},
                           index=X.columns).sort_values("acc_arbol_1_variable", ascending=False)
    rep.texto("""
Para cada variable, información mutua con el target y accuracy (CV de 3 folds) de un
árbol de profundidad 4 que solo ve esa variable:
""")
    rep.tabla(ranking.head(20))

    corr = tr[NUMERICAS + ["y"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Correlación de Spearman entre variables numéricas y el target")
    rep.figura(fig, "correlaciones")

    mejor = ranking.index[0]
    rep.texto(f"""
Las variables individuales más predictivas (`{mejor}` a la cabeza) son perfiles
agregados del vendedor y de la categoría, por encima de cualquier atributo del item.
Entre las del item lideran `initial_quantity`/`available_quantity` y luego el precio.

En la matriz de correlación:

- `initial_quantity` y `available_quantity` son casi la misma variable (0.99).
  `sold_quantity` va por otro lado (0.34).
- `seller_frac_iq_1` está muy correlacionada con `initial_quantity` (-0.86), pero
  solo moderadamente con `seller_frac_free` (0.43). Ambas tienen la misma
  correlación con el target (0.55), así que capturan facetas distintas del
  vendedor particular y conviene mantener las dos.
- Las variables de "profesionalismo" (`seller_frac_warranty`,
  `seller_frac_price_decimales`, `n_payment_methods`) apuntan en la misma dirección
  pero se correlacionan poco entre sí (0.17 a 0.33): son señales débiles e
  independientes que un modelo de árboles puede sumar.
""")
