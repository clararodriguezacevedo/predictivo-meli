"""Calidad de datos, target, ids y comparación train vs test."""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

import config
from carga import COLUMNAS_DESCARTADAS
from eda import graficos as gr


def calidad_datos(ctx):
    tr, te, rep = ctx.train, ctx.test, ctx.reporte
    rep.seccion("Calidad de datos")
    rep.texto(f"""
Train tiene {len(tr):,} filas y test {len(te):,}. Después de aplanar el JSON y
agregar features derivadas quedan {te.shape[1]} columnas. Se descartaron por ser
constantes o 100% nulas en train: {", ".join(f"`{c}`" for c in COLUMNAS_DESCARTADAS)}.
""")
    columnas = [c for c in tr.columns if c not in (config.TARGET, "y")]
    nulos = pd.DataFrame({
        "nulos_train": tr[columnas].isna().mean(),
        "nulos_test": te.reindex(columns=columnas).isna().mean(),
    })
    nulos = nulos[(nulos > 0).any(axis=1)].sort_values("nulos_train", ascending=False)
    # Tasa de usados según si el campo está presente o no
    nulos["usados_si_nulo"] = [tr.loc[tr[c].isna(), "y"].mean() for c in nulos.index]
    nulos["usados_si_presente"] = [tr.loc[tr[c].notna(), "y"].mean() for c in nulos.index]
    rep.texto("Columnas con faltantes y tasa de usados según la presencia del campo:")
    rep.tabla(nulos)
    rep.texto("""
La ausencia no es aleatoria. Tener `official_store_id`, `video_id` o `warranty`
cargados se asocia a items nuevos: son campos que completan los vendedores
profesionales. Las fotos (`pic_*`) faltan solo cuando el item no tiene ninguna, y en
ese caso el item suele ser usado.
""")


def target(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Target")
    conteo = tr["condition"].value_counts()
    rep.texto(f"""
Clases bastante balanceadas: {conteo["new"]:,} `new` ({conteo["new"] / len(tr):.1%}) y
{conteo["used"]:,} `used` ({conteo["used"] / len(tr):.1%}). Predecir siempre `new`
da un accuracy de {conteo["new"] / len(tr):.3f}, ese es el piso. Como la métrica es
accuracy y las clases están balanceadas, no hace falta reponderar.
""")


def ids(ctx):
    tr, te, rep = ctx.train, ctx.test, ctx.reporte
    rep.seccion("Identificadores")
    id_num = tr["id"].str[3:].astype(np.int64)
    corr = np.corrcoef(id_num, tr["date_created"].astype("int64"))[0, 1]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.5))
    gr.tasa_por_bins(tr.assign(id_numerico=id_num), "id_numerico", ejes[0], titulo="id del item")
    gr.tasa_por_bins(tr, "seller_id", ejes[1], titulo="seller_id numérico")
    rep.figura(fig, "ids", "Tasa de usados por deciles de los ids numéricos")
    padre_en_datos = tr["parent_item_id"].isin(pd.concat([tr["id"], te["id"]])).mean()
    rep.texto(f"""
Los ids están anonimizados: van uniformes de 1e9 a 1e10, no correlacionan con la
fecha de creación (r = {corr:.4f}) y la tasa de usados es plana en todos los deciles.
Lo mismo con `seller_id` y `parent_item_id` (que además casi nunca apunta a un item
del dataset: {padre_en_datos:.4%}). Su valor numérico no sirve, pero `seller_id` como
**clave de agrupación** es de lo más valioso del dataset (ver sección de vendedores).
""")


def train_vs_test(ctx):
    tr, te, rep = ctx.train, ctx.test, ctx.reporte
    rep.seccion("Train vs test (adversarial validation)")
    cols = [
        "log_price", "initial_quantity", "sold_quantity", "available_quantity",
        "n_pictures", "n_attributes", "n_variations", "warranty_len", "title_len",
        "dias_hasta_update", "duracion_dias", "hora_creacion", "seller_n_items",
        "seller_frac_free", "seller_frac_iq_1", "cat_n_items",
    ]
    fechas = pd.concat([tr["date_created"], te["date_created"]]).astype("int64")
    X = pd.concat([tr[cols], te[cols]], ignore_index=True).assign(fecha=fechas.values)
    es_test = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    modelo = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1,
                                random_state=config.SEED)
    proba = cross_val_predict(modelo, X, es_test, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(es_test, proba)
    vendedores_conocidos = te["seller_id"].isin(tr["seller_id"]).mean()
    categorias_conocidas = te["category_id"].isin(tr["category_id"]).mean()
    rep.texto(f"""
Un LightGBM entrenado para distinguir filas de train y de test obtiene AUC = {auc:.3f}
con validación cruzada. Un valor cercano a 0.5 indica que ambos vienen de la misma
distribución (split aleatorio, no temporal), así que una validación cruzada
estratificada sobre train debería ser representativa del leaderboard.

Solapamiento de grupos: {vendedores_conocidos:.1%} de los items de test tienen un
vendedor que aparece en train y {categorias_conocidas:.1%} una categoría que aparece
en train. Esto habilita target encoding por vendedor y por categoría (siempre
out-of-fold para no filtrar el target).
""")
