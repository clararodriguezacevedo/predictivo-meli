"""Análisis por variable: categóricas, stock, precio, tiempo y texto."""

import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

import features
from carga import MEDIOS_DE_PAGO
from eda import graficos as gr

CATEGORICAS = [
    "listing_type_id", "buying_mode", "ship_mode", "status", "automatic_relist",
    "accepts_mercadopago", "ship_free_shipping", "ship_local_pick_up", "currency_id",
    "tags", "es_tienda_oficial", "tiene_video",
]


def _rango_tasas(df, col, min_n=30):
    g = df.groupby(col, dropna=False, observed=True)["y"].agg(["mean", "size"])
    g = g[g["size"] >= min_n]
    return g["mean"].max() - g["mean"].min()


def categoricas(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Variables categóricas")
    fig, ejes = plt.subplots(4, 3, figsize=(15, 16))
    for col, ax in zip(CATEGORICAS, ejes.flat):
        gr.tasa_por_categoria(tr, col, ax)
    rep.figura(fig, "categoricas", "Tasa de usados por nivel (rojo: por encima de la tasa global)")

    rango = pd.Series({c: _rango_tasas(tr, c) for c in CATEGORICAS}).sort_values(ascending=False)
    rep.texto("Diferencia entre la tasa máxima y mínima de usados entre niveles (n >= 30):")
    rep.tabla(rango.rename("rango_tasa").to_frame())

    free = tr.loc[tr["listing_type_id"] == "free", "y"].mean()
    relist = tr.loc[tr["automatic_relist"], "y"].mean()
    rep.texto(f"""
- `listing_type_id` es la categórica más fuerte: las publicaciones `free` son
  {free:.1%} usadas. Publicar gratis es típico del particular que vende algo propio,
  mientras que `gold_special` (la más cara) es casi exclusivamente de nuevos.
- `automatic_relist` = True ({relist:.1%} usados), envío gratis, tienda oficial y
  video son marcas de vendedor profesional y empujan hacia "nuevo".
- `buying_mode` `classified` y `auction` son mayormente usados (clasificados de
  autos e inmuebles, subastas de objetos de colección).
""")

    medios = pd.DataFrame({
        f"{m} ({descripcion})": {
            "n": tr[f"pay_{m}"].sum(), "tasa_usados": tr.loc[tr[f"pay_{m}"], "y"].mean()
        }
        for m, descripcion in MEDIOS_DE_PAGO.items()
    }).T.sort_values("tasa_usados")
    rep.seccion("Medios de pago no MercadoPago", nivel=3)
    rep.tabla(medios)
    rep.texto("""
Aceptar tarjeta de crédito (`MLAOT`) o transferencia (`MLATB`) indica un vendedor más
formal (más nuevos). Giro postal (`MLAWT`) y cheque certificado (`MLABC`) son medios
viejos y se asocian a usados, igual que "acordar con el comprador" (`MLAWC`).
""")


def stock(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Stock y ventas")
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
    tr_c = tr.assign(
        initial_quantity_c=tr["initial_quantity"].clip(upper=10),
        stock_desbalance_c=tr["stock_desbalance"].clip(-5, 5),
    )
    gr.tasa_por_categoria(tr_c, "initial_quantity_c", ejes[0], top=10,
                          titulo="initial_quantity (10 = 10 o más)")
    gr.tasa_por_categoria(tr_c, "stock_desbalance_c", ejes[1], top=11,
                          titulo="available - (initial - sold)")
    gr.heatmap_tasa(tr, "listing_type_id", "iq_es_1", ejes[2],
                    titulo="listing_type x (initial_quantity == 1)")
    rep.figura(fig, "stock")

    iq1 = tr.loc[tr["iq_es_1"], "y"].mean()
    iq_mas = tr.loc[~tr["iq_es_1"], "y"].mean()
    desb = tr.loc[tr["stock_desbalance"] > 0, "y"].mean()
    n_desb = (tr["stock_desbalance"] > 0).sum()
    cruce = tr.groupby(["iq_es_1", "vendio_algo"])["y"].agg(["mean", "size"])
    rep.texto(f"""
- Con `initial_quantity` = 1 la tasa de usados es {iq1:.1%}, con más de una unidad
  cae a {iq_mas:.1%}. Un usado es casi siempre una unidad única.
- **Relación oculta, stock repuesto**: en teoría `available = initial - sold`. En
  {n_desb:,} items `available` es mayor a lo esperado (el vendedor repuso stock) y en
  esos casos la tasa de usados es {desb:.1%}. Es una huella de comercio.
- La interacción con `listing_type_id` es fuerte: un `bronze` con stock 1 es ~54%
  usado, pero con más stock baja a ~4%. Un `free` es usado casi siempre, incluso con
  varias unidades. Los árboles capturan este cruce solos.
- **Relación oculta, `gold_premium`**: sola parece neutra (~46% usados), pero
  esconde dos poblaciones opuestas. Con más de una unidad es 0% usado y con stock 1
  es ~84% usado (bienes caros de unidad única, como autos o inmuebles, publicados
  con la mejor exposición).
- Haber vendido algo también empuja a "nuevo":
""")
    rep.tabla(cruce)


def precio(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Precio")
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
    gr.tasa_por_bins(tr, "price", ejes[0], bins=20, log=True, titulo="precio (escala log)")
    gr.tasa_por_bins(tr, "log_price_rel_categoria", ejes[1], bins=10,
                     titulo="log(precio) - mediana de su categoría")
    gr.densidad_por_clase(tr, "log_price", ejes[2], titulo="log(1 + precio) por clase", clip=(0, 14))
    rep.figura(fig, "precio")

    psico = pd.DataFrame({
        f: {
            "n": tr[f].sum(),
            "usados_si_true": tr.loc[tr[f], "y"].mean(),
            "usados_si_false": tr.loc[~tr[f], "y"].mean(),
        }
        for f in ["price_con_decimales", "price_termina_en_9", "price_multiplo_100"]
    }).T
    rep.texto("""
La relación precio y condición no es monótona: los usados se concentran en precios
bajos (libros, revistas, discos entre 35 y 100 pesos) y vuelven a subir en precios
muy altos (autos e inmuebles clasificados). El tramo medio es mayormente nuevo.
""")
    rep.seccion("Psicología de precios", nivel=3)
    rep.tabla(psico)
    rep.texto("""
**Relación oculta**: el *formato* del precio delata al tipo de vendedor. Los
comercios usan precios con centavos o terminados en 9 (199.99, 349) y son
mayormente nuevos. Los particulares redondean a múltiplos de 100 y son mayormente
usados.
""")


def temporal(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Variables temporales")
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
    gr.tasa_por_bins(tr, "dias_hasta_update", ejes[0, 0], bins=10, log=True,
                     titulo="días entre creación y última actualización")
    gr.tasa_por_categoria(tr.assign(duracion=tr["duracion_dias"].clip(upper=400)), "duracion",
                          ejes[0, 1], top=8, titulo="duración de la publicación (días)")
    horas = tr.groupby("hora_creacion")["y"].mean()
    ejes[1, 0].plot(horas.index, horas.values, marker="o", color=gr.COLOR_USADO)
    ejes[1, 0].axhline(ctx.tasa_global, color=gr.COLOR_NEUTRO, ls="--")
    ejes[1, 0].set(title="hora de creación (Argentina)", xlabel="hora", ylabel="tasa de usados", ylim=(0, 1))
    dias = tr.groupby("dia_semana")["y"].mean()
    ejes[1, 1].bar(["lun", "mar", "mié", "jue", "vie", "sáb", "dom"], dias.values, color=gr.COLOR_NEUTRO)
    ejes[1, 1].axhline(ctx.tasa_global, color=gr.COLOR_USADO, ls="--")
    ejes[1, 1].set(title="día de la semana de creación", ylabel="tasa de usados", ylim=(0, 1))
    rep.figura(fig, "temporal")

    rango = tr["date_created"].agg(["min", "max"])
    rep.texto(f"""
- Los items se crearon entre {rango["min"]:%Y-%m-%d} y {rango["max"]:%Y-%m-%d}, casi
  todos entre agosto y octubre de 2015. La fecha en sí no tiene tendencia.
- Los items que se actualizaron días después de creados son más nuevos: el
  comerciante edita precio y stock, el particular publica y se olvida.
- La duración estándar es 60 días. Las de 30 días y 365 días son mayormente usados
  (formatos de clasificados y publicaciones gratuitas).
- Hay un patrón horario suave (hora argentina): de noche, entre las 19 y las 2, y
  los fines de semana publican más particulares (más usados); entre las 8 y las 18
  de lunes a viernes predominan los comercios.
""")


def _tokens(texto):
    return set(re.findall(r"[a-záéíóúñü0-9]+", texto.lower()))


def tasa_por_palabra(serie, y, minimo):
    tokens = serie.fillna("").map(_tokens)
    total = Counter(t for ts in tokens for t in ts)
    usados = Counter(t for ts, yi in zip(tokens, y) if yi for t in ts)
    df = pd.DataFrame({"n": total, "tasa_usados": usados}).fillna(0)
    df = df[df["n"] >= minimo]
    df["tasa_usados"] = df["tasa_usados"] / df["n"]
    return df


def texto(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Texto: título y garantía")
    for col, minimo in [("title", 300), ("warranty", 150)]:
        palabras = tasa_por_palabra(tr[col], tr["y"], minimo)
        extremos = pd.concat([
            palabras.nlargest(15, "tasa_usados"), palabras.nsmallest(15, "tasa_usados")
        ]).sort_values("tasa_usados")
        fig, ax = plt.subplots(figsize=(7, 8))
        colores = [gr.COLOR_USADO if t > ctx.tasa_global else gr.COLOR_NUEVO
                   for t in extremos["tasa_usados"]]
        ax.barh(extremos.index, extremos["tasa_usados"], color=colores)
        ax.axvline(ctx.tasa_global, color=gr.COLOR_NEUTRO, ls="--")
        ax.set(xlim=(0, 1), xlabel="tasa de usados",
               title=f"Palabras de `{col}` más asociadas a cada clase (n >= {minimo})")
        rep.figura(fig, f"palabras_{col}")

    fig, ejes = plt.subplots(1, 3, figsize=(16, 4))
    gr.tasa_por_bins(tr, "title_len", ejes[0], bins=8, titulo="largo del título")
    gr.tasa_por_bins(tr, "title_frac_mayus", ejes[1], bins=8, titulo="fracción de mayúsculas en título")
    gr.tasa_por_categoria(
        tr.assign(warranty_len_c=pd.cut(tr["warranty_len"], [-1, 0, 5, 15, 40, 100, 1e5])),
        "warranty_len_c", ejes[2], titulo="largo de la garantía",
    )
    rep.figura(fig, "texto_largos")

    rep.texto("""
El texto es muy informativo y complementa a las variables estructuradas:

- **Título**: aparecen palabras explícitas ("usado", "impecable", "excelente estado")
  pero sobre todo el *tipo de producto*. Revistas, vinilos, LPs, libros y cosas
  "antiguas" son usados; repuestos de auto (amortiguador, freno, kit), electrónica
  (led, usb, cargador) y textiles son nuevos. Artículos como "el", "la", "los"
  aparecen como usados porque son típicos de títulos de libros y discos.
- **Garantía**: los nuevos hablan de "fabricación", "fábrica", "oficial",
  "importador", "meses". Los usados escriben frases sobre "estado", "fotos",
  "reputación", "conforme" (el vendedor responde por su palabra, no por un
  fabricante). Una garantía corta o vacía no discrimina; una de 15 a 40 caracteres
  ("6 meses de garantía") es fuertemente nueva.
- Títulos cortos y con muchas mayúsculas son más de particulares.

Esto sugiere sumar TF-IDF de título y garantía (o un modelo lineal sobre texto cuya
predicción out-of-fold entre como feature al modelo de árboles).
""")


BANDERAS_SEMANTICAS = [
    *features.PATRONES_TITULO, *features.PATRONES_GARANTIA,
    "attr_es_0km", "attr_a_estrenar", "attr_numero_pieza",
]


def senales_semanticas(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Señales semánticas: palabras clave y ficha técnica")
    tabla = pd.DataFrame({
        b: {
            "n": int(tr[b].sum()),
            "usados_si_true": tr.loc[tr[b], "y"].mean(),
            "usados_si_false": tr.loc[~tr[b], "y"].mean(),
        }
        for b in BANDERAS_SEMANTICAS
    }).T.sort_values("usados_si_true")

    fig, ejes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    colores = [gr.COLOR_USADO if t > ctx.tasa_global else gr.COLOR_NUEVO
               for t in tabla["usados_si_true"]]
    ejes[0].barh(tabla.index, tabla["usados_si_true"], color=colores)
    for i, (tasa, n) in enumerate(zip(tabla["usados_si_true"], tabla["n"])):
        ejes[0].text(tasa + 0.01, i, f"n={int(n):,}", va="center", fontsize=8)
    ejes[0].axvline(ctx.tasa_global, color=gr.COLOR_NEUTRO, ls="--")
    ejes[0].set(xlim=(0, 1), xlabel="tasa de usados cuando la bandera es True",
                title="Banderas de título (t_), garantía (w_) y ficha técnica (attr_)")
    autos = tr[tr["attr_anios_de_uso"].notna()]
    autos = autos.assign(anios_de_uso=pd.cut(autos["attr_anios_de_uso"], [-2, 0, 1, 2, 5, 100],
                                             labels=["0", "1", "2", "3 a 5", "más de 5"]))
    gr.tasa_por_categoria(autos, "anios_de_uso", ejes[1], min_n=10,
                          titulo="Vehículos: años de uso según la ficha")
    rep.figura(fig, "senales_semanticas")
    rep.tabla(tabla)

    rep.texto("""
Banderas simples sobre el texto y la ficha técnica capturan buena parte de lo que
"dice" la publicación. Algunas son casi deterministas aunque cubren pocos items:

- **Título**: "usado" y las descripciones de estado ("como nuevo", "impecable",
  "poco uso") son ~97% usados. Hay que separar "como nuevo" de "nuevo": el primero
  describe algo usado, el segundo baja la tasa a ~12%. La venta mayorista ("x 10
  unidades", "por mayor") es casi siempre de nuevos. "Antiguo", "colección" o un año
  en el título apuntan a objetos usados (discos, revistas, antigüedades).
- **Garantía**: lo que importa es *quién* garantiza. Si la garantía es de fábrica,
  oficial o del importador, el item es nuevo (~1% usados). Si tiene un plazo
  concreto, también (~7%). Si el vendedor habla de su reputación o de que el
  producto funciona, es usado (~65%). Un "Sí" a secas no informa nada.
- **Ficha técnica**: en vehículos, los del año son mayormente nuevos (~12% usados),
  con un año de uso quedan en la mitad y desde 2 años de uso son ~99% usados. En inmuebles, "A estrenar" baja la tasa al
  ~21%. El número de pieza identifica repuestos nuevos (~3%).

Como cada bandera cubre pocos items, el aporte al accuracy global va a ser menor que
el de las variables de stock o vendedor, pero resuelven casos donde esas variables
dudan (por ejemplo un particular que vende algo nuevo en caja).
""")
