"""Señales de contexto: formato de fotos, comportamiento del vendedor y perfiles por palabra.

Todas las variables de este módulo se calculan sin el target. Las que agregan
información de otras publicaciones lo hacen sobre train + test.
"""

import matplotlib.pyplot as plt
import pandas as pd

import features
from eda import graficos as gr


def _tasa(tr, mascara):
    return tr.loc[mascara, "y"].mean()


def fotos(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Formato de las fotos")
    formato = pd.Series("otra proporción", index=tr.index)
    aspecto = tr["pic0_aspecto"].round(2)
    formato[aspecto == 1] = "cuadrada (1:1)"
    formato[aspecto.isin([1.33, 1.34])] = "horizontal 4:3"
    formato[aspecto.isin([0.75, 0.74])] = "vertical 3:4"
    formato[aspecto.isin([1.78, 1.79, 0.56])] = "16:9"
    formato[aspecto.isin([1.5, 0.67])] = "3:2 (réflex)"
    formato[tr["pic0_aspecto"].isna()] = "sin fotos"

    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
    gr.tasa_por_categoria(tr.assign(formato=formato), "formato", ejes[0],
                          titulo="proporción de la primera foto")
    gr.tasa_por_categoria(
        tr.assign(cuadradas=pd.cut(tr["pic_frac_cuadradas"], [-0.01, 0, 0.5, 0.99, 1],
                                   labels=["ninguna", "hasta la mitad", "la mayoría", "todas"])),
        "cuadradas", ejes[1], titulo="fotos cuadradas del item")
    gr.tasa_por_categoria(
        tr.assign(lado=pd.cut(tr["pic_lado_max"], [0, 400, 500, 800, 1199, 1200],
                              labels=["hasta 400", "401 a 500", "501 a 800", "801 a 1199", "1200"])),
        "lado", ejes[2], titulo="lado mayor de la foto original (px)")
    rep.figura(fig, "fotos_formato")

    cuadrada = _tasa(tr, formato == "cuadrada (1:1)")
    vertical = _tasa(tr, formato == "vertical 3:4")
    horizontal = _tasa(tr, formato == "horizontal 4:3")
    reflex = _tasa(tr, formato == "3:2 (réflex)")
    rep.texto(f"""
**Relación oculta**: la proporción de la foto delata cómo se sacó. Una foto cuadrada
es de catálogo o de estudio y solo el {cuadrada:.1%} de esos items es usado. Una foto
vertical 3:4 es de celular ({vertical:.1%} usados) y una horizontal 4:3 es de cámara
compacta ({horizontal:.1%}). La proporción 3:2, propia de cámaras réflex, vuelve a ser
más de comercio ({reflex:.1%}).

La resolución va en el mismo sentido: las fotos que llegan al tope de 1200 px son
fotos caseras subidas sin procesar, mientras que las de comercios suelen venir
recortadas y achicadas. Ninguna de estas variables mira el contenido de la imagen,
solo sus dimensiones.
""")


def comportamiento_vendedor(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Comportamiento del vendedor en sus otras publicaciones")
    m = tr["seller_n_items"] >= 3
    paneles = {
        "seller_frac_repuso_stock": ([-0.01, 0, 0.1, 0.3, 1], "fracción con reposición de stock"),
        "seller_frac_foto_cuadrada": ([-0.01, 0, 0.3, 0.7, 1], "fracción con fotos cuadradas"),
        "seller_max_initial_quantity": ([0, 1, 2, 10, 100, 1e5], "stock máximo publicado"),
        "seller_items_mismo_minuto": ([0, 1, 2, 5, 1e5], "items creados en el mismo minuto"),
        "seller_span_dias": ([-1, 0, 7, 30, 60, 1e4], "días entre primera y última publicación"),
        "seller_frac_pago_tarjeta": ([-0.01, 0, 0.5, 0.99, 1], "fracción que acepta tarjeta"),
    }
    fig, ejes = plt.subplots(2, 3, figsize=(16, 8.5))
    for (col, (cortes, titulo)), ax in zip(paneles.items(), ejes.flat):
        gr.tasa_por_categoria(tr[m].assign(**{col: pd.cut(tr.loc[m, col], cortes)}), col, ax,
                              top=6, titulo=titulo)
    rep.figura(fig, "vendedores_comportamiento", "Solo vendedores con 3 o más items (train + test)")

    nunca_repone = _tasa(tr, m & (tr["seller_frac_repuso_stock"] == 0))
    repone = _tasa(tr, m & (tr["seller_frac_repuso_stock"] > 0.3))
    stock_bajo = _tasa(tr, tr["seller_max_initial_quantity"] <= 2)
    stock_alto = _tasa(tr, tr["seller_max_initial_quantity"] > 10)
    rafaga = _tasa(tr, tr["seller_items_mismo_minuto"] > 5)
    sin_cuadrada = ~tr["pic_tiene_cuadrada"]
    sin_cuadrada_comercio = _tasa(tr, sin_cuadrada & (tr["seller_frac_foto_cuadrada"] > 0.3))
    sin_cuadrada_resto = _tasa(tr, sin_cuadrada & (tr["seller_frac_foto_cuadrada"] == 0))
    rep.texto(f"""
El perfil del vendedor se puede enriquecer mucho más allá de `free` y stock 1. Todo
se calcula sobre train + test juntos y sin el target, así que también existe para
los vendedores que solo aparecen en test.

- **Reposición de stock**: si el vendedor nunca repone, {nunca_repone:.1%} de sus
  items son usados. Si repone en más del 30% de sus publicaciones, {repone:.1%}.
- **Stock máximo**: un vendedor que nunca publicó más de 2 unidades de nada tiene
  {stock_bajo:.1%} de usados. Si alguna vez publicó más de 10, {stock_alto:.1%}.
- **Carga masiva**: más de 5 publicaciones creadas en el mismo minuto indican una
  herramienta de carga, propia de comercios ({rafaga:.1%} usados).
- **Fotos del vendedor**: quien usa fotos cuadradas en otras publicaciones es
  comercio aunque en este item no lo haga. Entre los items *sin* foto cuadrada, los
  de vendedores con más del 30% de publicaciones con fotos cuadradas son
  {sin_cuadrada_comercio:.1%} usados, contra {sin_cuadrada_resto:.1%} de los
  vendedores que nunca las usan.

Estas variables explican por qué un mismo item (stock 1, `bronze`, sin garantía)
puede ser nuevo o usado: depende de quién lo publica.
""")


def perfil_por_palabra(ctx):
    tr, rep = ctx.train, ctx.reporte
    rep.seccion("Perfil por palabra del título")
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
    for col, ax in zip(["pal_frac_iq_1_media", "pal_frac_free_media",
                        "pal_frac_foto_cuadrada_media"], ejes):
        gr.tasa_por_bins(tr, col, ax, bins=10)
    rep.figura(fig, "perfil_palabras")

    deciles = tr.groupby(pd.qcut(tr["pal_frac_iq_1_media"], 10, duplicates="drop"),
                         observed=True)["y"].mean()
    rep.texto(f"""
El título dice qué producto es, y hay productos que casi siempre son usados (un
vinilo, una revista) o nuevos (un repuesto, un cargador). Para capturarlo **sin
usar el target** se arma un perfil por palabra:

1. Para cada palabra se mira cómo se publican los *otros* items que la contienen:
   qué fracción es gratuita, de stock 1, con precio con centavos, con reposición de
   stock o con foto cuadrada. El propio item se excluye de ese cálculo.
2. Cada título se resume con el promedio y el máximo sobre sus palabras.

El resultado ordena muy bien los items: en el decil más bajo de
`pal_frac_iq_1_media` hay {deciles.iloc[0]:.1%} de usados y en el más alto
{deciles.iloc[-1]:.1%}. Solo se consideran palabras presentes en al menos
{features.MIN_TITULOS_POR_PALABRA} títulos.

Es la versión "legal" de un target encoding del título: usa como guía variables
que también existen en test (tipo de publicación, stock) en lugar de la condición.
""")


def control_de_leakage(ctx):
    rep = ctx.reporte
    rep.seccion("Control de leakage")
    rep.texto("""
Todas las features se construyen cumpliendo estas reglas:

- Solo usan campos presentes en el test. `condition` se descarta antes de calcular
  cualquier agregado.
- Los agregados por vendedor, categoría o palabra se calculan sobre train + test
  juntos, así que significan lo mismo en ambos y existen para vendedores y
  categorías que solo aparecen en test.
- El perfil por palabra excluye al propio item (leave-one-out).

`run_verificar_leakage.py` lo comprueba de forma automática con tres controles:

| Control | Qué verifica |
|---|---|
| Invarianza al target | las features no cambian si se mezcla `condition` al azar ni si se elimina la columna |
| Mismas columnas | train y test tienen exactamente las mismas features |
| Disponibilidad en test | ninguna feature tiene más nulos en test que en train |

Para confirmar que el control sirve, se le inyectó una feature tramposa (la tasa de
usados por vendedor) y la detectó por los tres caminos.

""")
    rep.seccion("Features que sí usan el target", nivel=3)
    rep.texto("""
La tasa de usados por vendedor, por categoría y un modelo de texto supervisado son
muy informativos, pero usan `condition`. Viven aparte, en `target_encoding.py`, y
se calculan con folds anidados:

- Las filas a predecir (validación o test) se codifican con las etiquetas del
  entrenamiento. Sus propias etiquetas no intervienen.
- Las filas de entrenamiento se codifican por sub-folds: cada una recibe valores
  calculados con las etiquetas de los otros sub-folds.

No se usa leave-one-out porque ahí el valor de cada fila depende, al revés, de su
propia etiqueta, y un modelo de árboles puede deducirla. El verificador suma dos
controles para esta parte:

| Control | Qué verifica | Qué error atrapa |
|---|---|---|
| Sin etiqueta propia | al invertir las etiquetas de un sub-fold, sus features no cambian | codificar el entrenamiento con sus propias etiquetas, o con leave-one-out |
| Target mezclado | con etiquetas al azar, el accuracy de validación cae al de la clase mayoritaria | calcular el encoding con todas las etiquetas antes de separar la validación |

Los dos controles se probaron contra implementaciones tramposas. Con el encoding
calculado de forma global, el modelo "acierta" 79% con etiquetas al azar (lo
esperable es 53%). Con la implementación real da 51%.
""")
