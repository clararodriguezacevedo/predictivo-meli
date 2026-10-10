"""Diccionario de variables: qué significa cada columna del dataset.

Cubre los campos crudos del JSON (incluso los descartados), las columnas que genera
el aplanado en `carga.py` y las features derivadas de `features.py`. Los
significados se apoyan en la documentación oficial de la API de MercadoLibre
(ver FUENTES) y en verificaciones sobre los datos, indicadas como tales.

`run_diccionario.py` combina estas definiciones con estadísticas reales y genera
docs/DICCIONARIO.md.
"""

from dataclasses import dataclass, field

import features
from carga import MEDIOS_DE_PAGO

# Qué mide cada indicador de `features.INDICADORES`, para describir los perfiles
DESCRIPCION_INDICADOR = {
    "free": "que son `free`",
    "bronze": "que son `bronze`",
    "iq_1": "con stock inicial 1",
    "vendio": "con al menos una venta",
    "warranty": "con garantía cargada",
    "price_decimales": "con precio con centavos",
    "repuso_stock": "donde se repuso stock (`stock_desbalance > 0`)",
    "foto_cuadrada": "con alguna foto cuadrada",
    "pago_tarjeta": "que aceptan tarjeta de crédito por fuera de MercadoPago",
    "t_nuevo": 'cuyo título dice "nuevo"',
    "t_estado_de_uso": 'cuyo título describe un estado de uso ("como nuevo", "impecable")',
    "t_antiguo_coleccion": 'cuyo título dice "antiguo", "vintage" o "colección"',
    "w_plazo": "cuya garantía indica un plazo",
}
assert set(DESCRIPCION_INDICADOR) == set(features.INDICADORES)

FUENTES = {
    "tipos": "https://developers.mercadolibre.com.ar/es_ar/tipos-de-publicacion-y-actualizaciones-de-articulos",
    "publicar": "https://developers.mercadolibre.com.ar/es_ar/publica-productos",
    "republicar": "https://developers.mercadolibre.com.ar/es_ar/re-publica",
    "me1": "https://developers.mercadolibre.com.ar/es_ar/mercadoenvios-modo-1",
    "envios_personalizados": "https://developers.mercadolibre.com.ar/es_ar/envios-personalizados",
}

# Origen de cada variable
CRUDA = "JSON"          # campo de primer nivel del JSON, sin cambios
APLANADA = "aplanado"   # extraída de un campo anidado en carga.py
DERIVADA = "feature"    # calculada en features.py
DESCARTADA = "descartada"
CON_TARGET = "con target"  # calculada con folds en target_encoding.py
# Orígenes que no son columnas del dataset de features y no tienen estadísticas
SIN_ESTADISTICAS = (DESCARTADA, CON_TARGET)

GRUPOS = [
    "Target",
    "Identificadores",
    "Tipo de publicación y estado",
    "Precio y moneda",
    "Stock y ventas",
    "Fechas",
    "Ubicación del vendedor",
    "Envío",
    "Medios de pago",
    "Contenido de la publicación",
    "Ficha técnica (vehículos, inmuebles y repuestos)",
    "Features derivadas del item",
    "Palabras clave en título y garantía",
    "Perfil del vendedor",
    "Perfil de la categoría",
    "Perfil por palabra del título",
    "Formato de las fotos",
    "Target encoding (con folds)",
    "Campos descartados",
]


@dataclass
class Variable:
    nombre: str
    grupo: str
    origen: str
    significado: str
    valores: dict = field(default_factory=dict)  # glosario de valores posibles
    fuentes: tuple = ()                           # claves de FUENTES


VARIABLES = [
    # Target
    Variable("condition", "Target", CRUDA,
             "Condición del artículo declarada por el vendedor al publicar. Es lo que hay "
             "que predecir. Solo está en train.",
             {"new": "artículo nuevo", "used": "artículo usado"}, ("publicar",)),

    # Identificadores
    Variable("id", "Identificadores", CRUDA,
             "Id de la publicación (`MLA` + número). En este dataset está anonimizado: el "
             "número es aleatorio y no guarda relación con la fecha ni con el target."),
    Variable("seller_id", "Identificadores", CRUDA,
             "Id del vendedor (anonimizado). Su valor numérico no significa nada, pero como "
             "clave para agrupar es de lo más informativo: los vendedores suelen vender "
             "solo nuevos o solo usados."),
    Variable("category_id", "Identificadores", CRUDA,
             "Categoría hoja del árbol de categorías de MercadoLibre Argentina (por ejemplo "
             "MLA1227 son libros, según sus títulos). El dataset no trae el nombre de la categoría."),
    Variable("parent_item_id", "Identificadores", CRUDA,
             "Si el item es una **republicación**, id de la publicación original (\"item "
             "padre\") que fue cerrada y vuelta a publicar. La republicación crea un item "
             "nuevo con otro id y puede heredar visitas y ventas del padre. Nulo si es una "
             "publicación original.", fuentes=("republicar",)),
    Variable("official_store_id", "Identificadores", CRUDA,
             "Id de la Tienda Oficial (marcas o comercios verificados por MercadoLibre) que "
             "publica el item. Nulo si no es tienda oficial."),
    Variable("catalog_product_id", "Identificadores", CRUDA,
             "Id del producto de catálogo de MercadoLibre al que está asociada la "
             "publicación. Casi siempre nulo (7 items en train)."),

    # Tipo de publicación y estado
    Variable("listing_type_id", "Tipo de publicación y estado", CRUDA,
             "Tipo de publicación contratado, que define la exposición en búsquedas y el "
             "costo. La publicación gratuita solo está disponible para usuarios con pocas "
             "ventas (la doc actual menciona hasta 5 transacciones en el último año), por "
             "eso identifica a vendedores particulares.",
             {"free": "Gratuita, la de menor exposición",
              "bronze": "Bronce",
              "silver": "Plata",
              "gold": "Oro",
              "gold_premium": "Oro Premium",
              "gold_special": "Clásica",
              "gold_pro": "Premium"}, ("tipos",)),
    Variable("buying_mode", "Tipo de publicación y estado", CRUDA,
             "Modalidad de compra de la publicación.",
             {"buy_it_now": "compra inmediata a precio fijo",
              "auction": "subasta, donde `base_price` es la oferta inicial y `price` la puja "
                         "actual (verificado en los datos)",
              "classified": "clasificado, el comprador contacta al vendedor y no hay compra "
                            "online. Se usa en autos, inmuebles y servicios"}, ("publicar",)),
    Variable("status", "Tipo de publicación y estado", CRUDA,
             "Estado de la publicación al momento de la extracción.",
             {"active": "activa", "paused": "pausada por el vendedor o por moderación",
              "closed": "finalizada", "not_yet_active": "creada pero todavía no activa"},
             ("publicar",)),
    Variable("sub_status", "Tipo de publicación y estado", APLANADA,
             "Lista de subestados que detallan el `status`, unidos con `|`. Vacío en la gran "
             "mayoría de los items. Significados interpretados por el nombre.",
             {"suspended": "suspendida por MercadoLibre (moderación)",
              "expired": "vencida", "deleted": "eliminada"}),
    Variable("tags", "Tipo de publicación y estado", APLANADA,
             "Etiquetas que MercadoLibre asigna a la publicación, ordenadas y unidas con `|`.",
             {"dragged_bids_and_visits": "republicación que hereda las ventas y visitas del "
                                         "item padre",
              "dragged_visits": "republicación que hereda solo las visitas del padre",
              "free_relist": "fue republicada gratuitamente",
              "good_quality_thumbnail": "la imagen principal es de buena calidad",
              "poor_quality_thumbnail": "la imagen principal es de mala calidad"}, ("publicar",)),
    Variable("automatic_relist", "Tipo de publicación y estado", CRUDA,
             "Si el vendedor activó la republicación automática cuando la publicación "
             "vence. Lo usan sobre todo comercios con stock permanente."),
    Variable("n_deal_ids", "Tipo de publicación y estado", APLANADA,
             "Cantidad de promociones o campañas (deals) en las que participa el item. Viene "
             "de la lista `deal_ids`."),

    # Precio y moneda
    Variable("price", "Precio y moneda", CRUDA,
             "Precio actual de la publicación, en la moneda de `currency_id`. En subastas es "
             "la puja actual."),
    Variable("base_price", "Precio y moneda", CRUDA,
             "Precio base. Igual a `price` salvo en 18 subastas, donde es la oferta inicial "
             "(muchas veces 1 peso)."),
    Variable("original_price", "Precio y moneda", CRUDA,
             "Precio anterior a un descuento (el \"precio tachado\"). Solo existe si el item "
             "está en oferta; en esos 103 casos `price` es menor. Todos son nuevos."),
    Variable("currency_id", "Precio y moneda", CRUDA,
             "Moneda del precio. Los 399 items en dólares son todos clasificados (autos, "
             "inmuebles).",
             {"ARS": "pesos argentinos", "USD": "dólares"}),

    # Stock y ventas
    Variable("initial_quantity", "Stock y ventas", CRUDA,
             "Unidades que el vendedor cargó al crear la publicación. Un usado suele ser una "
             "unidad única."),
    Variable("available_quantity", "Stock y ventas", CRUDA,
             "Unidades disponibles al momento de la extracción. En teoría es `initial - "
             "sold`, pero si el vendedor repone stock queda por encima."),
    Variable("sold_quantity", "Stock y ventas", CRUDA,
             "Unidades vendidas. En republicaciones con tag `dragged_bids_and_visits` "
             "incluye las ventas heredadas del item padre."),

    # Fechas
    Variable("date_created", "Fechas", CRUDA,
             "Fecha y hora (UTC) de creación de la publicación. Casi todo el dataset es de "
             "agosto a octubre de 2015."),
    Variable("last_updated", "Fechas", CRUDA,
             "Fecha y hora (UTC) de la última modificación de la publicación (precio, stock, "
             "descripción, estado)."),
    Variable("start_time", "Fechas", CRUDA,
             "Inicio de la vigencia de la publicación. Viene en milisegundos epoch y se "
             "convierte a fecha. Coincide con `date_created` (diferencia menor a 5 segundos) "
             "en el 99.5% de los casos."),
    Variable("stop_time", "Fechas", CRUDA,
             "Vencimiento programado de la publicación. En los datos la duración es de 60 "
             "días en el 97% de los casos; hay algunas de 7, 30 y 365 días."),

    # Ubicación
    Variable("seller_state", "Ubicación del vendedor", APLANADA,
             "Provincia del vendedor, desde `seller_address.state.name`. Más del 90% es "
             "Capital Federal o Buenos Aires."),
    Variable("seller_city", "Ubicación del vendedor", APLANADA,
             "Ciudad o barrio del vendedor, desde `seller_address.city.name`. Texto libre con "
             "más de 3.000 valores distintos."),

    # Envío
    Variable("ship_mode", "Envío", APLANADA,
             "Modalidad de envío (`shipping.mode`).",
             {"me2": "Mercado Envíos 2, MercadoLibre gestiona la logística y el vendedor "
                     "despacha con etiqueta prepaga",
              "me1": "Mercado Envíos 1, para productos pesados o voluminosos. Requiere "
                     "tener ME2 activo y un asesor comercial de MercadoLibre",
              "custom": "envío personalizado, el vendedor carga una tabla de costos por "
                        "región y hace la logística por su cuenta",
              "not_specified": "sin envío definido, comprador y vendedor acuerdan entre ellos"},
             ("envios_personalizados", "me1")),
    Variable("ship_local_pick_up", "Envío", APLANADA,
             "Si se permite retirar el producto en persona (`shipping.local_pick_up`)."),
    Variable("ship_free_shipping", "Envío", APLANADA,
             "Si el vendedor ofrece envío gratis (`shipping.free_shipping`)."),
    Variable("ship_has_dimensions", "Envío", APLANADA,
             "Si el envío tiene cargadas las dimensiones y el peso del paquete "
             "(`shipping.dimensions`). Solo 13 items en train."),
    Variable("ship_has_free_methods", "Envío", APLANADA,
             "Si la publicación tiene métodos de envío gratis configurados "
             "(`shipping.free_methods`, todos con alcance nacional)."),
    Variable("ship_tags", "Envío", APLANADA,
             "Etiquetas del envío (`shipping.tags`), unidas con `|`. Casi siempre vacío. "
             "Significados interpretados por el nombre.",
             {"relisted_with_modifications": "republicada con cambios en el envío",
              "free_method_upgraded": "se mejoró el método de envío gratis"}),

    # Medios de pago
    Variable("accepts_mercadopago", "Medios de pago", CRUDA,
             "Si la publicación acepta pagos con MercadoPago. Es False exactamente en los "
             "1.532 clasificados, donde no hay compra online."),
    Variable("n_payment_methods", "Medios de pago", APLANADA,
             "Cantidad de medios de pago por fuera de MercadoPago que acepta el vendedor "
             "(largo de `non_mercado_pago_payment_methods`)."),
    *[
        Variable(f"pay_{codigo}", "Medios de pago", APLANADA,
                 f"Si el vendedor acepta \"{descripcion}\" como medio de pago por fuera de "
                 "MercadoPago.")
        for codigo, descripcion in MEDIOS_DE_PAGO.items()
    ],

    # Contenido
    Variable("title", "Contenido de la publicación", CRUDA,
             "Título de la publicación escrito por el vendedor. En los datos llega hasta 100 "
             "caracteres."),
    Variable("warranty", "Contenido de la publicación", CRUDA,
             "Texto libre sobre la garantía (\"6 meses de garantía\", \"Sin garantía\"). Nulo "
             "si el vendedor no lo completó (61% de los items)."),
    Variable("video_id", "Contenido de la publicación", CRUDA,
             "Id de un video de YouTube asociado a la publicación. Nulo si no tiene."),
    Variable("n_descriptions", "Contenido de la publicación", APLANADA,
             "Cantidad de descripciones (0 o 1). El JSON solo trae el id de la descripción, "
             "no su texto."),
    Variable("n_attributes", "Contenido de la publicación", APLANADA,
             "Cantidad de atributos de la ficha técnica (`attributes`). Casi siempre 0; los "
             "valores altos corresponden a fichas de vehículos (65 atributos) e inmuebles."),
    Variable("attr_ids", "Contenido de la publicación", APLANADA,
             "Ids de los atributos de la ficha técnica, unidos con `|`. Los prefijos indican "
             "el rubro: `MLA1743-*` y `MLA1744-*` son vehículos (kilómetros, puertas, "
             "combustible) y `MLA1459-*` inmuebles. Los más comunes "
             "fuera de esos rubros son `GENDER` y `Season` (indumentaria)."),
    Variable("n_pictures", "Contenido de la publicación", APLANADA,
             "Cantidad de fotos de la publicación."),
    Variable("pic_max_ancho", "Contenido de la publicación", APLANADA,
             "Ancho máximo en píxeles (de `max_size`) entre las fotos del item. Refleja la "
             "resolución con la que se subió la foto."),
    Variable("pic_max_alto", "Contenido de la publicación", APLANADA,
             "Alto máximo en píxeles (de `max_size`) entre las fotos del item."),
    Variable("pic_min_lado", "Contenido de la publicación", APLANADA,
             "El lado más corto de la foto más chica. Detecta publicaciones con alguna foto "
             "de baja resolución."),
    Variable("pic_frac_id_fechado", "Contenido de la publicación", APLANADA,
             "Fracción de fotos cuyo id tiene el formato nuevo de MercadoLibre, que termina "
             "en el mes y año de subida (por ejemplo `..._052015`). El formato viejo no trae "
             "fecha."),
    Variable("pic_mes_mas_viejo", "Contenido de la publicación", APLANADA,
             "Mes de subida de la foto más vieja, en meses absolutos (año * 12 + mes - 1), "
             "a partir del id de la foto. Permite ver si las fotos son anteriores a la "
             "publicación (fotos reutilizadas)."),
    Variable("n_variations", "Contenido de la publicación", APLANADA,
             "Cantidad de variaciones del producto (combinaciones de color, talle, etc.), "
             "cada una con su stock propio. Típico de comercios de indumentaria."),
    Variable("var_sold_quantity", "Contenido de la publicación", APLANADA,
             "Suma de unidades vendidas entre todas las variaciones."),
    Variable("var_available_quantity", "Contenido de la publicación", APLANADA,
             "Suma de unidades disponibles entre todas las variaciones."),
    Variable("var_n_precios_distintos", "Contenido de la publicación", APLANADA,
             "Cantidad de precios distintos entre las variaciones. En los datos siempre es 0 "
             "o 1 (todas las variaciones de un item tienen el mismo precio), así que es "
             "redundante con `n_variations > 0`."),

    # Features derivadas del item
    Variable("iq_es_1", "Features derivadas del item", DERIVADA,
             "`initial_quantity == 1`: la publicación es de una unidad única."),
    Variable("vendio_algo", "Features derivadas del item", DERIVADA,
             "`sold_quantity > 0`."),
    Variable("stock_desbalance", "Features derivadas del item", DERIVADA,
             "`available - (initial - sold)`. Es 0 si el stock solo bajó por ventas; "
             "positivo si el vendedor repuso stock (huella de comercio)."),
    Variable("log_price", "Features derivadas del item", DERIVADA,
             "`log(1 + price)`. Comprime la escala, que va de menos de 1 peso a miles de "
             "millones."),
    Variable("price_con_decimales", "Features derivadas del item", DERIVADA,
             "El precio tiene centavos (199.99). Típico de comercios."),
    Variable("price_termina_en_9", "Features derivadas del item", DERIVADA,
             "La parte entera del precio termina en 9 (349, 1999). Precio psicológico de "
             "comercio."),
    Variable("price_multiplo_100", "Features derivadas del item", DERIVADA,
             "El precio es múltiplo de 100. Típico de particulares que redondean."),
    Variable("dias_hasta_update", "Features derivadas del item", DERIVADA,
             "Días entre `date_created` y `last_updated`. Mide cuánto tiempo después de "
             "publicar el vendedor siguió editando."),
    Variable("duracion_dias", "Features derivadas del item", DERIVADA,
             "Días entre `start_time` y `stop_time`: la vigencia programada."),
    Variable("hora_creacion", "Features derivadas del item", DERIVADA,
             "Hora de creación en horario argentino (UTC-3)."),
    Variable("dia_semana", "Features derivadas del item", DERIVADA,
             "Día de la semana de creación en horario argentino (0 = lunes)."),
    Variable("title_len", "Features derivadas del item", DERIVADA,
             "Largo del título en caracteres."),
    Variable("title_n_digitos", "Features derivadas del item", DERIVADA,
             "Cantidad de dígitos en el título. Los comercios incluyen modelos, medidas y "
             "códigos."),
    Variable("titulo_repetido", "Features derivadas del item", DERIVADA,
             "Cantidad de publicaciones (train + test) con exactamente el mismo título "
             "normalizado, contando la propia."),
    Variable("title_frac_mayus", "Features derivadas del item", DERIVADA,
             "Fracción de letras mayúsculas en el título."),
    Variable("warranty_len", "Features derivadas del item", DERIVADA,
             "Largo del texto de garantía (0 si es nulo)."),
    Variable("tiene_warranty", "Features derivadas del item", DERIVADA,
             "`warranty` no es nulo."),
    Variable("tiene_video", "Features derivadas del item", DERIVADA,
             "`video_id` no es nulo."),
    Variable("es_tienda_oficial", "Features derivadas del item", DERIVADA,
             "`official_store_id` no es nulo."),
    Variable("tiene_parent", "Features derivadas del item", DERIVADA,
             "`parent_item_id` no es nulo: el item es una republicación."),

    # Ficha técnica
    Variable("attr_kilometros", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "Kilómetros declarados en la ficha de un vehículo. Un 0 indica un vehículo 0 km."),
    Variable("attr_anio", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "Año de fabricación declarado en la ficha de un vehículo."),
    Variable("attr_antiguedad", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "Antigüedad de un inmueble, en rangos (\"A estrenar\", \"Hasta 5 años\", ..., "
             "\"Más de 40 años\")."),
    Variable("attr_estado", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "Estado declarado del inmueble o bien (\"Excelente\", \"Muy bueno\", \"Bueno\", "
             "\"A refaccionar\"). Algunas categorías usan códigos numéricos (0 a 6), todos de "
             "items nuevos."),
    Variable("attr_unico_duenio", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "Si el vehículo tuvo un único dueño (True / False). Nulo si no se informa."),
    Variable("attr_numero_pieza", "Ficha técnica (vehículos, inmuebles y repuestos)", APLANADA,
             "La ficha incluye el número de pieza del fabricante. Típico de repuestos "
             "nuevos de comercios."),
    Variable("attr_es_0km", "Ficha técnica (vehículos, inmuebles y repuestos)", DERIVADA,
             "`attr_kilometros == 0`."),
    Variable("attr_anios_de_uso", "Ficha técnica (vehículos, inmuebles y repuestos)", DERIVADA,
             "`2015 - attr_anio`: años de uso del vehículo al momento de publicar."),
    Variable("attr_a_estrenar", "Ficha técnica (vehículos, inmuebles y repuestos)", DERIVADA,
             "`attr_antiguedad == \"A estrenar\"`."),

    # Palabras clave (patrones en features.PATRONES_TITULO y PATRONES_GARANTIA)
    Variable("t_nuevo", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"nuevo/a/os/as\", excluyendo \"como nuevo\", \"igual a nuevo\" "
             "y \"casi nuevo\"."),
    Variable("t_usado", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"usado/a\" o \"segunda mano\"."),
    Variable("t_estado_de_uso", "Palabras clave en título y garantía", DERIVADA,
             "El título describe el estado de algo ya usado: \"como nuevo\", \"igual a nuevo\", "
             "\"poco uso\", \"impecable\", \"estado\"."),
    Variable("t_0km_a_estrenar", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"0 km\" o \"a estrenar\"."),
    Variable("t_sellado_en_caja", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"sellado\", \"blister\" o \"en caja\"."),
    Variable("t_mayorista", "Palabras clave en título y garantía", DERIVADA,
             "El título habla de venta por mayor o por cantidad (\"por mayor\", \"x 10 "
             "unidades\")."),
    Variable("t_antiguo_coleccion", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"antiguo\", \"vintage\", \"retro\", \"colección\" o \"de época\"."),
    Variable("t_original", "Palabras clave en título y garantía", DERIVADA,
             "El título dice \"original\"."),
    Variable("t_anio", "Palabras clave en título y garantía", DERIVADA,
             "El título menciona un año entre 1950 y 2015 (ediciones de discos y revistas, "
             "modelos de autos)."),
    Variable("w_fabrica", "Palabras clave en título y garantía", DERIVADA,
             "La garantía menciona fábrica, fabricante, oficial o importador."),
    Variable("w_plazo", "Palabras clave en título y garantía", DERIVADA,
             "La garantía indica un plazo (\"6 meses\", \"1 año\", \"30 días\")."),
    Variable("w_sin_garantia", "Palabras clave en título y garantía", DERIVADA,
             "La garantía dice explícitamente que no hay (\"sin garantía\", \"no tiene\")."),
    Variable("w_solo_si", "Palabras clave en título y garantía", DERIVADA,
             "La garantía es solo \"Sí\", sin más detalle."),
    Variable("w_vendedor_o_estado", "Palabras clave en título y garantía", DERIVADA,
             "La garantía habla del vendedor, su reputación o el estado del producto "
             "(\"funciona perfecto\", \"probado\"). El vendedor responde por su palabra, no un "
             "fabricante."),

    # Perfil del vendedor
    Variable("seller_n_items", "Perfil del vendedor", DERIVADA,
             "Cantidad de items del vendedor en train + test."),
    *[
        Variable(f"seller_frac_{ind}", "Perfil del vendedor", DERIVADA,
                 f"Fracción de las publicaciones del vendedor {desc}."
                 + (" Mayor a 0 indica un vendedor que califica para publicar gratis, o sea "
                    "un particular." if ind == "free" else ""))
        for ind, desc in DESCRIPCION_INDICADOR.items()
    ],
    Variable("seller_mediana_log_price", "Perfil del vendedor", DERIVADA,
             "Mediana del `log_price` de las publicaciones del vendedor."),
    Variable("seller_n_categorias", "Perfil del vendedor", DERIVADA,
             "Cantidad de categorías distintas en las que publica el vendedor."),
    Variable("seller_ratio_categorias", "Perfil del vendedor", DERIVADA,
             "`seller_n_categorias / seller_n_items`. Cerca de 1 es un vendedor que publica "
             "cosas variadas (alguien vaciando la casa); cerca de 0, un especialista."),

    Variable("seller_std_log_price", "Perfil del vendedor", DERIVADA,
             "Desvío estándar del `log_price` entre las publicaciones del vendedor. Nulo si "
             "tiene un solo item."),
    Variable("seller_max_initial_quantity", "Perfil del vendedor", DERIVADA,
             "Mayor stock inicial entre todas las publicaciones del vendedor. Si nunca "
             "publicó más de 2 unidades de algo, casi seguro es un particular."),
    Variable("seller_media_n_pictures", "Perfil del vendedor", DERIVADA,
             "Promedio de fotos por publicación del vendedor."),
    Variable("seller_n_garantias_distintas", "Perfil del vendedor", DERIVADA,
             "Cantidad de textos de garantía distintos que usa el vendedor."),
    Variable("seller_dias_activos", "Perfil del vendedor", DERIVADA,
             "Cantidad de días distintos en los que el vendedor creó publicaciones."),
    Variable("seller_span_dias", "Perfil del vendedor", DERIVADA,
             "Días entre la primera y la última publicación del vendedor en el dataset."),
    Variable("seller_items_mismo_dia", "Perfil del vendedor", DERIVADA,
             "Publicaciones que el vendedor creó el mismo día que este item (incluyéndolo)."),
    Variable("seller_items_mismo_minuto", "Perfil del vendedor", DERIVADA,
             "Publicaciones que el vendedor creó en el mismo minuto que este item. Valores "
             "altos indican carga masiva con una herramienta, típica de comercios."),
    Variable("seller_seg_desde_previo", "Perfil del vendedor", DERIVADA,
             "Segundos desde la publicación anterior del mismo vendedor. Nulo en su primera "
             "publicación."),

    # Perfil de la categoría
    Variable("cat_n_items", "Perfil de la categoría", DERIVADA,
             "Cantidad de items de la categoría en train + test."),
    *[
        Variable(f"cat_frac_{ind}", "Perfil de la categoría", DERIVADA,
                 f"Fracción de las publicaciones de la categoría {DESCRIPCION_INDICADOR[ind]}.")
        for ind in features.INDICADORES_CATEGORIA
    ],
    Variable("cat_n_sellers", "Perfil de la categoría", DERIVADA,
             "Cantidad de vendedores distintos en la categoría."),
    Variable("cat_ratio_sellers", "Perfil de la categoría", DERIVADA,
             "`cat_n_sellers / cat_n_items`. Cerca de 1 es una categoría donde cada vendedor "
             "publica un solo item; cerca de 0, una dominada por pocos vendedores."),
    Variable("cat_mediana_log_price", "Perfil de la categoría", DERIVADA,
             "Mediana del `log_price` de la categoría."),
    Variable("log_price_rel_categoria", "Perfil de la categoría", DERIVADA,
             "`log_price - cat_mediana_log_price`: qué tan caro es el item respecto de su "
             "categoría (0 = precio típico)."),

    # Perfil por palabra del título
    *[
        Variable(f"pal_frac_{ind}_{resumen}", "Perfil por palabra del título", DERIVADA,
                 f"Para cada palabra del título se calcula la fracción de las *otras* "
                 f"publicaciones que la contienen {DESCRIPCION_INDICADOR[ind]}. Esta variable "
                 f"es {texto} de esos valores entre las palabras del título.")
        for ind in features.INDICADORES_PALABRA
        for resumen, texto in [("media", "el promedio"), ("max", "el máximo")]
    ],

    # Formato de las fotos
    Variable("pic0_ancho", "Formato de las fotos", APLANADA,
             "Ancho en píxeles con que se muestra la primera foto (`size`, lado mayor de 500 "
             "como máximo). Junto con el alto da la proporción original de la imagen."),
    Variable("pic0_alto", "Formato de las fotos", APLANADA,
             "Alto en píxeles con que se muestra la primera foto."),
    Variable("pic_frac_cuadradas", "Formato de las fotos", APLANADA,
             "Fracción de las fotos del item que son cuadradas. Las fotos de catálogo o de "
             "estudio suelen ser cuadradas."),
    Variable("pic_n_tamanios", "Formato de las fotos", APLANADA,
             "Cantidad de tamaños distintos entre las fotos del item."),
    Variable("pic0_aspecto", "Formato de las fotos", DERIVADA,
             "`pic0_ancho / pic0_alto`. 1 es cuadrada, 1.33 horizontal 4:3, 0.75 vertical 3:4."),
    Variable("pic_tiene_cuadrada", "Formato de las fotos", DERIVADA,
             "`pic_frac_cuadradas > 0`, es decir, al menos una foto cuadrada."),
    Variable("pic_aspecto_camara", "Formato de las fotos", DERIVADA,
             "La primera foto tiene proporción 4:3 o 16:9 (horizontal o vertical), típica de "
             "una foto sacada con celular o cámara compacta sin recortar."),
    Variable("pic_lado_max", "Formato de las fotos", DERIVADA,
             "El lado más largo entre todas las fotos, en su resolución original "
             "(`max_size`). El tope de MercadoLibre es 1200 px."),

    # Target encoding: no están en features.construir, se calculan dentro de cada fold
    Variable("te_seller_id", "Target encoding (con folds)", CON_TARGET,
             "Tasa de usados del vendedor, suavizada hacia la tasa global cuando tiene "
             "pocos items. Se calcula solo con etiquetas de otras filas: por sub-folds para "
             "el entrenamiento y con todo el entrenamiento para validación o test. Nulo si "
             "el vendedor no tiene ningún item con etiqueta."),
    Variable("te_seller_id_n", "Target encoding (con folds)", CON_TARGET,
             "Cantidad de items con etiqueta del vendedor en los que se apoya `te_seller_id`. "
             "Le dice al modelo cuánto confiar en esa tasa."),
    Variable("te_category_id", "Target encoding (con folds)", CON_TARGET,
             "Tasa de usados de la categoría, calculada igual que `te_seller_id`."),
    Variable("te_category_id_n", "Target encoding (con folds)", CON_TARGET,
             "Cantidad de items con etiqueta de la categoría en los que se apoya "
             "`te_category_id`."),
    Variable("te_texto", "Target encoding (con folds)", CON_TARGET,
             "Probabilidad de usado según una regresión logística sobre las palabras y "
             "pares de palabras del título y de la garantía (TF-IDF). El modelo de texto se "
             "entrena solo con etiquetas de otras filas, con el mismo esquema de folds."),
    Variable("te_vecinos_tasa", "Target encoding (con folds)", CON_TARGET,
             "Tasa de usados entre los 30 títulos más parecidos (similitud coseno sobre "
             "TF-IDF) de las filas con etiqueta. Mismo esquema de folds que el resto. Nulo "
             "si ningún título comparte palabras con el del item."),
    Variable("te_vecinos_tasa_ponderada", "Target encoding (con folds)", CON_TARGET,
             "Igual que `te_vecinos_tasa`, pero cada vecino pesa según su similitud."),
    Variable("te_vecinos_similitud_max", "Target encoding (con folds)", CON_TARGET,
             "Similitud del título más parecido (0 a 1). Cerca de 1 indica que existe una "
             "publicación casi idéntica con etiqueta."),
    Variable("te_vecinos_similitud_media", "Target encoding (con folds)", CON_TARGET,
             "Similitud promedio de los vecinos encontrados."),
    Variable("te_vecinos_n", "Target encoding (con folds)", CON_TARGET,
             "Cantidad de vecinos encontrados (hasta 30)."),

    # Descartados
    Variable("site_id", "Campos descartados", DESCARTADA,
             "Sitio de MercadoLibre. Siempre `MLA` (Argentina)."),
    Variable("seller_address.country", "Campos descartados", DESCARTADA,
             "País del vendedor. Siempre Argentina (salvo 2 vacíos)."),
    Variable("listing_source", "Campos descartados", DESCARTADA,
             "Origen de la publicación. Siempre vacío."),
    Variable("international_delivery_mode", "Campos descartados", DESCARTADA,
             "Modo de envío internacional. Siempre `none`."),
    Variable("coverage_areas", "Campos descartados", DESCARTADA,
             "Zonas de cobertura (se usa en servicios). Siempre vacío."),
    Variable("differential_pricing", "Campos descartados", DESCARTADA,
             "Precios diferenciales por tipo de comprador. Siempre nulo."),
    Variable("subtitle", "Campos descartados", DESCARTADA,
             "Subtítulo de la publicación. Siempre nulo."),
    Variable("permalink", "Campos descartados", DESCARTADA,
             "URL pública de la publicación. Repite el id y el título, sin información "
             "nueva."),
    Variable("thumbnail", "Campos descartados", DESCARTADA,
             "URL de la miniatura de la foto principal. Su id ya está en `pictures`."),
    Variable("secure_thumbnail", "Campos descartados", DESCARTADA,
             "Igual que `thumbnail` pero por https."),
    Variable("shipping.methods", "Campos descartados", DESCARTADA,
             "Métodos de envío. Siempre vacío o nulo."),
    Variable("pictures.quality", "Campos descartados", DESCARTADA,
             "Calidad de cada foto. Siempre vacío."),
]


def por_nombre():
    return {v.nombre: v for v in VARIABLES}
