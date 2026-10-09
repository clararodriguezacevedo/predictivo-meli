"""Diccionario de variables: qué significa cada columna del dataset.

Cubre los campos crudos del JSON (incluso los descartados), las columnas que genera
el aplanado en `carga.py` y las features derivadas de `features.py`. Los
significados se apoyan en la documentación oficial de la API de MercadoLibre
(ver FUENTES) y en verificaciones sobre los datos, indicadas como tales.

`run_diccionario.py` combina estas definiciones con estadísticas reales y genera
docs/DICCIONARIO.md.
"""

from dataclasses import dataclass, field

from carga import MEDIOS_DE_PAGO

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
    "Features derivadas del item",
    "Perfil del vendedor",
    "Perfil de la categoría",
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

    # Perfil del vendedor
    Variable("seller_n_items", "Perfil del vendedor", DERIVADA,
             "Cantidad de items del vendedor en train + test."),
    Variable("seller_frac_free", "Perfil del vendedor", DERIVADA,
             "Fracción de las publicaciones del vendedor que son `free`. Mayor a 0 indica un "
             "vendedor que califica para publicar gratis, o sea un particular."),
    Variable("seller_frac_iq_1", "Perfil del vendedor", DERIVADA,
             "Fracción de las publicaciones del vendedor con stock inicial 1."),
    Variable("seller_frac_vendio", "Perfil del vendedor", DERIVADA,
             "Fracción de las publicaciones del vendedor con al menos una venta."),
    Variable("seller_frac_warranty", "Perfil del vendedor", DERIVADA,
             "Fracción de las publicaciones del vendedor con garantía cargada."),
    Variable("seller_frac_price_decimales", "Perfil del vendedor", DERIVADA,
             "Fracción de las publicaciones del vendedor con precios con centavos."),
    Variable("seller_mediana_log_price", "Perfil del vendedor", DERIVADA,
             "Mediana del `log_price` de las publicaciones del vendedor."),
    Variable("seller_n_categorias", "Perfil del vendedor", DERIVADA,
             "Cantidad de categorías distintas en las que publica el vendedor."),
    Variable("seller_ratio_categorias", "Perfil del vendedor", DERIVADA,
             "`seller_n_categorias / seller_n_items`. Cerca de 1 es un vendedor que publica "
             "cosas variadas (alguien vaciando la casa); cerca de 0, un especialista."),

    # Perfil de la categoría
    Variable("cat_n_items", "Perfil de la categoría", DERIVADA,
             "Cantidad de items de la categoría en train + test."),
    Variable("cat_frac_free", "Perfil de la categoría", DERIVADA,
             "Fracción de publicaciones `free` en la categoría."),
    Variable("cat_frac_iq_1", "Perfil de la categoría", DERIVADA,
             "Fracción de publicaciones con stock inicial 1 en la categoría."),
    Variable("cat_mediana_log_price", "Perfil de la categoría", DERIVADA,
             "Mediana del `log_price` de la categoría."),
    Variable("log_price_rel_categoria", "Perfil de la categoría", DERIVADA,
             "`log_price - cat_mediana_log_price`: qué tan caro es el item respecto de su "
             "categoría (0 = precio típico)."),

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
