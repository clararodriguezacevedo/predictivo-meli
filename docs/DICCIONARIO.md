# Diccionario de variables

Generado por `run_diccionario.py` a partir de `diccionario.py`. Las estadísticas se calculan sobre train + test juntos (100.000 items). La columna *Resumen* muestra mínimo / mediana / máximo para numéricas y los valores más frecuentes para el resto.

| Origen | Significado |
|---|---|
| JSON | campo de primer nivel del JSON original, sin cambios |
| aplanado | extraído de un campo anidado (listas o diccionarios) en `carga.py` |
| feature | calculada en `features.py`, sin usar el target |
| descartada | campo del JSON que no se carga por ser constante o redundante |

## Target

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `condition` | JSON | texto | 30.0% | 2 | `new`, `used` | Condición del artículo declarada por el vendedor al publicar. Es lo que hay que predecir. Solo está en train. |

| Valor de `condition` | Significado |
|---|---|
| `new` | artículo nuevo |
| `used` | artículo usado |

## Identificadores

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `id` | JSON | texto | 0.0% | 100,000 | `MLA3911152707`, `MLA1133955308`, `MLA3705853907` | Id de la publicación (`MLA` + número). En este dataset está anonimizado: el número es aleatorio y no guarda relación con la fecha ni con el target. |
| `seller_id` | JSON | numérica | 0.0% | 35,915 | 1e+09 / 5.522e+09 / 1e+10 | Id del vendedor (anonimizado). Su valor numérico no significa nada, pero como clave para agrupar es de lo más informativo: los vendedores suelen vender solo nuevos o solo usados. |
| `category_id` | JSON | texto | 0.0% | 10,907 | `MLA1227`, `MLA2044`, `MLA41287` | Categoría hoja del árbol de categorías de MercadoLibre Argentina (por ejemplo MLA1227 son libros, según sus títulos). El dataset no trae el nombre de la categoría. |
| `parent_item_id` | JSON | texto | 23.0% | 76,989 | `MLA8336077177`, `MLA2970364564`, `MLA8680365117` | Si el item es una **republicación**, id de la publicación original ("item padre") que fue cerrada y vuelta a publicar. La republicación crea un item nuevo con otro id y puede heredar visitas y ventas del padre. Nulo si es una publicación original. |
| `official_store_id` | JSON | numérica | 99.2% | 198 | 1024 / 6043 / 9996 | Id de la Tienda Oficial (marcas o comercios verificados por MercadoLibre) que publica el item. Nulo si no es tienda oficial. |
| `catalog_product_id` | JSON | numérica | 100.0% | 9 | `5093232.0`, `3051112.0`, `5126117.0` | Id del producto de catálogo de MercadoLibre al que está asociada la publicación. Casi siempre nulo (7 items en train). |

## Tipo de publicación y estado

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `listing_type_id` | JSON | texto | 0.0% | 7 | `bronze`, `free`, `silver` | Tipo de publicación contratado, que define la exposición en búsquedas y el costo. La publicación gratuita solo está disponible para usuarios con pocas ventas (la doc actual menciona hasta 5 transacciones en el último año), por eso identifica a vendedores particulares. |
| `buying_mode` | JSON | texto | 0.0% | 3 | `buy_it_now`, `classified`, `auction` | Modalidad de compra de la publicación. |
| `status` | JSON | texto | 0.0% | 4 | `active`, `paused`, `closed` | Estado de la publicación al momento de la extracción. |
| `sub_status` | aplanado | texto | 0.0% | 4 | `(vacío)`, `suspended`, `expired` | Lista de subestados que detallan el `status`, unidos con `\|`. Vacío en la gran mayoría de los items. Significados interpretados por el nombre. |
| `tags` | aplanado | texto | 0.0% | 8 | `dragged_bids_and_visits`, `(vacío)`, `dragged_bids_and_visits\|good_quality_thumbnail` | Etiquetas que MercadoLibre asigna a la publicación, ordenadas y unidas con `\|`. |
| `automatic_relist` | JSON | bool | 0.0% | 2 | `False`, `True` | Si el vendedor activó la republicación automática cuando la publicación vence. Lo usan sobre todo comercios con stock permanente. |
| `n_deal_ids` | aplanado | numérica | 0.0% | 2 | `0`, `1` | Cantidad de promociones o campañas (deals) en las que participa el item. Viene de la lista `deal_ids`. |

| Valor de `listing_type_id` | Significado |
|---|---|
| `free` | Gratuita, la de menor exposición |
| `bronze` | Bronce |
| `silver` | Plata |
| `gold` | Oro |
| `gold_premium` | Oro Premium |
| `gold_special` | Clásica |
| `gold_pro` | Premium |

| Valor de `buying_mode` | Significado |
|---|---|
| `buy_it_now` | compra inmediata a precio fijo |
| `auction` | subasta, donde `base_price` es la oferta inicial y `price` la puja actual (verificado en los datos) |
| `classified` | clasificado, el comprador contacta al vendedor y no hay compra online. Se usa en autos, inmuebles y servicios |

| Valor de `status` | Significado |
|---|---|
| `active` | activa |
| `paused` | pausada por el vendedor o por moderación |
| `closed` | finalizada |
| `not_yet_active` | creada pero todavía no activa |

| Valor de `sub_status` | Significado |
|---|---|
| `suspended` | suspendida por MercadoLibre (moderación) |
| `expired` | vencida |
| `deleted` | eliminada |

| Valor de `tags` | Significado |
|---|---|
| `dragged_bids_and_visits` | republicación que hereda las ventas y visitas del item padre |
| `dragged_visits` | republicación que hereda solo las visitas del padre |
| `free_relist` | fue republicada gratuitamente |
| `good_quality_thumbnail` | la imagen principal es de buena calidad |
| `poor_quality_thumbnail` | la imagen principal es de mala calidad |

## Precio y moneda

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `price` | JSON | numérica | 0.0% | 10,331 | 0.84 / 250 / 2.222e+09 | Precio actual de la publicación, en la moneda de `currency_id`. En subastas es la puja actual. |
| `base_price` | JSON | numérica | 0.0% | 10,330 | 0.84 / 250 / 2.222e+09 | Precio base. Igual a `price` salvo en 18 subastas, donde es la oferta inicial (muchas veces 1 peso). |
| `original_price` | JSON | numérica | 99.9% | 114 | 120 / 858 / 1.4e+04 | Precio anterior a un descuento (el "precio tachado"). Solo existe si el item está en oferta; en esos 103 casos `price` es menor. Todos son nuevos. |
| `currency_id` | JSON | texto | 0.0% | 2 | `ARS`, `USD` | Moneda del precio. Los 399 items en dólares son todos clasificados (autos, inmuebles). |

| Valor de `currency_id` | Significado |
|---|---|
| `ARS` | pesos argentinos |
| `USD` | dólares |

## Stock y ventas

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `initial_quantity` | JSON | numérica | 0.0% | 450 | 1 / 1 / 9999 | Unidades que el vendedor cargó al crear la publicación. Un usado suele ser una unidad única. |
| `available_quantity` | JSON | numérica | 0.0% | 468 | 1 / 1 / 9999 | Unidades disponibles al momento de la extracción. En teoría es `initial - sold`, pero si el vendedor repone stock queda por encima. |
| `sold_quantity` | JSON | numérica | 0.0% | 317 | 0 / 0 / 8676 | Unidades vendidas. En republicaciones con tag `dragged_bids_and_visits` incluye las ventas heredadas del item padre. |

## Fechas

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `date_created` | JSON | fecha | 0.0% | 87,075 | 2013-05-21 a 2015-10-15 | Fecha y hora (UTC) de creación de la publicación. Casi todo el dataset es de agosto a octubre de 2015. |
| `last_updated` | JSON | fecha | 0.0% | 95,124 | 2014-11-11 a 2015-10-15 | Fecha y hora (UTC) de la última modificación de la publicación (precio, stock, descripción, estado). |
| `start_time` | JSON | fecha | 0.0% | 86,628 | 2013-05-21 a 2015-10-15 | Inicio de la vigencia de la publicación. Viene en milisegundos epoch y se convierte a fecha. Coincide con `date_created` (diferencia menor a 5 segundos) en el 99.5% de los casos. |
| `stop_time` | JSON | fecha | 0.0% | 86,165 | 2015-10-14 a 2024-11-04 | Vencimiento programado de la publicación. En los datos la duración es de 60 días en el 97% de los casos; hay algunas de 7, 30 y 365 días. |

## Ubicación del vendedor

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `seller_state` | aplanado | texto | 0.0% | 24 | `Capital Federal`, `Buenos Aires`, `Santa Fe` | Provincia del vendedor, desde `seller_address.state.name`. Más del 90% es Capital Federal o Buenos Aires. |
| `seller_city` | aplanado | texto | 0.0% | 3,655 | `CABA`, `Buenos Aires`, `Capital Federal` | Ciudad o barrio del vendedor, desde `seller_address.city.name`. Texto libre con más de 3.000 valores distintos. |

## Envío

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `ship_mode` | aplanado | texto | 0.0% | 4 | `me2`, `not_specified`, `custom` | Modalidad de envío (`shipping.mode`). |
| `ship_local_pick_up` | aplanado | bool | 0.0% | 2 | `True`, `False` | Si se permite retirar el producto en persona (`shipping.local_pick_up`). |
| `ship_free_shipping` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor ofrece envío gratis (`shipping.free_shipping`). |
| `ship_has_dimensions` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el envío tiene cargadas las dimensiones y el peso del paquete (`shipping.dimensions`). Solo 13 items en train. |
| `ship_has_free_methods` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si la publicación tiene métodos de envío gratis configurados (`shipping.free_methods`, todos con alcance nacional). |
| `ship_tags` | aplanado | texto | 0.0% | 3 | `(vacío)`, `relisted_with_modifications`, `free_method_upgraded` | Etiquetas del envío (`shipping.tags`), unidas con `\|`. Casi siempre vacío. Significados interpretados por el nombre. |

| Valor de `ship_mode` | Significado |
|---|---|
| `me2` | Mercado Envíos 2, MercadoLibre gestiona la logística y el vendedor despacha con etiqueta prepaga |
| `me1` | Mercado Envíos 1, para productos pesados o voluminosos. Requiere tener ME2 activo y un asesor comercial de MercadoLibre |
| `custom` | envío personalizado, el vendedor carga una tabla de costos por región y hace la logística por su cuenta |
| `not_specified` | sin envío definido, comprador y vendedor acuerdan entre ellos |

| Valor de `ship_tags` | Significado |
|---|---|
| `relisted_with_modifications` | republicada con cambios en el envío |
| `free_method_upgraded` | se mejoró el método de envío gratis |

## Medios de pago

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `accepts_mercadopago` | JSON | bool | 0.0% | 2 | `True`, `False` | Si la publicación acepta pagos con MercadoPago. Es False exactamente en los 1.532 clasificados, donde no hay compra online. |
| `n_payment_methods` | aplanado | numérica | 0.0% | 13 | 0 / 2 / 12 | Cantidad de medios de pago por fuera de MercadoPago que acepta el vendedor (largo de `non_mercado_pago_payment_methods`). |
| `pay_MLAMO` | aplanado | bool | 0.0% | 2 | `True`, `False` | Si el vendedor acepta "Efectivo" como medio de pago por fuera de MercadoPago. |
| `pay_MLATB` | aplanado | bool | 0.0% | 2 | `True`, `False` | Si el vendedor acepta "Transferencia bancaria" como medio de pago por fuera de MercadoPago. |
| `pay_MLAOT` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Tarjeta de crédito" como medio de pago por fuera de MercadoPago. |
| `pay_MLAWC` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Acordar con el comprador" como medio de pago por fuera de MercadoPago. |
| `pay_MLAWT` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Giro postal" como medio de pago por fuera de MercadoPago. |
| `pay_MLAMP` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "MercadoPago" como medio de pago por fuera de MercadoPago. |
| `pay_MLAVS` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Visa" como medio de pago por fuera de MercadoPago. |
| `pay_MLAMC` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "MasterCard" como medio de pago por fuera de MercadoPago. |
| `pay_MLAVE` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Visa Electron" como medio de pago por fuera de MercadoPago. |
| `pay_MLAMS` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Mastercard Maestro" como medio de pago por fuera de MercadoPago. |
| `pay_MLACD` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Contra reembolso" como medio de pago por fuera de MercadoPago. |
| `pay_MLAAM` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "American Express" como medio de pago por fuera de MercadoPago. |
| `pay_MLADC` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Diners" como medio de pago por fuera de MercadoPago. |
| `pay_MLABC` | aplanado | bool | 0.0% | 2 | `False`, `True` | Si el vendedor acepta "Cheque certificado" como medio de pago por fuera de MercadoPago. |

## Contenido de la publicación

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `title` | JSON | texto | 0.0% | 98,823 | `50 Suspensores Para Short De Baño Tela Calada`, `50 Suspensores Para Short De Baño Nido De Abeja`, `Confeccion Short De Baño Bermudas Hombre Y Niño` | Título de la publicación escrito por el vendedor. En los datos llega hasta 100 caracteres. |
| `warranty` | JSON | texto | 60.9% | 10,263 | `Sí`, `Sin garantía`, `SI` | Texto libre sobre la garantía ("6 meses de garantía", "Sin garantía"). Nulo si el vendedor no lo completó (61% de los items). |
| `video_id` | JSON | texto | 97.0% | 2,077 | `QQNfOicE_o8`, `EVCQuwl7RIE`, `mYNRC5Ia1sk` | Id de un video de YouTube asociado a la publicación. Nulo si no tiene. |
| `n_descriptions` | aplanado | numérica | 0.0% | 2 | `1`, `0` | Cantidad de descripciones (0 o 1). El JSON solo trae el id de la descripción, no su texto. |
| `n_attributes` | aplanado | numérica | 0.0% | 55 | 0 / 0 / 81 | Cantidad de atributos de la ficha técnica (`attributes`). Casi siempre 0; los valores altos corresponden a fichas de vehículos (65 atributos) e inmuebles. |
| `attr_ids` | aplanado | texto | 0.0% | 484 | `(vacío)`, `GENDER\|Season`, `MLA-PART_NUMBER\|PART_NUMBER` | Ids de los atributos de la ficha técnica, unidos con `\|`. Los prefijos indican el rubro: `MLA1743-*` y `MLA1744-*` son vehículos (kilómetros, puertas, combustible) y `MLA1459-*` inmuebles. Los más comunes fuera de esos rubros son `GENDER` y `Season` (indumentaria). |
| `n_pictures` | aplanado | numérica | 0.0% | 32 | 0 / 2 / 36 | Cantidad de fotos de la publicación. |
| `pic_max_ancho` | aplanado | numérica | 0.8% | 1,141 | 1 / 900 / 1200 | Ancho máximo en píxeles (de `max_size`) entre las fotos del item. Refleja la resolución con la que se subió la foto. |
| `pic_max_alto` | aplanado | numérica | 0.8% | 1,140 | 1 / 900 / 1200 | Alto máximo en píxeles (de `max_size`) entre las fotos del item. |
| `pic_min_lado` | aplanado | numérica | 0.8% | 1,176 | 1 / 620 / 1200 | El lado más corto de la foto más chica. Detecta publicaciones con alguna foto de baja resolución. |
| `pic_frac_id_fechado` | aplanado | numérica | 0.8% | 20 | 0 / 1 / 1 | Fracción de fotos cuyo id tiene el formato nuevo de MercadoLibre, que termina en el mes y año de subida (por ejemplo `..._052015`). El formato viejo no trae fecha. |
| `pic_mes_mas_viejo` | aplanado | numérica | 8.2% | 48 | 2.414e+04 / 2.418e+04 / 2.419e+04 | Mes de subida de la foto más vieja, en meses absolutos (año * 12 + mes - 1), a partir del id de la foto. Permite ver si las fotos son anteriores a la publicación (fotos reutilizadas). |
| `n_variations` | aplanado | numérica | 0.0% | 32 | 0 / 0 / 50 | Cantidad de variaciones del producto (combinaciones de color, talle, etc.), cada una con su stock propio. Típico de comercios de indumentaria. |
| `var_sold_quantity` | aplanado | numérica | 0.0% | 35 | 0 / 0 / 61 | Suma de unidades vendidas entre todas las variaciones. |
| `var_available_quantity` | aplanado | numérica | 0.0% | 175 | 0 / 0 / 9999 | Suma de unidades disponibles entre todas las variaciones. |
| `var_n_precios_distintos` | aplanado | numérica | 0.0% | 2 | `0`, `1` | Cantidad de precios distintos entre las variaciones. En los datos siempre es 0 o 1 (todas las variaciones de un item tienen el mismo precio), así que es redundante con `n_variations > 0`. |

## Features derivadas del item

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `iq_es_1` | feature | bool | 0.0% | 2 | `True`, `False` | `initial_quantity == 1`: la publicación es de una unidad única. |
| `vendio_algo` | feature | bool | 0.0% | 2 | `False`, `True` | `sold_quantity > 0`. |
| `stock_desbalance` | feature | numérica | 0.0% | 313 | 0 / 0 / 8542 | `available - (initial - sold)`. Es 0 si el stock solo bajó por ventas; positivo si el vendedor repuso stock (huella de comercio). |
| `log_price` | feature | numérica | 0.0% | 10,331 | 0.6098 / 5.525 / 21.52 | `log(1 + price)`. Comprime la escala, que va de menos de 1 peso a miles de millones. |
| `price_con_decimales` | feature | bool | 0.0% | 2 | `False`, `True` | El precio tiene centavos (199.99). Típico de comercios. |
| `price_termina_en_9` | feature | bool | 0.0% | 2 | `False`, `True` | La parte entera del precio termina en 9 (349, 1999). Precio psicológico de comercio. |
| `price_multiplo_100` | feature | bool | 0.0% | 2 | `False`, `True` | El precio es múltiplo de 100. Típico de particulares que redondean. |
| `dias_hasta_update` | feature | numérica | 0.0% | 33,949 | -1.157e-05 / 5.787e-05 / 875.7 | Días entre `date_created` y `last_updated`. Mide cuánto tiempo después de publicar el vendedor siguió editando. |
| `duracion_dias` | feature | numérica | 0.0% | 275 | 0 / 60 / 3457 | Días entre `start_time` y `stop_time`: la vigencia programada. |
| `hora_creacion` | feature | numérica | 0.0% | 24 | 0 / 14 / 23 | Hora de creación en horario argentino (UTC-3). |
| `dia_semana` | feature | numérica | 0.0% | 7 | `1`, `0`, `2` | Día de la semana de creación en horario argentino (0 = lunes). |
| `title_len` | feature | numérica | 0.0% | 94 | 1 / 48 / 100 | Largo del título en caracteres. |
| `title_frac_mayus` | feature | numérica | 0.0% | 327 | 0 / 0.1404 / 0.4286 | Fracción de letras mayúsculas en el título. |
| `warranty_len` | feature | numérica | 0.0% | 258 | 0 / 0 / 266 | Largo del texto de garantía (0 si es nulo). |
| `tiene_warranty` | feature | bool | 0.0% | 2 | `False`, `True` | `warranty` no es nulo. |
| `tiene_video` | feature | bool | 0.0% | 2 | `False`, `True` | `video_id` no es nulo. |
| `es_tienda_oficial` | feature | bool | 0.0% | 2 | `False`, `True` | `official_store_id` no es nulo. |
| `tiene_parent` | feature | bool | 0.0% | 2 | `True`, `False` | `parent_item_id` no es nulo: el item es una republicación. |

## Perfil del vendedor

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `seller_n_items` | feature | numérica | 0.0% | 136 | 1 / 6 / 933 | Cantidad de items del vendedor en train + test. |
| `seller_frac_free` | feature | numérica | 0.0% | 254 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor que son `free`. Mayor a 0 indica un vendedor que califica para publicar gratis, o sea un particular. |
| `seller_frac_iq_1` | feature | numérica | 0.0% | 245 | 0 / 1 / 1 | Fracción de las publicaciones del vendedor con stock inicial 1. |
| `seller_frac_vendio` | feature | numérica | 0.0% | 307 | 0 / 0.001072 / 1 | Fracción de las publicaciones del vendedor con al menos una venta. |
| `seller_frac_warranty` | feature | numérica | 0.0% | 316 | 0 / 0.07692 / 1 | Fracción de las publicaciones del vendedor con garantía cargada. |
| `seller_frac_price_decimales` | feature | numérica | 0.0% | 218 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor con precios con centavos. |
| `seller_mediana_log_price` | feature | numérica | 0.0% | 7,671 | 0.6931 / 5.421 / 21.52 | Mediana del `log_price` de las publicaciones del vendedor. |
| `seller_n_categorias` | feature | numérica | 0.0% | 63 | 1 / 4 / 100 | Cantidad de categorías distintas en las que publica el vendedor. |
| `seller_ratio_categorias` | feature | numérica | 0.0% | 446 | 0.0059 / 0.8 / 1 | `seller_n_categorias / seller_n_items`. Cerca de 1 es un vendedor que publica cosas variadas (alguien vaciando la casa); cerca de 0, un especialista. |

## Perfil de la categoría

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `cat_n_items` | feature | numérica | 0.0% | 191 | 1 / 33 / 4593 | Cantidad de items de la categoría en train + test. |
| `cat_frac_free` | feature | numérica | 0.0% | 474 | 0 / 0.1818 / 1 | Fracción de publicaciones `free` en la categoría. |
| `cat_frac_iq_1` | feature | numérica | 0.0% | 531 | 0 / 0.75 / 1 | Fracción de publicaciones con stock inicial 1 en la categoría. |
| `cat_mediana_log_price` | feature | numérica | 0.0% | 5,093 | 0.6931 / 5.419 / 16.22 | Mediana del `log_price` de la categoría. |
| `log_price_rel_categoria` | feature | numérica | 0.0% | 51,687 | -11.38 / 0 / 16.65 | `log_price - cat_mediana_log_price`: qué tan caro es el item respecto de su categoría (0 = precio típico). |

## Campos descartados

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `site_id` | descartada |  |  |  |  | Sitio de MercadoLibre. Siempre `MLA` (Argentina). |
| `seller_address.country` | descartada |  |  |  |  | País del vendedor. Siempre Argentina (salvo 2 vacíos). |
| `listing_source` | descartada |  |  |  |  | Origen de la publicación. Siempre vacío. |
| `international_delivery_mode` | descartada |  |  |  |  | Modo de envío internacional. Siempre `none`. |
| `coverage_areas` | descartada |  |  |  |  | Zonas de cobertura (se usa en servicios). Siempre vacío. |
| `differential_pricing` | descartada |  |  |  |  | Precios diferenciales por tipo de comprador. Siempre nulo. |
| `subtitle` | descartada |  |  |  |  | Subtítulo de la publicación. Siempre nulo. |
| `permalink` | descartada |  |  |  |  | URL pública de la publicación. Repite el id y el título, sin información nueva. |
| `thumbnail` | descartada |  |  |  |  | URL de la miniatura de la foto principal. Su id ya está en `pictures`. |
| `secure_thumbnail` | descartada |  |  |  |  | Igual que `thumbnail` pero por https. |
| `shipping.methods` | descartada |  |  |  |  | Métodos de envío. Siempre vacío o nulo. |
| `pictures.quality` | descartada |  |  |  |  | Calidad de cada foto. Siempre vacío. |

## Fuentes

Documentación oficial de la API de MercadoLibre (consultada en octubre de 2026). Describe la API actual; el dataset es de 2015, así que algunos valores (por ejemplo los tipos de publicación) cambiaron de nombre o dejaron de existir.

- https://developers.mercadolibre.com.ar/es_ar/envios-personalizados
- https://developers.mercadolibre.com.ar/es_ar/mercadoenvios-modo-1
- https://developers.mercadolibre.com.ar/es_ar/publica-productos
- https://developers.mercadolibre.com.ar/es_ar/re-publica
- https://developers.mercadolibre.com.ar/es_ar/tipos-de-publicacion-y-actualizaciones-de-articulos
