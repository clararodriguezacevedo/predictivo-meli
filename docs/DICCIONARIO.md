# Diccionario de variables

Generado por `run_diccionario.py` a partir de `diccionario.py`. Las estadísticas se calculan sobre train + test juntos (100.000 items). La columna *Resumen* muestra mínimo / mediana / máximo para numéricas y los valores más frecuentes para el resto.

| Origen | Significado |
|---|---|
| JSON | campo de primer nivel del JSON original, sin cambios |
| aplanado | extraído de un campo anidado (listas o diccionarios) en `carga.py` |
| feature | calculada en `features.py`, sin usar el target |
| descartada | campo del JSON que no se carga por ser constante o redundante |
| con target | usa `condition`, se calcula con folds en `target_encoding.py` |

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

## Ficha técnica (vehículos, inmuebles y repuestos)

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `attr_kilometros` | aplanado | numérica | 98.8% | 378 | 0 / 6.5e+04 / 1.111e+09 | Kilómetros declarados en la ficha de un vehículo. Un 0 indica un vehículo 0 km. |
| `attr_anio` | aplanado | numérica | 98.7% | 53 | 1900 / 2010 / 2015 | Año de fabricación declarado en la ficha de un vehículo. |
| `attr_antiguedad` | aplanado | texto | 99.3% | 12 | `A estrenar`, `Hasta 5 años`, `Hasta 20 años` | Antigüedad de un inmueble, en rangos ("A estrenar", "Hasta 5 años", ..., "Más de 40 años"). |
| `attr_estado` | aplanado | texto | 99.5% | 11 | `Excelente`, `Muy bueno`, `Bueno` | Estado declarado del inmueble o bien ("Excelente", "Muy bueno", "Bueno", "A refaccionar"). Algunas categorías usan códigos numéricos (0 a 6), todos de items nuevos. |
| `attr_unico_duenio` | aplanado | texto | 99.3% | 2 | `True`, `False` | Si el vehículo tuvo un único dueño (True / False). Nulo si no se informa. |
| `attr_numero_pieza` | aplanado | bool | 0.0% | 2 | `False`, `True` | La ficha incluye el número de pieza del fabricante. Típico de repuestos nuevos de comercios. |
| `attr_es_0km` | feature | bool | 0.0% | 2 | `False`, `True` | `attr_kilometros == 0`. |
| `attr_anios_de_uso` | feature | numérica | 98.7% | 53 | 0 / 5 / 115 | `2015 - attr_anio`: años de uso del vehículo al momento de publicar. |
| `attr_a_estrenar` | feature | bool | 0.0% | 2 | `False`, `True` | `attr_antiguedad == "A estrenar"`. |

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
| `duracion_resto_seg` | feature | numérica | 0.0% | 2,507 | 0 / 0 / 8.64e+04 | Segundos que sobran de la duración (`stop_time - start_time`) después de contar los días enteros. El 96% de las publicaciones dura un número exacto de días (resto 0). Con un resto de 1 a 60 segundos el 98% son usadas (99.7% en las bronze). No se sabe qué proceso de MercadoLibre genera ese desfasaje. |
| `dias_update_a_fin` | feature | numérica | 0.0% | 33,776 | -0.0002662 / 60 / 3457 | Días entre `last_updated` y `stop_time`. Las publicaciones editadas cerca de su vencimiento tienen menos usados (37% contra 57%). |
| `seg_creacion_a_inicio` | feature | numérica | 0.0% | 426 | -7 / 0 / 2.994e+06 | Segundos entre `date_created` y `start_time`. Casi siempre es 0; cuando la publicación arranca después de creada hay más usados. |
| `dia_creacion` | feature | numérica | 0.0% | 304 | -590 / 259 / 287 | Días desde el 1 de enero de 2015 hasta la creación. |
| `hora_update` | feature | numérica | 0.0% | 24 | 0 / 14 / 23 | Hora de la última actualización en horario argentino (UTC-3). |
| `title_len` | feature | numérica | 0.0% | 94 | 1 / 48 / 100 | Largo del título en caracteres. |
| `title_n_digitos` | feature | numérica | 0.0% | 37 | 0 / 0 / 42 | Cantidad de dígitos en el título. Los comercios incluyen modelos, medidas y códigos. |
| `titulo_repetido` | feature | numérica | 0.0% | 18 | 1 / 1 / 78 | Cantidad de publicaciones (train + test) con exactamente el mismo título normalizado, contando la propia. |
| `title_frac_mayus` | feature | numérica | 0.0% | 327 | 0 / 0.1404 / 0.4286 | Fracción de letras mayúsculas en el título. |
| `warranty_len` | feature | numérica | 0.0% | 258 | 0 / 0 / 266 | Largo del texto de garantía (0 si es nulo). |
| `tiene_warranty` | feature | bool | 0.0% | 2 | `False`, `True` | `warranty` no es nulo. |
| `tiene_video` | feature | bool | 0.0% | 2 | `False`, `True` | `video_id` no es nulo. |
| `es_tienda_oficial` | feature | bool | 0.0% | 2 | `False`, `True` | `official_store_id` no es nulo. |
| `tiene_parent` | feature | bool | 0.0% | 2 | `True`, `False` | `parent_item_id` no es nulo: el item es una republicación. |

## Palabras clave en título y garantía

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `t_nuevo` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "nuevo/a/os/as", excluyendo "como nuevo", "igual a nuevo" y "casi nuevo". |
| `t_usado` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "usado/a" o "segunda mano". |
| `t_estado_de_uso` | feature | bool | 0.0% | 2 | `False`, `True` | El título describe el estado de algo ya usado: "como nuevo", "igual a nuevo", "poco uso", "impecable", "estado". |
| `t_0km_a_estrenar` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "0 km" o "a estrenar". |
| `t_sellado_en_caja` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "sellado", "blister" o "en caja". |
| `t_mayorista` | feature | bool | 0.0% | 2 | `False`, `True` | El título habla de venta por mayor o por cantidad ("por mayor", "x 10 unidades"). |
| `t_antiguo_coleccion` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "antiguo", "vintage", "retro", "colección" o "de época". |
| `t_original` | feature | bool | 0.0% | 2 | `False`, `True` | El título dice "original". |
| `t_anio` | feature | bool | 0.0% | 2 | `False`, `True` | El título menciona un año entre 1950 y 2015 (ediciones de discos y revistas, modelos de autos). |
| `w_fabrica` | feature | bool | 0.0% | 2 | `False`, `True` | La garantía menciona fábrica, fabricante, oficial o importador. |
| `w_plazo` | feature | bool | 0.0% | 2 | `False`, `True` | La garantía indica un plazo ("6 meses", "1 año", "30 días"). |
| `w_sin_garantia` | feature | bool | 0.0% | 2 | `False`, `True` | La garantía dice explícitamente que no hay ("sin garantía", "no tiene"). |
| `w_solo_si` | feature | bool | 0.0% | 2 | `False`, `True` | La garantía es solo "Sí", sin más detalle. |
| `w_vendedor_o_estado` | feature | bool | 0.0% | 2 | `False`, `True` | La garantía habla del vendedor, su reputación o el estado del producto ("funciona perfecto", "probado"). El vendedor responde por su palabra, no un fabricante. |

## Perfil del vendedor

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `seller_n_items` | feature | numérica | 0.0% | 136 | 1 / 6 / 933 | Cantidad de items del vendedor en train + test. |
| `seller_frac_free` | feature | numérica | 0.0% | 254 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor que son `free`. Mayor a 0 indica un vendedor que califica para publicar gratis, o sea un particular. |
| `seller_frac_bronze` | feature | numérica | 0.0% | 290 | 0 / 0.943 / 1 | Fracción de las publicaciones del vendedor que son `bronze`. |
| `seller_frac_iq_1` | feature | numérica | 0.0% | 245 | 0 / 1 / 1 | Fracción de las publicaciones del vendedor con stock inicial 1. |
| `seller_frac_vendio` | feature | numérica | 0.0% | 307 | 0 / 0.001072 / 1 | Fracción de las publicaciones del vendedor con al menos una venta. |
| `seller_frac_warranty` | feature | numérica | 0.0% | 316 | 0 / 0.07692 / 1 | Fracción de las publicaciones del vendedor con garantía cargada. |
| `seller_frac_price_decimales` | feature | numérica | 0.0% | 218 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor con precio con centavos. |
| `seller_frac_repuso_stock` | feature | numérica | 0.0% | 302 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor donde se repuso stock (`stock_desbalance > 0`). |
| `seller_frac_foto_cuadrada` | feature | numérica | 0.0% | 265 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor con alguna foto cuadrada. |
| `seller_frac_pago_tarjeta` | feature | numérica | 0.0% | 283 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor que aceptan tarjeta de crédito por fuera de MercadoPago. |
| `seller_frac_t_nuevo` | feature | numérica | 0.0% | 197 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor cuyo título dice "nuevo". |
| `seller_frac_t_estado_de_uso` | feature | numérica | 0.0% | 114 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor cuyo título describe un estado de uso ("como nuevo", "impecable"). |
| `seller_frac_t_antiguo_coleccion` | feature | numérica | 0.0% | 193 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor cuyo título dice "antiguo", "vintage" o "colección". |
| `seller_frac_w_plazo` | feature | numérica | 0.0% | 136 | 0 / 0 / 1 | Fracción de las publicaciones del vendedor cuya garantía indica un plazo. |
| `seller_mediana_log_price` | feature | numérica | 0.0% | 7,671 | 0.6931 / 5.421 / 21.52 | Mediana del `log_price` de las publicaciones del vendedor. |
| `seller_n_categorias` | feature | numérica | 0.0% | 63 | 1 / 4 / 100 | Cantidad de categorías distintas en las que publica el vendedor. |
| `seller_ratio_categorias` | feature | numérica | 0.0% | 446 | 0.0059 / 0.8 / 1 | `seller_n_categorias / seller_n_items`. Cerca de 1 es un vendedor que publica cosas variadas (alguien vaciando la casa); cerca de 0, un especialista. |
| `seller_std_log_price` | feature | numérica | 25.0% | 9,745 | 0 / 0.6624 / 7.173 | Desvío estándar del `log_price` entre las publicaciones del vendedor. Nulo si tiene un solo item. |
| `seller_max_initial_quantity` | feature | numérica | 0.0% | 341 | 1 / 1 / 9999 | Mayor stock inicial entre todas las publicaciones del vendedor. Si nunca publicó más de 2 unidades de algo, casi seguro es un particular. |
| `seller_media_n_pictures` | feature | numérica | 0.0% | 859 | 0 / 2.54 / 36 | Promedio de fotos por publicación del vendedor. |
| `seller_n_garantias_distintas` | feature | numérica | 0.0% | 24 | 0 / 1 / 61 | Cantidad de textos de garantía distintos que usa el vendedor. |
| `seller_dias_activos` | feature | numérica | 0.0% | 47 | 1 / 4 / 58 | Cantidad de días distintos en los que el vendedor creó publicaciones. |
| `seller_span_dias` | feature | numérica | 0.0% | 140 | 0 / 35 / 861 | Días entre la primera y la última publicación del vendedor en el dataset. |
| `seller_items_mismo_dia` | feature | numérica | 0.0% | 73 | 1 / 1 / 153 | Publicaciones que el vendedor creó el mismo día que este item (incluyéndolo). |
| `seller_items_mismo_minuto` | feature | numérica | 0.0% | 24 | 1 / 1 / 30 | Publicaciones que el vendedor creó en el mismo minuto que este item. Valores altos indican carga masiva con una herramienta, típica de comercios. |
| `seller_seg_desde_previo` | feature | numérica | 35.9% | 36,397 | 0 / 2.433e+04 / 7.339e+07 | Segundos desde la publicación anterior del mismo vendedor. Nulo en su primera publicación. |

## Perfil de la categoría

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `cat_n_items` | feature | numérica | 0.0% | 191 | 1 / 33 / 4593 | Cantidad de items de la categoría en train + test. |
| `cat_frac_free` | feature | numérica | 0.0% | 474 | 0 / 0.1818 / 1 | Fracción de las publicaciones de la categoría que son `free`. |
| `cat_frac_iq_1` | feature | numérica | 0.0% | 531 | 0 / 0.75 / 1 | Fracción de las publicaciones de la categoría con stock inicial 1. |
| `cat_frac_vendio` | feature | numérica | 0.0% | 445 | 0 / 0.08696 / 1 | Fracción de las publicaciones de la categoría con al menos una venta. |
| `cat_frac_warranty` | feature | numérica | 0.0% | 552 | 0 / 0.3883 / 1 | Fracción de las publicaciones de la categoría con garantía cargada. |
| `cat_frac_price_decimales` | feature | numérica | 0.0% | 440 | 0 / 0.08046 / 1 | Fracción de las publicaciones de la categoría con precio con centavos. |
| `cat_frac_repuso_stock` | feature | numérica | 0.0% | 432 | 0 / 0.075 / 1 | Fracción de las publicaciones de la categoría donde se repuso stock (`stock_desbalance > 0`). |
| `cat_frac_foto_cuadrada` | feature | numérica | 0.0% | 470 | 0 / 0.08126 / 1 | Fracción de las publicaciones de la categoría con alguna foto cuadrada. |
| `cat_frac_t_nuevo` | feature | numérica | 0.0% | 261 | 0 / 0.008403 / 1 | Fracción de las publicaciones de la categoría cuyo título dice "nuevo". |
| `cat_frac_t_estado_de_uso` | feature | numérica | 0.0% | 225 | 0 / 0 / 1 | Fracción de las publicaciones de la categoría cuyo título describe un estado de uso ("como nuevo", "impecable"). |
| `cat_frac_t_antiguo_coleccion` | feature | numérica | 0.0% | 285 | 0 / 0 / 1 | Fracción de las publicaciones de la categoría cuyo título dice "antiguo", "vintage" o "colección". |
| `cat_frac_w_plazo` | feature | numérica | 0.0% | 354 | 0 / 0.006006 / 1 | Fracción de las publicaciones de la categoría cuya garantía indica un plazo. |
| `cat_n_sellers` | feature | numérica | 0.0% | 137 | 1 / 23 / 719 | Cantidad de vendedores distintos en la categoría. |
| `cat_ratio_sellers` | feature | numérica | 0.0% | 512 | 0.01562 / 0.7742 / 1 | `cat_n_sellers / cat_n_items`. Cerca de 1 es una categoría donde cada vendedor publica un solo item; cerca de 0, una dominada por pocos vendedores. |
| `cat_mediana_log_price` | feature | numérica | 0.0% | 5,093 | 0.6931 / 5.419 / 16.22 | Mediana del `log_price` de la categoría. |
| `log_price_rel_categoria` | feature | numérica | 0.0% | 51,687 | -11.38 / 0 / 16.65 | `log_price - cat_mediana_log_price`: qué tan caro es el item respecto de su categoría (0 = precio típico). |

## Perfil por palabra del título

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `pal_frac_free_media` | feature | numérica | 0.3% | 96,423 | 0 / 0.197 / 0.8 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen que son `free`. Esta variable es el promedio de esos valores entre las palabras del título. |
| `pal_frac_free_max` | feature | numérica | 0.3% | 1,899 | 0 / 0.3 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen que son `free`. Esta variable es el máximo de esos valores entre las palabras del título. |
| `pal_frac_iq_1_media` | feature | numérica | 0.3% | 96,421 | 0 / 0.6885 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con stock inicial 1. Esta variable es el promedio de esos valores entre las palabras del título. |
| `pal_frac_iq_1_max` | feature | numérica | 0.3% | 2,249 | 0 / 0.8783 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con stock inicial 1. Esta variable es el máximo de esos valores entre las palabras del título. |
| `pal_frac_price_decimales_media` | feature | numérica | 0.3% | 95,794 | 0 / 0.1384 / 0.8 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con precio con centavos. Esta variable es el promedio de esos valores entre las palabras del título. |
| `pal_frac_price_decimales_max` | feature | numérica | 0.3% | 1,886 | 0 / 0.2371 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con precio con centavos. Esta variable es el máximo de esos valores entre las palabras del título. |
| `pal_frac_repuso_stock_media` | feature | numérica | 0.3% | 95,967 | 0 / 0.1562 / 0.75 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen donde se repuso stock (`stock_desbalance > 0`). Esta variable es el promedio de esos valores entre las palabras del título. |
| `pal_frac_repuso_stock_max` | feature | numérica | 0.3% | 1,954 | 0 / 0.2639 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen donde se repuso stock (`stock_desbalance > 0`). Esta variable es el máximo de esos valores entre las palabras del título. |
| `pal_frac_foto_cuadrada_media` | feature | numérica | 0.3% | 95,945 | 0 / 0.1641 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con alguna foto cuadrada. Esta variable es el promedio de esos valores entre las palabras del título. |
| `pal_frac_foto_cuadrada_max` | feature | numérica | 0.3% | 2,080 | 0 / 0.2807 / 1 | Para cada palabra del título se calcula la fracción de las *otras* publicaciones que la contienen con alguna foto cuadrada. Esta variable es el máximo de esos valores entre las palabras del título. |

## Formato de las fotos

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `pic0_ancho` | aplanado | numérica | 0.8% | 452 | 1 / 491 / 500 | Ancho en píxeles con que se muestra la primera foto (`size`, lado mayor de 500 como máximo). Junto con el alto da la proporción original de la imagen. |
| `pic0_alto` | aplanado | numérica | 0.8% | 459 | 1 / 377 / 500 | Alto en píxeles con que se muestra la primera foto. |
| `pic_frac_cuadradas` | aplanado | numérica | 0.8% | 45 | 0 / 0 / 1 | Fracción de las fotos del item que son cuadradas. Las fotos de catálogo o de estudio suelen ser cuadradas. |
| `pic_n_tamanios` | aplanado | numérica | 0.0% | 22 | 0 / 1 / 28 | Cantidad de tamaños distintos entre las fotos del item. |
| `pic0_aspecto` | feature | numérica | 0.8% | 6,873 | 0.05 / 1 / 15.62 | `pic0_ancho / pic0_alto`. 1 es cuadrada, 1.33 horizontal 4:3, 0.75 vertical 3:4. |
| `pic_tiene_cuadrada` | feature | bool | 0.0% | 2 | `False`, `True` | `pic_frac_cuadradas > 0`, es decir, al menos una foto cuadrada. |
| `pic_aspecto_camara` | feature | bool | 0.0% | 2 | `False`, `True` | La primera foto tiene proporción 4:3 o 16:9 (horizontal o vertical), típica de una foto sacada con celular o cámara compacta sin recortar. |
| `pic_lado_max` | feature | numérica | 0.8% | 1,122 | 1 / 1200 / 1200 | El lado más largo entre todas las fotos, en su resolución original (`max_size`). El tope de MercadoLibre es 1200 px. |

## Target encoding (con folds)

| Variable | Origen | Tipo | Nulos | Únicos | Resumen | Significado |
|---|---|---|---|---|---|---|
| `te_seller_id` | con target |  |  |  |  | Tasa de usados del vendedor, suavizada hacia la tasa global cuando tiene pocos items. Se calcula solo con etiquetas de otras filas: por sub-folds para el entrenamiento y con todo el entrenamiento para validación o test. Nulo si el vendedor no tiene ningún item con etiqueta. |
| `te_seller_id_n` | con target |  |  |  |  | Cantidad de items con etiqueta del vendedor en los que se apoya `te_seller_id`. Le dice al modelo cuánto confiar en esa tasa. |
| `te_category_id` | con target |  |  |  |  | Tasa de usados de la categoría, calculada igual que `te_seller_id`. |
| `te_category_id_n` | con target |  |  |  |  | Cantidad de items con etiqueta de la categoría en los que se apoya `te_category_id`. |
| `te_vecinos_tasa` | con target |  |  |  |  | Tasa de usados entre los 30 títulos más parecidos (similitud coseno sobre TF-IDF) de las filas con etiqueta. Mismo esquema de folds que el resto. Nulo si ningún título comparte palabras con el del item. |
| `te_vecinos_tasa_ponderada` | con target |  |  |  |  | Igual que `te_vecinos_tasa`, pero cada vecino pesa según su similitud. |
| `te_vecinos_similitud_max` | con target |  |  |  |  | Similitud del título más parecido (0 a 1). Cerca de 1 indica que existe una publicación casi idéntica con etiqueta. |
| `te_vecinos_similitud_media` | con target |  |  |  |  | Similitud promedio de los vecinos encontrados. |
| `te_vecinos_n` | con target |  |  |  |  | Cantidad de vecinos encontrados (hasta 30). |
| `te_seller_id_x_listing_type_id` | con target |  |  |  |  | Tasa de usados del vendedor entre sus publicaciones del mismo tipo (`listing_type_id`), calculada igual que `te_seller_id`. |
| `te_seller_id_x_listing_type_id_n` | con target |  |  |  |  | Cantidad de items con etiqueta en los que se apoya la tasa anterior. |
| `te_seller_id_x_iq_es_1` | con target |  |  |  |  | Tasa de usados del vendedor separando las publicaciones de una sola unidad del resto. |
| `te_seller_id_x_iq_es_1_n` | con target |  |  |  |  | Cantidad de items con etiqueta en los que se apoya la tasa anterior. |
| `te_seller_id_x_tiene_warranty` | con target |  |  |  |  | Tasa de usados del vendedor separando las publicaciones con y sin garantía. |
| `te_seller_id_x_tiene_warranty_n` | con target |  |  |  |  | Cantidad de items con etiqueta en los que se apoya la tasa anterior. |
| `te_seller_id_x_title` | con target |  |  |  |  | Tasa de usados de las publicaciones del vendedor con exactamente el mismo título (normalizado). Captura republicaciones del mismo producto. |
| `te_seller_id_x_title_n` | con target |  |  |  |  | Cantidad de items con etiqueta en los que se apoya la tasa anterior. |
| `te_vv_similitud_max` | con target |  |  |  |  | Similitud de título (coseno sobre TF-IDF) con el item con etiqueta más parecido del mismo vendedor. Nulo si el vendedor no tiene items con etiqueta. |
| `te_vv_y_mas_parecido` | con target |  |  |  |  | Condición (1 = usado) del item con etiqueta del mismo vendedor con el título más parecido. |
| `te_vv_tasa_ponderada` | con target |  |  |  |  | Tasa de usados del vendedor donde cada item pesa según el cuadrado de la similitud de su título con el del item. |
| `te_vv_y_mas_cercano_en_tiempo` | con target |  |  |  |  | Condición del item con etiqueta del mismo vendedor publicado más cerca en el tiempo (promedio si hay empate). |
| `te_vv_horas_al_mas_cercano` | con target |  |  |  |  | Horas entre el item y esa publicación más cercana del vendedor. |
| `te_vv_tasa_precio_parecido` | con target |  |  |  |  | Tasa de usados entre los items del vendedor con precio parecido (diferencia de log del precio menor a 0.2). |
| `te_vv_n_precio_parecido` | con target |  |  |  |  | Cantidad de items con etiqueta del vendedor con precio parecido. |
| `te_vv_tasa_mismo_tipo_y_stock` | con target |  |  |  |  | Tasa de usados entre los items del vendedor con el mismo tipo de publicación y la misma condición de stock (una unidad o más). |
| `te_vv_n_mismo_tipo_y_stock` | con target |  |  |  |  | Cantidad de items con etiqueta en los que se apoya la tasa anterior. |

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
