"""La lista de decisiones del modelo y la prueba con la que se justifica cada una."""

from target_encoding import EncoderDeGrupo, EncoderDeVecinos, crear_encoders

# Se quedan si sacarlas empeora de forma significativa: (código, qué es, grupos del diccionario o regex)
ABLACIONES = [
    ("D10", "Tipo de publicación y estado", ["Tipo de publicación y estado"]),
    ("D11", "Precio y moneda", ["Precio y moneda"]),
    ("D12", "Stock y ventas", ["Stock y ventas"]),
    ("D14", "Ubicación del vendedor", ["Ubicación del vendedor"]),
    ("D15", "Envío", ["Envío"]),
    ("D16", "Medios de pago", ["Medios de pago"]),
    ("D17", "Contenido de la publicación", ["Contenido de la publicación"]),
    ("D18", "Ficha técnica", ["Ficha técnica (vehículos, inmuebles y repuestos)"]),
    ("D19", "Features derivadas del item", ["Features derivadas del item"]),
    ("D20", "Palabras clave en título y garantía", ["Palabras clave en título y garantía"]),
    ("D21", "Perfil del vendedor", ["Perfil del vendedor"]),
    ("D22", "Perfil de la categoría", ["Perfil de la categoría"]),
    ("D23", "Perfil por palabra del título", ["Perfil por palabra del título"]),
    ("D24", "Formato de las fotos", ["Formato de las fotos"]),
    ("D25", "Desfasaje en la duración", [r"duracion_resto_seg$"]),
    ("D26", "Fechas agregadas después", [r"(dias_update_a_fin|seg_creacion_a_inicio|dia_creacion|hora_update)$"]),
    ("D30", "Tasa de usados del vendedor", [r"te_seller_id(_n)?$"]),
    ("D31", "Tasa de usados de la categoría", [r"te_category_id"]),
    ("D32", "Vecinos por título", [r"te_vecinos_"]),
    ("D33", "Tasas por vendedor cruzado con tipo, stock, garantía y título", [r"te_seller_id_x_"]),
    ("D34", "Vecinos del mismo vendedor", [r"te_vv_"]),
    # De a una, las piezas parecidas se tapan entre sí: estas se prueban juntas
    ("D27", "Los tres perfiles juntos", ["Perfil del vendedor", "Perfil de la categoría", "Perfil por palabra del título"]),
    ("D35", "Todo el target encoding junto", [r"te_"]),
    ("D36", "Todas las piezas de efecto casi nulo juntas", [
        "Precio y moneda", "Stock y ventas", "Ubicación del vendedor", "Envío", "Medios de pago",
        "Formato de las fotos", "Perfil del vendedor", r"te_seller_id(_n)?$", r"te_category_id", r"te_seller_id_x_",
    ]),
]


def _con_vecinos(k):
    return lambda: [EncoderDeVecinos(k=k)]


def _con_suavizado(m):
    def fabrica():
        grupos = [e for e in crear_encoders() if isinstance(e, EncoderDeGrupo)]
        for e in grupos:
            e.suavizado = m
        return grupos
    return fabrica


# Valores alternativos de un encoder: el actual se queda salvo que otro gane de forma significativa
VARIANTES = [
    ("D40", "Vecinos por título: k = 30 y no 10", _con_vecinos(10)),
    ("D41", "Suavizado de las tasas: 2 y no 10", _con_suavizado(10.0)),
    ("D42", "Suavizado de las tasas: 2 y no 0.5", _con_suavizado(0.5)),
]

# Comparaciones ya registradas: (código, decisión, corrida con la pieza, sin la pieza, se_queda)
HISTORICAS = [
    ("D50", "Usar solo LightGBM y no promediar cuatro modelos", "20261102-115951-validacion-ensamble",
     "20261102-134308-validacion-lightgbm", False),
    ("D51", "No adoptar los hiperparámetros de la última búsqueda", "20261102-144135-validacion-lightgbm",
     "20261102-134308-validacion-lightgbm", False),
    ("D53", "Conservar fotos, ubicación, tasa del vendedor y fechas: sacarlas juntas pierde más de 0.02 puntos",
     "simplificar: confirmacion completo", "simplificar: confirmacion reducido", True),
    ("D60", "Sacar el modelo de texto (en LightGBM; además fue pedido tuyo)",
     "20261102-010038-validacion-lightgbm", "20261102-134308-validacion-lightgbm", False),
    ("D61", "Sacar las pseudo-etiquetas", "20261025-212052-validacion-ensamble",
     "20261102-004740-validacion-ensamble", False),
    ("D62", "Sacar Random Forest frente a LightGBM", "20261011-145903-validacion-random_forest",
     "20261011-144222-validacion-lightgbm", False),
    ("D63", "Sacar Extra Trees frente a LightGBM", "20261011-150900-validacion-extra_trees",
     "20261011-144222-validacion-lightgbm", False),
    ("D64", "Sacar regresión logística frente a LightGBM", "20261011-151058-validacion-regresion_logistica",
     "20261011-144222-validacion-lightgbm", False),
    ("D65", "No usar red neuronal frente a LightGBM", "20261102-131508-validacion-red_neuronal",
     "20261102-134308-validacion-lightgbm", False),
    ("D66", "No usar kNN frente a LightGBM", "20261102-130950-validacion-knn",
     "20261102-134308-validacion-lightgbm", False),
    ("D67", "No usar clustering frente a LightGBM", "20261102-130632-validacion-clustering",
     "20261102-134308-validacion-lightgbm", False),
    ("D68", "No sumar kNN ni clustering como variables", "20261102-132849-validacion-lightgbm",
     "20261102-134308-validacion-lightgbm", False),
    ("D69", "Sacar las palabras clave nuevas", "20261102-013639-validacion-lightgbm",
     "20261102-014400-validacion-lightgbm", False),
    ("D70", "Sacar el subdominio de la URL", "20261102-013639-validacion-lightgbm",
     "20261102-015031-validacion-lightgbm", False),
    ("D71", "Sacar el consenso del vendedor", "20261011-014927-validacion-lightgbm",
     "20261011-014517-validacion-lightgbm", False),
    ("D80", "No sumar URL contra título", "explorar: url contra titulo", "explorar: base", False),
    ("D81", "No sumar más datos de la ficha técnica", "explorar: atributos", "explorar: base", False),
    ("D82", "No sumar más datos de las variaciones", "explorar: variaciones", "explorar: base", False),
    ("D83", "No sumar rastros del JSON crudo", "explorar: rastros", "explorar: base", False),
    ("D84", "No sumar tasas por garantía, ciudad y categoría cruzada", "explorar: grupos", "explorar: base", False),
    ("D85", "No limpiar etiquetas: sacar dudosas al 99%", "explorar: quitar 0.99", "explorar: base", False),
    ("D86", "No limpiar etiquetas: sacar dudosas al 95%", "explorar: quitar 0.95", "explorar: base", False),
    ("D87", "No limpiar etiquetas: sacar dudosas al 90%", "explorar: quitar 0.9", "explorar: base", False),
    ("D88", "No limpiar etiquetas: sacar dudosas al 80%", "explorar: quitar 0.8", "explorar: base", False),
    ("D89", "No limpiar etiquetas: invertir dudosas al 95%", "explorar: invertir 0.95", "explorar: base", False),
    ("D90", "No hacer stacking de los cuatro modelos", "explorar: stacking",
     "20261102-115951-validacion-ensamble", False),
    ("D91", "No entrenar un modelo por segmento de stock", "explorar: segmentos", "explorar: base", False),
    ("D92", "No promediar varias semillas de LightGBM", "explorar: tres semillas", "explorar: base", False),
    ("D93", "5 sub-folds y no 10 en el target encoding", "explorar: 10 subfolds", "explorar: base", False),
]
