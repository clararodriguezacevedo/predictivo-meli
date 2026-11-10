"""Las pruebas. Tests estadísticos, la lista de decisiones, los controles de leakage y docs/DECISIONES.md.

Se lee en orden. Primero los tests, después cada decisión con la prueba que la justifica, los cinco
controles de que el modelo no filtra el target y al final cómo se corren las pruebas de las decisiones.
"""

from functools import lru_cache
import json
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_predict

import datos
import diccionario
import modelo


# ----------------------------------------------------------------------------
# Tests estadísticos
# ----------------------------------------------------------------------------

# 1 fila de validación por cada 4 de entrenamiento, por usar 5 folds
FILAS_VAL_SOBRE_TRAIN = 1 / 4


def t_corregido(con, sin):
    """Test t pareado de Nadeau y Bengio. Devuelve (diferencia, error estándar, p), en proporción."""
    d = np.asarray(con) - np.asarray(sin)
    # La corrección importa porque los folds comparten filas y el t común daría p demasiado chicos
    error = np.sqrt(d.var(ddof=1) * (1 / len(d) + FILAS_VAL_SOBRE_TRAIN))
    if error == 0:
        return float(d.mean()), 0.0, 1.0
    p = 2 * stats.t.sf(abs(d.mean() / error), len(d) - 1)
    return float(d.mean()), float(error), float(p)


def holm(p):
    """Corrección de Holm, porque con muchos tests alguno da p < 0.05 por azar."""
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    ajustado = np.maximum.accumulate(p[orden] * (len(p) - np.arange(len(p)))).clip(max=1)
    resultado = np.empty(len(p))
    resultado[orden] = ajustado
    return resultado


def chi2_independencia(a, b):
    """p de chi cuadrado, que mide si dos variables categóricas son independientes."""
    return float(stats.chi2_contingency(pd.crosstab(a, b))[1])


def mcnemar(aciertos_a, aciertos_b):
    """p exacto de McNemar, que compara dos clasificadores solo en las filas donde difieren."""
    solo_a = int((aciertos_a & ~aciertos_b).sum())
    solo_b = int((~aciertos_a & aciertos_b).sum())
    if solo_a + solo_b == 0:
        return 1.0
    return float(stats.binomtest(solo_a, solo_a + solo_b, 0.5).pvalue)


def mann_whitney(puntajes_a, puntajes_b):
    """Mide si los puntajes de un grupo tienden a ser mayores que los del otro. Devuelve (AUC, p)."""
    resultado = stats.mannwhitneyu(puntajes_a, puntajes_b)
    return float(resultado.statistic / (len(puntajes_a) * len(puntajes_b))), float(resultado.pvalue)

# ----------------------------------------------------------------------------
# Las decisiones y la prueba que justifica cada una
# ----------------------------------------------------------------------------

# Se quedan si sacarlas empeora de forma significativa. Cada una es (código, qué es, grupos del diccionario o regex)
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
    # De a una, las piezas parecidas se tapan entre sí, por eso estas se prueban juntas
    ("D27", "Los tres perfiles juntos", ["Perfil del vendedor", "Perfil de la categoría", "Perfil por palabra del título"]),
    ("D35", "Todo el target encoding junto", [r"te_"]),
    ("D36", "Todas las piezas de efecto casi nulo juntas", [
        "Precio y moneda", "Stock y ventas", "Ubicación del vendedor", "Envío", "Medios de pago",
        "Formato de las fotos", "Perfil del vendedor", r"te_seller_id(_n)?$", r"te_category_id", r"te_seller_id_x_",
    ]),
]


def _con_vecinos(k):
    return lambda: [modelo.EncoderDeVecinos(k=k)]


def _con_suavizado(m):
    def fabrica():
        grupos = [e for e in modelo.crear_encoders() if isinstance(e, modelo.EncoderDeGrupo)]
        for e in grupos:
            e.suavizado = m
        return grupos
    return fabrica


# Valores alternativos de un encoder. El actual se queda salvo que otro gane de forma significativa
# (código, decisión, nombre de la corrida en el registro, encoders alternativos)
VARIANTES = [
    ("D40", "Usar 30 vecinos por título y no 10", "Vecinos por título: k = 30 y no 10", _con_vecinos(10)),
    ("D41", "Suavizar las tasas con 2 y no con 10", "Suavizado de las tasas: 2 y no 10", _con_suavizado(10.0)),
    ("D42", "Suavizar las tasas con 2 y no con 0.5", "Suavizado de las tasas: 2 y no 0.5", _con_suavizado(0.5)),
]

# Comparaciones ya registradas, como (código, decisión, corrida con la pieza, sin la pieza, se_queda)
HISTORICAS = [
    ("D50", "Usar solo LightGBM y no promediar cuatro modelos", "20261102-115951-validacion-ensamble",
     "20261102-134308-validacion-lightgbm", False),
    ("D51", "No adoptar los hiperparámetros de la última búsqueda", "20261102-144135-validacion-lightgbm",
     "20261102-134308-validacion-lightgbm", False),
    ("D53", "Conservar fotos, ubicación, tasa del vendedor y fechas, porque sacarlas juntas pierde más de 0.02 puntos",
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
    ("D85", "No limpiar etiquetas sacando las dudosas al 99%", "explorar: quitar 0.99", "explorar: base", False),
    ("D86", "No limpiar etiquetas sacando las dudosas al 95%", "explorar: quitar 0.95", "explorar: base", False),
    ("D87", "No limpiar etiquetas sacando las dudosas al 90%", "explorar: quitar 0.9", "explorar: base", False),
    ("D88", "No limpiar etiquetas sacando las dudosas al 80%", "explorar: quitar 0.8", "explorar: base", False),
    ("D89", "No limpiar etiquetas invirtiendo las dudosas al 95%", "explorar: invertir 0.95", "explorar: base", False),
    ("D90", "No hacer stacking de los cuatro modelos", "explorar: stacking",
     "20261102-115951-validacion-ensamble", False),
    ("D91", "No entrenar un modelo por segmento de stock", "explorar: segmentos", "explorar: base", False),
    ("D92", "No promediar varias semillas de LightGBM", "explorar: tres semillas", "explorar: base", False),
    ("D93", "5 sub-folds y no 10 en el target encoding", "explorar: 10 subfolds", "explorar: base", False),
]

# ----------------------------------------------------------------------------
# Controles de leakage, tienen que dar OK antes de commitear
# ----------------------------------------------------------------------------

# Una feature con muchos más nulos en test sugiere que depende de algo que solo existe en train
TOLERANCIA_NULOS = 0.02
# Muestra para que los controles del target encoding corran rápido
MUESTRA_TARGET_ENCODING = 15000
# Margen sobre la clase mayoritaria, porque con 5.000 filas de validación el azar mueve hasta ~1.5 puntos
TOLERANCIA_AZAR = 0.015


def invarianza_al_target(train, test, tr, te):
    """Mezclar o borrar `condition` no puede cambiar ninguna feature."""
    mezclado = train.copy()
    rng = np.random.default_rng(datos.SEED)
    mezclado[datos.TARGET] = rng.permutation(mezclado[datos.TARGET].to_numpy())
    variantes = {
        "target mezclado": datos.construir(mezclado, test),
        "sin target": datos.construir(train.drop(columns=[datos.TARGET]), test),
    }
    errores = []
    for nombre, (tr_var, te_var) in variantes.items():
        for split, real, var in [("train", tr, tr_var), ("test", te, te_var)]:
            for col in real.columns.drop(datos.TARGET, errors="ignore"):
                if col not in var or not real[col].equals(var[col]):
                    errores.append(f"`{col}` cambia en {split} con {nombre}")
    return errores


def mismas_columnas(train, test, tr, te):
    """Train y test tienen las mismas features."""
    distintas = set(tr.columns.drop(datos.TARGET)) ^ set(te.columns)
    return [f"hay columnas que no están en los dos {sorted(distintas)}"] if distintas else []


def disponibilidad_en_test(train, test, tr, te):
    """Ninguna feature tiene muchos más nulos en test que en train."""
    diferencia = te.isna().mean() - tr[te.columns].isna().mean()
    return [f"`{col}` tiene {dif:.1%} más de nulos en test"
            for col, dif in diferencia.items() if dif > TOLERANCIA_NULOS]


def _muestra(tr):
    muestra = tr.sample(MUESTRA_TARGET_ENCODING, random_state=datos.SEED).reset_index(drop=True)
    return muestra, modelo.y_binario(muestra)


def te_sin_etiqueta_propia(train, test, tr, te):
    """Invertir las etiquetas de un sub-fold no cambia el target encoding de sus filas."""
    muestra, y = _muestra(tr)
    original, _ = modelo.codificar(muestra, y, muestra.head(1))
    errores = []
    for k, (_, i_fold) in enumerate(modelo.subfolds(len(muestra))):
        y_invertido = y.copy()
        y_invertido.iloc[i_fold] = 1 - y_invertido.iloc[i_fold]
        alterado, _ = modelo.codificar(muestra, y_invertido, muestra.head(1))
        for col in original.columns:
            if not np.allclose(original[col].iloc[i_fold], alterado[col].iloc[i_fold], equal_nan=True):
                errores.append(f"`{col}` cambia en el sub-fold {k} al invertir sus etiquetas")
    return errores


def target_mezclado(train, test, tr, te):
    """Con etiquetas al azar, el modelo completo cae al nivel de la clase mayoritaria."""
    muestra, y = _muestra(tr)
    rng = np.random.default_rng(datos.SEED)
    y_mezclado = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    accuracy, _ = modelo.validar(muestra, y_mezclado, protocolo={"n_folds": 3, "semillas": [0]},
                                 anotar=False)
    azar = max(y.mean(), 1 - y.mean())
    print(f"      accuracy con target mezclado {accuracy:.3f}, clase mayoritaria {azar:.3f}")
    return [f"acierta {accuracy:.3f} con el target mezclado"] if accuracy > azar + TOLERANCIA_AZAR else []

# ----------------------------------------------------------------------------
# Pruebas de las decisiones, que generan docs/DECISIONES.md
# ----------------------------------------------------------------------------

ALFA = 0.05  # Umbral habitual; en las comparaciones se aplica después de corregir por cantidad de tests
# Las descartadas por constantes. Las URL varían y se prueban aparte (D70, D80)
CONSTANTES = [c for c in datos.COLUMNAS_DESCARTADAS if c not in ("permalink", "thumbnail", "secure_thumbnail")]
FALTANTES = ["warranty", "video_id", "official_store_id", "parent_item_id"]
SEMILLAS_DE_CONTROL = [2, 3]  # Folds nuevos, para confirmar que una mejora no fue suerte


def puntajes_adversarios(tr, te):
    """Probabilidad de "ser de test" de cada fila de train y de test, predicha fuera de muestra."""
    columnas = [
        "log_price", "initial_quantity", "sold_quantity", "available_quantity",
        "n_pictures", "n_attributes", "n_variations", "warranty_len", "title_len",
        "dias_hasta_update", "duracion_dias", "hora_creacion", "seller_n_items",
        "seller_frac_free", "seller_frac_iq_1", "cat_n_items",
    ]
    fechas = pd.concat([tr["date_created"], te["date_created"]]).astype("int64")
    X = pd.concat([tr[columnas], te[columnas]], ignore_index=True).assign(fecha=fechas.values)
    es_test = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    adversario = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=datos.SEED)
    proba = cross_val_predict(adversario, X, es_test, cv=5, method="predict_proba")[:, 1]
    return proba[:len(tr)], proba[len(tr):]


def pruebas_de_datos(train, test):
    """Decisiones que se prueban sobre los datos, sin entrenar el modelo final."""
    filas = []
    en_train, en_test = puntajes_adversarios(train, test)
    auc, p = mann_whitney(en_test, en_train)
    # Acá se busca un p grande, que indica que no se distingue train de test
    filas.append(("D01", "Validar con folds al azar, porque train y test son intercambiables",
                  f"Mann-Whitney, AUC {auc:.3f}", p, "respaldada" if p >= ALFA else "contradicha"))

    valores = {c: set() for c in CONSTANTES}
    for ruta in (datos.TRAIN_RAW, datos.TEST_RAW):
        with open(ruta, encoding="utf8") as f:
            for linea in f:
                r = json.loads(linea)
                for c in CONSTANTES:
                    valores[c].add(str(r.get(c)))
    no_constantes = [c for c, v in valores.items() if len(v) > 1]
    filas.append(("D02", "Descartar columnas constantes",
                  f"sin test, pero no son constantes {no_constantes}" if no_constantes
                  else "sin test, porque una constante no puede asociarse",
                  np.nan, "contradicha" if no_constantes else "respaldada"))

    y = modelo.y_binario(train)
    for col in FALTANTES:
        p = chi2_independencia(train[col].notna(), y)
        filas.append(("D03", f"Usar la ausencia de `{col}` como señal",
                      f"chi cuadrado, usados {y[train[col].isna()].mean():.1%} si falta "
                      f"y {y[train[col].notna()].mean():.1%} si está",
                      p, "respaldada" if p < ALFA else "sin respaldo"))

    ruta = datos.OUTPUTS / "errores" / "predicciones_validacion.parquet"
    if ruta.exists():
        pred = pd.read_parquet(ruta)
        aciertos = lambda corte: ((pred["proba_usado"] > corte).astype(int) == pred["y"]).to_numpy()
        p = mcnemar(aciertos(0.48), aciertos(0.5))
        filas.append(("D52", "Cortar en 0.5 y no en 0.48, el mejor valor visto",
                      f"McNemar, diferencia {100 * (aciertos(0.48).mean() - aciertos(0.5).mean()):+.3f} puntos",
                      p, "respaldada" if p >= ALFA else "contradicha"))
    return filas


def columnas_de(patrones, todas):
    """Columnas de los grupos del diccionario, o las que coinciden con una regex."""
    grupos = {}
    for v in diccionario.VARIABLES:
        grupos.setdefault(v.grupo, []).append(v.nombre)
    columnas = []
    for patron in patrones:
        columnas += ([c for c in grupos[patron] if c in todas] if patron in grupos
                     else [c for c in todas if re.match(patron, c)])
    return columnas


def controlar_leakage():
    """Corre los cinco controles e imprime el resultado. Devuelve True si todos dan OK."""
    train, test = datos.cargar_train_test()
    tr, te = datos.construir(train, test)
    todo_ok = True
    for control in [invarianza_al_target, mismas_columnas, disponibilidad_en_test,
                    te_sin_etiqueta_propia, target_mezclado]:
        errores = control(train, test, tr, te)
        print(f"{'FALLA' if errores else 'OK':5} {control.__doc__}")
        for error in errores:
            print(f"      {error}")
        todo_ok = todo_ok and not errores
    return todo_ok


def comparar(train, semillas):
    """Prueba cada decisión con los folds de esas semillas, reutilizando las corridas ya registradas.

    Devuelve un diccionario que lleva cada código a (decisión, prueba, p, veredicto).
    """
    y = modelo.y_binario(train)
    estandar = semillas == modelo.PROTOCOLO_VALIDACION["semillas"]
    # Las semillas de siempre se registran como validacion y las nuevas como control
    tipo_corrida = "validacion" if estandar else "control"
    sufijo = "" if estandar else f" (semillas {semillas})"
    protocolo = {**modelo.PROTOCOLO_VALIDACION, "semillas": semillas}

    @lru_cache(maxsize=1)
    def folds():
        return modelo.armar_folds(train, y, protocolo)  # Es lo caro, así que solo se arma si falta alguna corrida

    def corrida(nota, transformar):
        """Accuracy por fold de una prueba. Usa la registrada si ya está, o la corre y la registra."""
        for r in reversed(modelo.leer()):
            if r["tipo"] == tipo_corrida and r.get("nota") == nota:
                return np.array(r["accuracy_por_fold"])
        inicio = time.time()
        variante, detalle = transformar(folds())
        aciertos = modelo.validar_en_folds(variante)
        accuracy, error = modelo.resumir(aciertos)
        modelo.registrar(
            tipo_corrida, "lightgbm", modelo.PARAMETROS, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio), nota=nota,
            ablacion=detalle, **modelo.descripcion_comun(variante, protocolo),
        )
        print(f"{nota} da {accuracy:.4f}", flush=True)
        return np.array(aciertos)

    def sin_columnas(patrones):
        def transformar(fs):
            columnas = columnas_de(patrones, list(fs[0][0].columns))
            return [(X_tr.drop(columns=columnas), y_tr, X_val.drop(columns=columnas), y_val, s)
                    for X_tr, y_tr, X_val, y_val, s in fs], {"quitado": columnas}
        return transformar

    def con_encoder(fabrica):
        def transformar(fs):
            nuevos = []
            for X_tr, y_tr, X_val, y_val, s in fs:
                te_tr, te_val = modelo.codificar(train.loc[X_tr.index], y_tr, train.loc[X_val.index], s, fabrica)
                nuevos.append((X_tr.assign(**te_tr), y_tr, X_val.assign(**te_val), y_val, s))
            return nuevos, {"reemplazado": list(te_tr.columns)}
        return transformar

    completo = corrida(f"decisiones: modelo completo{sufijo}", sin_columnas([]))

    # (código, decisión, accuracy con la pieza, sin la pieza, se queda)
    comparaciones = []
    for codigo, nombre, patrones in ABLACIONES:
        comparaciones.append((codigo, f"Conservar {nombre[0].lower()}{nombre[1:]}", completo,
                              corrida(f"decisiones: sin {nombre}{sufijo}", sin_columnas(patrones)), True))
    for codigo, nombre, nota, fabrica in VARIANTES:
        comparaciones.append((codigo, nombre, corrida(f"decisiones: {nota}{sufijo}", con_encoder(fabrica)),
                              completo, False))

    # Se busca por id o por nota, porque las exploraciones se identifican por nota
    por_clave = {}
    for r in modelo.leer():
        if r.get("accuracy_por_fold"):
            por_clave[r["id"]] = por_clave[r.get("nota")] = r["accuracy_por_fold"]
    resultado = {}
    for codigo, nombre, con, sin, se_queda in HISTORICAS:
        if con in por_clave and sin in por_clave:
            comparaciones.append((codigo, nombre, por_clave[con], por_clave[sin], se_queda))
        else:
            resultado[codigo] = (nombre, "sin corrida con resultados por fold", np.nan, "sin test")

    ajustados = holm([t_corregido(c[2], c[3])[2] for c in comparaciones])
    for (codigo, nombre, con, sin, se_queda), p in zip(comparaciones, ajustados):
        dif, error, _ = t_corregido(con, sin)
        significativo = p < ALFA
        if se_queda:
            veredicto = "respaldada" if significativo and dif > 0 else "sin respaldo"
        else:
            veredicto = "contradicha" if significativo and dif > 0 else "respaldada"
        resultado[codigo] = (nombre, f"t corregido, {100 * dif:+.3f} ± {100 * error:.3f} puntos", p, veredicto)
    return resultado


def _p_legible(p):
    return "" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def generar_decisiones():
    """Corre las pruebas con los folds de siempre y con folds nuevos, y escribe docs/DECISIONES.md."""
    train, test = datos.construir(*datos.cargar_train_test())
    siempre = comparar(train, modelo.PROTOCOLO_VALIDACION["semillas"])
    nuevos = comparar(train, SEMILLAS_DE_CONTROL)
    # Las comparaciones registradas antes tienen una sola medición, que no depende de las semillas
    historicas = {codigo for codigo, *_ in HISTORICAS}

    filas = [[codigo, nombre, prueba, p, veredicto, "", np.nan, ""]
             for codigo, nombre, prueba, p, veredicto in pruebas_de_datos(train, test)]
    for codigo, (nombre, prueba, p, veredicto) in siempre.items():
        otra = ("", np.nan, "") if codigo in historicas else nuevos[codigo][1:]
        filas.append([codigo, nombre, prueba, p, veredicto, *otra])
    tabla = pd.DataFrame(sorted(filas, key=lambda f: f[:2]), columns=[
        "Código", "Decisión", "Folds de siempre", "p", "Veredicto",
        "Folds nuevos", "p con folds nuevos", "Veredicto con folds nuevos"])
    for columna in ["p", "p con folds nuevos"]:
        tabla[columna] = tabla[columna].map(_p_legible)

    def resumen(columna):
        conteo = tabla.loc[tabla[columna] != "", columna].value_counts()
        return ", ".join(f"{n} {veredicto}" for veredicto, n in conteo.items())

    texto = (
        "# Decisiones y sus tests\n\nGenerado por `python correr.py decisiones`.\n\n"
        "Cada decisión se prueba dos veces. Primero con los folds de siempre (semillas 0 y 1) y después "
        "con folds nuevos (semillas 2 y 3), para confirmar que una mejora no fue suerte de cómo cayeron "
        "los folds. Las decisiones sobre los datos y las comparaciones registradas antes tienen una sola "
        "medición.\n\n"
        "Las comparaciones de modelos usan el t pareado corregido de Nadeau y Bengio sobre los 10 "
        "folds, con corrección de Holm por la cantidad de tests (su p ya está ajustado). La "
        "diferencia es con la pieza menos sin la pieza, en puntos de accuracy, casi siempre "
        "medida con LightGBM.\n\n"
        "Los veredictos posibles son **respaldada** (al conservar, la pieza suma de forma significativa; "
        "al descartar, no se demostró que sume), **sin respaldo** (se conserva algo sin una mejora "
        "demostrada), **contradicha** (el test indica lo contrario) y **sin test**.\n\n"
        "Un p alto no prueba que una pieza no aporte, porque con 10 folds y tantos tests solo se detectan "
        "las diferencias grandes. Además, sacar una pieza sola subestima su aporte si otra lleva "
        "información parecida, por eso algunas se prueban juntas.\n\n"
        f"Con los folds de siempre hay {resumen('Veredicto')}. Con folds nuevos hay "
        f"{resumen('Veredicto con folds nuevos')}.\n\n" + tabla.to_markdown(index=False) + "\n"
    )
    datos.DOCS.mkdir(exist_ok=True)
    (datos.DOCS / "DECISIONES.md").write_text(texto, encoding="utf8")
    print(f"Folds de siempre {resumen('Veredicto')}. Folds nuevos {resumen('Veredicto con folds nuevos')}.")
    print("Guardado en docs/DECISIONES.md")

