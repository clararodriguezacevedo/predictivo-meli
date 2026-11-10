"""Tests estadísticos y controles de leakage"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_predict

import datos
import modelo


# ----------------------------------------------------------------------------
# Tests estadísticos
# ----------------------------------------------------------------------------

# 1 fila de validación por cada 4 de entrenamiento, por usar 5 folds
FILAS_VAL_SOBRE_TRAIN = 1 / 4


def t_corregido(con, sin):
    """Test t pareado de Nadeau y Bengio. Devuelve (diferencia, error estándar, p), en proporción"""
    d = np.asarray(con) - np.asarray(sin)
    # Se corrige porque los folds comparten filas y el t común daría p demasiado chicos
    error = np.sqrt(d.var(ddof=1) * (1 / len(d) + FILAS_VAL_SOBRE_TRAIN))
    if error == 0:
        return float(d.mean()), 0.0, 1.0
    p = 2 * stats.t.sf(abs(d.mean() / error), len(d) - 1)
    return float(d.mean()), float(error), float(p)


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
    """Mezclar o borrar `condition` no puede cambiar ninguna feature"""
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
    """Train y test tienen las mismas features"""
    distintas = set(tr.columns.drop(datos.TARGET)) ^ set(te.columns)
    return [f"hay columnas que no están en los dos {sorted(distintas)}"] if distintas else []


def disponibilidad_en_test(train, test, tr, te):
    """Ninguna feature tiene muchos más nulos en test que en train"""
    diferencia = te.isna().mean() - tr[te.columns].isna().mean()
    return [f"`{col}` tiene {dif:.1%} más de nulos en test"
            for col, dif in diferencia.items() if dif > TOLERANCIA_NULOS]


def _muestra(tr):
    muestra = tr.sample(MUESTRA_TARGET_ENCODING, random_state=datos.SEED).reset_index(drop=True)
    return muestra, modelo.y_binario(muestra)


def te_sin_etiqueta_propia(train, test, tr, te):
    """Invertir las etiquetas de un sub-fold no cambia el target encoding de sus filas"""
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
    """Con etiquetas al azar, el modelo completo cae al nivel de la clase mayoritaria"""
    muestra, y = _muestra(tr)
    rng = np.random.default_rng(datos.SEED)
    y_mezclado = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    accuracy, _ = modelo.validar(muestra, y_mezclado, protocolo={"n_folds": 3, "semillas": [0]},
                                 anotar=False)
    azar = max(y.mean(), 1 - y.mean())
    print(f"      accuracy con target mezclado {accuracy:.3f}, clase mayoritaria {azar:.3f}")
    return [f"acierta {accuracy:.3f} con el target mezclado"] if accuracy > azar + TOLERANCIA_AZAR else []

def controlar_leakage():
    """Corre los cinco controles e imprime el resultado. Devuelve True si todos dan OK"""
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


# ----------------------------------------------------------------------------
# Validación adversaria, si train y test se pueden distinguir
# ----------------------------------------------------------------------------


def puntajes_adversarios(tr, te):
    """Probabilidad de "ser de test" de cada fila de train y de test, predicha fuera de muestra"""
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
