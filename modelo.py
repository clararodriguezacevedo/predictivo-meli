"""El modelo: elige las columnas, une features y target encoding, entrena LightGBM y valida."""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
import registro
import target_encoding

# Identificadores, texto libre, fechas crudas y target: ya están resumidos en features
NO_SON_FEATURES = [
    config.TARGET, "id", "seller_id", "category_id", "parent_item_id",
    "official_store_id", "catalog_product_id", "title", "warranty", "video_id",
    "date_created", "last_updated", "start_time", "stop_time",
]
CATEGORICAS = [
    "listing_type_id", "buying_mode", "ship_mode", "status", "sub_status", "tags",
    "ship_tags", "currency_id", "seller_state", "attr_estado", "attr_antiguedad",
]
# Un solo modelo: promediar cuatro no mejora de forma significativa (D50); última búsqueda sin mejora (D51)
PARAMETROS = dict(
    n_estimators=1500, learning_rate=0.034309580195791355, num_leaves=149,
    min_child_samples=5, colsample_bytree=0.6874594323703044, subsample=0.8344766462089817,
    subsample_freq=1, reg_lambda=0.07827034783412899, reg_alpha=0.029235419526953025,
)
# 5 folds y 2 semillas: toda corrida "validacion" usa estas particiones, así se comparan de a pares
PROTOCOLO_VALIDACION = {"n_folds": 5, "semillas": [0, 1]}


def y_binario(df):
    return (df[config.TARGET] == config.CLASE_POSITIVA).astype(int)


def crear_clasificador(semilla):
    return lgb.LGBMClassifier(**PARAMETROS, random_state=semilla, n_jobs=8, verbose=-1)


def preparar(df, niveles=None):
    """Deja solo las features y tipa las categóricas."""
    X = df[[c for c in df.columns if c not in NO_SON_FEATURES]].copy()
    for col in CATEGORICAS:
        X[col] = pd.Categorical(X[col], categories=niveles[col] if niveles else None)
    X["attr_unico_duenio"] = X["attr_unico_duenio"].astype(float)
    return X


def preparar_juntos(train, test):
    """Train y test con los mismos niveles en las categóricas, para predecir el test."""
    niveles = {c: sorted(set(train[c].dropna()) | set(test[c].dropna())) for c in CATEGORICAS}
    return preparar(train, niveles), preparar(test, niveles)


def armar_matrices(X_train, train, y, X_pred, a_predecir, semilla):
    """Agrega el target encoding. Validar y predecir el test pasan por esta misma función."""
    te_train, te_pred = target_encoding.codificar(train, y, a_predecir, semilla)
    return pd.concat([X_train, te_train], axis=1), pd.concat([X_pred, te_pred], axis=1)


def armar_folds(train, y, protocolo):
    """Lista de (X_train, y_train, X_val, y_val, semilla). Lo caro: se arma una vez y se reutiliza."""
    X = preparar(train)
    folds = []
    for semilla in protocolo["semillas"]:
        particion = StratifiedKFold(protocolo["n_folds"], shuffle=True, random_state=semilla)
        for i_train, i_val in particion.split(X, y):
            # Las etiquetas de validación no entran a ningún cálculo
            X_tr, X_val = armar_matrices(
                X.iloc[i_train], train.iloc[i_train], y.iloc[i_train],
                X.iloc[i_val], train.iloc[i_val], semilla,
            )
            folds.append((X_tr, y.iloc[i_train], X_val, y.iloc[i_val], semilla))
    return folds


def validar_en_folds(folds):
    """Accuracy de cada fold. Corta en 0.5: moverlo no mejora (D52)."""
    aciertos = []
    for X_tr, y_tr, X_val, y_val, semilla in folds:
        clasificador = crear_clasificador(semilla).fit(X_tr, y_tr)
        aciertos.append(float((clasificador.predict(X_val) == y_val).mean()))
    return aciertos


def resumir(aciertos):
    return float(np.mean(aciertos)), float(np.std(aciertos) / np.sqrt(len(aciertos)))


def descripcion_comun(folds, protocolo):
    """Detalles que van al registro además del modelo y sus parámetros."""
    return {
        "protocolo": protocolo,
        "target_encoding": target_encoding.descripcion(),
        "features": registro.describir_features(folds[0][0].columns),
    }


def validar(train, y, protocolo=PROTOCOLO_VALIDACION, registrar=True, nota=""):
    """Validación cruzada del modelo completo. Devuelve (accuracy, error estándar)."""
    inicio = time.time()
    folds = armar_folds(train, y, protocolo)
    aciertos = validar_en_folds(folds)
    accuracy, error = resumir(aciertos)
    if registrar:
        registro.registrar(
            "validacion", "lightgbm", PARAMETROS, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            nota=nota, **descripcion_comun(folds, protocolo),
        )
    return accuracy, error


def predecir_en_folds(folds):
    """Probabilidad de cada fila, predicha por un modelo que no la vio. Para analizar errores."""
    partes = []
    for X_tr, y_tr, X_val, y_val, semilla in folds:
        clasificador = crear_clasificador(semilla).fit(X_tr, y_tr)
        parte = X_val[[c for c in X_val.columns if c.startswith("te_")]].copy()
        parte["proba_usado"] = clasificador.predict_proba(X_val)[:, 1]
        parte["y"] = y_val
        parte["semilla"] = semilla
        partes.append(parte)
    return pd.concat(partes)
