"""Verifica que el pipeline de features no filtre información del target.

Corre tres controles sobre `features.construir`:

1. Invarianza al target: las features de train y test tienen que ser idénticas si
   se mezcla al azar la columna `condition` o si directamente se la elimina. Si
   alguna feature cambia, es que depende del target.
2. Mismas columnas: train y test tienen que tener exactamente las mismas features.
3. Disponibilidad en test: ninguna feature puede tener muchos más nulos en test
   que en train (sería señal de que depende de algo que solo existe en train).

Y dos controles sobre el target encoding y el modelo completo:

4. Ninguna fila ve su propia etiqueta: si se invierten las etiquetas de un
   sub-fold, las features de target encoding de esas filas no pueden cambiar.
5. Target mezclado: si se entrena el modelo completo con las etiquetas mezcladas al
   azar, el accuracy de validación tiene que caer al nivel de predecir siempre la
   clase mayoritaria. Si queda por encima, alguna feature está filtrando el target.

Uso: python run_verificar_leakage.py
"""

import sys

import numpy as np
import pandas as pd

import config
import evaluacion
import features
import modelo
import target_encoding
from carga import cargar_train_test

TOLERANCIA_NULOS = 0.02
# Filas usadas en los controles del target encoding, para que corran rápido
MUESTRA_TARGET_ENCODING = 15000
# Margen sobre el accuracy de la clase mayoritaria en el control del target mezclado
TOLERANCIA_AZAR = 0.015


def _features(train, test):
    tr, te = features.construir(train, test)
    return tr.drop(columns=[config.TARGET], errors="ignore"), te


def verificar_invarianza_al_target(train, test):
    tr_real, te_real = _features(train, test)

    mezclado = train.copy()
    rng = np.random.default_rng(config.SEED)
    mezclado[config.TARGET] = rng.permutation(mezclado[config.TARGET].to_numpy())
    variantes = {
        "target mezclado": _features(mezclado, test),
        "sin target": _features(train.drop(columns=[config.TARGET]), test),
    }
    errores = []
    for nombre, (tr_var, te_var) in variantes.items():
        for split, real, var in [("train", tr_real, tr_var), ("test", te_real, te_var)]:
            for col in real.columns:
                if col not in var:
                    errores.append(f"`{col}` no se puede calcular en {split} con {nombre}")
                elif not real[col].equals(var[col]):
                    errores.append(f"`{col}` cambia en {split} con {nombre}")
    return errores


def verificar_mismas_columnas(train, test):
    tr, te = _features(train, test)
    solo_train = sorted(set(tr.columns) - set(te.columns))
    solo_test = sorted(set(te.columns) - set(tr.columns))
    errores = []
    if solo_train:
        errores.append(f"columnas que solo existen en train: {solo_train}")
    if solo_test:
        errores.append(f"columnas que solo existen en test: {solo_test}")
    return errores


def verificar_disponibilidad_en_test(train, test):
    tr, te = _features(train, test)
    diferencia = te.isna().mean() - tr[te.columns].isna().mean()
    return [
        f"`{col}` tiene {dif:.1%} más de nulos en test que en train"
        for col, dif in diferencia.items() if dif > TOLERANCIA_NULOS
    ]


def _muestra_con_features(train, test):
    tr, _ = features.construir(train, test)
    tr = tr.sample(MUESTRA_TARGET_ENCODING, random_state=config.SEED).reset_index(drop=True)
    return tr, evaluacion.y_binario(tr)


def verificar_te_sin_etiqueta_propia(train, test):
    tr, y = _muestra_con_features(train, test)
    original, _ = target_encoding.codificar(tr, y, tr.head(1))
    errores = []
    for k, (_, i_fold) in enumerate(target_encoding.subfolds(len(tr))):
        y_invertido = y.copy()
        y_invertido.iloc[i_fold] = 1 - y_invertido.iloc[i_fold]
        alterado, _ = target_encoding.codificar(tr, y_invertido, tr.head(1))
        for col in original.columns:
            iguales = np.allclose(original[col].iloc[i_fold], alterado[col].iloc[i_fold],
                                  equal_nan=True)
            if not iguales:
                errores.append(f"`{col}` cambia en el sub-fold {k} al invertir sus etiquetas")
    return errores


def verificar_target_mezclado(train, test):
    tr, y = _muestra_con_features(train, test)
    rng = np.random.default_rng(config.SEED)
    y_mezclado = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    accuracy, _ = modelo.validar(
        tr, y_mezclado, protocolo={"n_folds": 3, "semillas": [0]}, registrar=False
    )
    azar = max(y.mean(), 1 - y.mean())
    print(f"      accuracy con target mezclado {accuracy:.3f}, clase mayoritaria {azar:.3f}")
    if accuracy > azar + TOLERANCIA_AZAR:
        return [f"el modelo acierta {accuracy:.3f} con el target mezclado (azar {azar:.3f})"]
    return []


def main():
    train, test = cargar_train_test()
    controles = {
        "Invarianza al target": verificar_invarianza_al_target,
        "Mismas columnas en train y test": verificar_mismas_columnas,
        "Disponibilidad en test": verificar_disponibilidad_en_test,
        "Target encoding sin etiqueta propia": verificar_te_sin_etiqueta_propia,
        "Target mezclado da accuracy de azar": verificar_target_mezclado,
    }
    hay_errores = False
    for nombre, control in controles.items():
        errores = control(train, test)
        print(f"{'FALLA' if errores else 'OK':5} {nombre}")
        for error in errores:
            print(f"      {error}")
        hay_errores = hay_errores or bool(errores)
    sys.exit(1 if hay_errores else 0)


if __name__ == "__main__":
    main()
