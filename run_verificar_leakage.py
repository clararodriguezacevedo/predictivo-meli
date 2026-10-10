"""Verifica que el pipeline de features no filtre información del target.

Corre tres controles sobre `features.construir`:

1. Invarianza al target: las features de train y test tienen que ser idénticas si
   se mezcla al azar la columna `condition` o si directamente se la elimina. Si
   alguna feature cambia, es que depende del target.
2. Mismas columnas: train y test tienen que tener exactamente las mismas features.
3. Disponibilidad en test: ninguna feature puede tener muchos más nulos en test
   que en train (sería señal de que depende de algo que solo existe en train).

Uso: python run_verificar_leakage.py
"""

import sys

import numpy as np
import pandas as pd

import config
import features
from carga import cargar_train_test

TOLERANCIA_NULOS = 0.02


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


def main():
    train, test = cargar_train_test()
    controles = {
        "Invarianza al target": verificar_invarianza_al_target,
        "Mismas columnas en train y test": verificar_mismas_columnas,
        "Disponibilidad en test": verificar_disponibilidad_en_test,
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
