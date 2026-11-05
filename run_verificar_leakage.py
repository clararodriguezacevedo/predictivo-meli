"""Cinco controles de que el modelo no filtra el target. Tienen que dar OK antes de commitear.

Uso: python run_verificar_leakage.py
"""

import sys

import numpy as np
import pandas as pd

from carga import cargar_train_test
import config
import features
import modelo
import target_encoding

# Una feature con muchos más nulos en test sugiere que depende de algo que solo existe en train
TOLERANCIA_NULOS = 0.02
# Muestra para que los controles del target encoding corran rápido
MUESTRA_TARGET_ENCODING = 15000
# Margen sobre la clase mayoritaria: con 5.000 filas de validación el azar mueve hasta ~1.5 puntos
TOLERANCIA_AZAR = 0.015


def invarianza_al_target(train, test, tr, te):
    """Mezclar o borrar `condition` no puede cambiar ninguna feature."""
    mezclado = train.copy()
    rng = np.random.default_rng(config.SEED)
    mezclado[config.TARGET] = rng.permutation(mezclado[config.TARGET].to_numpy())
    variantes = {
        "target mezclado": features.construir(mezclado, test),
        "sin target": features.construir(train.drop(columns=[config.TARGET]), test),
    }
    errores = []
    for nombre, (tr_var, te_var) in variantes.items():
        for split, real, var in [("train", tr, tr_var), ("test", te, te_var)]:
            for col in real.columns.drop(config.TARGET, errors="ignore"):
                if col not in var or not real[col].equals(var[col]):
                    errores.append(f"`{col}` cambia en {split} con {nombre}")
    return errores


def mismas_columnas(train, test, tr, te):
    """Train y test tienen las mismas features."""
    distintas = set(tr.columns.drop(config.TARGET)) ^ set(te.columns)
    return [f"columnas que no están en los dos: {sorted(distintas)}"] if distintas else []


def disponibilidad_en_test(train, test, tr, te):
    """Ninguna feature tiene muchos más nulos en test que en train."""
    diferencia = te.isna().mean() - tr[te.columns].isna().mean()
    return [f"`{col}` tiene {dif:.1%} más de nulos en test"
            for col, dif in diferencia.items() if dif > TOLERANCIA_NULOS]


def _muestra(tr):
    muestra = tr.sample(MUESTRA_TARGET_ENCODING, random_state=config.SEED).reset_index(drop=True)
    return muestra, modelo.y_binario(muestra)


def te_sin_etiqueta_propia(train, test, tr, te):
    """Invertir las etiquetas de un sub-fold no cambia el target encoding de sus filas."""
    muestra, y = _muestra(tr)
    original, _ = target_encoding.codificar(muestra, y, muestra.head(1))
    errores = []
    for k, (_, i_fold) in enumerate(target_encoding.subfolds(len(muestra))):
        y_invertido = y.copy()
        y_invertido.iloc[i_fold] = 1 - y_invertido.iloc[i_fold]
        alterado, _ = target_encoding.codificar(muestra, y_invertido, muestra.head(1))
        for col in original.columns:
            if not np.allclose(original[col].iloc[i_fold], alterado[col].iloc[i_fold], equal_nan=True):
                errores.append(f"`{col}` cambia en el sub-fold {k} al invertir sus etiquetas")
    return errores


def target_mezclado(train, test, tr, te):
    """Con etiquetas al azar, el modelo completo cae al nivel de la clase mayoritaria."""
    muestra, y = _muestra(tr)
    rng = np.random.default_rng(config.SEED)
    y_mezclado = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    accuracy, _ = modelo.validar(muestra, y_mezclado, protocolo={"n_folds": 3, "semillas": [0]},
                                 registrar=False)
    azar = max(y.mean(), 1 - y.mean())
    print(f"      accuracy con target mezclado {accuracy:.3f}, clase mayoritaria {azar:.3f}")
    return [f"acierta {accuracy:.3f} con el target mezclado"] if accuracy > azar + TOLERANCIA_AZAR else []


def main():
    train, test = cargar_train_test()
    tr, te = features.construir(train, test)
    hay_errores = False
    for control in [invarianza_al_target, mismas_columnas, disponibilidad_en_test,
                    te_sin_etiqueta_propia, target_mezclado]:
        errores = control(train, test, tr, te)
        print(f"{'FALLA' if errores else 'OK':5} {control.__doc__}")
        for error in errores:
            print(f"      {error}")
        hay_errores = hay_errores or bool(errores)
    sys.exit(1 if hay_errores else 0)


if __name__ == "__main__":
    main()
