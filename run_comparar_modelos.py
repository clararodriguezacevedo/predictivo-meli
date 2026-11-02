"""Compara varios modelos sobre exactamente los mismos folds y registra cada uno.

Arma una sola vez los datos de cada fold (protocolo estándar, con target encoding)
y entrena cada modelo sobre ellos, así las diferencias de accuracy se deben solo
al modelo. Después evalúa promedios entre los mejores modelos reutilizando las
probabilidades ya calculadas, sin reentrenar.

Para LightGBM y XGBoost usa los hiperparámetros de su mejor validación registrada;
para el resto, los valores por defecto de `clasificadores.py`.

Uso:
    python run_comparar_modelos.py
    python run_comparar_modelos.py --modelos catboost random_forest --con-consenso
"""

import argparse
import itertools
import time

import numpy as np

import clasificadores
import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test

# Cuántos de los mejores modelos se combinan entre sí
MODELOS_A_COMBINAR = 4
MODELOS = [m for m in clasificadores.PARAMETROS_POR_DEFECTO]


def parametros_de(tipo):
    """Mejores hiperparámetros validados del modelo, o los valores por defecto."""
    try:
        return registro.mejor_validacion(tipo)["parametros"]
    except LookupError:
        return clasificadores.PARAMETROS_POR_DEFECTO[tipo]


def probabilidades(folds, tipo, parametros):
    """Probabilidad de usado en la parte de validación de cada fold."""
    resultado = []
    for X_tr, y_tr, X_val, _, semilla in folds:
        clasificador = clasificadores.crear_clasificador(tipo, parametros, semilla).fit(X_tr, y_tr)
        resultado.append(clasificador.predict_proba(X_val)[:, 1])
    return resultado


def aciertos_por_fold(probas, folds):
    return [float(((p > 0.5).astype(int) == y_val).mean())
            for p, (_, _, _, y_val, _) in zip(probas, folds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=MODELOS)
    parser.add_argument("--con-consenso", action="store_true")
    parser.add_argument("--nota", default="comparación de modelos sobre los mismos folds")
    args = parser.parse_args()

    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    protocolo = modelo.PROTOCOLO_VALIDACION
    folds = modelo.armar_folds(train, y, protocolo, usar_consenso=args.con_consenso)
    comun = modelo.descripcion_comun(folds, protocolo, True, args.con_consenso)

    probas, parametros, accuracy = {}, {}, {}
    for tipo in args.modelos:
        inicio = time.time()
        parametros[tipo] = parametros_de(tipo)
        try:
            probas[tipo] = probabilidades(folds, tipo, parametros[tipo])
        except Exception as error:  # un modelo que falla no frena al resto
            print(f"{tipo}: FALLÓ ({type(error).__name__}: {str(error)[:200]})", flush=True)
            continue
        aciertos = aciertos_por_fold(probas[tipo], folds)
        accuracy[tipo], error = modelo.resumir(aciertos)
        registro.registrar(
            "validacion", tipo, parametros[tipo], accuracy=accuracy[tipo], error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            nota=args.nota, **comun,
        )
        print(f"{tipo}: {accuracy[tipo]:.4f} (error estándar {error:.4f}, "
              f"{time.time() - inicio:.0f} s)", flush=True)

    mejores = sorted(accuracy, key=accuracy.get, reverse=True)[:MODELOS_A_COMBINAR]
    print(f"\nPromedios entre los {len(mejores)} mejores: {mejores}", flush=True)
    for cantidad in range(2, len(mejores) + 1):
        for combinacion in itertools.combinations(mejores, cantidad):
            promedio = [np.mean([probas[t][i] for t in combinacion], axis=0)
                        for i in range(len(folds))]
            aciertos = aciertos_por_fold(promedio, folds)
            acc, error = modelo.resumir(aciertos)
            registro.registrar(
                "validacion", "ensamble", {t: parametros[t] for t in combinacion},
                accuracy=acc, error_estandar=error, accuracy_por_fold=aciertos,
                nota="promedio de " + " + ".join(combinacion), **comun,
            )
            print(f"{' + '.join(combinacion)}: {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
