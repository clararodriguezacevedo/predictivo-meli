"""Reporte de en qué items se equivoca el modelo final: outputs/errores/REPORTE_ERRORES.md.

Uso:
    python run_analizar_errores.py              (recalcula las predicciones)
    python run_analizar_errores.py --reusar     (usa las ya guardadas)
"""

import argparse
import time

import pandas as pd

from carga import cargar_train_test
import config
import errores
import features
import modelo
import registro

CARPETA = config.OUTPUTS / "errores"
PREDICCIONES = CARPETA / "predicciones_validacion.parquet"
PROTOCOLO = {"n_folds": 5, "semillas": [0]}  # Una semilla: cada item se predice una sola vez


def calcular_predicciones(train, y):
    inicio = time.time()
    folds = modelo.armar_folds(train, y, PROTOCOLO)
    pred = modelo.predecir_en_folds(folds)
    accuracy = float(((pred["proba_usado"] > 0.5).astype(int) == pred["y"]).mean())
    registro.registrar(
        "analisis", "lightgbm", modelo.PARAMETROS, accuracy=accuracy,
        nota="predicciones de validación para análisis de errores",
        duracion_seg=round(time.time() - inicio), **modelo.descripcion_comun(folds, PROTOCOLO),
    )
    CARPETA.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PREDICCIONES)
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reusar", action="store_true")
    args = parser.parse_args()

    train, _ = features.construir(*cargar_train_test())
    y = modelo.y_binario(train)
    pred = pd.read_parquet(PREDICCIONES) if args.reusar else calcular_predicciones(train, y)
    datos = train.join(pred.drop(columns=["y"])).assign(y=y)
    print(f"Reporte guardado en {errores.generar_reporte(datos, CARPETA)}")


if __name__ == "__main__":
    main()
