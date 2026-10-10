"""Analiza en qué items se equivoca el mejor modelo validado.

Calcula la predicción de validación de cada item de train (hecha por un modelo que
no lo vio), la guarda y genera outputs/errores/REPORTE_ERRORES.md con los errores
desglosados por tipo de vendedor, de publicación y de producto.

Uso:
    python run_analizar_errores.py              (recalcula las predicciones)
    python run_analizar_errores.py --reusar     (usa las predicciones ya guardadas)
"""

import argparse
import time

import pandas as pd

import config
import errores
import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test

CARPETA = config.OUTPUTS / "errores"
PREDICCIONES = CARPETA / "predicciones_validacion.parquet"
# Una sola semilla: cada item se predice exactamente una vez
PROTOCOLO = {"n_folds": 5, "semillas": [0]}


def calcular_predicciones(train, y):
    mejor = registro.mejor_validacion()
    inicio = time.time()
    folds = modelo.armar_folds(train, y, PROTOCOLO)
    pred = modelo.predecir_en_folds(folds, mejor["modelo"], mejor["parametros"])
    accuracy = float(((pred["proba_usado"] > 0.5).astype(int) == pred["y"]).mean())
    registro.registrar(
        "analisis", mejor["modelo"], mejor["parametros"], accuracy=accuracy,
        nota=f"predicciones de validación para análisis de errores, parámetros de {mejor['id']}",
        duracion_seg=round(time.time() - inicio),
        **modelo.descripcion_comun(folds, PROTOCOLO, usar_target_encoding=True),
    )
    CARPETA.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PREDICCIONES)
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reusar", action="store_true")
    args = parser.parse_args()

    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    pred = pd.read_parquet(PREDICCIONES) if args.reusar else calcular_predicciones(train, y)
    datos = train.join(pred.drop(columns=["y"])).assign(y=y)
    print(f"Reporte guardado en {errores.generar_reporte(datos, CARPETA)}")


if __name__ == "__main__":
    main()
