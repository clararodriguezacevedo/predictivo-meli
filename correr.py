"""Tareas que entrenan o registran algo

    python correr.py leakage                       los cinco controles de leakage
    python correr.py validar --nota "una prueba"   valida el modelo y lo anota en el registro
    python correr.py baseline                      valida el baseline con los mismos folds
    python correr.py errores                       guarda las predicciones que analiza el notebook 07
    python correr.py entrenar --validacion ID      entrena el modelo final con todo train y lo guarda
    python correr.py predecir                      aplica el modelo guardado al test y escribe la submission
    python correr.py anotar ID --puntaje 0.9476    anota el puntaje de Kaggle de una submission

Tardan de minutos a horas y cada corrida queda en registro/runs.jsonl, que leen los notebooks
"""

import argparse
from datetime import datetime
import sys
import time

import joblib

import datos
import modelo
import pruebas

ARCHIVO_MODELO = datos.OUTPUTS / "modelo" / "lightgbm.joblib"
CARPETA_SUBMISSIONS = datos.OUTPUTS / "submissions"
PREDICCIONES_DE_VALIDACION = datos.OUTPUTS / "errores" / "predicciones_validacion.parquet"


def leakage(args):
    # Tiene que dar OK antes de commitear cambios en datos.py o modelo.py
    sys.exit(0 if pruebas.controlar_leakage() else 1)


def validar(args):
    train, _ = datos.construir(*datos.cargar_train_test())
    accuracy, error = modelo.validar(train, modelo.y_binario(train), nota=args.nota)
    print(f"accuracy {accuracy:.4f} (error estándar {error:.4f})")


def baseline(args):
    train, _ = datos.construir(*datos.cargar_train_test())
    accuracy, error, mayoritaria = modelo.validar_baseline(train, modelo.y_binario(train))
    print(f"baseline {accuracy:.4f} (error estándar {error:.4f}), clase mayoritaria {mayoritaria:.4f}")


def errores(args):
    """Predice cada item de train con un modelo que no lo vio, para el análisis de errores"""
    inicio = time.time()
    train, _ = datos.construir(*datos.cargar_train_test())
    y = modelo.y_binario(train)
    protocolo = {"n_folds": 5, "semillas": [0]}  # Una sola seed, así cada item se predice una sola vez
    folds = modelo.armar_folds(train, y, protocolo)
    pred = modelo.predecir_en_folds(folds)
    accuracy = float(((pred["proba_usado"] > 0.5).astype(int) == pred["y"]).mean())
    modelo.registrar(
        "analisis", "lightgbm", modelo.PARAMETROS, accuracy=accuracy,
        nota="predicciones de validación para análisis de errores",
        duracion_seg=round(time.time() - inicio), **modelo.descripcion_comun(folds, protocolo),
    )
    PREDICCIONES_DE_VALIDACION.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PREDICCIONES_DE_VALIDACION)
    print(f"Predicciones guardadas en {PREDICCIONES_DE_VALIDACION}")


def entrenar(args):
    """Entrena el modelo final con todo train y lo guarda, atado a la validación que lo respalda"""
    validacion = modelo.buscar(args.validacion)
    # El modelo final tiene que usar la configuración que se validó
    if validacion["target_encoding"] != modelo.descripcion() or validacion["parametros"] != modelo.PARAMETROS:
        sys.exit(f"{validacion['id']} se validó con otra configuración, hay que validar de nuevo.")
    train, test = datos.construir(*datos.cargar_train_test())
    clasificador = modelo.entrenar_final(train, test, modelo.y_binario(train))
    ARCHIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clasificador": clasificador, "validacion": validacion["id"]}, ARCHIVO_MODELO)
    print(f"Modelo guardado en {ARCHIVO_MODELO}")


def predecir(args):
    """Aplica el modelo guardado al test, escribe el archivo para Kaggle y lo anota en el registro"""
    inicio = time.time()
    guardado = joblib.load(ARCHIVO_MODELO)
    clasificador, validacion = guardado["clasificador"], modelo.buscar(guardado["validacion"])
    train, test = datos.construir(*datos.cargar_train_test())
    y = modelo.y_binario(train)
    submission = modelo.predecir_test(clasificador, train, test, y)

    CARPETA_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    archivo = CARPETA_SUBMISSIONS / f"{datetime.now():%Y%m%d-%H%M%S}-lightgbm.csv"
    submission.to_csv(archivo, index=False)

    run_id = modelo.registrar(
        "submission", "lightgbm", modelo.PARAMETROS,
        run_validacion=validacion["id"],
        accuracy_validacion=validacion["accuracy"],
        target_encoding=validacion["target_encoding"],
        features=modelo.describir_features(clasificador.feature_name_),
        seed=datos.SEED,
        archivo=archivo.relative_to(datos.RAIZ).as_posix(),
        filas=len(submission),
        fraccion_predicha_usado=float((submission[datos.TARGET] == datos.CLASE_POSITIVA).mean()),
        fraccion_usado_en_train=float(y.mean()),
        duracion_seg=round(time.time() - inicio),
    )
    print(f"Submission {run_id} guardada en {archivo}")


def anotar(args):
    modelo.buscar(args.submission)  # Falla si el id no existe
    modelo.registrar("leaderboard", "", {}, submission=args.submission, leaderboard_publico=args.puntaje)
    print(f"Puntaje {args.puntaje} anotado para {args.submission}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ordenes = parser.add_subparsers(dest="orden", required=True)
    ordenes.add_parser("leakage").set_defaults(tarea=leakage)
    orden = ordenes.add_parser("validar")
    orden.add_argument("--nota", default="")
    orden.set_defaults(tarea=validar)
    ordenes.add_parser("baseline").set_defaults(tarea=baseline)
    ordenes.add_parser("errores").set_defaults(tarea=errores)
    orden = ordenes.add_parser("entrenar")
    orden.add_argument("--validacion", required=True, metavar="ID")
    orden.set_defaults(tarea=entrenar)
    ordenes.add_parser("predecir").set_defaults(tarea=predecir)
    orden = ordenes.add_parser("anotar")
    orden.add_argument("submission", metavar="ID")
    orden.add_argument("--puntaje", type=float, required=True)
    orden.set_defaults(tarea=anotar)
    args = parser.parse_args()
    args.tarea(args)


if __name__ == "__main__":
    main()
