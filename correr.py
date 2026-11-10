"""Las tareas que entrenan o registran algo. Cada una se corre con su orden.

    python correr.py leakage                       los cinco controles de leakage
    python correr.py validar --nota "una prueba"   valida el modelo y lo anota en el registro
    python correr.py decisiones                    corre las pruebas y genera docs/DECISIONES.md
    python correr.py errores                       guarda las predicciones que analiza el notebook 07
    python correr.py submission --validacion ID    genera el archivo para Kaggle
    python correr.py anotar ID --puntaje 0.9476    anota el puntaje de Kaggle de una submission

Son scripts y no notebooks porque tardan (de minutos a horas) y cada corrida queda anotada en
registro/runs.jsonl. Los notebooks leen ese registro.
"""

import argparse
from datetime import datetime
import sys
import time

import pandas as pd

import datos
import modelo
import pruebas

CARPETA_SUBMISSIONS = datos.OUTPUTS / "submissions"
PREDICCIONES_DE_VALIDACION = datos.OUTPUTS / "errores" / "predicciones_validacion.parquet"


def leakage(args):
    # Tiene que dar OK antes de commitear cambios en datos.py o modelo.py
    sys.exit(0 if pruebas.controlar_leakage() else 1)


def validar(args):
    train, _ = datos.construir(*datos.cargar_train_test())
    accuracy, error = modelo.validar(train, modelo.y_binario(train), nota=args.nota)
    print(f"accuracy {accuracy:.4f} (error estándar {error:.4f})")


def decisiones(args):
    pruebas.generar_decisiones()


def errores(args):
    """Predice cada item de train con un modelo que no lo vio, para el análisis de errores."""
    inicio = time.time()
    train, _ = datos.construir(*datos.cargar_train_test())
    y = modelo.y_binario(train)
    protocolo = {"n_folds": 5, "semillas": [0]}  # Una sola semilla, así cada item se predice una sola vez
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


def submission(args):
    """Entrena con todo train, predice el test y registra la corrida, atada a una validación."""
    validacion = modelo.buscar(args.validacion)
    # La submission tiene que usar la configuración que se validó
    if validacion["target_encoding"] != modelo.descripcion() or validacion["parametros"] != modelo.PARAMETROS:
        sys.exit(f"{validacion['id']} se validó con otra configuración, hay que validar de nuevo.")
    semilla = datos.SEED
    inicio = time.time()

    train, test = datos.construir(*datos.cargar_train_test())
    y = modelo.y_binario(train)
    X_base_train, X_base_test = modelo.preparar_juntos(train, test)
    X_train, X_test = modelo.armar_matrices(X_base_train, train, y, X_base_test, test, semilla)
    es_usado = modelo.crear_clasificador(semilla).fit(X_train, y).predict(X_test) == 1

    CARPETA_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    archivo = CARPETA_SUBMISSIONS / f"{datetime.now():%Y%m%d-%H%M%S}-lightgbm.csv"
    # Kaggle numera las filas del test desde 1, no usa el id de MercadoLibre
    pd.DataFrame({
        "ID": range(1, len(test) + 1),
        datos.TARGET: pd.Series(es_usado).map({True: datos.CLASE_POSITIVA, False: "new"}),
    }).to_csv(archivo, index=False)

    run_id = modelo.registrar(
        "submission", "lightgbm", modelo.PARAMETROS,
        run_validacion=validacion["id"],
        accuracy_validacion=validacion["accuracy"],
        target_encoding=validacion["target_encoding"],
        features=modelo.describir_features(X_train.columns),
        semilla=semilla,
        archivo=archivo.relative_to(datos.RAIZ).as_posix(),
        filas=len(test),
        fraccion_predicha_usado=float(es_usado.mean()),
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
    ordenes.add_parser("decisiones").set_defaults(tarea=decisiones)
    ordenes.add_parser("errores").set_defaults(tarea=errores)
    orden = ordenes.add_parser("submission")
    orden.add_argument("--validacion", required=True, metavar="ID")
    orden.set_defaults(tarea=submission)
    orden = ordenes.add_parser("anotar")
    orden.add_argument("submission", metavar="ID")
    orden.add_argument("--puntaje", type=float, required=True)
    orden.set_defaults(tarea=anotar)
    args = parser.parse_args()
    args.tarea(args)


if __name__ == "__main__":
    main()
