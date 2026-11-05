"""Entrena con todo train, predice el test y registra la corrida como "submission".

Toda submission queda atada a una validación del registro con la misma configuración.

Uso:
    python run_submission.py --validacion <id de una validación>
    python run_submission.py --anotar <id de la submission> --puntaje 0.9412
"""

import argparse
import time
from datetime import datetime

import pandas as pd

from carga import cargar_train_test
import config
import features
import modelo
import registro
import target_encoding

CARPETA = config.OUTPUTS / "submissions"


def generar(validacion):
    # La submission tiene que usar la configuración que se validó
    if (validacion["target_encoding"] != target_encoding.descripcion()
            or validacion["parametros"] != modelo.PARAMETROS):
        raise SystemExit(f"{validacion['id']} se validó con otra configuración: validar de nuevo.")
    semilla = config.SEED
    inicio = time.time()

    train, test = features.construir(*cargar_train_test())
    y = modelo.y_binario(train)
    X_base_train, X_base_test = modelo.preparar_juntos(train, test)
    X_train, X_test = modelo.armar_matrices(X_base_train, train, y, X_base_test, test, semilla)
    es_usado = modelo.crear_clasificador(semilla).fit(X_train, y).predict(X_test) == 1

    CARPETA.mkdir(parents=True, exist_ok=True)
    archivo = CARPETA / f"{datetime.now():%Y%m%d-%H%M%S}-lightgbm.csv"
    # Kaggle numera las filas del test desde 1, no usa el id de MercadoLibre
    pd.DataFrame({
        "ID": range(1, len(test) + 1),
        config.TARGET: pd.Series(es_usado).map({True: config.CLASE_POSITIVA, False: "new"}),
    }).to_csv(archivo, index=False)

    return archivo, registro.registrar(
        "submission", "lightgbm", modelo.PARAMETROS,
        run_validacion=validacion["id"],
        accuracy_validacion=validacion["accuracy"],
        target_encoding=validacion["target_encoding"],
        features=registro.describir_features(X_train.columns),
        semilla=semilla,
        archivo=archivo.relative_to(config.RAIZ).as_posix(),
        filas=len(test),
        fraccion_predicha_usado=float(es_usado.mean()),
        fraccion_usado_en_train=float(y.mean()),
        duracion_seg=round(time.time() - inicio),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--validacion", metavar="RUN_ID")
    parser.add_argument("--anotar", metavar="RUN_ID", help="submission a la que cargarle el puntaje")
    parser.add_argument("--puntaje", type=float)
    args = parser.parse_args()

    if args.anotar:
        registro.buscar(args.anotar)
        registro.registrar("leaderboard", "", {}, submission=args.anotar, leaderboard_publico=args.puntaje)
        print(f"Puntaje {args.puntaje} anotado para {args.anotar}")
    elif args.validacion:
        archivo, run_id = generar(registro.buscar(args.validacion))
        print(f"Submission {run_id} guardada en {archivo}")
    else:
        parser.error("indicar --validacion o --anotar")


if __name__ == "__main__":
    main()
