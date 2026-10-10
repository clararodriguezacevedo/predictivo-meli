"""Entrena con todo train, predice el test y registra la corrida como "submission".

Toma el modelo y los hiperparámetros de una corrida de validación del registro (por
defecto, la de mayor accuracy), así toda submission queda atada a una validación.

Uso:
    python run_submission.py
    python run_submission.py --validacion <id de una corrida de validación>

Después de subir el archivo a Kaggle, anotar el puntaje:
    python run_submission.py --anotar <id de la submission> --puntaje 0.9412
"""

import argparse
import time
from datetime import datetime

import pandas as pd

import config
import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test

CARPETA = config.OUTPUTS / "submissions"


def generar(run_validacion):
    tipo, parametros = run_validacion["modelo"], run_validacion["parametros"]
    usar_te = run_validacion["target_encoding"]["activo"]
    semilla = config.SEED
    inicio = time.time()

    train, test = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    # Mismos niveles de las categóricas en train y test
    niveles = {
        c: sorted(set(train[c].dropna()) | set(test[c].dropna())) for c in evaluacion.CATEGORICAS
    }
    X_train, X_test = modelo.armar_matrices(
        evaluacion.preparar(train, niveles), train, y,
        evaluacion.preparar(test, niveles), test, semilla, usar_te,
    )
    clasificador = modelo.crear_clasificador(tipo, parametros, semilla).fit(X_train, y)
    proba = clasificador.predict_proba(X_test)[:, 1]
    es_usado = proba > 0.5

    # El archivo se escribe antes de registrar, así el registro guarda su ruta real
    CARPETA.mkdir(parents=True, exist_ok=True)
    nombre = f"{datetime.now():%Y%m%d-%H%M%S}-{tipo}"
    archivo = CARPETA / f"{nombre}.csv"
    pd.DataFrame({
        "id": test["id"],
        config.TARGET: pd.Series(es_usado).map({True: config.CLASE_POSITIVA, False: "new"}),
    }).to_csv(archivo, index=False)
    # Las probabilidades se guardan aparte, por si después se combinan modelos
    pd.DataFrame({"id": test["id"], "proba_usado": proba}).to_csv(
        CARPETA / f"{nombre}_proba.csv", index=False
    )

    return archivo, registro.registrar(
        "submission", tipo, parametros,
        run_validacion=run_validacion["id"],
        accuracy_validacion=run_validacion["accuracy"],
        target_encoding=run_validacion["target_encoding"],
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
        registro.anotar_leaderboard(args.anotar, args.puntaje)
        print(f"Puntaje {args.puntaje} anotado para {args.anotar}")
        return

    run_validacion = (registro.buscar(args.validacion) if args.validacion
                      else registro.mejor_validacion())
    print(f"Usando {run_validacion['id']} (accuracy {run_validacion['accuracy']:.4f})")
    archivo, run_id = generar(run_validacion)
    print(f"Submission {run_id} guardada en {archivo}")


if __name__ == "__main__":
    main()
