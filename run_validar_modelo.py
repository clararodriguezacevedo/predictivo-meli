"""Valida un modelo con el protocolo estándar y lo anota en el registro.

La validación es anidada: en cada fold el target encoding se ajusta solo con la
parte de entrenamiento, así que el accuracy es comparable al del leaderboard.

Uso:
    python run_validar_modelo.py --modelo lightgbm
    python run_validar_modelo.py --modelo xgboost --nota "primera prueba"
    python run_validar_modelo.py --parametros-de <id de una corrida del registro>
    python run_validar_modelo.py --modelo lightgbm --sin-target-encoding
"""

import argparse

import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modelo", choices=list(modelo.PARAMETROS_POR_DEFECTO), default="lightgbm")
    parser.add_argument("--parametros-de", metavar="RUN_ID",
                        help="usar el modelo y los hiperparámetros de una corrida registrada")
    parser.add_argument("--sin-target-encoding", action="store_true")
    parser.add_argument("--nota", default="")
    args = parser.parse_args()

    tipo, parametros = args.modelo, None
    nota = args.nota
    if args.parametros_de:
        origen = registro.buscar(args.parametros_de)
        tipo, parametros = origen["modelo"], origen["parametros"]
        nota = nota or f"parámetros de {origen['id']}"

    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    accuracy, error = modelo.validar(
        train, y, tipo, parametros, usar_target_encoding=not args.sin_target_encoding, nota=nota
    )
    print(f"{tipo}: accuracy {accuracy:.4f} (error estándar {error:.4f})")


if __name__ == "__main__":
    main()
