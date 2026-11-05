"""Valida el modelo y lo anota en el registro.

Uso: python run_validar_modelo.py --nota "primera prueba"
"""

import argparse

from carga import cargar_train_test
import features
import modelo


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nota", default="")
    args = parser.parse_args()

    train, _ = features.construir(*cargar_train_test())
    accuracy, error = modelo.validar(train, modelo.y_binario(train), nota=args.nota)
    print(f"accuracy {accuracy:.4f} (error estándar {error:.4f})")


if __name__ == "__main__":
    main()
