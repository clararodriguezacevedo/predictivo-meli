"""Valida el modelo completo con y sin target encoding.

La validación es anidada: en cada fold el target encoding se ajusta solo con la
parte de entrenamiento, así que el accuracy que reporta es comparable al del
leaderboard.

Uso: python run_validar_modelo.py
"""

import evaluacion
import features
import modelo
from carga import cargar_train_test


def main():
    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    for nombre, usar in [("Sin target encoding", False), ("Con target encoding", True)]:
        accuracy, error = modelo.validar(train, y, usar_target_encoding=usar)
        print(f"{nombre}: accuracy {accuracy:.4f} (error estándar {error:.4f})", flush=True)


if __name__ == "__main__":
    main()
