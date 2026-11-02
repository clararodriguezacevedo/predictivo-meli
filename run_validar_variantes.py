"""Valida variantes de hiperparámetros de un modelo sobre los mismos folds.

Parte de los hiperparámetros de una corrida registrada y cambia los que se
indiquen en cada variante. Arma los folds una sola vez (protocolo estándar), así
las diferencias entre variantes se deben solo a los hiperparámetros. Cada variante
queda registrada como "validacion".

Uso:
    python run_validar_variantes.py --base <RUN_ID> \\
        --variante n_estimators=3000 \\
        --variante n_estimators=3000,learning_rate=0.02,min_child_samples=2
"""

import argparse
import ast
import time

import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test


def leer_variante(texto):
    """Convierte "a=1,b=0.5" en {"a": 1, "b": 0.5}."""
    cambios = {}
    for parte in texto.split(","):
        clave, valor = parte.split("=")
        cambios[clave.strip()] = ast.literal_eval(valor.strip())
    return cambios


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, metavar="RUN_ID")
    parser.add_argument("--variante", action="append", required=True,
                        help='cambios sobre la base, por ejemplo "n_estimators=3000,num_leaves=255"')
    args = parser.parse_args()

    base = registro.buscar(args.base)
    tipo = base["modelo"]
    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    protocolo = modelo.PROTOCOLO_VALIDACION
    folds = modelo.armar_folds(train, y, protocolo)
    comun = modelo.descripcion_comun(folds, protocolo, usar_target_encoding=True)

    for texto in args.variante:
        cambios = leer_variante(texto)
        parametros = {**base["parametros"], **cambios}
        inicio = time.time()
        aciertos = modelo.validar_en_folds(folds, tipo, parametros)
        accuracy, error = modelo.resumir(aciertos)
        registro.registrar(
            "validacion", tipo, parametros, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            nota=f"variante de {base['id']}: {texto}", **comun,
        )
        print(f"{texto}: {accuracy:.4f} (error estándar {error:.4f}, "
              f"{time.time() - inicio:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
