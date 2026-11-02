"""Ablación: cuánto aporta cada grupo de features al modelo completo.

Arma una sola vez los folds del protocolo estándar (con target encoding) y valida
el mejor LightGBM con todas las features y sin cada grupo, sobre los mismos folds.
Los grupos son los del diccionario; el target encoding se separa por encoder.
Cada corrida queda registrada como "validacion".

Con `--semillas` usa folds nuevos, que ninguna prueba anterior miró, y registra las
corridas como "control". Sirve para confirmar que una mejora elegida sobre los folds
estándar no fue suerte de esos folds.

Uso:
    python run_evaluar_features.py                      (quita cada grupo por separado)
    python run_evaluar_features.py --juntos "Envío" "Formato de las fotos"
                                                        (quita esos grupos a la vez)
    python run_evaluar_features.py --semillas 2 3 --juntos te_vv_ duracion_resto_seg
                                                        (también acepta prefijos de columnas)
"""

import argparse
import time

import numpy as np

import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test
from diccionario import VARIABLES

# Corrida de la que se toman los hiperparámetros: el LightGBM de la primera submission
BASE = "20261010-195055-validacion-lightgbm"
# Prefijo de las columnas de cada encoder, para quitarlos de a uno
ENCODERS = {
    "Target encoding: vendedor": "te_seller_id",
    "Target encoding: categoría": "te_category_id",
    "Target encoding: vecinos por título": "te_vecinos",
}


def grupos(columnas):
    """{nombre del grupo: sus columnas}, en el orden del diccionario."""
    resultado = {}
    for v in VARIABLES:
        if v.nombre in columnas and not v.nombre.startswith("te_"):
            resultado.setdefault(v.grupo, []).append(v.nombre)
    for nombre, prefijo in ENCODERS.items():
        resultado[nombre] = [c for c in columnas if c.startswith(prefijo)]
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--juntos", nargs="+", action="append", metavar="GRUPO",
                        help="grupos o prefijos de columnas a quitar a la vez; se puede repetir")
    parser.add_argument("--semillas", nargs="+", type=int,
                        help="semillas de folds nuevos; las corridas se registran como control")
    args = parser.parse_args()

    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    base = registro.buscar(BASE)
    parametros = base["parametros"]
    protocolo = modelo.PROTOCOLO_VALIDACION
    tipo = "validacion"
    if args.semillas:
        protocolo, tipo = {**protocolo, "semillas": args.semillas}, "control"
    folds = modelo.armar_folds(train, y, protocolo)

    def validar(quitar, nota):
        inicio = time.time()
        sin = [(X_tr.drop(columns=quitar), y_tr, X_val.drop(columns=quitar), y_val, s)
               for X_tr, y_tr, X_val, y_val, s in folds]
        aciertos = modelo.validar_en_folds(sin, "lightgbm", parametros)
        accuracy, error = modelo.resumir(aciertos)
        registro.registrar(
            tipo, "lightgbm", parametros, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio), nota=nota,
            ablacion={"quitado": quitar},
            **modelo.descripcion_comun(sin, protocolo),
        )
        return np.array(aciertos)

    todas = folds[0][0].columns
    por_grupo = grupos(todas)

    def columnas_de(nombre):
        """Columnas de un grupo del diccionario, o las que empiezan con `nombre`."""
        if nombre in por_grupo:
            return por_grupo[nombre]
        return [c for c in todas if c.startswith(nombre)]

    a_quitar = ({" + ".join(g): sum((columnas_de(n) for n in g), []) for g in args.juntos}
                if args.juntos else por_grupo)

    completo = validar([], f"ablación: modelo completo (parámetros de {base['id']})")
    print(f"Completo: {completo.mean():.4f}", flush=True)
    print("\nAl quitar cada grupo (diferencia negativa: el grupo aporta):", flush=True)
    for grupo, columnas in a_quitar.items():
        sin_grupo = validar(columnas, f"ablación: sin {grupo}")
        diferencia = sin_grupo - completo
        # Error estándar de la diferencia pareada: los dos se validan en los mismos folds
        error = diferencia.std() / np.sqrt(len(diferencia))
        print(f"  sin {grupo} ({len(columnas)} variables): {sin_grupo.mean():.4f} "
              f"({diferencia.mean():+.4f} ± {error:.4f})", flush=True)


if __name__ == "__main__":
    main()
