"""Búsqueda de hiperparámetros con Optuna. Cada prueba queda en el registro.

La búsqueda usa un protocolo propio (menos folds y otra semilla que la validación
estándar) para no elegir hiperparámetros mirando los mismos folds con los que
después se compara. Al terminar, valida la mejor prueba con el protocolo estándar
y la registra como "validacion".

Uso:
    python run_optimizar.py --modelo lightgbm --minutos 30
    python run_optimizar.py --modelo xgboost --pruebas 40
"""

import argparse
import time

import optuna

import evaluacion
import features
import modelo
import registro
from carga import cargar_train_test


def sugerir_lightgbm(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
    )


def sugerir_xgboost(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 2, log=True),
    )


ESPACIOS = {"lightgbm": sugerir_lightgbm, "xgboost": sugerir_xgboost}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modelo", choices=list(ESPACIOS), required=True)
    parser.add_argument("--pruebas", type=int, default=None)
    parser.add_argument("--minutos", type=float, default=None)
    args = parser.parse_args()
    if args.pruebas is None and args.minutos is None:
        parser.error("indicar --pruebas, --minutos o ambos")

    train, _ = features.construir(*cargar_train_test())
    y = evaluacion.y_binario(train)
    protocolo = modelo.PROTOCOLO_BUSQUEDA
    # El target encoding no depende de los hiperparámetros: se arma una sola vez
    folds = modelo.armar_folds(train, y, protocolo)
    comun = modelo.descripcion_comun(folds, protocolo, usar_target_encoding=True)

    def objetivo(trial):
        parametros = ESPACIOS[args.modelo](trial)
        inicio = time.time()
        aciertos = modelo.validar_en_folds(folds, args.modelo, parametros)
        accuracy, error = modelo.resumir(aciertos)
        registro.registrar(
            "busqueda", args.modelo, parametros, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            numero_de_prueba=trial.number, **comun,
        )
        print(f"prueba {trial.number}: {accuracy:.4f}", flush=True)
        return accuracy

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=0))
    # Si ya hubo búsquedas de este modelo con los mismos datos, sus pruebas se cargan
    # como resultados conocidos, así el buscador sigue desde lo aprendido sin repetirlas
    prueba_modelo = optuna.create_study().ask()
    ESPACIOS[args.modelo](prueba_modelo)
    distribuciones = prueba_modelo.distributions
    previas = [
        r for r in registro.leer() if r["tipo"] == "busqueda" and r["modelo"] == args.modelo
        and r["protocolo"] == protocolo and r["features"] == comun["features"]
        and r["target_encoding"] == comun["target_encoding"]
    ]
    for run in previas:
        estudio.add_trial(optuna.trial.create_trial(
            params={k: run["parametros"][k] for k in distribuciones},
            distributions=distribuciones, value=run["accuracy"],
        ))
    print(f"Pruebas anteriores retomadas: {len(previas)}", flush=True)
    if not previas:
        # La primera prueba parte de los parámetros por defecto, como punto de
        # referencia (los que el defecto no fija se sortean)
        estudio.enqueue_trial({k: v for k, v in modelo.PARAMETROS_POR_DEFECTO[args.modelo].items()
                               if k != "subsample_freq"})
    estudio.optimize(objetivo, n_trials=args.pruebas,
                     timeout=args.minutos * 60 if args.minutos else None)

    mejores = ESPACIOS[args.modelo](optuna.trial.FixedTrial(estudio.best_params))
    print(f"\nMejor prueba de la búsqueda: {estudio.best_value:.4f}")
    accuracy, error = modelo.validar(
        train, y, args.modelo, mejores,
        nota=f"mejor de {len(estudio.trials)} pruebas de búsqueda",
    )
    print(f"Validación estándar de la mejor: {accuracy:.4f} (error estándar {error:.4f})")


if __name__ == "__main__":
    main()
