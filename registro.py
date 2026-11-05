"""Registro de corridas: `runs.jsonl` solo agrega líneas y `RUNS.md` es su resumen legible.

Tipos: validacion (5 folds y 2 semillas, comparables), control (semillas nuevas), submission,
leaderboard (puntaje de Kaggle) y analisis (errores). Las filas busqueda y las de varios
modelos son históricas: el código que las generó ya no está.
"""

import hashlib
import json
import subprocess
from datetime import datetime

import pandas as pd

import config

CARPETA = config.RAIZ / "registro"
ARCHIVO = CARPETA / "runs.jsonl"
RESUMEN = CARPETA / "RUNS.md"


def _git(*args):
    return subprocess.run(["git", *args], cwd=config.RAIZ, capture_output=True, text=True).stdout.strip()


def describir_features(columnas):
    """Cantidad y huella de las columnas, para saber si dos corridas usaron las mismas."""
    columnas = sorted(columnas)
    return {"n": len(columnas), "huella": hashlib.sha1("|".join(columnas).encode()).hexdigest()[:10]}


def registrar(tipo, modelo, parametros, **detalles):
    """Agrega una corrida al registro y devuelve su id."""
    ahora = datetime.now()
    run = {
        "id": f"{ahora:%Y%m%d-%H%M%S}-{tipo}-{modelo}",
        "fecha": ahora.isoformat(timespec="seconds"),
        "tipo": tipo,
        "modelo": modelo,
        "parametros": parametros,
        **detalles,
        "commit": _git("rev-parse", "--short", "HEAD"),
        # Con cambios sin commitear, el commit solo no alcanza para reproducir
        "cambios_sin_commitear": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    with open(ARCHIVO, "a", encoding="utf8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    generar_resumen()
    return run["id"]


def leer():
    with open(ARCHIVO, encoding="utf8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def buscar(run_id):
    for run in leer():
        if run["id"] == run_id:
            return run
    raise KeyError(f"No existe la corrida {run_id}")


def _parametros_cortos(parametros):
    if parametros and all(isinstance(v, dict) for v in parametros.values()):
        return "; ".join(f"{m}: {_parametros_cortos(p)}" for m, p in parametros.items())
    return ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )


def _tabla(filas, columnas):
    return [pd.DataFrame(filas, columns=columnas).to_markdown(index=False), ""]


def generar_resumen():
    runs = leer()
    puntajes = {r["submission"]: r["leaderboard_publico"] for r in runs if r["tipo"] == "leaderboard"}
    lineas = ["# Registro de corridas", "", "Resumen generado desde `runs.jsonl`.", "",
              "## Validaciones (comparables entre sí)", ""]
    validaciones = sorted((r for r in runs if r["tipo"] == "validacion"), key=lambda r: -r["accuracy"])
    lineas += _tabla([{
        "id": r["id"], "modelo": r["modelo"], "accuracy": f"{r['accuracy']:.4f}",
        "features": r["features"]["n"], "nota": r.get("nota", ""),
        "parámetros": _parametros_cortos(r["parametros"]),
    } for r in validaciones], ["id", "modelo", "accuracy", "features", "nota", "parámetros"])

    lineas += ["## Controles con folds nuevos", ""]
    lineas += _tabla([{
        "id": r["id"], "semillas": r["protocolo"]["semillas"],
        "accuracy": f"{r['accuracy']:.4f}", "nota": r.get("nota", ""),
    } for r in runs if r["tipo"] == "control"], ["id", "semillas", "accuracy", "nota"])

    lineas += ["## Submissions", ""]
    lineas += _tabla([{
        "id": r["id"], "archivo": r["archivo"], "validación": r["run_validacion"],
        "accuracy de validación": f"{r['accuracy_validacion']:.4f}",
        "leaderboard público": puntajes.get(r["id"], "sin cargar"),
    } for r in runs if r["tipo"] == "submission"],
        ["id", "archivo", "validación", "accuracy de validación", "leaderboard público"])

    lineas += ["## Búsqueda de hiperparámetros (las 15 mejores de cada modelo)", ""]
    busquedas = [r for r in runs if r["tipo"] == "busqueda"]
    for modelo in sorted({r["modelo"] for r in busquedas}):
        del_modelo = sorted((r for r in busquedas if r["modelo"] == modelo), key=lambda r: -r["accuracy"])
        lineas += [f"### {modelo} ({len(del_modelo)} pruebas)", ""]
        lineas += _tabla([{
            "id": r["id"], "accuracy": f"{r['accuracy']:.4f}",
            "parámetros": _parametros_cortos(r["parametros"]),
        } for r in del_modelo[:15]], ["id", "accuracy", "parámetros"])
    RESUMEN.write_text("\n".join(lineas), encoding="utf8")
