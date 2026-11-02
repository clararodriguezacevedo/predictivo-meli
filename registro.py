"""Registro de corridas: cada validación, búsqueda y submission queda anotada.

`registro/runs.jsonl` es la fuente de verdad: una línea JSON por corrida, con todo
lo necesario para reproducirla (modelo, hiperparámetros, target encoding, features,
protocolo de validación, commit) y su resultado. Nunca se edita a mano ni se
reescribe: solo se agregan líneas. `registro/RUNS.md` es un resumen legible que se
regenera a partir del jsonl.

Tipos de corrida:

- validacion: protocolo estándar (5 folds, 2 semillas). Son comparables entre sí
  y son las que se usan para elegir el modelo.
- busqueda: una prueba de la búsqueda de hiperparámetros, con un protocolo más
  liviano y otras semillas. Sirven para elegir candidatos, no para comparar con
  las de validación.
- control: mismo esquema que validacion pero con semillas nuevas, que no se usaron
  para elegir nada. Confirman que una mejora no fue suerte de los folds estándar.
- submission: entrenamiento con todo train y predicción del test.
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
    try:
        salida = subprocess.run(["git", *args], cwd=config.RAIZ, capture_output=True, text=True)
        return salida.stdout.strip()
    except OSError:
        return ""


def describir_features(columnas):
    """Cantidad y huella de las columnas, para saber si dos corridas usaron las mismas."""
    columnas = sorted(columnas)
    huella = hashlib.sha1("|".join(columnas).encode()).hexdigest()[:10]
    return {"n": len(columnas), "huella": huella}


def registrar(tipo, modelo, parametros, **detalles):
    """Agrega una corrida al registro y devuelve su id."""
    CARPETA.mkdir(exist_ok=True)
    ahora = datetime.now()
    run = {
        "id": f"{ahora:%Y%m%d-%H%M%S}-{tipo}-{modelo}",
        "fecha": ahora.isoformat(timespec="seconds"),
        "tipo": tipo,
        "modelo": modelo,
        "parametros": parametros,
        **detalles,
        "commit": _git("rev-parse", "--short", "HEAD"),
        # Si hay cambios sin commitear, el commit solo no alcanza para reproducir
        "cambios_sin_commitear": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    existentes = {r["id"] for r in leer()}
    sufijo = 2
    id_base = run["id"]
    while run["id"] in existentes:
        run["id"] = f"{id_base}-{sufijo}"
        sufijo += 1
    with open(ARCHIVO, "a", encoding="utf8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    generar_resumen()
    return run["id"]


def leer():
    if not ARCHIVO.exists():
        return []
    with open(ARCHIVO, encoding="utf8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def buscar(run_id):
    for run in leer():
        if run["id"] == run_id:
            return run
    raise KeyError(f"No existe la corrida {run_id}")


def mejor_validacion(modelo=None):
    """La corrida de validación estándar con mayor accuracy (opcionalmente de un modelo).

    Ignora las ablaciones: les falta un grupo de features y no sirven para una submission.
    """
    candidatas = [r for r in leer() if r["tipo"] == "validacion" and "ablacion" not in r
                  and (modelo is None or r["modelo"] == modelo)]
    if not candidatas:
        raise LookupError("Todavía no hay corridas de validación registradas")
    return max(candidatas, key=lambda r: r["accuracy"])


def anotar_leaderboard(run_id, puntaje, nota=""):
    """Agrega una línea con el puntaje público de una submission ya registrada."""
    buscar(run_id)
    return registrar(
        "leaderboard", "", {}, submission=run_id, leaderboard_publico=puntaje, nota=nota
    )


def _parametros_cortos(parametros):
    if parametros and all(isinstance(v, dict) for v in parametros.values()):
        # Ensamble: un diccionario de hiperparámetros por modelo
        return "; ".join(f"{m}: {_parametros_cortos(p)}" for m, p in parametros.items())
    return ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )


def _tabla(filas, columnas):
    if not filas:
        return ["Todavía no hay corridas de este tipo.", ""]
    df = pd.DataFrame(filas)[columnas]
    return [df.to_markdown(index=False), ""]


def generar_resumen():
    runs = leer()
    puntajes = {r["submission"]: r["leaderboard_publico"] for r in runs if r["tipo"] == "leaderboard"}
    lineas = [
        "# Registro de corridas",
        "",
        "Resumen generado desde `runs.jsonl`, que tiene el detalle completo de cada "
        "corrida (hiperparámetros exactos, accuracy por fold, features, commit).",
        "",
        "## Validaciones (protocolo estándar, comparables entre sí)",
        "",
    ]
    validaciones = sorted((r for r in runs if r["tipo"] == "validacion"),
                          key=lambda r: -r["accuracy"])
    lineas += _tabla([{
        "id": r["id"], "modelo": r["modelo"], "accuracy": f"{r['accuracy']:.4f}",
        "error estándar": f"{r['error_estandar']:.4f}",
        "target encoding": (f"{len(r['target_encoding']['encoders'])} encoders"
                            if r["target_encoding"]["activo"] else "no"),
        "consenso": "sí" if r.get("consenso", {}).get("activo") else "no",
        "pseudo-etiquetas": "sí" if r.get("pseudo_etiquetas", {}).get("activo") else "no",
        "features": r["features"]["n"], "nota": r.get("nota", ""),
        "parámetros": _parametros_cortos(r["parametros"]),
    } for r in validaciones], ["id", "modelo", "accuracy", "error estándar", "target encoding",
                               "consenso", "pseudo-etiquetas", "features", "nota", "parámetros"])

    lineas += ["## Controles con folds nuevos", "",
               "Semillas que no se usaron para elegir nada. Se comparan solo entre sí.", ""]
    lineas += _tabla([{
        "id": r["id"], "semillas": r["protocolo"]["semillas"],
        "accuracy": f"{r['accuracy']:.4f}", "error estándar": f"{r['error_estandar']:.4f}",
        "nota": r.get("nota", ""),
    } for r in runs if r["tipo"] == "control"],
        ["id", "semillas", "accuracy", "error estándar", "nota"])

    lineas += ["## Submissions", ""]
    lineas += _tabla([{
        "id": r["id"], "modelo": r["modelo"], "archivo": r["archivo"],
        "validación de origen": r["run_validacion"],
        "accuracy de validación": f"{r['accuracy_validacion']:.4f}",
        "leaderboard público": puntajes.get(r["id"], "sin cargar"),
    } for r in runs if r["tipo"] == "submission"],
        ["id", "modelo", "archivo", "validación de origen", "accuracy de validación",
         "leaderboard público"])

    lineas += ["## Búsqueda de hiperparámetros", "",
               "Protocolo más liviano que el estándar. Se muestran las 15 mejores pruebas "
               "de cada modelo.", ""]
    busquedas = [r for r in runs if r["tipo"] == "busqueda"]
    for modelo in sorted({r["modelo"] for r in busquedas}):
        del_modelo = sorted((r for r in busquedas if r["modelo"] == modelo),
                            key=lambda r: -r["accuracy"])
        lineas += [f"### {modelo} ({len(del_modelo)} pruebas)", ""]
        lineas += _tabla([{
            "id": r["id"], "accuracy": f"{r['accuracy']:.4f}",
            "parámetros": _parametros_cortos(r["parametros"]),
        } for r in del_modelo[:15]], ["id", "accuracy", "parámetros"])
    RESUMEN.write_text("\n".join(lineas), encoding="utf8")
