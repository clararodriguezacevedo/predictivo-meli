"""Corre los tests de cada decisión y genera docs/DECISIONES.md.

Las corridas ya registradas se reutilizan, así que regenerar el documento es rápido.

Uso:
    python run_decisiones.py
    python run_decisiones.py --semillas 2 3     (repite las pruebas con folds nuevos, como control)
"""

import argparse
from functools import lru_cache
import json
import re
import time

import numpy as np
import pandas as pd

from carga import COLUMNAS_DESCARTADAS, cargar_train_test
import config
import decisiones as dec
from diccionario import VARIABLES
from eda.calidad import puntajes_adversarios
import estadistica as est
import features
import modelo
import registro
import target_encoding

ALFA = 0.05  # Umbral habitual; en las comparaciones se aplica después de corregir por cantidad de tests
# Las descartadas por constantes: las URL varían y se prueban aparte (D70, D80)
CONSTANTES = [c for c in COLUMNAS_DESCARTADAS if c not in ("permalink", "thumbnail", "secure_thumbnail")]
FALTANTES = ["warranty", "video_id", "official_store_id", "parent_item_id"]


def pruebas_de_datos(train, test):
    """Decisiones que se prueban sobre los datos, sin entrenar el modelo final."""
    filas = []
    en_train, en_test = puntajes_adversarios(train, test)
    auc, p = est.mann_whitney(en_test, en_train)
    # Acá un p grande es lo que se busca: no se distingue train de test
    filas.append(("D01", "Validar con folds al azar: train y test son intercambiables",
                  f"Mann-Whitney, AUC {auc:.3f}", p, "respaldada" if p >= ALFA else "contradicha"))

    valores = {c: set() for c in CONSTANTES}
    for ruta in (config.TRAIN_RAW, config.TEST_RAW):
        with open(ruta, encoding="utf8") as f:
            for linea in f:
                r = json.loads(linea)
                for c in CONSTANTES:
                    valores[c].add(str(r.get(c)))
    no_constantes = [c for c, v in valores.items() if len(v) > 1]
    filas.append(("D02", "Descartar columnas constantes",
                  f"sin test: una constante no puede asociarse; no constantes: {no_constantes or 'ninguna'}",
                  np.nan, "contradicha" if no_constantes else "respaldada"))

    y = modelo.y_binario(train)
    for col in FALTANTES:
        p = est.chi2_independencia(train[col].notna(), y)
        filas.append(("D03", f"Usar la ausencia de `{col}` como señal",
                      f"chi cuadrado, usados {y[train[col].isna()].mean():.1%} si falta "
                      f"y {y[train[col].notna()].mean():.1%} si está",
                      p, "respaldada" if p < ALFA else "sin respaldo"))

    ruta = config.OUTPUTS / "errores" / "predicciones_validacion.parquet"
    if ruta.exists():
        pred = pd.read_parquet(ruta)
        aciertos = lambda corte: ((pred["proba_usado"] > corte).astype(int) == pred["y"]).to_numpy()
        p = est.mcnemar(aciertos(0.48), aciertos(0.5))
        filas.append(("D52", "Cortar en 0.5 y no en 0.48, el mejor valor visto",
                      f"McNemar, diferencia {100 * (aciertos(0.48).mean() - aciertos(0.5).mean()):+.3f} puntos",
                      p, "respaldada" if p >= ALFA else "contradicha"))
    return filas


def columnas_de(patrones, todas):
    """Columnas de los grupos del diccionario, o las que coinciden con una regex."""
    grupos = {}
    for v in VARIABLES:
        grupos.setdefault(v.grupo, []).append(v.nombre)
    columnas = []
    for patron in patrones:
        columnas += ([c for c in grupos[patron] if c in todas] if patron in grupos
                     else [c for c in todas if re.match(patron, c)])
    return columnas


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--semillas", nargs="+", type=int)
    args = parser.parse_args()

    train, test = features.construir(*cargar_train_test())
    y = modelo.y_binario(train)
    tipo_corrida, sufijo, protocolo = "validacion", "", modelo.PROTOCOLO_VALIDACION
    if args.semillas:
        tipo_corrida, sufijo = "control", f" (semillas {args.semillas})"
        protocolo = {**protocolo, "semillas": args.semillas}

    @lru_cache(maxsize=1)
    def folds():
        return modelo.armar_folds(train, y, protocolo)  # Lo caro: solo si falta alguna corrida

    def corrida(nota, transformar):
        """Accuracy por fold de una prueba: la registrada si ya está, o la corre y la registra."""
        for r in reversed(registro.leer()):
            if r["tipo"] == tipo_corrida and r.get("nota") == nota:
                return np.array(r["accuracy_por_fold"])
        inicio = time.time()
        variante, detalle = transformar(folds())
        aciertos = modelo.validar_en_folds(variante)
        accuracy, error = modelo.resumir(aciertos)
        registro.registrar(
            tipo_corrida, "lightgbm", modelo.PARAMETROS, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio), nota=nota,
            ablacion=detalle, **modelo.descripcion_comun(variante, protocolo),
        )
        print(f"{nota}: {accuracy:.4f}", flush=True)
        return np.array(aciertos)

    def sin_columnas(patrones):
        def transformar(fs):
            columnas = columnas_de(patrones, list(fs[0][0].columns))
            return [(X_tr.drop(columns=columnas), y_tr, X_val.drop(columns=columnas), y_val, s)
                    for X_tr, y_tr, X_val, y_val, s in fs], {"quitado": columnas}
        return transformar

    def con_encoder(fabrica):
        def transformar(fs):
            nuevos = []
            for X_tr, y_tr, X_val, y_val, s in fs:
                te_tr, te_val = target_encoding.codificar(
                    train.loc[X_tr.index], y_tr, train.loc[X_val.index], s, fabrica)
                nuevos.append((X_tr.assign(**te_tr), y_tr, X_val.assign(**te_val), y_val, s))
            return nuevos, {"reemplazado": list(te_tr.columns)}
        return transformar

    completo = corrida(f"decisiones: modelo completo{sufijo}", sin_columnas([]))

    # (código, decisión, accuracy con la pieza, sin la pieza, se queda)
    comparaciones = []
    for codigo, nombre, patrones in dec.ABLACIONES:
        comparaciones.append((codigo, f"Conservar: {nombre}", completo,
                              corrida(f"decisiones: sin {nombre}{sufijo}", sin_columnas(patrones)), True))
    for codigo, nombre, fabrica in dec.VARIANTES:
        comparaciones.append((codigo, nombre, corrida(f"decisiones: {nombre}{sufijo}", con_encoder(fabrica)),
                              completo, False))

    # Se busca por id o por nota: las exploraciones se identifican por nota
    por_clave = {}
    for r in registro.leer():
        if r.get("accuracy_por_fold"):
            por_clave[r["id"]] = por_clave[r.get("nota")] = r["accuracy_por_fold"]
    faltantes = []
    for codigo, nombre, con, sin, se_queda in dec.HISTORICAS:
        if con in por_clave and sin in por_clave:
            comparaciones.append((codigo, nombre, por_clave[con], por_clave[sin], se_queda))
        else:
            faltantes.append((codigo, nombre))

    ajustados = est.holm([est.t_corregido(c[2], c[3])[2] for c in comparaciones])
    filas = pruebas_de_datos(train, test)
    for (codigo, nombre, con, sin, se_queda), p in zip(comparaciones, ajustados):
        dif, error, _ = est.t_corregido(con, sin)
        significativo = p < ALFA
        if se_queda:
            veredicto = "respaldada" if significativo and dif > 0 else "sin respaldo"
        else:
            veredicto = "contradicha" if significativo and dif > 0 else "respaldada"
        filas.append((codigo, nombre, f"t corregido, {100 * dif:+.3f} ± {100 * error:.3f} puntos", p, veredicto))
    filas += [(c, n, "sin corrida con resultados por fold", np.nan, "sin test")
              for c, n in faltantes]

    tabla = pd.DataFrame(sorted(filas, key=lambda f: f[:2]),
                         columns=["Código", "Decisión", "Prueba", "p", "Veredicto"])
    tabla["p"] = tabla["p"].map(lambda p: "" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}"))
    conteo = tabla["Veredicto"].value_counts().to_dict()
    texto = (
        "# Decisiones y sus tests\n\nGenerado por `run_decisiones.py`.\n\n"
        "Las comparaciones de modelos usan el t pareado corregido de Nadeau y Bengio sobre los 10 "
        "folds, con corrección de Holm por la cantidad de tests (su p ya está ajustado). La "
        "diferencia es con la pieza menos sin la pieza, en puntos de accuracy, casi siempre "
        "medida con LightGBM.\n\n"
        "Veredictos: **respaldada** (al conservar, la pieza suma de forma significativa; al descartar, "
        "no se demostró que sume), **sin respaldo** (se conserva algo sin una mejora demostrada), "
        "**contradicha** (el test indica lo contrario), **sin test**.\n\n"
        "Un p alto no prueba que una pieza no aporte: con 10 folds y tantos tests solo se detectan "
        "las diferencias grandes. Además, sacar una pieza sola subestima su aporte si otra lleva "
        "información parecida, por eso algunas se prueban juntas.\n\n"
        f"Resumen: {conteo}\n\n" + tabla.to_markdown(index=False) + "\n"
    )
    config.DOCS.mkdir(exist_ok=True)
    nombre = "DECISIONES_CONTROL.md" if args.semillas else "DECISIONES.md"
    (config.DOCS / nombre).write_text(texto, encoding="utf8")
    print(f"Resumen: {conteo}. Guardado en docs/{nombre}")


if __name__ == "__main__":
    main()
