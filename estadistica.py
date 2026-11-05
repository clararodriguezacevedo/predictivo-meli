"""Los tests estadísticos con los que se justifica cada decisión de docs/DECISIONES.md."""

import numpy as np
import pandas as pd
from scipy import stats

# 1 fila de validación por cada 4 de entrenamiento, por usar 5 folds
FILAS_VAL_SOBRE_TRAIN = 1 / 4


def t_corregido(con, sin):
    """Test t pareado de Nadeau y Bengio: (diferencia, error estándar, p), en proporción."""
    d = np.asarray(con) - np.asarray(sin)
    # La corrección importa: los folds comparten filas y el t común daría p demasiado chicos
    error = np.sqrt(d.var(ddof=1) * (1 / len(d) + FILAS_VAL_SOBRE_TRAIN))
    if error == 0:
        return float(d.mean()), 0.0, 1.0
    p = 2 * stats.t.sf(abs(d.mean() / error), len(d) - 1)
    return float(d.mean()), float(error), float(p)


def holm(p):
    """Corrección de Holm: con muchos tests, alguno da p < 0.05 por azar."""
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    ajustado = np.maximum.accumulate(p[orden] * (len(p) - np.arange(len(p)))).clip(max=1)
    resultado = np.empty(len(p))
    resultado[orden] = ajustado
    return resultado


def chi2_independencia(a, b):
    """p de chi cuadrado: ¿dos variables categóricas son independientes?"""
    return float(stats.chi2_contingency(pd.crosstab(a, b))[1])


def mcnemar(aciertos_a, aciertos_b):
    """p exacto de McNemar: compara dos clasificadores solo en las filas donde difieren."""
    solo_a = int((aciertos_a & ~aciertos_b).sum())
    solo_b = int((~aciertos_a & aciertos_b).sum())
    if solo_a + solo_b == 0:
        return 1.0
    return float(stats.binomtest(solo_a, solo_a + solo_b, 0.5).pvalue)


def mann_whitney(puntajes_a, puntajes_b):
    """(AUC, p): ¿los puntajes de un grupo tienden a ser mayores que los del otro?"""
    resultado = stats.mannwhitneyu(puntajes_a, puntajes_b)
    return float(resultado.statistic / (len(puntajes_a) * len(puntajes_b))), float(resultado.pvalue)
