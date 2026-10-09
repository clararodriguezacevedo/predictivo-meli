"""Genera el reporte de EDA en outputs/eda/REPORTE_EDA.md (con figuras).

Uso: python run_eda.py
"""

import time

import config
import features
from carga import cargar_train_test
from eda import Contexto, calidad, relaciones, variables
from eda.graficos import estilo
from eda.reporte import Reporte

SECCIONES = [
    calidad.calidad_datos,
    calidad.target,
    calidad.ids,
    calidad.train_vs_test,
    variables.categoricas,
    variables.stock,
    variables.precio,
    variables.temporal,
    variables.texto,
    relaciones.vendedores,
    relaciones.categorias,
    relaciones.ranking_global,
]


def main():
    estilo()
    train, test = features.construir(*cargar_train_test())
    train["y"] = (train[config.TARGET] == config.CLASE_POSITIVA).astype(int)

    reporte = Reporte(config.OUTPUTS / "eda", "EDA: items nuevos vs usados de MercadoLibre")
    ctx = Contexto(train=train, test=test, reporte=reporte)
    for seccion in SECCIONES:
        inicio = time.time()
        seccion(ctx)
        print(f"{seccion.__module__}.{seccion.__name__}: {time.time() - inicio:.1f}s")
    print(f"Reporte guardado en {reporte.guardar()}")


if __name__ == "__main__":
    main()
