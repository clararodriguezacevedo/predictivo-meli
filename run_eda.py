"""Genera el reporte de EDA en outputs/eda/REPORTE_EDA.md (con figuras).

Uso: python run_eda.py
"""

import time

import config
import features
from carga import cargar_train_test
from eda import Contexto, calidad, contexto, relaciones, variables
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
    variables.senales_semanticas,
    contexto.fotos,
    relaciones.vendedores,
    contexto.comportamiento_vendedor,
    relaciones.categorias,
    contexto.perfil_por_palabra,
    relaciones.ranking_global,
    contexto.control_de_leakage,
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
