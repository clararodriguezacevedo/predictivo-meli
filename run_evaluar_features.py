"""Mide el accuracy de validación cruzada con las features actuales.

Compara el set completo contra la versión sin cada grupo de features, para ver
cuánto aporta cada uno, y guarda la importancia de cada variable.

Uso: python run_evaluar_features.py
"""

import config
import evaluacion
import features
from carga import cargar_train_test
from diccionario import VARIABLES

GRUPOS_A_EVALUAR = [
    "Palabras clave en título y garantía",
    "Ficha técnica (vehículos, inmuebles y repuestos)",
    "Formato de las fotos",
    "Perfil del vendedor",
    "Perfil de la categoría",
    "Perfil por palabra del título",
]


def main():
    train, _ = features.construir(*cargar_train_test())
    X, y = evaluacion.preparar(train), evaluacion.y_binario(train)

    completo, error = evaluacion.validar(X, y)
    print(f"Todas las features ({X.shape[1]}): accuracy {completo:.4f} (error estándar {error:.4f})")

    print("\nAccuracy al quitar cada grupo (cuanto más baja, más aporta el grupo):")
    for grupo in GRUPOS_A_EVALUAR:
        columnas = [v.nombre for v in VARIABLES if v.grupo == grupo and v.nombre in X]
        sin_grupo, _ = evaluacion.validar(X.drop(columns=columnas), y)
        print(f"  sin {grupo} ({len(columnas)} variables): "
              f"{sin_grupo:.4f} ({sin_grupo - completo:+.4f})")

    salida = config.OUTPUTS / "evaluacion"
    salida.mkdir(parents=True, exist_ok=True)
    imp = evaluacion.importancias(X, y)
    imp.rename("importancia").to_csv(salida / "importancia_features.csv", index_label="variable")
    print("\nTop 20 por importancia:")
    print(imp.head(20).round(4).to_string())


if __name__ == "__main__":
    main()
