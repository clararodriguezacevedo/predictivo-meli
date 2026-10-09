"""EDA de la competencia. Cada módulo agrupa secciones del reporte por tema.

Todas las funciones de sección reciben un `Contexto` con el train (con la columna
binaria `y`, 1 = usado), el test y el reporte donde escriben.
"""

from dataclasses import dataclass

import pandas as pd

from eda.reporte import Reporte


@dataclass
class Contexto:
    train: pd.DataFrame
    test: pd.DataFrame
    reporte: Reporte

    @property
    def tasa_global(self):
        return self.train["y"].mean()
