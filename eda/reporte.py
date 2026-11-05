"""Acumula texto, tablas y figuras del EDA y los vuelca a un reporte markdown."""

from pathlib import Path
import re

import matplotlib.pyplot as plt


def mostrar_markdown(texto, carpeta):
    """Muestra un reporte en un notebook, con sus figuras dentro de la celda."""
    from IPython.display import Image, Markdown, display  # Solo hace falta dentro de un notebook
    for trozo in re.split(r"(!\[.*?\]\(figuras/[^)]+\))", texto):
        figura = re.search(r"\((figuras/[^)]+)\)", trozo) if trozo.startswith("![") else None
        display(Image(filename=Path(carpeta) / figura.group(1)) if figura else Markdown(trozo))


class Reporte:
    def __init__(self, carpeta, titulo):
        self.carpeta = Path(carpeta)
        self.carpeta_figuras = self.carpeta / "figuras"
        self.carpeta_figuras.mkdir(parents=True, exist_ok=True)
        self.partes = [f"# {titulo}\n"]
        self.n_secciones = 0
        self.ya_mostradas = 0

    def mostrar(self):
        """Muestra en el notebook lo agregado desde la última vez."""
        mostrar_markdown("\n".join(self.partes[self.ya_mostradas:]), self.carpeta)
        self.ya_mostradas = len(self.partes)

    def seccion(self, titulo, nivel=2):
        # Las secciones principales se numeran solas, en el orden en que se agregan
        if nivel == 2:
            self.n_secciones += 1
            titulo = f"{self.n_secciones}. {titulo}"
        self.partes.append(f"\n{'#' * nivel} {titulo}\n")

    def texto(self, md):
        self.partes.append(md.strip() + "\n")

    def tabla(self, df, decimales=3, index=True):
        self.partes.append(df.round(decimales).to_markdown(index=index) + "\n")

    def figura(self, fig, nombre, epigrafe=""):
        ruta = self.carpeta_figuras / f"{nombre}.png"
        fig.tight_layout()
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        self.partes.append(f"![{epigrafe or nombre}](figuras/{ruta.name})\n")
        if epigrafe:
            self.partes.append(f"*{epigrafe}*\n")

    def guardar(self, nombre="REPORTE_EDA.md"):
        ruta = self.carpeta / nombre
        ruta.write_text("\n".join(self.partes), encoding="utf8")
        return ruta
