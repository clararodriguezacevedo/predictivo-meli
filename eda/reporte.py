"""Acumula texto, tablas y figuras del EDA y los vuelca a un reporte markdown."""

from pathlib import Path

import matplotlib.pyplot as plt


class Reporte:
    def __init__(self, carpeta, titulo):
        self.carpeta = Path(carpeta)
        self.carpeta_figuras = self.carpeta / "figuras"
        self.carpeta_figuras.mkdir(parents=True, exist_ok=True)
        self.partes = [f"# {titulo}\n"]

    def seccion(self, titulo, nivel=2):
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
