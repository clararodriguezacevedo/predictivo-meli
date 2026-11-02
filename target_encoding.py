"""Features que usan el target, calculadas sin filtrarlo (target encoding con folds).

A diferencia de `features.py`, acá todo depende de `condition`, así que nunca se
calcula sobre el dataset completo. Cada encoder aprende de un conjunto de filas con
etiqueta (`fit`) y se aplica a otras filas (`transform`). La función `codificar`
arma las features respetando dos reglas:

- Las filas a predecir (validación o test) se codifican con las etiquetas de todo
  el entrenamiento, y sus propias etiquetas nunca intervienen.
- Las filas de entrenamiento se codifican por sub-folds: cada una recibe valores
  calculados con las etiquetas de los otros sub-folds. Así ninguna fila ve su
  propia etiqueta y la feature se comporta igual que en las filas a predecir.

Se usan folds y no leave-one-out porque con leave-one-out el valor de cada fila
depende (al revés) de su propia etiqueta, y un modelo de árboles puede deducirla.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

import config
from features import normalizar_texto

N_SUBFOLDS = 5


class EncoderDeGrupo:
    """Tasa de usados suavizada de un grupo (vendedor, categoría, etc.).

    El suavizado acerca la tasa al promedio global cuando el grupo tiene pocos
    casos: tasa = (usados + m * global) / (n + m). También devuelve `n`, la cantidad
    de filas con etiqueta en las que se apoya, para que el modelo sepa cuánto
    confiar. Los grupos sin ninguna fila con etiqueta quedan en nulo.
    """

    def __init__(self, *columnas, suavizado=2.0, normalizar=False):
        # Varias columnas definen el grupo por su combinación (vendedor y categoría).
        # `normalizar` pasa el texto a minúsculas sin tildes antes de agrupar.
        self.columnas = list(columnas)
        self.suavizado = suavizado
        self.normalizar = normalizar

    @property
    def nombre(self):
        return "te_" + "_x_".join(self.columnas)

    def _clave(self, df):
        partes = [df[c].astype("string").fillna("(nulo)") for c in self.columnas]
        if self.normalizar:
            partes = [normalizar_texto(parte) for parte in partes]
        clave = partes[0]
        for parte in partes[1:]:
            clave = clave + "|" + parte
        return clave.to_numpy()

    def fit(self, df, y):
        self.tasa_global_ = y.mean()
        self.stats_ = y.groupby(self._clave(df)).agg(["sum", "count"])
        return self

    def transform(self, df):
        stats = self.stats_.reindex(self._clave(df))
        n = stats["count"].to_numpy()
        tasa = (stats["sum"].to_numpy() + self.suavizado * self.tasa_global_) / (
            n + self.suavizado
        )
        return pd.DataFrame({
            self.nombre: tasa,
            f"{self.nombre}_n": np.nan_to_num(n),
        }, index=df.index)


class EncoderDeVecinos:
    """Condición de los títulos más parecidos entre las filas con etiqueta.

    Representa cada título con TF-IDF de palabras y pares de palabras, y para cada
    fila busca los `k` títulos de entrenamiento con mayor similitud coseno. Devuelve
    la tasa de usados de esos vecinos (simple y ponderada por similitud) y qué tan
    parecido es el más cercano. Captura "este producto exacto suele venderse
    usado", algo que un modelo lineal sobre palabras sueltas no ve.

    Las palabras muy frecuentes se descartan (`max_df`): no ayudan a encontrar
    productos parecidos y hacen la búsqueda mucho más lenta.
    """

    def __init__(self, k=10, max_df=0.02, bloque=2000):
        self.k = k
        self.max_df = max_df
        self.bloque = bloque

    def fit(self, df, y):
        self.vectorizador_ = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=self.max_df, sublinear_tf=True,
            token_pattern=r"[a-z0-9]+",
        )
        # TF-IDF ya normaliza cada fila, así que el producto punto es la similitud coseno
        self.matriz_ = self.vectorizador_.fit_transform(normalizar_texto(df["title"])).T.tocsr()
        self.y_ = y.to_numpy().astype(float)
        self.tasa_global_ = self.y_.mean()
        return self

    def _vecinos(self, consultas):
        """Similitud y etiqueta de los k vecinos más cercanos de cada consulta."""
        n = consultas.shape[0]
        similitud = np.zeros((n, self.k))
        etiqueta = np.full((n, self.k), np.nan)
        for inicio in range(0, n, self.bloque):
            producto = (consultas[inicio:inicio + self.bloque] @ self.matriz_).tocsr()
            for i in range(producto.shape[0]):
                fila = slice(producto.indptr[i], producto.indptr[i + 1])
                valores, indices = producto.data[fila], producto.indices[fila]
                if len(valores) > self.k:
                    mejores = np.argpartition(valores, -self.k)[-self.k:]
                    valores, indices = valores[mejores], indices[mejores]
                similitud[inicio + i, :len(valores)] = valores
                etiqueta[inicio + i, :len(valores)] = self.y_[indices]
        return similitud, etiqueta

    def transform(self, df):
        consultas = self.vectorizador_.transform(normalizar_texto(df["title"]))
        similitud, etiqueta = self._vecinos(consultas)
        hay_vecino = ~np.isnan(etiqueta)
        n_vecinos = hay_vecino.sum(axis=1)
        usados = np.where(hay_vecino, etiqueta, 0)
        peso = similitud.sum(axis=1)
        sin_vecinos = n_vecinos == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            tasa = usados.sum(axis=1) / n_vecinos
            tasa_ponderada = (usados * similitud).sum(axis=1) / peso
        return pd.DataFrame({
            "te_vecinos_tasa": np.where(sin_vecinos, np.nan, tasa),
            "te_vecinos_tasa_ponderada": np.where(sin_vecinos, np.nan, tasa_ponderada),
            "te_vecinos_similitud_max": similitud.max(axis=1),
            "te_vecinos_similitud_media": np.where(sin_vecinos, 0, peso / np.maximum(n_vecinos, 1)),
            "te_vecinos_n": n_vecinos,
        }, index=df.index)


class EncoderVecinosDelVendedor:
    """Etiquetas de los items del mismo vendedor que más se parecen a cada item.

    La tasa de usados del vendedor trata igual a todas sus publicaciones. Pero un
    vendedor mixto (por ejemplo, una librería que vende libros nuevos y usados)
    suele publicar parecido los items de la misma condición. Para cada fila se
    miran los items con etiqueta de su vendedor y se devuelve la etiqueta del de
    título más parecido, la del publicado más cerca en el tiempo y la tasa entre
    los de precio parecido o mismo tipo de publicación y stock.
    """

    # Diferencia máxima de log del precio para considerar dos precios parecidos
    PRECIO_PARECIDO = 0.2

    @staticmethod
    def _tabla(df):
        return pd.DataFrame({
            "seller_id": df["seller_id"].to_numpy(),
            "log_price": df["log_price"].to_numpy(),
            "horas": (df["date_created"] - pd.Timestamp("2015-01-01")).dt.total_seconds().to_numpy()
                     / 3600,
            "tipo": df["listing_type_id"].astype(str).to_numpy() + df["iq_es_1"].astype(str).to_numpy(),
            "fila": np.arange(len(df)),
        })

    def fit(self, df, y):
        self.vectorizador_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                             token_pattern=r"[a-z0-9]+")
        self.titulos_ = self.vectorizador_.fit_transform(normalizar_texto(df["title"])).tocsr()
        self.etiquetados_ = self._tabla(df).assign(y=y.to_numpy().astype(float))
        return self

    def transform(self, df):
        titulos = self.vectorizador_.transform(normalizar_texto(df["title"])).tocsr()
        # Un par por cada combinación de fila a codificar e item con etiqueta del mismo vendedor
        p = self._tabla(df).merge(self.etiquetados_, on="seller_id", suffixes=("", "_e"))
        p["similitud"] = np.asarray(
            titulos[p["fila"].to_numpy()].multiply(self.titulos_[p["fila_e"].to_numpy()]).sum(axis=1)
        ).ravel()
        p["distancia_horas"] = (p["horas"] - p["horas_e"]).abs()
        g = p.groupby("fila")

        out = pd.DataFrame(index=np.arange(len(df)))
        mas_parecido = p.loc[g["similitud"].idxmax()].set_index("fila")
        out["te_vv_similitud_max"] = mas_parecido["similitud"]
        out["te_vv_y_mas_parecido"] = mas_parecido["y"]
        peso = p["similitud"] ** 2 + 1e-3
        out["te_vv_tasa_ponderada"] = (p["y"] * peso).groupby(p["fila"]).sum() / peso.groupby(p["fila"]).sum()
        mas_cercano = p[p["distancia_horas"] == g["distancia_horas"].transform("min")].groupby("fila")
        out["te_vv_y_mas_cercano_en_tiempo"] = mas_cercano["y"].mean()
        out["te_vv_horas_al_mas_cercano"] = mas_cercano["distancia_horas"].min()
        precio = p[(p["log_price"] - p["log_price_e"]).abs() < self.PRECIO_PARECIDO].groupby("fila")["y"]
        out["te_vv_tasa_precio_parecido"] = precio.mean()
        out["te_vv_n_precio_parecido"] = precio.size()
        tipo = p[p["tipo"] == p["tipo_e"]].groupby("fila")["y"]
        out["te_vv_tasa_mismo_tipo_y_stock"] = tipo.mean()
        out["te_vv_n_mismo_tipo_y_stock"] = tipo.size()

        out = out.reindex(np.arange(len(df)))
        out[["te_vv_n_precio_parecido", "te_vv_n_mismo_tipo_y_stock"]] = (
            out[["te_vv_n_precio_parecido", "te_vv_n_mismo_tipo_y_stock"]].fillna(0)
        )
        out.index = df.index
        return out


def crear_encoders():
    """Encoders nuevos, sin ajustar. Se crean de nuevo para cada fold."""
    return [
        EncoderDeGrupo("seller_id"),
        EncoderDeGrupo("category_id"),
        EncoderDeVecinos(k=30),
        # Dentro del vendedor: separa sus publicaciones por tipo, stock, garantía y título
        EncoderDeGrupo("seller_id", "listing_type_id"),
        EncoderDeGrupo("seller_id", "iq_es_1"),
        EncoderDeGrupo("seller_id", "tiene_warranty"),
        EncoderDeGrupo("seller_id", "title", normalizar=True),
        EncoderVecinosDelVendedor(),
    ]


def _ajustar_y_aplicar(train, y, destino):
    partes = [encoder.fit(train, y).transform(destino) for encoder in crear_encoders()]
    return pd.concat(partes, axis=1)


def subfolds(n_filas, semilla=config.SEED):
    """Partición al azar de las filas de entrenamiento. No mira las etiquetas."""
    return list(KFold(N_SUBFOLDS, shuffle=True, random_state=semilla).split(np.zeros(n_filas)))


def codificar(train, y, a_predecir, semilla=config.SEED):
    """Devuelve las features de target encoding de `train` y de `a_predecir`.

    `train` se codifica por sub-folds y `a_predecir` con las etiquetas de todo
    `train`. Las etiquetas de `a_predecir`, si existen, no se usan nunca.
    """
    partes = []
    for i_ajuste, i_resto in subfolds(len(train), semilla):
        partes.append(
            _ajustar_y_aplicar(train.iloc[i_ajuste], y.iloc[i_ajuste], train.iloc[i_resto])
        )
    te_train = pd.concat(partes).loc[train.index]
    te_prediccion = _ajustar_y_aplicar(train, y, a_predecir)
    return te_train, te_prediccion


def descripcion():
    """Configuración del target encoding, para dejarla anotada en el registro."""
    return {
        "activo": True,
        "n_subfolds": N_SUBFOLDS,
        "encoders": [{"tipo": type(e).__name__, **vars(e)} for e in crear_encoders()],
    }
