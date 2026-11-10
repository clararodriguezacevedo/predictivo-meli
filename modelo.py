"""Target encoding, registro de corridas, LightGBM y el baseline, en ese orden"""

from datetime import datetime
import hashlib
import json
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import datos


# ----------------------------------------------------------------------------
# Target encoding, features que usan el target
# ----------------------------------------------------------------------------

# Folds y no leave-one-out, porque con leave-one-out un árbol puede deducir la etiqueta de cada fila
N_SUBFOLDS = 5


class EncoderDeGrupo:
    """Tasa de usados de un grupo, suavizada hacia la tasa global, y cuántas filas la respaldan"""

    def __init__(self, *columnas, suavizado=2.0, normalizar=False):
        # Se suaviza porque la tasa de un grupo chico es poco confiable (D41)
        self.columnas = list(columnas)
        self.suavizado = suavizado
        self.normalizar = normalizar

    @property
    def nombre(self):
        return "te_" + "_x_".join(self.columnas)

    def _clave(self, df):
        partes = [df[c].astype("string").fillna("(nulo)") for c in self.columnas]
        if self.normalizar:
            partes = [datos.normalizar_texto(parte) for parte in partes]
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
    """Tasa de usados de los k títulos más parecidos (D32)"""

    def __init__(self, k=10, max_df=0.02, bloque=2000):
        self.k = k
        self.max_df = max_df  # Las palabras muy frecuentes no distinguen productos y hacen todo lento
        self.bloque = bloque  # Se buscan de a bloques para no llenar la memoria

    def fit(self, df, y):
        self.vectorizador_ = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=self.max_df, sublinear_tf=True,
            token_pattern=r"[a-z0-9]+",
        )
        # TF-IDF ya normaliza cada fila, así que el producto punto es la similitud coseno
        self.matriz_ = self.vectorizador_.fit_transform(datos.normalizar_texto(df["title"])).T.tocsr()
        self.y_ = y.to_numpy().astype(float)
        return self

    def _vecinos(self, consultas):
        """Similitud y etiqueta de los k vecinos de cada consulta"""
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
        consultas = self.vectorizador_.transform(datos.normalizar_texto(df["title"]))
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
    """Etiquetas de los items más parecidos del mismo vendedor, útiles con vendedores mixtos (D34)"""

    PRECIO_PARECIDO = 0.2  # Diferencia máxima de log del precio, unos 20%

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
        self.titulos_ = self.vectorizador_.fit_transform(datos.normalizar_texto(df["title"])).tocsr()
        self.etiquetados_ = self._tabla(df).assign(y=y.to_numpy().astype(float))
        return self

    def transform(self, df):
        titulos = self.vectorizador_.transform(datos.normalizar_texto(df["title"])).tocsr()
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
        peso = p["similitud"] ** 2 + 1e-3  # El cuadrado premia a los muy parecidos y el 1e-3 evita dividir por 0
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
    """Encoders sin ajustar. Se crean de nuevo para cada fold"""
    return [
        EncoderDeGrupo("seller_id"),  # D30
        EncoderDeGrupo("category_id"),  # D31
        EncoderDeVecinos(k=30),  # D32 y D40
        # Dentro de cada vendedor, separa sus publicaciones por tipo, stock, garantía y título (D33)
        EncoderDeGrupo("seller_id", "listing_type_id"),
        EncoderDeGrupo("seller_id", "iq_es_1"),
        EncoderDeGrupo("seller_id", "tiene_warranty"),
        EncoderDeGrupo("seller_id", "title", normalizar=True),
        EncoderVecinosDelVendedor(),  # D34
    ]


def subfolds(n_filas, seed=datos.SEED):
    """Partición al azar de las filas de entrenamiento. No mira las etiquetas"""
    return list(KFold(N_SUBFOLDS, shuffle=True, random_state=seed).split(np.zeros(n_filas)))


def codificar_destino(train, y, destino, fabrica=crear_encoders):
    """Features de `destino`, con encoders ajustados sobre las etiquetas de todo `train`"""
    return pd.concat([e.fit(train, y).transform(destino) for e in fabrica()], axis=1)


def codificar_train(train, y, seed=datos.SEED, fabrica=crear_encoders):
    """Features de `train`, donde cada fila se codifica con los sub-folds que no la contienen"""
    partes = [codificar_destino(train.iloc[i_ajuste], y.iloc[i_ajuste], train.iloc[i_resto], fabrica)
              for i_ajuste, i_resto in subfolds(len(train), seed)]
    return pd.concat(partes).loc[train.index]


def codificar(train, y, a_predecir, seed=datos.SEED, fabrica=crear_encoders):
    """Features de `train` (por sub-folds) y de `a_predecir` (con las etiquetas de todo `train`)"""
    return codificar_train(train, y, seed, fabrica), codificar_destino(train, y, a_predecir, fabrica)


def descripcion():
    """Configuración del target encoding, para dejarla anotada en el registro"""
    return {
        "n_subfolds": N_SUBFOLDS,
        "encoders": [{"tipo": type(e).__name__, **vars(e)} for e in crear_encoders()],
    }

# ----------------------------------------------------------------------------
# Registro de corridas, runs.jsonl solo agrega líneas y RUNS.md es su resumen
# ----------------------------------------------------------------------------

CARPETA_REGISTRO = datos.RAIZ / "registro"
ARCHIVO = CARPETA_REGISTRO / "runs.jsonl"
RESUMEN = CARPETA_REGISTRO / "RUNS.md"


def _git(*args):
    return subprocess.run(["git", *args], cwd=datos.RAIZ, capture_output=True, text=True).stdout.strip()


def describir_features(columnas):
    """Cantidad y huella de las columnas, para saber si dos corridas usaron las mismas"""
    columnas = sorted(columnas)
    return {"n": len(columnas), "huella": hashlib.sha1("|".join(columnas).encode()).hexdigest()[:10]}


def registrar(tipo, modelo, parametros, **detalles):
    """Agrega una corrida al registro y devuelve su id"""
    ahora = datetime.now()
    run = {
        "id": f"{ahora:%Y%m%d-%H%M%S}-{tipo}-{modelo}",
        "fecha": ahora.isoformat(timespec="seconds"),
        "tipo": tipo,
        "modelo": modelo,
        "parametros": parametros,
        **detalles,
        "commit": _git("rev-parse", "--short", "HEAD"),
        # Con cambios sin commitear, el commit solo no alcanza para reproducir
        "cambios_sin_commitear": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    with open(ARCHIVO, "a", encoding="utf8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    generar_resumen()
    return run["id"]


def leer():
    with open(ARCHIVO, encoding="utf8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def buscar(run_id):
    for run in leer():
        if run["id"] == run_id:
            return run
    raise KeyError(f"No existe la corrida {run_id}")


def _parametros_cortos(parametros):
    if parametros and all(isinstance(v, dict) for v in parametros.values()):
        return "; ".join(f"{m} ({_parametros_cortos(p)})" for m, p in parametros.items())
    return ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )


def _tabla(filas, columnas):
    return [pd.DataFrame(filas, columns=columnas).to_markdown(index=False), ""]


def generar_resumen():
    runs = leer()
    puntajes = {r["submission"]: r["leaderboard_publico"] for r in runs if r["tipo"] == "leaderboard"}
    lineas = ["# Registro de corridas", "", "Resumen generado desde `runs.jsonl`.", "",
              "## Validaciones (comparables entre sí)", ""]
    validaciones = sorted((r for r in runs if r["tipo"] == "validacion"), key=lambda r: -r["accuracy"])
    lineas += _tabla([{
        "id": r["id"], "modelo": r["modelo"], "accuracy": f"{r['accuracy']:.4f}",
        "features": r["features"]["n"], "nota": r.get("nota", ""),
        "parámetros": _parametros_cortos(r["parametros"]),
    } for r in validaciones], ["id", "modelo", "accuracy", "features", "nota", "parámetros"])

    lineas += ["## Controles con folds nuevos", ""]
    lineas += _tabla([{
        "id": r["id"], "semillas": r["protocolo"]["semillas"],
        "accuracy": f"{r['accuracy']:.4f}", "nota": r.get("nota", ""),
    } for r in runs if r["tipo"] == "control"], ["id", "semillas", "accuracy", "nota"])

    lineas += ["## Submissions", ""]
    lineas += _tabla([{
        "id": r["id"], "archivo": r["archivo"], "validación": r["run_validacion"],
        "accuracy de validación": f"{r['accuracy_validacion']:.4f}",
        "leaderboard público": puntajes.get(r["id"], "sin cargar"),
    } for r in runs if r["tipo"] == "submission"],
        ["id", "archivo", "validación", "accuracy de validación", "leaderboard público"])

    lineas += ["## Búsqueda de hiperparámetros (las 15 mejores de cada modelo)", ""]
    busquedas = [r for r in runs if r["tipo"] == "busqueda"]
    for modelo in sorted({r["modelo"] for r in busquedas}):
        del_modelo = sorted((r for r in busquedas if r["modelo"] == modelo), key=lambda r: -r["accuracy"])
        lineas += [f"### {modelo} ({len(del_modelo)} pruebas)", ""]
        lineas += _tabla([{
            "id": r["id"], "accuracy": f"{r['accuracy']:.4f}",
            "parámetros": _parametros_cortos(r["parametros"]),
        } for r in del_modelo[:15]], ["id", "accuracy", "parámetros"])
    RESUMEN.write_text("\n".join(lineas), encoding="utf8")

# ----------------------------------------------------------------------------
# El modelo
# ----------------------------------------------------------------------------

# Identificadores, texto libre, fechas crudas y target, que ya están resumidos en features
NO_SON_FEATURES = [
    datos.TARGET, "id", "seller_id", "category_id", "parent_item_id",
    "official_store_id", "catalog_product_id", "title", "warranty", "video_id",
    "date_created", "last_updated", "start_time", "stop_time",
]
CATEGORICAS = [
    "listing_type_id", "buying_mode", "ship_mode", "status", "sub_status", "tags",
    "ship_tags", "currency_id", "seller_state", "attr_estado", "attr_antiguedad",
]
# Un solo modelo, porque promediar cuatro no mejora de forma significativa (D50) y la última búsqueda tampoco (D51)
PARAMETROS = dict(
    n_estimators=1500, learning_rate=0.034309580195791355, num_leaves=149,
    min_child_samples=5, colsample_bytree=0.6874594323703044, subsample=0.8344766462089817,
    subsample_freq=1, reg_lambda=0.07827034783412899, reg_alpha=0.029235419526953025,
)
# Toda corrida "validacion" usa estos 5 folds y 2 seeds, así se comparan de a pares
PROTOCOLO_VALIDACION = {"n_folds": 5, "semillas": [0, 1]}


def y_binario(df):
    return (df[datos.TARGET] == datos.CLASE_POSITIVA).astype(int)


def crear_clasificador(seed):
    # Modo determinista, para que el modelo final salga igual en cualquier máquina y con cualquier cantidad de núcleos
    return lgb.LGBMClassifier(**PARAMETROS, random_state=seed, n_jobs=8, verbose=-1,
                              deterministic=True, force_row_wise=True)


def preparar(df, niveles=None):
    """Deja solo las features y tipa las categóricas"""
    X = df[[c for c in df.columns if c not in NO_SON_FEATURES]].copy()
    for col in CATEGORICAS:
        X[col] = pd.Categorical(X[col], categories=niveles[col] if niveles else None)
    X["attr_unico_duenio"] = X["attr_unico_duenio"].astype(float)
    return X


def preparar_juntos(train, test):
    """Train y test con los mismos niveles en las categóricas, para predecir el test"""
    niveles = {c: sorted(set(train[c].dropna()) | set(test[c].dropna())) for c in CATEGORICAS}
    return preparar(train, niveles), preparar(test, niveles)


def armar_matrices(X_train, train, y, X_pred, a_predecir, seed):
    """Agrega el target encoding. Validar y predecir el test pasan por esta misma función"""
    te_train, te_pred = codificar(train, y, a_predecir, seed)
    return pd.concat([X_train, te_train], axis=1), pd.concat([X_pred, te_pred], axis=1)


def armar_folds(train, y, protocolo):
    """Lista de (X_train, y_train, X_val, y_val, seed). Se arma una vez porque es lento"""
    X = preparar(train)
    folds = []
    for seed in protocolo["semillas"]:
        particion = StratifiedKFold(protocolo["n_folds"], shuffle=True, random_state=seed)
        for i_train, i_val in particion.split(X, y):
            # Las etiquetas de validación no entran a ningún cálculo
            X_tr, X_val = armar_matrices(
                X.iloc[i_train], train.iloc[i_train], y.iloc[i_train],
                X.iloc[i_val], train.iloc[i_val], seed,
            )
            folds.append((X_tr, y.iloc[i_train], X_val, y.iloc[i_val], seed))
    return folds


def validar_en_folds(folds):
    """Accuracy de cada fold. Corta en 0.5 porque moverlo no mejora (D52)"""
    aciertos = []
    for X_tr, y_tr, X_val, y_val, seed in folds:
        clasificador = crear_clasificador(seed).fit(X_tr, y_tr)
        aciertos.append(float((clasificador.predict(X_val) == y_val).mean()))
    return aciertos


def resumir(aciertos):
    return float(np.mean(aciertos)), float(np.std(aciertos) / np.sqrt(len(aciertos)))


def descripcion_comun(folds, protocolo):
    """Detalles que van al registro además del modelo y sus parámetros"""
    return {
        "protocolo": protocolo,
        "target_encoding": descripcion(),
        "features": describir_features(folds[0][0].columns),
    }


def validar(train, y, protocolo=PROTOCOLO_VALIDACION, anotar=True, nota=""):
    """Validación cruzada del modelo completo. Devuelve (accuracy, error estándar)"""
    inicio = time.time()
    folds = armar_folds(train, y, protocolo)
    aciertos = validar_en_folds(folds)
    accuracy, error = resumir(aciertos)
    if anotar:
        registrar(
            "validacion", "lightgbm", PARAMETROS, accuracy=accuracy, error_estandar=error,
            accuracy_por_fold=aciertos, duracion_seg=round(time.time() - inicio),
            nota=nota, **descripcion_comun(folds, protocolo),
        )
    return accuracy, error


def predecir_en_folds(folds):
    """Probabilidad de cada fila, predicha por un modelo que no la vio. Para analizar errores"""
    partes = []
    for X_tr, y_tr, X_val, y_val, seed in folds:
        clasificador = crear_clasificador(seed).fit(X_tr, y_tr)
        parte = X_val[[c for c in X_val.columns if c.startswith("te_")]].copy()
        parte["proba_usado"] = clasificador.predict_proba(X_val)[:, 1]
        parte["y"] = y_val
        parte["seed"] = seed
        partes.append(parte)
    return pd.concat(partes)


def entrenar_final(train, test, y, seed=datos.SEED):
    """El modelo final, entrenado con todo train. El test entra sin etiquetas, solo para los niveles"""
    X_train, _ = preparar_juntos(train, test)
    # Cada fila de train se codifica con sub-folds que no la contienen, igual que en la validación
    X_train = X_train.join(codificar_train(train, y, seed))
    return crear_clasificador(seed).fit(X_train, y)


def predecir_test(clasificador, train, test, y):
    """Predice el test y lo deja con el formato de Kaggle"""
    _, X_test = preparar_juntos(train, test)
    # El test se codifica con las etiquetas de todo train
    X_test = X_test.join(codificar_destino(train, y, test))
    es_usado = clasificador.predict(X_test[clasificador.feature_name_]) == 1
    # Kaggle numera las filas del test desde 1, no usa el id de MercadoLibre
    return pd.DataFrame({
        "ID": range(1, len(test) + 1),
        datos.TARGET: pd.Series(es_usado).map({True: datos.CLASE_POSITIVA, False: "new"}),
    })

# ----------------------------------------------------------------------------
# Baseline, la referencia contra la que se mide todo lo demás
# ----------------------------------------------------------------------------

# Seis variables crudas del item, sin perfiles ni target encoding. Se entrena en segundos
BASELINE_CATEGORICAS = ["listing_type_id", "buying_mode"]
BASELINE_NUMERICAS = ["tiene_warranty", "es_tienda_oficial", "initial_quantity", "sold_quantity"]


def crear_baseline():
    """Regresión logística, con las cantidades en escala logarítmica"""
    numericas = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    columnas = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), BASELINE_CATEGORICAS),
        ("numericas", numericas, BASELINE_NUMERICAS),
    ])
    return make_pipeline(columnas, LogisticRegression(max_iter=1000))


def validar_baseline(train, y, protocolo=PROTOCOLO_VALIDACION):
    """Valida el baseline con los mismos folds que el modelo final y la anota en el registro"""
    X = train[BASELINE_CATEGORICAS + BASELINE_NUMERICAS].copy()
    X[BASELINE_CATEGORICAS] = X[BASELINE_CATEGORICAS].astype(str)
    X[BASELINE_NUMERICAS] = X[BASELINE_NUMERICAS].astype(float)
    aciertos, mayoritaria = [], []
    for seed in protocolo["semillas"]:
        particion = StratifiedKFold(protocolo["n_folds"], shuffle=True, random_state=seed)
        for i_train, i_val in particion.split(X, y):
            clasificador = crear_baseline().fit(X.iloc[i_train], y.iloc[i_train])
            aciertos.append(float((clasificador.predict(X.iloc[i_val]) == y.iloc[i_val]).mean()))
            # El piso, predecir siempre la clase más frecuente del fold de entrenamiento
            clase = int(y.iloc[i_train].mean() > 0.5)
            mayoritaria.append(float((y.iloc[i_val] == clase).mean()))
    accuracy, error = resumir(aciertos)
    registrar(
        "validacion", "baseline", {"modelo": "regresion_logistica", "max_iter": 1000},
        accuracy=accuracy, error_estandar=error, accuracy_por_fold=aciertos,
        accuracy_clase_mayoritaria=float(np.mean(mayoritaria)), protocolo=protocolo,
        features=describir_features(BASELINE_CATEGORICAS + BASELINE_NUMERICAS),
        nota="baseline, regresión logística con seis variables crudas",
    )
    return accuracy, error, float(np.mean(mayoritaria))
