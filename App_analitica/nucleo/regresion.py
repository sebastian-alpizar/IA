"""
regresion.py
Métodos de regresión del Laboratorio 3, integrados al framework de la app:

    EDA -> SupervisadoReg(EDA) -> RegresionLab(SupervisadoReg)

- Familia lineal: regresión lineal simple y múltiple, Lasso, LassoCV, Ridge y RidgeCV
- Familia de núcleos: SVR (kernels rbf, lineal, polinomial y sigmoide)
- Familia de árboles: árbol de regresión, Random Forest y XGBoost
- Además (resto del curso): Gradient Boosting (potenciación del código guía) y KNN

- SupervisadoReg: partición aleatoria (la respuesta es continua, no se estratifica),
  preprocesamiento dentro de un Pipeline (imputación, escalado, one-hot y selección de
  atributos), entrenamiento, índices RMSE / MAE / ER / R², historial de experimentos,
  validación cruzada y benchmarking (validación cruzada repetida + Friedman y Wilcoxon).
- RegresionLab: los algoritmos. Cada uno sin parámetros es la configuración estándar
  (valores por defecto de la librería); los parámetros extra van directo al estimador.
Todo devuelve tablas o figuras de Plotly.
"""

import textwrap
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text

from . import graficos as g
from .eda import EDA

try:
    from xgboost import XGBRegressor
    HAY_XGB = True
except ImportError:
    HAY_XGB = False


# ----------------------------------------------------------------------
# Algoritmos (misma estructura que en clasificacion.py)
#   lab: True para los que pide el Laboratorio 3
#   familia: lineal / núcleos / árboles (para comparar dentro y entre familias)
#   metodo: nombre del método de RegresionLab
#   escalar: valor por defecto (penalizados, SVR y KNN dependen de la escala)
#   params: (nombre, etiqueta, tipo, default de la librería, extra)
#   variaciones: configuración estándar + variaciones (las del notebook del Lab 3)
# ----------------------------------------------------------------------
def _p(nombre, etiqueta, tipo, default, extra, ayuda=""):
    return {"nombre": nombre, "etiqueta": etiqueta, "tipo": tipo, "default": default, "extra": extra, "ayuda": ayuda}


LINEAL, NUCLEOS, ARBOLES, VECINOS = "Lineal", "Núcleos (SVR)", "Árboles", "Vecinos"

ALGORITMOS = {
    "Lineal simple": {
        "lab": True, "familia": LINEAL, "metodo": "lineal_simple", "escalar": False,
        "descripcion": "Mínimos cuadrados con una sola predictora: una recta ŷ = w0 + w1·x. "
                       "Por defecto usa la variable más correlacionada con la respuesta.",
        "params": [],
        "variaciones": [{}, {"variable": 2}],
    },
    "Lineal múltiple": {
        "lab": True, "familia": LINEAL, "metodo": "lineal_multiple", "escalar": False,
        "descripcion": "Mínimos cuadrados ordinarios con todas las predictoras.",
        "params": [],
        "variaciones": [{}, {"n_atributos": 50}, {"n_atributos": 20}],
    },
    "Lasso": {
        "lab": True, "familia": LINEAL, "metodo": "lasso", "escalar": True,
        "descripcion": "Mínimos cuadrados con penalización L1: encoge los coeficientes y anula los menos útiles "
                       "(selecciona variables).",
        "params": [_p("alpha", "Penalización (alpha)", "float", 1.0, (0.00001, 100.0, 0.0005),
                      "Más grande = más coeficientes en cero. Con alpha = 0 es mínimos cuadrados.")],
        "variaciones": [{}, {"alpha": 0.1}, {"alpha": 0.01}, {"alpha": 0.001}, {"alpha": 0.001, "escalar": False}],
    },
    "LassoCV": {
        "lab": True, "familia": LINEAL, "metodo": "lasso_cv", "escalar": True,
        "descripcion": "Lasso que escoge alpha solo, con validación cruzada sobre el entrenamiento "
                       "(prueba 100 valores de alpha).",
        "params": [_p("cv", "Particiones de la validación cruzada", "int", 5, (2, 20, 1))],
        "variaciones": [{}, {"cv": 10}],
    },
    "Ridge": {
        "lab": True, "familia": LINEAL, "metodo": "ridge", "escalar": True,
        "descripcion": "Mínimos cuadrados con penalización L2: encoge los coeficientes sin anularlos; "
                       "estabiliza el modelo cuando hay multicolinealidad.",
        "params": [_p("alpha", "Penalización (alpha)", "float", 1.0, (0.0, 100000.0, 1.0))],
        "variaciones": [{}, {"alpha": 10.0}, {"alpha": 100.0}, {"alpha": 1000.0}],
    },
    "RidgeCV": {
        "lab": True, "familia": LINEAL, "metodo": "ridge_cv", "escalar": True,
        "descripcion": "Ridge que escoge alpha solo con validación cruzada (prueba 100 valores entre 0,001 y "
                       "10 000). Con 0 particiones usa dejar-uno-fuera, que en Ridge es muy rápido.",
        "params": [_p("cv", "Particiones (0 = dejar uno fuera)", "int", 0, (0, 20, 1))],
        "variaciones": [{}, {"cv": 10}],
    },
    "SVR": {
        "lab": True, "familia": NUCLEOS, "metodo": "svr", "escalar": True, "lento": True,
        "descripcion": "Regresión con máquinas de soporte vectorial: ignora los errores menores que epsilon y "
                       "penaliza los demás con C. Lenta con muchas filas (más de ~20 000).",
        "params": [_p("kernel", "Núcleo", "select", "rbf", ["rbf", "linear", "poly", "sigmoid"]),
                   _p("C", "Penalización C", "float", 1.0, (0.001, 1000.0, 0.01)),
                   _p("epsilon", "Ancho del tubo (epsilon)", "float", 0.1, (0.0, 10.0, 0.01)),
                   _p("gamma", "gamma (scale, auto o un número)", "text", "scale", None,
                      "Flexibilidad del kernel rbf/poly: más grande = más local."),
                   _p("degree", "Grado (kernel poly)", "int", 3, (1, 6, 1))],
        "variaciones": [{}, {"C": 100.0}, {"epsilon": 0.01},
                        {"C": 1.0, "epsilon": 0.05, "gamma": "0.002"},
                        {"C": 3.0, "epsilon": 0.05, "gamma": "0.001"},
                        {"kernel": "linear"}, {"kernel": "linear", "C": 0.01, "epsilon": 0.01},
                        {"kernel": "poly"}, {"kernel": "poly", "degree": 2, "C": 0.1, "epsilon": 0.01}],
    },
    "DT": {
        "lab": True, "familia": ARBOLES, "metodo": "arbol", "escalar": False,
        "descripcion": "Árbol de regresión: divide los datos con reglas sí/no y predice el promedio de cada hoja.",
        "params": [_p("criterion", "Criterio", "select", "squared_error",
                      ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
                   _p("max_depth", "Profundidad máxima (0 = sin límite)", "int", None, (0, 50, 1)),
                   _p("min_samples_split", "Mínimo para dividir", "int", 2, (2, 500, 1)),
                   _p("min_samples_leaf", "Mínimo por hoja", "int", 1, (1, 500, 1)),
                   _p("ccp_alpha", "Poda costo-complejidad (ccp_alpha)", "float", 0.0, (0.0, 0.1, 0.0005))],
        "variaciones": [{}, {"max_depth": 3}, {"max_depth": 4}, {"max_depth": 5, "min_samples_leaf": 20},
                        {"min_samples_leaf": 30}, {"ccp_alpha": 0.0005}],
    },
    "RF": {
        "lab": True, "familia": ARBOLES, "metodo": "bosque", "escalar": False,
        "descripcion": "Random Forest: promedio de muchos árboles entrenados con muestras y variables al azar.",
        "params": [_p("n_estimators", "Cantidad de árboles", "int", 100, (10, 2000, 10)),
                   _p("max_features", "Variables por división", "select", 1.0, [1.0, 0.33, "sqrt", "log2"],
                      "1.0 = todas (bagging puro, el estándar en regresión); 0.33 = p/3, lo que recomienda Breiman."),
                   _p("max_depth", "Profundidad máxima (0 = sin límite)", "int", None, (0, 50, 1)),
                   _p("min_samples_leaf", "Mínimo por hoja", "int", 1, (1, 500, 1))],
        "variaciones": [{}, {"max_depth": 2}, {"n_estimators": 500}, {"n_estimators": 500, "max_features": 0.33},
                        {"n_estimators": 500, "max_features": 0.33, "min_samples_leaf": 5}],
    },
    "XGBoost": {
        "lab": True, "familia": ARBOLES, "metodo": "xgboost", "escalar": False,
        "descripcion": "Gradient boosting optimizado: árboles en secuencia, cada uno ajustado a los residuos "
                       "de los anteriores.",
        "params": [_p("n_estimators", "Cantidad de árboles", "int", 100, (10, 3000, 10)),
                   _p("learning_rate", "Tasa de aprendizaje", "float", 0.3, (0.001, 1.0, 0.01)),
                   _p("max_depth", "Profundidad de cada árbol", "int", 6, (1, 20, 1)),
                   _p("subsample", "Proporción de filas (subsample)", "float", 1.0, (0.1, 1.0, 0.05)),
                   _p("colsample_bytree", "Proporción de columnas por árbol", "float", 1.0, (0.1, 1.0, 0.05)),
                   _p("min_child_weight", "Peso mínimo por hoja", "float", 1.0, (0.0, 100.0, 1.0))],
        "variaciones": [{}, {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
                        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 3, "subsample": 0.8,
                         "colsample_bytree": 0.5},
                        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 2, "subsample": 0.8,
                         "colsample_bytree": 0.5, "min_child_weight": 5.0}],
    },
    "Gradient Boosting": {
        "lab": False, "familia": ARBOLES, "metodo": "gradient_boosting", "escalar": False,
        "descripcion": "Potenciación de scikit-learn (la del código guía del curso).",
        "params": [_p("n_estimators", "Iteraciones", "int", 100, (10, 3000, 10)),
                   _p("learning_rate", "Tasa de aprendizaje", "float", 0.1, (0.001, 1.0, 0.01)),
                   _p("max_depth", "Profundidad de cada árbol", "int", 3, (1, 15, 1)),
                   _p("min_samples_split", "Mínimo para dividir", "int", 2, (2, 500, 1)),
                   _p("subsample", "Proporción de filas (subsample)", "float", 1.0, (0.1, 1.0, 0.05))],
        "variaciones": [{}, {"n_estimators": 500, "max_depth": 4, "min_samples_split": 5},
                        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 3, "subsample": 0.8}],
    },
    "KNN": {
        "lab": False, "familia": VECINOS, "metodo": "knn", "escalar": True,
        "descripcion": "Predice el promedio de la variable en los k vecinos más cercanos.",
        "params": [_p("n_neighbors", "Vecinos (k)", "int", 5, (1, 200, 1)),
                   _p("weights", "Pesos", "select", "uniform", ["uniform", "distance"]),
                   _p("p", "Distancia (p): 1 = Manhattan, 2 = Euclídea", "int", 2, (1, 5, 1))],
        "variaciones": [{}, {"n_neighbors": 15}, {"n_neighbors": 15, "weights": "distance"}],
    },
}
if not HAY_XGB:
    del ALGORITMOS["XGBoost"]
ALGORITMOS_LAB = [a for a, i in ALGORITMOS.items() if i["lab"]]
FAMILIAS = {a: i["familia"] for a, i in ALGORITMOS.items()}

CRITERIOS = ["RMSE", "MAE", "ER", "R²"]
MENOR_ES_MEJOR = ("RMSE", "MAE", "ER")
# métricas de la validación cruzada: nombre -> (scorer de sklearn, signo para la escala natural)
METRICAS_CV = {"RMSE": ("neg_root_mean_squared_error", -1), "MAE": ("neg_mean_absolute_error", -1),
               "R²": ("r2", 1)}


def _corta(nombre, ancho=42):
    """Parte los nombres largos de los modelos en varias líneas para los gráficos."""
    return "<br>".join(textwrap.wrap(str(nombre), ancho)) or str(nombre)


def _gamma(valor):
    """gamma de SVR puede venir como texto: 'scale', 'auto' o un número."""
    if isinstance(valor, str) and valor.strip().lower() not in ("scale", "auto"):
        return float(valor.replace(",", "."))
    return valor


# ======================================================================
class SupervisadoReg(EDA):
    """
    Parte común de los métodos de regresión. Hereda de EDA.
    - Separa predictoras / variable a predecir y hace la partición entrenamiento-prueba
      (aleatoria simple: la variable a predecir es continua)
    - Preprocesamiento dentro de un Pipeline (sin fuga de información del conjunto de prueba)
    - Entrena, evalúa con RMSE, MAE, ER y R², y guarda el historial de experimentos
    """

    def __init__(self, datos: pd.DataFrame, variable_predecir, predictoras=None, test_size=0.25,
                 eliminar_duplicados=True, semilla=42):
        datos = datos.dropna(subset=[variable_predecir])
        if not pd.api.types.is_numeric_dtype(datos[variable_predecir]):
            raise ValueError("La variable a predecir tiene que ser numérica.")
        super().__init__(datos)
        self.__variable_predecir = variable_predecir
        self.__predictoras = list(predictoras or [c for c in datos.columns if c != variable_predecir])
        self.__semilla = int(semilla)
        self.__resultados = []      # una fila de índices por experimento
        self.__experimentos = {}    # nombre -> pipeline, predicciones, índices y configuración
        self.__ultimo = None
        self.benchmark_tabla = None
        self.particionar(test_size, eliminar_duplicados)

    # --- Propiedades ---
    @property
    def variable_predecir(self):
        return self.__variable_predecir

    @property
    def predictoras(self):
        return self.__predictoras

    @property
    def semilla(self):
        return self.__semilla

    @property
    def X_train(self):
        return self.__X_train

    @property
    def X_test(self):
        return self.__X_test

    @property
    def y_train(self):
        return self.__y_train

    @property
    def y_test(self):
        return self.__y_test

    @property
    def experimentos(self):
        return self.__experimentos

    @property
    def ultimo(self):
        return self.__ultimo

    # =================================================================
    # Preparación de los datos
    # =================================================================
    def particionar(self, test_size=0.25, eliminar_duplicados=True):
        datos = self.datos[self.__predictoras + [self.__variable_predecir]]
        if eliminar_duplicados:
            datos = datos.drop_duplicates()
        y = datos[self.__variable_predecir].astype(float)
        X = datos[self.__predictoras]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.__semilla)
        self.__resultados, self.__experimentos, self.__ultimo = [], {}, None
        self.benchmark_tabla = None

    def resumen_particion(self):
        filas = []
        for nombre, y in [("Entrenamiento", self.__y_train), ("Prueba", self.__y_test)]:
            filas.append({"Conjunto": nombre, "Filas": len(y), "Predictoras": self.__X_train.shape[1],
                          "Media": y.mean(), "Desv. estándar": y.std(), "Mediana": y.median(),
                          "Mínimo": y.min(), "Máximo": y.max()})
        return pd.DataFrame(filas).set_index("Conjunto")

    def resumen_variable(self):
        """Mínimo, cuartiles, máximo, media, desviación y asimetría de la variable a predecir."""
        y = self.datos[self.__variable_predecir]
        return pd.Series({"Mínimo": y.min(), "Q1": y.quantile(0.25), "Mediana": y.median(),
                          "Q3": y.quantile(0.75), "Máximo": y.max(), "Media": y.mean(),
                          "Desv. estándar": y.std(), "Asimetría": y.skew()}, name=self.__variable_predecir)

    def correlacion_respuesta(self, top=None):
        """Correlación de Pearson de cada predictora numérica con la respuesta (en el entrenamiento)."""
        num = self.__X_train.select_dtypes(include="number")
        if num.empty:
            return pd.Series(dtype=float, name="Correlación")
        corr = num.corrwith(self.__y_train).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).rename("Correlación")
        return corr.head(top) if top else corr

    def preprocesador(self, escalar=False, imputacion="median", columnas=None):
        # numéricas: imputación (+ escalado); categóricas: moda + one-hot
        X = self.__X_train if columnas is None else self.__X_train[columnas]
        numericas = X.select_dtypes(include="number").columns.tolist()
        categoricas = X.select_dtypes(exclude="number").columns.tolist()
        pasos_num = [("imputar", SimpleImputer(strategy=imputacion))]
        if escalar:
            pasos_num.append(("escalar", StandardScaler()))
        transformadores = []
        if numericas:
            transformadores.append(("num", Pipeline(pasos_num), numericas))
        if categoricas:
            transformadores.append(("cat", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categoricas))
        return ColumnTransformer(transformadores, verbose_feature_names_out=False)

    def crear_pipeline(self, estimador, escalar=False, imputacion="median", n_atributos=None, columnas=None):
        pasos = [("prep", self.preprocesador(escalar, imputacion, columnas))]
        if n_atributos:
            pasos.append(("seleccion", SelectKBest(f_regression, k=n_atributos)))
        pasos.append(("modelo", estimador))
        return Pipeline(pasos)

    # =================================================================
    # Entrenamiento y evaluación
    # =================================================================
    def entrenar(self, nombre, estimador, algoritmo=None, escalar=False, imputacion="median",
                 n_atributos=None, config=None, columnas=None):
        """
        Entrena el estimador (dentro del pipeline) con el conjunto de entrenamiento,
        lo evalúa en el de prueba y lo guarda en el historial. Retorna los índices de calidad.
        """
        if n_atributos:
            total = len(self.preprocesador(columnas=columnas).fit(self.__X_train).get_feature_names_out())
            n_atributos = min(int(n_atributos), total)
        inicio = time.time()
        pipe = self.crear_pipeline(estimador, escalar, imputacion, n_atributos, columnas)
        pipe.fit(self.__X_train, self.__y_train)
        y_pred = pipe.predict(self.__X_test)
        indices = self.indices_calidad(self.__y_test, y_pred)
        indices["Tiempo (s)"] = round(time.time() - inicio, 2)
        algoritmo = algoritmo or type(estimador).__name__
        fila = {"Modelo": nombre, "Algoritmo": algoritmo, "Familia": FAMILIAS.get(algoritmo, "Otro"), **indices}
        # si se repite el nombre se reemplaza el experimento anterior
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre] + [fila]
        self.__experimentos[nombre] = {"pipeline": pipe, "y_pred": y_pred, "algoritmo": algoritmo,
                                       "indices": indices,
                                       "config": dict(config or {}, escalar=escalar, imputacion=imputacion,
                                                      n_atributos=n_atributos)}
        self.__ultimo = nombre
        return pd.Series(indices, name=nombre)

    @staticmethod
    def indices_calidad(y_real, y_pred):
        """
        RMSE = raíz del error cuadrático medio      MAE = error absoluto medio
        ER   = suma |error| / suma |real| (error relativo, como en el código guía)
        R²   = 1 − SCres / SCtot (proporción de la varianza explicada)
        """
        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        suma = np.abs(y_real).sum()
        indices = {"RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
                   "MAE": mean_absolute_error(y_real, y_pred),
                   "ER": np.abs(y_real - y_pred).sum() / suma if suma else np.nan,
                   "R²": r2_score(y_real, y_pred)}
        return {k: round(float(v), 4) for k, v in indices.items()}

    def validacion_cruzada(self, nombre=None, k=5):
        """Validación cruzada de k particiones (sobre entrenamiento) de un experimento ya corrido."""
        pipe = clone(self.__experimentos[nombre or self.__ultimo]["pipeline"])
        cv = KFold(n_splits=k, shuffle=True, random_state=self.__semilla)
        res = cross_validate(pipe, self.__X_train, self.__y_train, cv=cv, n_jobs=-1,
                             scoring={m: s for m, (s, _) in METRICAS_CV.items()})
        tabla = pd.DataFrame({m: res[f"test_{m}"] * signo for m, (_, signo) in METRICAS_CV.items()})
        tabla.index = [f"Fold {i + 1}" for i in range(k)]
        tabla.loc["Promedio"] = tabla.iloc[:k].mean()
        tabla.loc["Desv. estándar"] = tabla.iloc[:k].std()
        return tabla.round(4)

    def predecir(self, nombre, nuevos: pd.DataFrame):
        faltan = [c for c in self.__predictoras if c not in nuevos.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en los datos nuevos: {faltan}")
        return self.__experimentos[nombre]["pipeline"].predict(nuevos[self.__predictoras])

    def residuos(self, nombre=None):
        exp = self.__experimentos[nombre or self.__ultimo]
        return pd.DataFrame({"Real": self.__y_test.values, "Predicción": exp["y_pred"],
                             "Residuo": self.__y_test.values - exp["y_pred"]}, index=self.__y_test.index)

    # =================================================================
    # Resultados
    # =================================================================
    def resultados(self, algoritmo=None, familia=None):
        tabla = pd.DataFrame(self.__resultados)
        if tabla.empty:
            return tabla
        if algoritmo is not None:
            tabla = tabla[tabla["Algoritmo"] == algoritmo]
        if familia is not None:
            tabla = tabla[tabla["Familia"] == familia]
        return tabla.reset_index(drop=True)

    def mejor_modelo(self, criterio="RMSE", algoritmo=None, familia=None):
        """Nombre del mejor experimento según el criterio (en RMSE, MAE y ER el mejor es el menor)."""
        tabla = self.resultados(algoritmo, familia)
        if tabla.empty or tabla[criterio].isna().all():
            return None
        fila = tabla[criterio].idxmin() if criterio in MENOR_ES_MEJOR else tabla[criterio].idxmax()
        return tabla.loc[fila, "Modelo"]

    def mejores(self, criterio="RMSE", por="Algoritmo"):
        """Mejor modelo de cada algoritmo (por='Algoritmo') o de cada familia (por='Familia')."""
        tabla = self.resultados()
        if tabla.empty:
            return tabla
        tabla = tabla.dropna(subset=[criterio])
        tabla = tabla.sort_values(criterio, ascending=criterio in MENOR_ES_MEJOR)
        return tabla.groupby(por, sort=False).head(1).reset_index(drop=True)

    def borrar_experimento(self, nombre):
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre]
        self.__experimentos.pop(nombre, None)
        if self.__ultimo == nombre:
            self.__ultimo = self.__resultados[-1]["Modelo"] if self.__resultados else None

    # =================================================================
    # Benchmarking
    # =================================================================
    def benchmark(self, nombres, k=10, repeticiones=3, progreso=None):
        """
        Validación cruzada repetida (k particiones x repeticiones) sobre el entrenamiento.
        Todos los modelos usan exactamente las mismas particiones, así sus resultados por
        partición se pueden comparar en pares (Friedman y Wilcoxon).
        progreso: función opcional (i, total, nombre) para mostrar el avance.
        """
        cv = RepeatedKFold(n_splits=k, n_repeats=repeticiones, random_state=self.__semilla)
        filas = []
        for i, nombre in enumerate(nombres):
            if progreso:
                progreso(i, len(nombres), nombre)
            exp = self.__experimentos[nombre]
            res = cross_validate(clone(exp["pipeline"]), self.__X_train, self.__y_train, cv=cv, n_jobs=-1,
                                 scoring={m: s for m, (s, _) in METRICAS_CV.items()})
            for j in range(k * repeticiones):
                filas.append({"Modelo": nombre, "Algoritmo": exp["algoritmo"],
                              "Familia": FAMILIAS.get(exp["algoritmo"], "Otro"), "Partición": j + 1,
                              **{m: res[f"test_{m}"][j] * signo for m, (_, signo) in METRICAS_CV.items()}})
        self.benchmark_tabla = pd.DataFrame(filas)
        return self.benchmark_tabla

    def resumen_benchmark(self, criterio="RMSE"):
        tabla = self.benchmark_tabla
        res = tabla.groupby(["Modelo", "Familia"], sort=False)[list(METRICAS_CV)].agg(["mean", "std"])
        res.columns = [f"{m} {'promedio' if e == 'mean' else 'desv.'}" for m, e in res.columns]
        res = res.reset_index()
        # rango promedio en las particiones (1 = el mejor), el que usa la prueba de Friedman
        ancho = tabla.pivot(index="Partición", columns="Modelo", values=criterio)
        rangos = ancho.rank(axis=1, ascending=criterio in MENOR_ES_MEJOR).mean()
        res["Rango promedio"] = res["Modelo"].map(rangos)
        return res.sort_values(f"{criterio} promedio", ascending=criterio in MENOR_ES_MEJOR).reset_index(drop=True)

    def prueba_estadistica(self, criterio="RMSE", alfa=0.05):
        """
        Prueba de Friedman (¿hay diferencias entre los modelos?) y prueba de rangos con signo
        de Wilcoxon entre el mejor modelo y cada uno de los demás, con la corrección de Holm.
        """
        ancho = self.benchmark_tabla.pivot(index="Partición", columns="Modelo", values=criterio)
        asc = criterio in MENOR_ES_MEJOR
        mejor = ancho.mean().sort_values(ascending=asc).index[0]
        friedman = None
        if ancho.shape[1] >= 3:
            estad, p = stats.friedmanchisquare(*[ancho[c] for c in ancho.columns])
            friedman = {"Chi² de Friedman": float(estad), "p": float(p), "Modelos": ancho.shape[1],
                        "Particiones": ancho.shape[0]}
        filas = []
        for otro in ancho.columns.drop(mejor):
            dif = ancho[otro] - ancho[mejor]
            p = stats.wilcoxon(ancho[mejor], ancho[otro])[1] if (dif != 0).any() else 1.0
            filas.append({"Mejor": mejor, "Contra": otro, f"Diferencia promedio de {criterio}": dif.mean(), "p": p})
        comp = pd.DataFrame(filas)
        if not comp.empty:
            comp = comp.sort_values("p").reset_index(drop=True)
            m = len(comp)
            comp["p Holm"] = np.minimum(1, np.maximum.accumulate(comp["p"] * (m - np.arange(m))))
            comp["Significativa"] = np.where(comp["p Holm"] < alfa, "sí", "no")
        return friedman, comp

    # =================================================================
    # Gráficos
    # =================================================================
    def fig_distribucion_variable(self):
        y = self.datos[self.__variable_predecir]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
        fig.add_trace(go.Histogram(x=y, nbinsx=40, marker_color=g.AZUL, opacity=0.85, name="Frecuencia"), 1, 1)
        fig.add_trace(go.Box(x=y, marker_color=g.AZUL, name="", boxpoints="outliers"), 2, 1)
        fig.add_vline(x=y.mean(), line_dash="dash", line_color=g.ROJO, annotation_text=f"media {y.mean():.3f}",
                      row=1, col=1)
        fig.update_layout(height=380, showlegend=False, title=f"Distribución de {self.__variable_predecir}")
        fig.update_yaxes(visible=False, row=2, col=1)
        return fig

    def fig_correlacion_respuesta(self, top=20):
        corr = self.correlacion_respuesta(top).iloc[::-1]
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                               marker_color=[g.ROJO if v < 0 else g.AZUL for v in corr.values],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(height=max(320, 24 * len(corr) + 100), xaxis=dict(range=[-1, 1]),
                          title=f"Correlación con {self.__variable_predecir} (las {len(corr)} más fuertes)")
        return fig

    def fig_comparacion(self, metricas=("RMSE", "MAE", "ER", "R²"), algoritmo=None, nombres=None,
                        titulo="Comparación de modelos"):
        tabla = self.resultados(algoritmo).set_index("Modelo")
        if nombres is not None:
            tabla = tabla.loc[nombres]
        metricas = list(metricas)
        etiquetas = [_corta(n) for n in tabla.index]
        fig = make_subplots(rows=1, cols=len(metricas), shared_yaxes=True, horizontal_spacing=0.035,
                            subplot_titles=[m + (" ↓" if m in MENOR_ES_MEJOR else " ↑") for m in metricas])
        for j, m in enumerate(metricas, start=1):
            valores = tabla[m]
            mejor = valores.min() if m in MENOR_ES_MEJOR else valores.max()
            fig.add_trace(go.Bar(x=valores, y=etiquetas, orientation="h", showlegend=False, cliponaxis=False,
                                 marker_color=[g.CATEGORICOS[1] if v == mejor else g.AZUL for v in valores],
                                 text=[f"{v:.4f}" for v in valores], textposition="outside", textfont_size=11,
                                 hovertemplate="%{y}<br>" + m + ": %{x:.4f}<extra></extra>"), 1, j)
            fig.update_xaxes(range=[min(0, valores.min() * 1.1), valores.max() * 1.55], row=1, col=j)
        fig.update_yaxes(autorange="reversed", automargin=True, tickfont_size=11)
        lineas = max(e.count("<br>") + 1 for e in etiquetas)
        fig.update_layout(height=max(320, (24 + 13 * lineas) * len(tabla) + 130), title=titulo,
                          margin=dict(t=80, r=40), bargap=0.3)
        return fig

    def fig_real_vs_predicho(self, nombres=None, columnas=3):
        nombres = nombres or [self.__ultimo]
        columnas = min(columnas, len(nombres))
        filas = int(np.ceil(len(nombres) / columnas))
        fig = make_subplots(rows=filas, cols=columnas, subplot_titles=[
            f"{_corta(n, 55)}<br><sup>RMSE {self.__experimentos[n]['indices']['RMSE']:.4f} · "
            f"R² {self.__experimentos[n]['indices']['R²']:.4f}</sup>" for n in nombres],
            horizontal_spacing=0.06, vertical_spacing=0.16)
        lo = float(min(self.__y_test.min(), min(self.__experimentos[n]["y_pred"].min() for n in nombres)))
        hi = float(max(self.__y_test.max(), max(self.__experimentos[n]["y_pred"].max() for n in nombres)))
        for i, n in enumerate(nombres):
            fila, col = i // columnas + 1, i % columnas + 1
            fig.add_trace(go.Scatter(x=self.__y_test, y=self.__experimentos[n]["y_pred"], mode="markers",
                                     marker=dict(size=5, color=g.AZUL, opacity=0.5), showlegend=False,
                                     hovertemplate="real %{x:.3f}<br>predicción %{y:.3f}<extra></extra>"), fila, col)
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=g.ROJO, dash="dash", width=1)), fila, col)
            fig.update_xaxes(title_text="Valor real", row=fila, col=col)
            fig.update_yaxes(title_text="Predicción" if col == 1 else None, row=fila, col=col)
        fig.update_annotations(font_size=12)
        fig.update_layout(height=380 * filas + 40, title="Valores reales contra predicciones (conjunto de prueba)",
                          margin=dict(t=110))
        return fig

    def fig_residuos(self, nombre=None):
        nombre = nombre or self.__ultimo
        r = self.residuos(nombre)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.08,
                            subplot_titles=["Residuos contra predicción", "Distribución de los residuos"])
        fig.add_trace(go.Scatter(x=r["Predicción"], y=r["Residuo"], mode="markers", showlegend=False,
                                 marker=dict(size=5, color=g.AZUL, opacity=0.5),
                                 hovertemplate="predicción %{x:.3f}<br>residuo %{y:.3f}<extra></extra>"), 1, 1)
        fig.add_hline(y=0, line_dash="dash", line_color=g.ROJO, row=1, col=1)
        fig.add_trace(go.Histogram(x=r["Residuo"], nbinsx=40, marker_color=g.AZUL, showlegend=False), 1, 2)
        fig.update_xaxes(title_text="Predicción", row=1, col=1)
        fig.update_yaxes(title_text="Residuo (real − predicción)", row=1, col=1)
        fig.update_xaxes(title_text="Residuo", row=1, col=2)
        fig.update_layout(height=420, title=f"Residuos: {nombre}", margin=dict(t=90))
        return fig


# ======================================================================
class RegresionLab(SupervisadoReg):
    """
    Métodos de regresión del laboratorio. Hereda de SupervisadoReg (y este de EDA).
    Cada método sin parámetros es la configuración estándar; los parámetros extra
    (**parametros) van directo al estimador. Opciones de preprocesamiento:
    escalar, imputacion y n_atributos (selección de atributos con el F de la regresión).
    """

    @staticmethod
    def nombre_config(algoritmo, parametros, escalar=None, n_atributos=None, imputacion="median"):
        # ej. "Lasso (alpha=0.01)" o "Ridge (estándar)"
        partes = [f"{k}={v}" for k, v in parametros.items()]
        if escalar is False:
            partes.append("sin escalar")
        if escalar is True and not ALGORITMOS.get(algoritmo, {}).get("escalar"):
            partes.append("escalado")
        if imputacion != "median":
            partes.append(f"imputación {imputacion}")
        if n_atributos:
            partes.append(f"{n_atributos} atributos")
        return f"{algoritmo} ({', '.join(partes) if partes else 'estándar'})"

    # =================================================================
    # Algoritmos: familia lineal
    # =================================================================
    def lineal_simple(self, nombre=None, variable=None, imputacion="median", escalar=False, n_atributos=None):
        """
        Una sola predictora. variable: nombre de la columna o su posición (1, 2, ...) en el ranking
        de correlación con la respuesta; por defecto la más correlacionada.
        """
        ranking = self.correlacion_respuesta()
        if ranking.empty:
            raise ValueError("La regresión simple necesita al menos una predictora numérica.")
        if variable is None or isinstance(variable, (int, np.integer)):
            variable = ranking.index[min(int(variable or 1), len(ranking)) - 1]
        nombre = nombre or f"Lineal simple ({variable})"
        return self.entrenar(nombre, LinearRegression(), "Lineal simple", escalar, imputacion, None,
                             {"variable": variable}, columnas=[variable])

    def lineal_multiple(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Lineal múltiple", parametros, escalar or None, n_atributos, imputacion)
        return self.entrenar(nombre, LinearRegression(**parametros), "Lineal múltiple", escalar, imputacion,
                             n_atributos, parametros)

    def lasso(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # la penalización depende de la escala de los coeficientes, por eso escala por defecto
        nombre = nombre or self.nombre_config("Lasso", parametros, escalar, n_atributos, imputacion)
        modelo = Lasso(max_iter=50000, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "Lasso", escalar, imputacion, n_atributos, parametros)

    def lasso_cv(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, cv=5, **parametros):
        etiqueta = dict(parametros, cv=cv) if cv != 5 else parametros
        nombre = nombre or self.nombre_config("LassoCV", etiqueta, escalar, n_atributos, imputacion)
        modelo = LassoCV(cv=KFold(int(cv), shuffle=True, random_state=self.semilla), max_iter=50000,
                         n_jobs=-1, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "LassoCV", escalar, imputacion, n_atributos, etiqueta)

    def ridge(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Ridge", parametros, escalar, n_atributos, imputacion)
        return self.entrenar(nombre, Ridge(**parametros), "Ridge", escalar, imputacion, n_atributos, parametros)

    def ridge_cv(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, cv=0, **parametros):
        # cv = 0: dejar uno fuera (eficiente en Ridge); cv = k: k particiones
        etiqueta = dict(parametros, cv=cv) if cv else parametros
        nombre = nombre or self.nombre_config("RidgeCV", etiqueta, escalar, n_atributos, imputacion)
        particion = KFold(int(cv), shuffle=True, random_state=self.semilla) if cv else None
        modelo = RidgeCV(alphas=np.logspace(-3, 4, 100), cv=particion, **parametros)
        return self.entrenar(nombre, modelo, "RidgeCV", escalar, imputacion, n_atributos, etiqueta)

    # =================================================================
    # Algoritmos: núcleos y vecinos
    # =================================================================
    def svr(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # SVR usa productos punto / distancias, por eso escala por defecto
        nombre = nombre or self.nombre_config("SVR", parametros, escalar, n_atributos, imputacion)
        estimador = dict(parametros)
        if "gamma" in estimador:
            estimador["gamma"] = _gamma(estimador["gamma"])
        return self.entrenar(nombre, SVR(**estimador), "SVR", escalar, imputacion, n_atributos, parametros)

    def knn(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("KNN", parametros, escalar, n_atributos, imputacion)
        modelo = KNeighborsRegressor(n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "KNN", escalar, imputacion, n_atributos, parametros)

    # =================================================================
    # Algoritmos: árboles
    # =================================================================
    def arbol(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("DT", parametros, escalar or None, n_atributos, imputacion)
        modelo = DecisionTreeRegressor(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "DT", escalar, imputacion, n_atributos, parametros)

    def bosque(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("RF", parametros, escalar or None, n_atributos, imputacion)
        modelo = RandomForestRegressor(random_state=self.semilla, n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "RF", escalar, imputacion, n_atributos, parametros)

    def xgboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        if not HAY_XGB:
            raise ImportError("Falta la librería xgboost (pip install xgboost).")
        nombre = nombre or self.nombre_config("XGBoost", parametros, escalar or None, n_atributos, imputacion)
        modelo = XGBRegressor(random_state=self.semilla, n_jobs=-1, verbosity=0, **parametros)
        return self.entrenar(nombre, modelo, "XGBoost", escalar, imputacion, n_atributos, parametros)

    def gradient_boosting(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Gradient Boosting", parametros, escalar or None, n_atributos,
                                              imputacion)
        modelo = GradientBoostingRegressor(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "Gradient Boosting", escalar, imputacion, n_atributos, parametros)

    def ejecutar(self, algoritmo, configuracion=None):
        """Corre una configuración ({parámetros + opciones de preprocesamiento}) de un algoritmo."""
        metodo = getattr(self, ALGORITMOS[algoritmo]["metodo"])
        return metodo(**dict(configuracion or {}))

    def comparar(self, algoritmo, configuraciones):
        """Corre varias configuraciones de un mismo algoritmo y retorna sus índices."""
        return pd.DataFrame([self.ejecutar(algoritmo, conf) for conf in configuraciones])

    # =================================================================
    # Análisis del modelo
    # =================================================================
    def nombres_atributos(self, nombre=None):
        # nombres de las variables después del one-hot y la selección de atributos
        pipe = self.experimentos[nombre or self.ultimo]["pipeline"]
        nombres = pipe.named_steps["prep"].get_feature_names_out()
        if "seleccion" in pipe.named_steps:
            nombres = pipe.named_steps["seleccion"].get_feature_names_out(nombres)
        return list(nombres)

    def es_lineal(self, nombre=None):
        # modelos con coeficientes: lineal, Lasso, Ridge (y sus CV) y SVR con kernel lineal
        modelo = self.experimentos[nombre or self.ultimo]["pipeline"].named_steps["modelo"]
        if isinstance(modelo, SVR):
            return modelo.kernel == "linear"
        return hasattr(modelo, "coef_")

    def coeficientes(self, nombre=None):
        """Coeficientes de un modelo lineal, ordenados por magnitud (con las variables del pipeline)."""
        exp = self.experimentos[nombre or self.ultimo]
        modelo = exp["pipeline"].named_steps["modelo"]
        coef = pd.Series(np.ravel(modelo.coef_), index=self.nombres_atributos(nombre), name="Coeficiente")
        return coef.reindex(coef.abs().sort_values(ascending=False).index)

    def intercepto(self, nombre=None):
        modelo = self.experimentos[nombre or self.ultimo]["pipeline"].named_steps["modelo"]
        return float(np.ravel(modelo.intercept_)[0])

    def parametro_optimo(self, nombre=None):
        """alpha que escogió LassoCV o RidgeCV y cuántos coeficientes quedaron distintos de cero."""
        modelo = self.experimentos[nombre or self.ultimo]["pipeline"].named_steps["modelo"]
        if not hasattr(modelo, "alpha_"):
            return None
        return {"alpha": float(modelo.alpha_), "no_cero": int(np.sum(np.abs(modelo.coef_) > 1e-10)),
                "total": int(len(modelo.coef_))}

    def importancia_variables(self, nombre=None):
        """
        Importancia de variables: la propia del modelo si la tiene (árboles y potenciación);
        si no (SVR, KNN, lineales), por permutación: cuánto sube el RMSE al desordenar cada variable.
        """
        exp = self.experimentos[nombre or self.ultimo]
        if "importancia" in exp:
            return exp["importancia"]
        pipe = exp["pipeline"]
        modelo = pipe.named_steps["modelo"]
        if hasattr(modelo, "feature_importances_"):
            imp = pd.Series(modelo.feature_importances_, index=self.nombres_atributos(nombre))
        else:
            muestra = min(len(self.X_test), 1500)
            r = permutation_importance(pipe, self.X_test.iloc[:muestra], self.y_test.iloc[:muestra], n_repeats=5,
                                       scoring="neg_root_mean_squared_error", random_state=self.semilla, n_jobs=-1)
            imp = pd.Series(r.importances_mean, index=self.X_test.columns)
        exp["importancia"] = imp.rename("Importancia").sort_values(ascending=False).round(4)
        return exp["importancia"]

    def importancia_propia(self, nombre=None):
        modelo = self.experimentos[nombre or self.ultimo]["pipeline"].named_steps["modelo"]
        return hasattr(modelo, "feature_importances_")

    def fig_importancia(self, nombre=None, top=15):
        imp = self.importancia_variables(nombre).head(top).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=g.AZUL,
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(height=max(320, 26 * len(imp) + 100),
                          title=f"Importancia de variables: {nombre or self.ultimo}")
        return fig

    def fig_coeficientes(self, nombre=None, top=20):
        coef = self.coeficientes(nombre).head(top).iloc[::-1]
        fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation="h",
                               marker_color=[g.ROJO if v < 0 else g.AZUL for v in coef.values],
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.add_vline(x=0, line_color=g.TEXTO_SUAVE, line_width=1)
        fig.update_layout(height=max(320, 24 * len(coef) + 100),
                          title=f"Los {len(coef)} coeficientes de mayor magnitud: {nombre or self.ultimo}")
        return fig

    def reglas_arbol(self, nombre=None, profundidad=3):
        pipe = self.experimentos[nombre or self.ultimo]["pipeline"]
        return export_text(pipe.named_steps["modelo"], feature_names=self.nombres_atributos(nombre),
                           max_depth=profundidad, decimals=3)

    def fig_arbol(self, nombre=None, profundidad=3):
        """Primeros niveles del árbol de regresión (regla, muestras y promedio de cada nodo)."""
        pipe = self.experimentos[nombre or self.ultimo]["pipeline"]
        arbol = pipe.named_steps["modelo"].tree_
        atributos = self.nombres_atributos(nombre)
        nodos, aristas = [], []

        def recorrer(n, prof, x0, x1, padre=None):
            x = (x0 + x1) / 2
            nodos.append((n, x, -prof))
            if padre is not None:
                aristas.append((padre, (x, -prof)))
            izq, der = arbol.children_left[n], arbol.children_right[n]
            if izq != -1 and prof < profundidad:
                recorrer(izq, prof + 1, x0, x, (x, -prof))
                recorrer(der, prof + 1, x, x1, (x, -prof))

        recorrer(0, 0, 0.0, 1.0)
        fig = go.Figure()
        for (px, py), (hx, hy) in aristas:
            fig.add_trace(go.Scatter(x=[px, hx], y=[py, hy], mode="lines", line=dict(color=g.BORDE, width=1.5),
                                     hoverinfo="skip", showlegend=False))
        valores = [float(arbol.value[n][0][0]) for n, _, _ in nodos]
        textos = []
        for (n, _, _), v in zip(nodos, valores):
            hoja = arbol.children_left[n] == -1
            regla = "hoja" if hoja else f"{atributos[arbol.feature[n]]} ≤ {arbol.threshold[n]:.3f}"
            textos.append(f"<b>{regla}</b><br>n = {arbol.n_node_samples[n]}<br>promedio: {v:.3f}")
        fig.add_trace(go.Scatter(x=[x for _, x, _ in nodos], y=[y for _, _, y in nodos], mode="markers+text",
                                 text=textos, textposition="bottom center", textfont=dict(size=10),
                                 hovertext=textos, hoverinfo="text", showlegend=False,
                                 marker=dict(size=18, color=valores, colorscale=g.SECUENCIAL, showscale=True,
                                             colorbar=dict(title="promedio", thickness=12),
                                             line=dict(color="#FFFFFF", width=1.5))))
        fig.update_layout(height=170 * (profundidad + 1) + 80, title=f"Primeros {profundidad} niveles del árbol",
                          xaxis=dict(visible=False, range=[-0.02, 1.02]),
                          yaxis=dict(visible=False, range=[-profundidad - 0.6, 0.3]), plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # =================================================================
    # Curvas de hiperparámetros y regularización (no se guardan en el historial)
    # =================================================================
    CURVAS = {"DT": ("max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]),
              "SVR": ("C", [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]),
              "RF": ("n_estimators", [10, 25, 50, 100, 200, 400]),
              "XGBoost": ("n_estimators", [25, 50, 100, 200, 400, 800]),
              "KNN": ("n_neighbors", [1, 3, 5, 10, 15, 25, 40, 60]),
              "Lasso": ("alpha", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]),
              "Ridge": ("alpha", [0.01, 0.1, 1, 10, 100, 1000, 10000])}

    def _estimador(self, algoritmo, **parametros):
        fabricas = {"DT": lambda: DecisionTreeRegressor(random_state=self.semilla, **parametros),
                    "RF": lambda: RandomForestRegressor(random_state=self.semilla, n_jobs=-1, **parametros),
                    "XGBoost": lambda: XGBRegressor(random_state=self.semilla, n_jobs=-1, verbosity=0, **parametros),
                    "SVR": lambda: SVR(**{k: _gamma(v) if k == "gamma" else v for k, v in parametros.items()}),
                    "KNN": lambda: KNeighborsRegressor(n_jobs=-1, **parametros),
                    "Lasso": lambda: Lasso(max_iter=50000, **parametros),
                    "Ridge": lambda: Ridge(**parametros)}
        return fabricas[algoritmo]()

    def curva_parametro(self, algoritmo, parametro, valores, k=5, escalar=None, **fijos):
        """RMSE de entrenamiento y de validación cruzada según un hiperparámetro (muestra el sobreajuste)."""
        escalar = ALGORITMOS[algoritmo]["escalar"] if escalar is None else escalar
        filas = []
        for v in valores:
            pipe = self.crear_pipeline(self._estimador(algoritmo, **{**fijos, parametro: v}), escalar)
            res = cross_validate(pipe, self.X_train, self.y_train, n_jobs=-1, return_train_score=True,
                                 cv=KFold(k, shuffle=True, random_state=self.semilla),
                                 scoring="neg_root_mean_squared_error")
            filas.append({parametro: v, "RMSE entrenamiento": -res["train_score"].mean(),
                          "RMSE validación": -res["test_score"].mean(), "Desv. validación": res["test_score"].std()})
        return pd.DataFrame(filas)

    @staticmethod
    def fig_curva_parametro(tabla, titulo=""):
        parametro = tabla.columns[0]
        mejor = tabla.loc[tabla["RMSE validación"].idxmin(), parametro]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tabla[parametro], y=tabla["RMSE entrenamiento"], mode="lines+markers",
                                 name="Entrenamiento", line=dict(color=g.AZUL_CLARO, width=2)))
        fig.add_trace(go.Scatter(x=tabla[parametro], y=tabla["RMSE validación"], mode="lines+markers",
                                 name="Validación cruzada", line=dict(color=g.AZUL, width=2.5),
                                 error_y=dict(type="data", array=tabla["Desv. validación"], thickness=1)))
        fig.add_vline(x=mejor, line_dash="dash", line_color=g.CATEGORICOS[1],
                      annotation_text=f"mínimo en {parametro} = {mejor:g}")
        tipo = "log" if parametro in ("alpha", "C") else "linear"
        fig.update_layout(height=420, title=titulo or f"RMSE según {parametro}", xaxis_title=parametro,
                          yaxis_title="RMSE", xaxis_type=tipo, legend=dict(orientation="h", y=-0.2))
        return fig

    def ruta_regularizacion(self, tipo="Lasso", k=10, puntos=40):
        """Coeficientes (variables estandarizadas) y RMSE de validación cruzada según alpha."""
        alphas = np.logspace(-5, 0, puntos) if tipo == "Lasso" else np.logspace(-3, 5, puntos)
        nombres = self.preprocesador(escalar=True).fit(self.X_train).get_feature_names_out()
        coefs, errores, desv = [], [], []
        for a in alphas:
            pipe = self.crear_pipeline(self._estimador(tipo, alpha=a), escalar=True)
            res = cross_validate(pipe, self.X_train, self.y_train, n_jobs=-1,
                                 cv=KFold(k, shuffle=True, random_state=self.semilla),
                                 scoring="neg_root_mean_squared_error")
            errores.append(-res["test_score"].mean())
            desv.append(res["test_score"].std())
            coefs.append(pipe.fit(self.X_train, self.y_train).named_steps["modelo"].coef_)
        coefs = pd.DataFrame(coefs, index=alphas, columns=nombres)
        resumen = pd.DataFrame({"alpha": alphas, "RMSE validación": errores, "Desv.": desv,
                                "Coeficientes ≠ 0": (coefs.abs() > 1e-10).sum(axis=1).values})
        return coefs, resumen

    @staticmethod
    def fig_ruta_regularizacion(coefs, resumen, tipo="Lasso", destacadas=8):
        mejor = resumen.loc[resumen["RMSE validación"].idxmin(), "alpha"]
        top = coefs.loc[mejor].abs().sort_values(ascending=False).head(destacadas).index
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                            subplot_titles=["Ruta de los coeficientes", "RMSE de validación cruzada según alpha"])
        for col in coefs.columns:
            if col not in top:
                fig.add_trace(go.Scatter(x=coefs.index, y=coefs[col], mode="lines", showlegend=False,
                                         line=dict(color="#D9DDD9", width=1), hoverinfo="skip"), 1, 1)
        for i, col in enumerate(top):
            fig.add_trace(go.Scatter(x=coefs.index, y=coefs[col], mode="lines", name=col,
                                     line=dict(color=g.color_cluster(i), width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=resumen["alpha"], y=resumen["RMSE validación"], mode="lines+markers",
                                 showlegend=False, line=dict(color=g.AZUL, width=2),
                                 error_y=dict(type="data", array=resumen["Desv."], thickness=1)), 1, 2)
        for c in (1, 2):
            fig.add_vline(x=mejor, line_dash="dash", line_color=g.TEXTO_SUAVE, row=1, col=c)
            fig.update_xaxes(type="log", title_text="alpha", row=1, col=c)
        fig.update_yaxes(title_text="Coeficiente", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_layout(height=480, title=f"{tipo}: mejor alpha = {mejor:.4g}", margin=dict(t=90),
                          legend=dict(orientation="h", y=-0.22, font=dict(size=11)))
        return fig

    def fig_benchmark(self, criterio="RMSE"):
        tabla = self.benchmark_tabla
        asc = criterio in MENOR_ES_MEJOR
        orden = tabla.groupby("Modelo")[criterio].mean().sort_values(ascending=asc).index
        familias = list(dict.fromkeys(tabla["Familia"]))
        fig = go.Figure()
        vistas = set()
        for nombre in orden:
            sub = tabla[tabla["Modelo"] == nombre]
            fam = sub["Familia"].iloc[0]
            fig.add_trace(go.Box(x=sub[criterio], y=[_corta(nombre)] * len(sub), name=fam, legendgroup=fam,
                                 showlegend=fam not in vistas, orientation="h", boxpoints="all", jitter=0.4,
                                 pointpos=0, marker=dict(size=4, opacity=0.6),
                                 line=dict(color=g.color_cluster(familias.index(fam)))))
            vistas.add(fam)
        fig.update_layout(height=max(360, 52 * len(orden) + 120), yaxis=dict(autorange="reversed", automargin=True,
                                                                              tickfont_size=11),
                          title=f"{criterio} en cada partición de la validación cruzada repetida",
                          xaxis_title=criterio, legend=dict(orientation="h", y=-0.15))
        return fig
