"""
regresion.py
Métodos de regresión (Laboratorio 3 + resto del curso). Hereda de Supervisado (y este de EDA):

    EDA -> Supervisado(EDA) -> Regresion(Supervisado)

- Familia lineal: regresión lineal simple y múltiple, Lasso, LassoCV, Ridge y RidgeCV
- Familia de núcleos: SVR (kernels rbf, lineal, polinomial y sigmoide)
- Familia de árboles: árbol de regresión, Random Forest y XGBoost
- Además: Gradient Boosting (potenciación del código guía) y KNN
- Índices RMSE, MAE, ER y R²; partición aleatoria simple (la respuesta es continua);
  curvas de hiperparámetros, ruta de regularización y benchmarking (validación cruzada
  repetida + pruebas de Friedman y Wilcoxon con corrección de Holm).
Cada método sin parámetros es la configuración estándar (valores por defecto de la librería).
Todo devuelve tablas o figuras de Plotly.
"""

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text

from shared import graficos as g
from ..supervisado import Supervisado          # clase padre, carpeta supervisado/
from .algoritmos import ALGORITMOS, CRITERIOS, FAMILIAS, HAY_XGB, MENOR_ES_MEJOR, METRICAS_CV

if HAY_XGB:
    from xgboost import XGBRegressor


def _corta(nombre, ancho=42):
    """Parte los nombres largos de los modelos en varias líneas para los gráficos."""
    return "<br>".join(textwrap.wrap(str(nombre), ancho)) or str(nombre)


def _gamma(valor):
    """gamma de SVR puede venir como texto: 'scale', 'auto' o un número."""
    if isinstance(valor, str) and valor.strip().lower() not in ("scale", "auto"):
        return float(valor.replace(",", "."))
    return valor



class Regresion(Supervisado):
    """
    Métodos de regresión. Hereda de Supervisado (y este de EDA).
    Opciones de preprocesamiento de cada algoritmo: escalar, imputacion y n_atributos
    (selección de atributos con el F de la regresión).
    """

    ALGORITMOS = ALGORITMOS
    CRITERIOS = CRITERIOS
    MENOR_ES_MEJOR = MENOR_ES_MEJOR
    SCORING_IMPORTANCIA = "neg_root_mean_squared_error"   # importancia = cuánto sube el RMSE

    def __init__(self, datos: pd.DataFrame, variable_predecir, predictoras=None, test_size=0.25,
                 eliminar_duplicados=True, semilla=42):
        if not pd.api.types.is_numeric_dtype(datos[variable_predecir]):
            raise ValueError("La variable a predecir tiene que ser numérica.")
        self.benchmark_tabla = None
        super().__init__(datos, variable_predecir, predictoras, test_size, eliminar_duplicados, semilla)

    # =================================================================
    # Ganchos de Supervisado para regresión
    # =================================================================
    def _preparar_y(self, y):
        return y.astype(float)

    def _puntaje_seleccion(self):
        return f_regression

    def _evaluar(self, pipe):
        y_pred = pipe.predict(self.X_test)
        return self.indices_calidad(self.y_test, y_pred), {"y_pred": y_pred}

    def _particiones_cv(self, k):
        return KFold(n_splits=k, shuffle=True, random_state=self.semilla)

    def _metricas_cv(self):
        return METRICAS_CV

    def _columnas_extra(self, algoritmo):
        return {"Familia": FAMILIAS.get(algoritmo, "Otro")}

    def particionar(self, test_size=0.25, eliminar_duplicados=True):
        super().particionar(test_size, eliminar_duplicados)
        self.benchmark_tabla = None

    # =================================================================
    # Variable a predecir y partición
    # =================================================================

    def resumen_particion(self):
        filas = []
        for nombre, y in [("Entrenamiento", self.y_train), ("Prueba", self.y_test)]:
            filas.append({"Conjunto": nombre, "Filas": len(y), "Predictoras": self.X_train.shape[1],
                          "Media": y.mean(), "Desv. estándar": y.std(), "Mediana": y.median(),
                          "Mínimo": y.min(), "Máximo": y.max()})
        return pd.DataFrame(filas).set_index("Conjunto")

    def resumen_variable(self):
        """Mínimo, cuartiles, máximo, media, desviación y asimetría de la variable a predecir."""
        y = self.datos[self.variable_predecir]
        return pd.Series({"Mínimo": y.min(), "Q1": y.quantile(0.25), "Mediana": y.median(),
                          "Q3": y.quantile(0.75), "Máximo": y.max(), "Media": y.mean(),
                          "Desv. estándar": y.std(), "Asimetría": y.skew()}, name=self.variable_predecir)

    def correlacion_respuesta(self, top=None):
        """Correlación de Pearson de cada predictora numérica con la respuesta (en el entrenamiento)."""
        num = self.X_train.select_dtypes(include="number")
        if num.empty:
            return pd.Series(dtype=float, name="Correlación")
        corr = num.corrwith(self.y_train).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).rename("Correlación")
        return corr.head(top) if top else corr

    # =================================================================
    # Índices de calidad y residuos
    # =================================================================

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

    def residuos(self, nombre=None):
        exp = self.experimentos[nombre or self.ultimo]
        return pd.DataFrame({"Real": self.y_test.values, "Predicción": exp["y_pred"],
                             "Residuo": self.y_test.values - exp["y_pred"]}, index=self.y_test.index)

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

    # =================================================================
    # Análisis del modelo
    # =================================================================

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

    # =================================================================
    # Benchmarking: validación cruzada repetida + Friedman y Wilcoxon
    # =================================================================

    def benchmark(self, nombres, k=10, repeticiones=3, progreso=None):
        """
        Validación cruzada repetida (k particiones x repeticiones) sobre el entrenamiento.
        Todos los modelos usan exactamente las mismas particiones, así sus resultados por
        partición se pueden comparar en pares (Friedman y Wilcoxon).
        progreso: función opcional (i, total, nombre) para mostrar el avance.
        """
        cv = RepeatedKFold(n_splits=k, n_repeats=repeticiones, random_state=self.semilla)
        filas = []
        for i, nombre in enumerate(nombres):
            if progreso:
                progreso(i, len(nombres), nombre)
            exp = self.experimentos[nombre]
            res = cross_validate(clone(exp["pipeline"]), self.X_train, self.y_train, cv=cv, n_jobs=-1,
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

    # =================================================================
    # Gráficos
    # =================================================================

    def fig_distribucion_variable(self):
        y = self.datos[self.variable_predecir]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
        fig.add_trace(go.Histogram(x=y, nbinsx=40, marker_color=g.AZUL, opacity=0.85, name="Frecuencia"), 1, 1)
        fig.add_trace(go.Box(x=y, marker_color=g.AZUL, name="", boxpoints="outliers"), 2, 1)
        fig.add_vline(x=y.mean(), line_dash="dash", line_color=g.ROJO, annotation_text=f"media {y.mean():.3f}",
                      row=1, col=1)
        fig.update_layout(height=380, showlegend=False, title=f"Distribución de {self.variable_predecir}")
        fig.update_yaxes(visible=False, row=2, col=1)
        return fig

    def fig_correlacion_respuesta(self, top=20):
        corr = self.correlacion_respuesta(top).iloc[::-1]
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                               marker_color=[g.ROJO if v < 0 else g.AZUL for v in corr.values],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(height=max(320, 24 * len(corr) + 100), xaxis=dict(range=[-1, 1]),
                          title=f"Correlación con {self.variable_predecir} (las {len(corr)} más fuertes)")
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
        nombres = nombres or [self.ultimo]
        columnas = min(columnas, len(nombres))
        filas = int(np.ceil(len(nombres) / columnas))
        fig = make_subplots(rows=filas, cols=columnas, subplot_titles=[
            f"{_corta(n, 55)}<br><sup>RMSE {self.experimentos[n]['indices']['RMSE']:.4f} · "
            f"R² {self.experimentos[n]['indices']['R²']:.4f}</sup>" for n in nombres],
            horizontal_spacing=0.06, vertical_spacing=0.16)
        lo = float(min(self.y_test.min(), min(self.experimentos[n]["y_pred"].min() for n in nombres)))
        hi = float(max(self.y_test.max(), max(self.experimentos[n]["y_pred"].max() for n in nombres)))
        for i, n in enumerate(nombres):
            fila, col = i // columnas + 1, i % columnas + 1
            fig.add_trace(go.Scatter(x=self.y_test, y=self.experimentos[n]["y_pred"], mode="markers",
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
        nombre = nombre or self.ultimo
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
