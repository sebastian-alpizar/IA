import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, cross_validate
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from supervisado import Supervisado   # clase padre, carpeta supervisado/


class Regresion(Supervisado):
    """
    Métodos de regresión del framework. Hereda de Supervisado (y este de EDA).
    - Familia lineal: regresión lineal simple y múltiple (LinearRegression),
      Lasso, LassoCV, Ridge y RidgeCV
    - Familia de núcleos: SVR (Support Vector Regression) con kernel rbf, lineal o polinomial
    - Familia de árboles: DecisionTreeRegressor, RandomForestRegressor y XGBRegressor
    - Igual que en Clasificacion, cada método sin parámetros es la configuración estándar
      (valores por defecto de la librería) y los parámetros extra van directo al estimador.
    - Índices de calidad: RMSE, MAE, ER (error relativo) y R². El criterio por defecto es el RMSE.
    - Benchmarking: validación cruzada repetida con las mismas particiones para todos los
      modelos y pruebas de Friedman y Wilcoxon para ver si las diferencias son significativas.
    """

    MENOR_ES_MEJOR = ("RMSE", "MAE", "ER")

    # familia (categoría) de cada algoritmo
    FAMILIAS = {"Lineal simple": "Lineal", "Lineal múltiple": "Lineal", "Lasso": "Lineal",
                "LassoCV": "Lineal", "Ridge": "Lineal", "RidgeCV": "Lineal",
                "SVR": "Núcleos (SVR)",
                "DT": "Árboles", "RF": "Árboles", "XGBoost": "Árboles"}

    def __init__(self, dataframe: pd.DataFrame, variable_predecir, test_size=0.25,
                 eliminar_duplicados=True, semilla=42, excluir=None):
        super().__init__(dataframe, variable_predecir, test_size, eliminar_duplicados, semilla, excluir)

    # =================================================================
    # Ganchos de Supervisado para regresión
    # =================================================================
    def _estratificar(self):
        return False   # la variable a predecir es continua

    def _puntaje_seleccion(self):
        return f_regression

    def _resumen_particion(self, X, y):
        filas = []
        for nombre, yy in [("Entrenamiento", self.y_train), ("Prueba", self.y_test)]:
            filas.append({"Conjunto": nombre, "Filas": len(yy), "Predictoras": X.shape[1],
                          "Media y": round(yy.mean(), 4), "Desv. y": round(yy.std(), 4),
                          "Mediana y": round(yy.median(), 4)})
        return pd.DataFrame(filas)

    def _evaluar(self, pipe):
        y_pred = pipe.predict(self.X_test)
        return self.indices_calidad(self.y_test, y_pred), {"y_pred": y_pred}

    def _particiones_cv(self, k):
        return KFold(n_splits=k, shuffle=True, random_state=self.semilla)

    def _metricas_cv(self):
        return {"RMSE": ("neg_root_mean_squared_error", -1),
                "MAE": ("neg_mean_absolute_error", -1),
                "R²": ("r2", 1)}

    @staticmethod
    def indices_calidad(y_real, y_pred, *args):
        y_real = np.asarray(y_real, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        indices = {"RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
                   "MAE": mean_absolute_error(y_real, y_pred),
                   # error relativo: suma de errores absolutos / suma de valores reales
                   "ER": np.abs(y_real - y_pred).sum() / np.abs(y_real).sum(),
                   "R²": r2_score(y_real, y_pred)}
        return {k: round(float(v), 4) for k, v in indices.items()}

    # =================================================================
    # Variable a predecir
    # =================================================================
    def resumen_variable(self):
        # mínimo, cuartiles, máximo, media y desviación de la variable a predecir
        y = self.mostrar()[self.variable_predecir]
        return pd.Series({"Mínimo": y.min(), "Q1": y.quantile(0.25), "Mediana": y.median(),
                          "Q3": y.quantile(0.75), "Máximo": y.max(), "Media": y.mean(),
                          "Desv. estándar": y.std(), "Asimetría": y.skew()},
                         name=self.variable_predecir).round(4)

    def correlacion_respuesta(self, top=None):
        # correlación de Pearson de cada predictora con la variable a predecir (entrenamiento)
        num = self.X_train.select_dtypes(include="number")
        corr = num.corrwith(self.y_train).dropna()
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).round(4)
        return corr.head(top) if top else corr

    # =================================================================
    # Algoritmos: familia lineal
    # =================================================================
    def lineal_simple(self, variable=None, nombre=None, imputacion="median"):
        # una sola predictora; por defecto la más correlacionada con y en el entrenamiento
        variable = variable or self.correlacion_respuesta().index[0]
        nombre = nombre or f"Lineal simple ({variable})"
        return self.entrenar(nombre, LinearRegression(), "Lineal simple", False, imputacion,
                             columnas=[variable])

    def lineal_multiple(self, nombre=None, escalar=False, imputacion="median", n_atributos=None,
                        **parametros):
        nombre = nombre or self.nombre_config("Lineal múltiple", parametros, None, n_atributos, imputacion)
        return self.entrenar(nombre, LinearRegression(**parametros), "Lineal múltiple", escalar,
                             imputacion, n_atributos)

    def lasso(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # la penalización depende de la escala de los coeficientes, por eso escala por defecto
        nombre = nombre or self.nombre_config("Lasso", parametros, escalar, n_atributos, imputacion)
        modelo = Lasso(max_iter=50000, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "Lasso", escalar, imputacion, n_atributos)

    def lasso_cv(self, nombre=None, escalar=True, imputacion="median", n_atributos=None,
                 alphas=None, cv=10, **parametros):
        # escoge alpha por validación cruzada (cv particiones) y reajusta con todo el entrenamiento
        etiqueta = dict(parametros, cv=cv)
        nombre = nombre or self.nombre_config("LassoCV", etiqueta, escalar, n_atributos, imputacion)
        alphas = np.logspace(-5, 0, 100) if alphas is None else alphas
        modelo = LassoCV(alphas=alphas, cv=KFold(cv, shuffle=True, random_state=self.semilla),
                         max_iter=50000, n_jobs=-1, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "LassoCV", escalar, imputacion, n_atributos)

    def ridge(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Ridge", parametros, escalar, n_atributos, imputacion)
        return self.entrenar(nombre, Ridge(**parametros), "Ridge", escalar, imputacion, n_atributos)

    def ridge_cv(self, nombre=None, escalar=True, imputacion="median", n_atributos=None,
                 alphas=None, cv=None, **parametros):
        # cv=None: validación cruzada dejando uno fuera (eficiente en Ridge); cv=k: k particiones
        etiqueta = dict(parametros, cv=cv if cv else "LOO")
        nombre = nombre or self.nombre_config("RidgeCV", etiqueta, escalar, n_atributos, imputacion)
        alphas = np.logspace(-3, 4, 100) if alphas is None else alphas
        particion = KFold(cv, shuffle=True, random_state=self.semilla) if cv else None
        modelo = RidgeCV(alphas=alphas, cv=particion, **parametros)
        return self.entrenar(nombre, modelo, "RidgeCV", escalar, imputacion, n_atributos)

    # =================================================================
    # Algoritmos: núcleos (SVR)
    # =================================================================
    def svr(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # SVR usa productos punto / distancias, por eso escala por defecto
        nombre = nombre or self.nombre_config("SVR", parametros, escalar, n_atributos, imputacion)
        return self.entrenar(nombre, SVR(**parametros), "SVR", escalar, imputacion, n_atributos)

    # =================================================================
    # Algoritmos: árboles
    # =================================================================
    def arbol(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("DT", parametros, None, n_atributos, imputacion)
        modelo = DecisionTreeRegressor(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "DT", escalar, imputacion, n_atributos)

    def bosque(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("RF", parametros, None, n_atributos, imputacion)
        modelo = RandomForestRegressor(random_state=self.semilla, n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "RF", escalar, imputacion, n_atributos)

    def xgboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("XGBoost", parametros, None, n_atributos, imputacion)
        modelo = XGBRegressor(random_state=self.semilla, n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "XGBoost", escalar, imputacion, n_atributos)

    # =================================================================
    # Resultados por familia
    # =================================================================
    def resultados(self, algoritmo=None, familia=None):
        tabla = super().resultados(algoritmo)
        if tabla.empty:
            return tabla
        tabla.insert(2, "Familia", tabla["Algoritmo"].map(self.FAMILIAS))
        if familia is not None:
            tabla = tabla[tabla["Familia"] == familia].reset_index(drop=True)
        return tabla

    def mejor_modelo(self, criterio="RMSE", algoritmo=None, familia=None):
        tabla = self.resultados(algoritmo, familia)
        fila = tabla[criterio].idxmin() if criterio in self.MENOR_ES_MEJOR else tabla[criterio].idxmax()
        return tabla.loc[fila]

    def mejores(self, criterio="RMSE", por="Algoritmo"):
        # mejor modelo de cada algoritmo (por="Algoritmo") o de cada familia (por="Familia")
        tabla = self.resultados()
        asc = criterio in self.MENOR_ES_MEJOR
        tabla = tabla.sort_values(criterio, ascending=asc).groupby(por, sort=False).head(1)
        return tabla.reset_index(drop=True)

    def parametro_optimo(self, nombre=None):
        # alpha escogido por LassoCV / RidgeCV y cantidad de coeficientes distintos de cero
        exp = self.experimentos[nombre] if nombre else self.ultimo
        modelo = exp["pipeline"].named_steps["modelo"]
        if not hasattr(modelo, "alpha_"):
            raise ValueError("El modelo no escoge alpha por validación cruzada (LassoCV o RidgeCV).")
        return pd.Series({"alpha": modelo.alpha_,
                          "Coeficientes ≠ 0": int(np.sum(np.abs(modelo.coef_) > 1e-10)),
                          "Coeficientes totales": len(modelo.coef_)})

    # =================================================================
    # Análisis del modelo
    # =================================================================
    def coeficientes(self, nombre=None, top=None):
        # coeficientes de un modelo lineal (Lineal, Lasso, Ridge, SVR lineal), ordenados por |coef|
        exp = self.experimentos[nombre] if nombre else self.ultimo
        modelo = exp["pipeline"].named_steps["modelo"]
        if not hasattr(modelo, "coef_"):
            raise ValueError("El modelo no es lineal: no tiene coeficientes.")
        coef = pd.Series(np.ravel(modelo.coef_), index=self.nombres_atributos(nombre), name="coeficiente")
        coef = coef.reindex(coef.abs().sort_values(ascending=False).index).round(4)
        return coef.head(top) if top else coef

    def residuos(self, nombre=None):
        exp = self.experimentos[nombre] if nombre else self.ultimo
        return pd.DataFrame({"Real": self.y_test.values, "Predicción": exp["y_pred"],
                             "Residuo": self.y_test.values - exp["y_pred"]}, index=self.y_test.index)

    def _fabrica(self, algoritmo):
        # constructor del estimador y si se escala por defecto, según el nombre del método
        fabricas = {"arbol": (lambda **p: DecisionTreeRegressor(random_state=self.semilla, **p), False),
                    "bosque": (lambda **p: RandomForestRegressor(random_state=self.semilla, n_jobs=-1, **p), False),
                    "xgboost": (lambda **p: XGBRegressor(random_state=self.semilla, n_jobs=-1, **p), False),
                    "svr": (lambda **p: SVR(**p), True),
                    "lasso": (lambda **p: Lasso(max_iter=50000, **p), True),
                    "ridge": (lambda **p: Ridge(**p), True)}
        return fabricas[algoritmo]

    def buscar_hiperparametros(self, algoritmo, rejilla, k=5, escalar=None, top=10):
        """
        Búsqueda en rejilla (GridSearchCV) con validación cruzada de k particiones sobre el
        entrenamiento. No toca el conjunto de prueba ni se guarda en el historial: sirve para
        escoger qué configuraciones vale la pena correr después con el método del algoritmo.
        rejilla: dict parámetro -> lista de valores, ej. {"C": [0.3, 1, 3], "gamma": [0.001, 0.002]}
        """
        fabrica, escalar_def = self._fabrica(algoritmo)
        escalar = escalar_def if escalar is None else escalar
        busqueda = GridSearchCV(self.crear_pipeline(fabrica(), escalar),
                                {f"modelo__{p}": v for p, v in rejilla.items()},
                                cv=self._particiones_cv(k), scoring="neg_root_mean_squared_error",
                                n_jobs=-1).fit(self.X_train, self.y_train)
        res = pd.DataFrame(busqueda.cv_results_)
        tabla = pd.DataFrame([{p.replace("modelo__", ""): v for p, v in d.items()} for d in res["params"]])
        tabla["RMSE validación"] = -res["mean_test_score"]
        tabla["Desv. validación"] = res["std_test_score"]
        return tabla.sort_values("RMSE validación").head(top).reset_index(drop=True).round(4)

    def curva_parametro(self, algoritmo, parametro, valores, k=5, escalar=None, **fijos):
        """
        RMSE de entrenamiento y de validación cruzada (k particiones) según un hiperparámetro.
        Sirve para ver el sobreajuste (ej. max_depth del árbol, C del SVR, alpha del Lasso).
        No se guarda en el historial. algoritmo: 'arbol', 'bosque', 'xgboost', 'svr', 'lasso' o 'ridge'.
        """
        fabrica, escalar_def = self._fabrica(algoritmo)
        escalar = escalar_def if escalar is None else escalar
        filas = []
        for v in valores:
            pipe = self.crear_pipeline(fabrica(**{**fijos, parametro: v}), escalar)
            res = cross_validate(pipe, self.X_train, self.y_train, cv=self._particiones_cv(k),
                                 scoring="neg_root_mean_squared_error", return_train_score=True)
            filas.append({parametro: v, "RMSE entrenamiento": -res["train_score"].mean(),
                          "RMSE validación": -res["test_score"].mean(),
                          "Desv. validación": res["test_score"].std()})
        tabla = pd.DataFrame(filas)
        x = [str(v) for v in tabla[parametro]] if tabla[parametro].dtype == object else tabla[parametro]
        plt.figure(figsize=(8, 4))
        plt.plot(x, tabla["RMSE entrenamiento"], marker="o", label="Entrenamiento")
        plt.errorbar(x, tabla["RMSE validación"], yerr=tabla["Desv. validación"], marker="o",
                     capsize=3, label=f"Validación cruzada ({k} particiones)")
        if parametro in ("alpha", "C", "gamma"):
            plt.xscale("log")
        plt.xlabel(parametro)
        plt.ylabel("RMSE")
        plt.title(f"{algoritmo}: RMSE según {parametro}")
        plt.legend()
        return tabla.round(4)

    def ruta_regularizacion(self, tipo="lasso", alphas=None, k=10, top=8):
        """
        Ruta de los coeficientes y RMSE de validación cruzada según alpha (Lasso o Ridge),
        con las variables estandarizadas. Marca el alpha de menor error.
        """
        alphas = alphas if alphas is not None else (np.logspace(-5, 0, 50) if tipo == "lasso"
                                                    else np.logspace(-3, 5, 50))
        coefs, errores, desv = [], [], []
        for a in alphas:
            est = Lasso(alpha=a, max_iter=50000) if tipo == "lasso" else Ridge(alpha=a)
            pipe = self.crear_pipeline(est, escalar=True)
            res = cross_validate(pipe, self.X_train, self.y_train, cv=self._particiones_cv(k),
                                 scoring="neg_root_mean_squared_error")
            errores.append(-res["test_score"].mean())
            desv.append(res["test_score"].std())
            coefs.append(pipe.fit(self.X_train, self.y_train).named_steps["modelo"].coef_)
        nombres = self.crear_pipeline(LinearRegression()).named_steps["prep"].fit(self.X_train).get_feature_names_out()
        coefs = pd.DataFrame(coefs, index=alphas, columns=nombres)
        mejor = alphas[int(np.argmin(errores))]
        # las variables con mayor |coef| en el alpha óptimo
        destacadas = coefs.loc[mejor].abs().sort_values(ascending=False).head(top).index
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for col in coefs.columns:
            if col not in destacadas:
                ax[0].plot(alphas, coefs[col], color="lightgrey", linewidth=0.7)
        for col in destacadas:
            ax[0].plot(alphas, coefs[col], linewidth=1.8, label=col)
        ax[0].axvline(mejor, color="black", linestyle="--")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("Coeficiente (variables estandarizadas)")
        ax[0].set_title(f"{tipo.capitalize()}: ruta de los coeficientes")
        ax[0].legend(fontsize=7, loc="best")
        ax[1].errorbar(alphas, errores, yerr=desv, marker=".", capsize=2)
        ax[1].axvline(mejor, color="black", linestyle="--", label=f"mejor alpha = {mejor:.4g}")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel(f"RMSE validación cruzada ({k} particiones)")
        ax[1].set_title(f"{tipo.capitalize()}: error según alpha")
        ax[1].legend()
        plt.tight_layout()
        no_cero = (coefs.abs() > 1e-10).sum(axis=1)
        return pd.DataFrame({"alpha": alphas, "RMSE validación": errores,
                             "Coeficientes ≠ 0": no_cero.values}).round(5)

    # =================================================================
    # Benchmarking entre modelos y familias
    # =================================================================
    def benchmark(self, nombres=None, k=10, repeticiones=3):
        """
        Validación cruzada repetida (k particiones x repeticiones) sobre el entrenamiento.
        Todos los modelos usan exactamente las mismas particiones, así que los resultados
        por partición se pueden comparar en pares (pruebas de Friedman y Wilcoxon).
        Retorna una tabla larga: una fila por modelo y partición.
        """
        nombres = nombres or list(self.experimentos)
        cv = RepeatedKFold(n_splits=k, n_repeats=repeticiones, random_state=self.semilla)
        filas = []
        for nombre in nombres:
            pipe = clone(self.experimentos[nombre]["pipeline"])
            res = cross_validate(pipe, self.X_train, self.y_train, cv=cv, n_jobs=-1,
                                 scoring={m: s for m, (s, _) in self._metricas_cv().items()})
            algoritmo = self.experimentos[nombre]["algoritmo"]
            for i in range(k * repeticiones):
                filas.append({"Modelo": nombre, "Algoritmo": algoritmo,
                              "Familia": self.FAMILIAS.get(algoritmo), "Partición": i + 1,
                              **{m: res[f"test_{m}"][i] * signo
                                 for m, (_, signo) in self._metricas_cv().items()}})
        self._benchmark = pd.DataFrame(filas)
        return self._benchmark

    def resumen_benchmark(self, tabla=None, criterio="RMSE"):
        tabla = self._benchmark if tabla is None else tabla
        res = tabla.groupby(["Modelo", "Familia"], sort=False)[["RMSE", "MAE", "R²"]].agg(["mean", "std"])
        res.columns = [f"{m} {'promedio' if e == 'mean' else 'desv.'}" for m, e in res.columns]
        # rango promedio de cada modelo en las particiones (1 = el mejor), como en la prueba de Friedman
        ancho = tabla.pivot(index="Partición", columns="Modelo", values=criterio)
        rangos = ancho.rank(axis=1, ascending=criterio in self.MENOR_ES_MEJOR).mean()
        res["Rango promedio"] = rangos.reindex(res.index.get_level_values("Modelo")).values
        return res.sort_values(f"{criterio} promedio",
                               ascending=criterio in self.MENOR_ES_MEJOR).round(4).reset_index()

    def prueba_estadistica(self, tabla=None, criterio="RMSE", alfa=0.05):
        """
        Prueba de Friedman (¿hay diferencias entre los modelos?) y luego Wilcoxon de rangos con
        signo entre el mejor modelo y cada uno de los demás, con la corrección de Holm.
        """
        tabla = self._benchmark if tabla is None else tabla
        ancho = tabla.pivot(index="Partición", columns="Modelo", values=criterio)
        estad, p_friedman = stats.friedmanchisquare(*[ancho[c] for c in ancho.columns])
        asc = criterio in self.MENOR_ES_MEJOR
        mejor = ancho.mean().sort_values(ascending=asc).index[0]
        filas = []
        for otro in ancho.columns.drop(mejor):
            _, p = stats.wilcoxon(ancho[mejor], ancho[otro])
            filas.append({"Mejor": mejor, "Contra": otro,
                          f"Diferencia {criterio}": (ancho[otro] - ancho[mejor]).mean(), "p": p})
        comp = pd.DataFrame(filas).sort_values("p").reset_index(drop=True)
        # corrección de Holm por comparaciones múltiples
        m = len(comp)
        comp["p Holm"] = np.minimum(1, np.maximum.accumulate(comp["p"] * (m - np.arange(m))))
        comp["Significativa"] = np.where(comp["p Holm"] < alfa, "sí", "no")
        friedman = pd.Series({"Chi² de Friedman": estad, "p": p_friedman, "Modelos": ancho.shape[1],
                              "Particiones": ancho.shape[0]}, name="Friedman")
        return friedman, comp.round(6)

    # =================================================================
    # Gráficos
    # =================================================================
    def graficar_comparacion(self, metricas=("RMSE", "MAE", "ER", "R²"), algoritmo=None,
                             nombres=None, titulo="Comparación de modelos"):
        tabla = self.resultados(algoritmo).set_index("Modelo")
        if nombres is not None:
            tabla = tabla.loc[nombres]
        tabla.index = [textwrap.fill(n, 40) for n in tabla.index]
        fig, ejes = plt.subplots(1, len(metricas), figsize=(4 * len(metricas), max(3.5, len(tabla) * 0.55)),
                                 sharey=True)
        for eje, m in zip(np.atleast_1d(ejes), metricas):
            mejor = tabla[m].min() if m in self.MENOR_ES_MEJOR else tabla[m].max()
            colores = ["darkorange" if v == mejor else "steelblue" for v in tabla[m]]
            eje.barh(tabla.index, tabla[m], color=colores)
            eje.set_title(m + (" (menor es mejor)" if m in self.MENOR_ES_MEJOR else " (mayor es mejor)"),
                          fontsize=9)
            for i, v in enumerate(tabla[m]):
                eje.text(v, i, f" {v:.4f}", va="center", fontsize=7)
            eje.set_xlim(min(0, tabla[m].min() * 1.1), tabla[m].max() * 1.3)
        np.atleast_1d(ejes)[0].invert_yaxis()
        fig.suptitle(titulo)
        plt.tight_layout()

    def graficar_real_vs_predicho(self, nombres=None, columnas=3):
        nombres = nombres or [self.nombre_ultimo]
        filas = int(np.ceil(len(nombres) / columnas))
        fig, ejes = plt.subplots(filas, min(columnas, len(nombres)),
                                 figsize=(4.6 * min(columnas, len(nombres)), 4.3 * filas), squeeze=False)
        lim = [min(self.y_test.min(), 0), max(self.y_test.max(), 1)]
        for eje, nombre in zip(ejes.flat, nombres):
            exp = self.experimentos[nombre]
            eje.scatter(self.y_test, exp["y_pred"], s=10, alpha=0.5)
            eje.plot(lim, lim, color="red", linestyle="--", linewidth=1)
            ind = exp["indices"]
            eje.set_title(textwrap.fill(nombre, 38) + f"\nRMSE={ind['RMSE']:.4f}  R²={ind['R²']:.4f}",
                          fontsize=8)
            eje.set_xlabel("Valor real")
            eje.set_ylabel("Predicción")
        for eje in ejes.flat[len(nombres):]:
            eje.set_visible(False)
        plt.tight_layout()

    def graficar_residuos(self, nombre=None):
        nombre = nombre or self.nombre_ultimo
        r = self.residuos(nombre)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].scatter(r["Predicción"], r["Residuo"], s=10, alpha=0.5)
        ax[0].axhline(0, color="red", linestyle="--")
        ax[0].set_xlabel("Predicción")
        ax[0].set_ylabel("Residuo (real - predicción)")
        ax[0].set_title("Residuos contra predicción")
        sns.histplot(r["Residuo"], kde=True, ax=ax[1])
        ax[1].set_title("Distribución de los residuos")
        fig.suptitle(f"Residuos: {nombre}")
        plt.tight_layout()

    def graficar_coeficientes(self, nombre=None, top=20, titulo=None):
        coef = self.coeficientes(nombre, top)
        plt.figure(figsize=(8, max(3, len(coef) * 0.35)))
        plt.barh(coef.index[::-1], coef.values[::-1],
                 color=["indianred" if v < 0 else "steelblue" for v in coef.values[::-1]])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(titulo or f"Coeficientes: {nombre or self.nombre_ultimo}")
        plt.xlabel("Coeficiente")

    def graficar_arbol(self, nombre=None, profundidad=3):
        pipe = (self.experimentos[nombre] if nombre else self.ultimo)["pipeline"]
        plt.figure(figsize=(20, 9))
        plot_tree(pipe.named_steps["modelo"], max_depth=profundidad,
                  feature_names=self.nombres_atributos(nombre), filled=True, rounded=True,
                  fontsize=8, precision=3)
        plt.title(f"Primeros {profundidad} niveles del árbol: {nombre or self.nombre_ultimo}")

    def graficar_benchmark(self, tabla=None, criterio="RMSE", titulo=None):
        tabla = self._benchmark if tabla is None else tabla
        asc = criterio in self.MENOR_ES_MEJOR
        orden = tabla.groupby("Modelo")[criterio].mean().sort_values(ascending=asc).index
        etiquetas = {n: textwrap.fill(n, 40) for n in orden}
        datos = tabla.assign(Modelo=tabla["Modelo"].map(etiquetas))
        plt.figure(figsize=(10, max(4, len(orden) * 0.6)))
        sns.boxplot(data=datos, y="Modelo", x=criterio, hue="Familia", order=[etiquetas[n] for n in orden],
                    dodge=False)
        sns.stripplot(data=datos, y="Modelo", x=criterio, order=[etiquetas[n] for n in orden],
                      color="black", size=2.5, alpha=0.5)
        plt.ylabel("")
        plt.title(titulo or f"Benchmarking: {criterio} en validación cruzada repetida")
        plt.legend(fontsize=8, loc="best")
