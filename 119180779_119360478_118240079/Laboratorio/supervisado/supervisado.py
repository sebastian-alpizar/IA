import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

from eda import EDA   # clase base, carpeta eda/


class Supervisado(EDA):
    """
    Métodos supervisados del framework. Hereda de EDA, así que
    también tiene todos los métodos de análisis exploratorio.
    Se encarga de lo que comparten Clasificación y Regresión:
    - Separar predictoras / variable a predecir y la partición entrenamiento-prueba
    - Preprocesamiento (imputación, escalado, one-hot, selección de atributos)
      dentro de un Pipeline, para que no haya fuga de información del conjunto de prueba
    - Entrenar, evaluar, comparar configuraciones y guardar el historial de experimentos
    Lo que cambia entre clasificación y regresión (estratificar o no, los índices de
    calidad, la validación cruzada y el sentido del criterio) está en métodos "gancho"
    (_estratificar, _evaluar, _metricas_cv, ...) que cada clase hija redefine.
    Por defecto los ganchos son los de clasificación (Laboratorio 02).
    """

    # índices de error: para estos el mejor modelo es el de MENOR valor
    MENOR_ES_MEJOR = ()

    def __init__(self, dataframe: pd.DataFrame, variable_predecir, test_size=0.25,
                 eliminar_duplicados=True, semilla=42, excluir=None):
        super().__init__(dataframe)
        self.__variable_predecir = variable_predecir
        self.__semilla = semilla
        self.__excluir = list(excluir) if excluir else []   # columnas que no son predictoras (ej. id)
        self.__resultados = []      # una fila de índices por experimento
        self.__experimentos = {}    # nombre -> pipeline, predicciones y probabilidades
        self.__ultimo = None        # nombre del último experimento
        self.particionar(test_size, eliminar_duplicados)

    # --- Propiedades ---
    @property
    def variable_predecir(self):
        return self.__variable_predecir

    @property
    def semilla(self):
        return self.__semilla

    @property
    def excluir(self):
        return self.__excluir

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
        return self.__experimentos.get(self.__ultimo)

    @property
    def nombre_ultimo(self):
        return self.__ultimo

    # =================================================================
    # Ganchos (los redefine Regresion; por defecto: clasificación)
    # =================================================================
    def _estratificar(self):
        return True

    def _puntaje_seleccion(self):
        # estadístico para SelectKBest
        return f_classif

    def _resumen_particion(self, X, y):
        return pd.DataFrame({
            "Conjunto": ["Entrenamiento", "Prueba"],
            "Filas": [len(self.__X_train), len(self.__X_test)],
            "Predictoras": [X.shape[1], X.shape[1]],
            **{f"% clase {c}": [round(self.__y_train.eq(c).mean() * 100, 2),
                                round(self.__y_test.eq(c).mean() * 100, 2)]
               for c in sorted(y.unique())}
        })

    def _evaluar(self, pipe):
        # predice el conjunto de prueba; retorna (índices, datos a guardar del experimento)
        proba = pipe.predict_proba(self.__X_test)
        y_pred = pipe.classes_[proba.argmax(axis=1)]
        indices = self.indices_calidad(self.__y_test, y_pred, proba[:, 1], pipe.classes_)
        return indices, {"y_pred": y_pred, "proba": proba[:, 1], "clases": pipe.classes_}

    def _particiones_cv(self, k):
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=self.__semilla)

    def _metricas_cv(self):
        # nombre -> (scorer de sklearn, signo para dejarlo en su escala natural)
        return {"Precisión Global": ("accuracy", 1), "F1": ("f1", 1),
                "F1 macro": ("f1_macro", 1), "AUC": ("roc_auc", 1)}

    # =================================================================
    # Preparación de los datos
    # =================================================================
    def particionar(self, test_size=0.25, eliminar_duplicados=True, variables=None):
        """
        Separa X / y y hace la partición entrenamiento-prueba
        (estratificada en clasificación, aleatoria simple en regresión).
        variables: lista de predictoras a usar (selección manual de atributos); None = todas.
        """
        datos = self.mostrar()
        if self.__excluir:
            datos = datos.drop(columns=self.__excluir)
        if eliminar_duplicados:
            datos = datos.drop_duplicates()
        y = datos[self.__variable_predecir]
        X = datos.drop(columns=[self.__variable_predecir])
        if variables is not None:
            X = X[variables]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=test_size, stratify=y if self._estratificar() else None,
            random_state=self.__semilla)
        return self._resumen_particion(X, y)

    def distribucion_clases(self):
        y = self.mostrar()[self.__variable_predecir]
        return pd.DataFrame({"cantidad": y.value_counts(),
                             "porcentaje": (y.value_counts(normalize=True) * 100).round(2)})

    def preprocesador(self, escalar=False, imputacion="median", columnas=None):
        # numéricas: imputación (+ escalado); categóricas: moda + one-hot
        # columnas: usar solo esas predictoras (ej. regresión lineal simple); None = todas
        X = self.__X_train if columnas is None else self.__X_train[columnas]
        numericas = X.select_dtypes(include="number").columns.tolist()
        categoricas = X.select_dtypes(exclude="number").columns.tolist()
        pasos_num = [("imputar", SimpleImputer(strategy=imputacion))]
        if escalar:
            pasos_num.append(("escalar", StandardScaler()))
        transformadores = [("num", Pipeline(pasos_num), numericas)]
        if categoricas:
            transformadores.append(("cat", Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categoricas))
        return ColumnTransformer(transformadores, verbose_feature_names_out=False)

    def crear_pipeline(self, estimador, escalar=False, imputacion="median", n_atributos=None,
                       columnas=None):
        pasos = [("prep", self.preprocesador(escalar, imputacion, columnas))]
        if n_atributos is not None:
            pasos.append(("seleccion", SelectKBest(self._puntaje_seleccion(), k=n_atributos)))
        pasos.append(("modelo", estimador))
        return Pipeline(pasos)

    # =================================================================
    # Entrenamiento y evaluación
    # =================================================================
    def entrenar(self, nombre, estimador, algoritmo=None, escalar=False, imputacion="median",
                 n_atributos=None, columnas=None):
        """
        Entrena el estimador (dentro del pipeline) con el conjunto de entrenamiento,
        lo evalúa en el de prueba y guarda el resultado en el historial.
        Retorna los índices de calidad del modelo.
        """
        pipe = self.crear_pipeline(estimador, escalar, imputacion, n_atributos, columnas)
        pipe.fit(self.__X_train, self.__y_train)
        indices, salida = self._evaluar(pipe)
        fila = {"Modelo": nombre, "Algoritmo": algoritmo or type(estimador).__name__, **indices}
        # si se repite el nombre se reemplaza el experimento anterior
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre] + [fila]
        self.__experimentos[nombre] = {"pipeline": pipe, "algoritmo": fila["Algoritmo"],
                                       "indices": indices, **salida}
        self.__ultimo = nombre
        return pd.Series(indices, name=nombre)

    @staticmethod
    def nombre_config(algoritmo, parametros, escalar=None, n_atributos=None, imputacion="median"):
        # ej. "KNN (n_neighbors=15, weights=distance)" o "KNN (estándar)"
        partes = [f"{k}={v}" for k, v in parametros.items()]
        if escalar is False:
            partes.append("sin escalar")
        if imputacion != "median":
            partes.append(f"imputación {imputacion}")
        if n_atributos is not None:
            partes.append(f"{n_atributos} atributos")
        return f"{algoritmo} ({', '.join(partes) if partes else 'estándar'})"

    def comparar(self, algoritmo, configuraciones):
        """
        Corre varias configuraciones de un mismo algoritmo y retorna sus índices.
        algoritmo: nombre del método (ej. 'knn', 'arbol', 'lasso', 'svr')
        configuraciones: lista de dicts, ej. [{}, {"n_neighbors": 15}, {"escalar": False}]
        """
        metodo = getattr(self, algoritmo)
        filas = [metodo(**conf) for conf in configuraciones]
        return pd.DataFrame(filas)

    @staticmethod
    def indices_calidad(y_real, y_pred, proba_positiva, clases):
        # índices de calidad para clasificación binaria (la clase positiva es clases[1])
        mc = confusion_matrix(y_real, y_pred, labels=clases)
        indices = {"Precisión Global": np.trace(mc) / mc.sum(),
                   "Error Global": 1 - np.trace(mc) / mc.sum()}
        for i, c in enumerate(clases):
            # precisión por categoría (aciertos de la clase / total real de la clase)
            indices[f"Precisión clase {c}"] = mc[i, i] / mc[i, :].sum()
        positiva = clases[1]
        indices["Precision (VPP)"] = precision_score(y_real, y_pred, pos_label=positiva, zero_division=0)
        indices["Recall"] = recall_score(y_real, y_pred, pos_label=positiva)
        indices["F1"] = f1_score(y_real, y_pred, pos_label=positiva)
        indices["F1 macro"] = f1_score(y_real, y_pred, average="macro")
        indices["AUC"] = roc_auc_score(y_real, proba_positiva)
        return {k: round(float(v), 4) for k, v in indices.items()}

    def validacion_cruzada(self, nombre=None, k=5):
        # validación cruzada (sobre entrenamiento) de un experimento ya corrido
        pipe = clone(self.__experimentos[nombre or self.__ultimo]["pipeline"])
        metricas = self._metricas_cv()
        res = cross_validate(pipe, self.__X_train, self.__y_train, cv=self._particiones_cv(k),
                             scoring={m: s for m, (s, _) in metricas.items()})
        tabla = pd.DataFrame({m: res[f"test_{m}"] * signo for m, (_, signo) in metricas.items()})
        tabla.index = [f"Fold {i+1}" for i in range(k)]
        tabla.loc["Promedio"] = tabla.mean()
        tabla.loc["Desv. estándar"] = tabla.iloc[:k].std()
        return tabla.round(4)

    # =================================================================
    # Resultados
    # =================================================================
    def resultados(self, algoritmo=None):
        tabla = pd.DataFrame(self.__resultados)
        if algoritmo is not None and not tabla.empty:
            tabla = tabla[tabla["Algoritmo"] == algoritmo]
        return tabla.reset_index(drop=True)

    def mejor_modelo(self, criterio="AUC", algoritmo=None):
        tabla = self.resultados(algoritmo)
        # en los índices de error (RMSE, MAE, ...) el mejor es el mínimo
        fila = tabla[criterio].idxmin() if criterio in self.MENOR_ES_MEJOR else tabla[criterio].idxmax()
        return tabla.loc[fila]

    def matriz_confusion(self, nombre=None):
        exp = self.__experimentos[nombre or self.__ultimo]
        mc = confusion_matrix(self.__y_test, exp["y_pred"], labels=exp["clases"])
        return pd.DataFrame(mc, index=[f"Real {c}" for c in exp["clases"]],
                            columns=[f"Pred {c}" for c in exp["clases"]])

    # =================================================================
    # Análisis del modelo
    # =================================================================
    def nombres_atributos(self, nombre=None):
        # nombres de las variables después del one-hot y la selección de atributos
        pipe = (self.__experimentos[nombre] if nombre else self.ultimo)["pipeline"]
        nombres = pipe.named_steps["prep"].get_feature_names_out()
        if "seleccion" in pipe.named_steps:
            nombres = pipe.named_steps["seleccion"].get_feature_names_out(nombres)
        return list(nombres)

    def importancia_variables(self, nombre=None):
        pipe = (self.__experimentos[nombre] if nombre else self.ultimo)["pipeline"]
        modelo = pipe.named_steps["modelo"]
        if not hasattr(modelo, "feature_importances_"):
            raise ValueError("El modelo no calcula importancia de variables (ej. KNN o SVR).")
        return pd.Series(modelo.feature_importances_, index=self.nombres_atributos(nombre),
                         name="importancia").sort_values(ascending=False).round(4)

    # =================================================================
    # Gráficos
    # =================================================================
    def graficar_importancia(self, nombre=None, top=15, titulo=None):
        imp = self.importancia_variables(nombre).head(top)
        plt.figure(figsize=(8, max(3, len(imp) * 0.4)))
        plt.barh(imp.index[::-1], imp.values[::-1], color="steelblue")
        plt.title(titulo or f"Importancia de variables: {nombre or 'último modelo'}")
        plt.xlabel("Importancia")

    def graficar_matriz_confusion(self, nombre=None, normalizar=False):
        nombre = nombre or self.__ultimo
        mc = self.matriz_confusion(nombre)
        if normalizar:
            mc = mc.div(mc.sum(axis=1), axis=0)
        plt.figure(figsize=(5, 4))
        sns.heatmap(mc, annot=True, fmt=".2%" if normalizar else "d", cmap="Blues", cbar=False)
        plt.title(f"Matriz de confusión: {nombre}")
        plt.ylabel("Real")
        plt.xlabel("Predicción")

    def graficar_roc(self, nombres=None, titulo="Curvas ROC"):
        nombres = nombres or list(self.__experimentos)
        plt.figure(figsize=(7, 6))
        for nombre in nombres:
            exp = self.__experimentos[nombre]
            fpr, tpr, _ = roc_curve(self.__y_test, exp["proba"], pos_label=exp["clases"][1])
            plt.plot(fpr, tpr, label=f"{nombre} (AUC={exp['indices']['AUC']:.3f})")
        plt.plot([0, 1], [0, 1], color="dimgrey", linestyle="--")
        plt.xlabel("Tasa de falsos positivos")
        plt.ylabel("Tasa de verdaderos positivos")
        plt.title(titulo)
        plt.legend(loc="lower right", fontsize=8)

    def graficar_comparacion(self, metricas=("Precisión Global", "F1", "F1 macro", "AUC"),
                             algoritmo=None, nombres=None, titulo="Comparación de modelos"):
        tabla = self.resultados(algoritmo).set_index("Modelo")
        if nombres is not None:
            tabla = tabla.loc[nombres]
        # nombres largos en varias líneas para que no aplasten el gráfico
        tabla.index = [textwrap.fill(n, 40) for n in tabla.index]
        ax = tabla[list(metricas)].plot(kind="barh", figsize=(10, max(4, len(tabla) * 0.9)), width=0.8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(titulo)
        ax.legend(loc="lower right", fontsize=8)
