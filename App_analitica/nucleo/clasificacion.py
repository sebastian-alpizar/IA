"""
clasificacion.py
Métodos de clasificación del Laboratorio 2 (KNN, Árboles de Decisión, Random Forest,
XGBoost y AdaBoost) y los demás vistos en el curso (Gradient Boosting, SVM, Bayes,
regresión logística, LDA, QDA y red neuronal), integrados al framework de la app:

    EDA -> SupervisadoLab(EDA) -> ClasificacionLab(SupervisadoLab)

- SupervisadoLab: partición estratificada, preprocesamiento dentro de un Pipeline
  (imputación, escalado, one-hot y selección de atributos), entrenamiento,
  índices de calidad, historial de experimentos y validación cruzada.
- ClasificacionLab: los algoritmos. Cada uno sin parámetros es la
  configuración estándar (valores por defecto de la librería); los parámetros
  extra van directo al estimador, así se prueban las variaciones de configuración.
Todo devuelve tablas o figuras de Plotly.
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from . import graficos as g
from .eda import EDA

try:
    from xgboost import XGBClassifier
    HAY_XGB = True
except ImportError:
    HAY_XGB = False


# ----------------------------------------------------------------------
# Algoritmos
#   lab: True para los cinco que pide el Laboratorio 2
#   metodo: nombre del método de ClasificacionLab
#   escalar: valor por defecto (KNN usa distancias, por eso escala)
#   params: (nombre, etiqueta, tipo, default de la librería, extra)
#           tipo "int"/"float" -> extra = (min, max, paso); "select" -> extra = opciones; "text" -> texto libre
#           un default None significa "sin límite" / "no usar" (se muestra como 0 o "ninguno")
#   variaciones: configuración estándar + variaciones (en los del Lab 2, las del notebook)
# ----------------------------------------------------------------------
def _p(nombre, etiqueta, tipo, default, extra, ayuda=""):
    return {"nombre": nombre, "etiqueta": etiqueta, "tipo": tipo, "default": default, "extra": extra, "ayuda": ayuda}


ALGORITMOS = {
    "KNN": {
        "lab": True, "metodo": "knn", "escalar": True,
        "descripcion": "Clasifica según la clase mayoritaria de los k vecinos más cercanos.",
        "params": [_p("n_neighbors", "Vecinos (k)", "int", 5, (1, 200, 1)),
                   _p("weights", "Pesos", "select", "uniform", ["uniform", "distance"]),
                   _p("p", "Distancia (p): 1 = Manhattan, 2 = Euclídea", "int", 2, (1, 5, 1))],
        "variaciones": [{}, {"n_neighbors": 15}, {"n_neighbors": 25},
                        {"n_neighbors": 25, "weights": "distance"}, {"n_neighbors": 25, "p": 1},
                        {"n_neighbors": 25, "imputacion": "mean"}, {"n_neighbors": 25, "escalar": False},
                        {"n_neighbors": 25, "n_atributos": 5}],
    },
    "DT": {
        "lab": True, "metodo": "arbol", "escalar": False,
        "descripcion": "Árbol de decisión: divide los datos con reglas sí/no sobre las variables.",
        "params": [_p("criterion", "Criterio", "select", "gini", ["gini", "entropy", "log_loss"]),
                   _p("max_depth", "Profundidad máxima (0 = sin límite)", "int", None, (0, 50, 1)),
                   _p("min_samples_split", "Mínimo para dividir", "int", 2, (2, 500, 1)),
                   _p("min_samples_leaf", "Mínimo por hoja", "int", 1, (1, 500, 1)),
                   _p("class_weight", "Peso de clases", "select", None, [None, "balanced"]),
                   _p("ccp_alpha", "Poda costo-complejidad (ccp_alpha)", "float", 0.0, (0.0, 0.1, 0.0005))],
        "variaciones": [{}, {"max_depth": 5}, {"max_depth": 8, "min_samples_leaf": 20},
                        {"criterion": "entropy", "max_depth": 8, "min_samples_leaf": 20},
                        {"max_depth": 8, "min_samples_leaf": 20, "class_weight": "balanced"},
                        {"ccp_alpha": 0.002}],
    },
    "RF": {
        "lab": True, "metodo": "bosque", "escalar": False,
        "descripcion": "Random Forest: muchos árboles con muestras y variables al azar; vota la mayoría.",
        "params": [_p("n_estimators", "Cantidad de árboles", "int", 100, (10, 2000, 10)),
                   _p("max_features", "Variables por división", "select", "sqrt", ["sqrt", "log2", None]),
                   _p("max_depth", "Profundidad máxima (0 = sin límite)", "int", None, (0, 50, 1)),
                   _p("min_samples_leaf", "Mínimo por hoja", "int", 1, (1, 500, 1)),
                   _p("class_weight", "Peso de clases", "select", None, [None, "balanced", "balanced_subsample"])],
        "variaciones": [{}, {"n_estimators": 500}, {"n_estimators": 500, "max_features": None},
                        {"n_estimators": 500, "min_samples_leaf": 3}, {"n_estimators": 500, "max_depth": 12},
                        {"n_estimators": 500, "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
                        {"n_atributos": 5}],
    },
    "XGBoost": {
        "lab": True, "metodo": "xgboost", "escalar": False,
        "descripcion": "Gradient boosting optimizado: árboles en secuencia que corrigen el error del anterior.",
        "params": [_p("n_estimators", "Cantidad de árboles", "int", 100, (10, 3000, 10)),
                   _p("learning_rate", "Tasa de aprendizaje", "float", 0.3, (0.001, 1.0, 0.01)),
                   _p("max_depth", "Profundidad de cada árbol", "int", 6, (1, 20, 1)),
                   _p("subsample", "Proporción de filas (subsample)", "float", 1.0, (0.1, 1.0, 0.05)),
                   _p("colsample_bytree", "Proporción de columnas por árbol", "float", 1.0, (0.1, 1.0, 0.05)),
                   _p("scale_pos_weight", "Peso de la clase positiva", "float", 1.0, (0.1, 100.0, 0.1),
                      "Para clases desbalanceadas: negativos / positivos (solo binaria).")],
        "variaciones": [{}, {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
                        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                         "colsample_bytree": 0.8},
                        {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 6, "subsample": 0.8,
                         "colsample_bytree": 0.8},
                        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                         "colsample_bytree": 0.8, "scale_pos_weight": "ratio"}],
    },
    "AdaBoost": {
        "lab": True, "metodo": "adaboost", "escalar": False,
        "descripcion": "Potenciación: árboles pequeños en secuencia; cada uno da más peso a los errores del anterior.",
        "params": [_p("n_estimators", "Iteraciones", "int", 50, (10, 2000, 10)),
                   _p("learning_rate", "Tasa de aprendizaje", "float", 1.0, (0.01, 2.0, 0.01)),
                   _p("profundidad_base", "Profundidad del árbol base (1 = stump)", "int", 1, (1, 10, 1))],
        "variaciones": [{}, {"n_estimators": 200}, {"n_estimators": 200, "learning_rate": 0.5},
                        {"n_estimators": 300, "learning_rate": 0.5, "profundidad_base": 2},
                        {"n_estimators": 300, "learning_rate": 0.1, "profundidad_base": 3}],
    },
    "Gradient Boosting": {
        "lab": False, "metodo": "gradient_boosting", "escalar": False,
        "descripcion": "Árboles en secuencia que minimizan el error por gradiente (versión de scikit-learn).",
        "params": [_p("n_estimators", "Iteraciones", "int", 100, (10, 2000, 10)),
                   _p("learning_rate", "Tasa de aprendizaje", "float", 0.1, (0.001, 1.0, 0.01)),
                   _p("max_depth", "Profundidad de cada árbol", "int", 3, (1, 15, 1)),
                   _p("subsample", "Proporción de filas (subsample)", "float", 1.0, (0.1, 1.0, 0.05))],
        "variaciones": [{}, {"n_estimators": 300, "learning_rate": 0.05},
                        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8}],
    },
    "SVM": {
        "lab": False, "metodo": "svm", "escalar": True, "lento": True,
        "descripcion": "Máquinas de soporte vectorial: la frontera que separa las clases con el mayor margen. "
                       "Lenta con muchas filas (más de ~20 000).",
        "params": [_p("kernel", "Núcleo", "select", "rbf", ["rbf", "linear", "poly", "sigmoid"]),
                   _p("C", "Penalización C", "float", 1.0, (0.01, 1000.0, 0.01)),
                   _p("class_weight", "Peso de clases", "select", None, [None, "balanced"])],
        "variaciones": [{}, {"C": 10.0}, {"kernel": "linear"}, {"class_weight": "balanced"}],
    },
    "Bayes": {
        "lab": False, "metodo": "bayes", "escalar": False,
        "descripcion": "Bayes ingenuo (gaussiano): probabilidades suponiendo variables independientes.",
        "params": [],
        "variaciones": [{}],
    },
    "Logística": {
        "lab": False, "metodo": "logistica", "escalar": True,
        "descripcion": "Regresión logística: modelo lineal que estima la probabilidad de cada clase.",
        "params": [_p("C", "Inverso de la regularización (C)", "float", 1.0, (0.001, 1000.0, 0.01)),
                   _p("class_weight", "Peso de clases", "select", None, [None, "balanced"])],
        "variaciones": [{}, {"C": 0.1}, {"class_weight": "balanced"}],
    },
    "LDA": {
        "lab": False, "metodo": "lda", "escalar": False,
        "descripcion": "Análisis discriminante lineal: combinaciones lineales que mejor separan las clases.",
        "params": [],
        "variaciones": [{}],
    },
    "QDA": {
        "lab": False, "metodo": "qda", "escalar": False,
        "descripcion": "Análisis discriminante cuadrático: como LDA pero con fronteras curvas.",
        "params": [_p("reg_param", "Regularización", "float", 0.0, (0.0, 1.0, 0.01),
                      "Súbala (ej. 0.1) si da error de colinealidad.")],
        "variaciones": [{}, {"reg_param": 0.1}],
    },
    "Red neuronal": {
        "lab": False, "metodo": "red_neuronal", "escalar": True,
        "descripcion": "Perceptrón multicapa.",
        "params": [_p("hidden_layer_sizes", "Neuronas por capa (ej. 32,16)", "text", "100", None),
                   _p("max_iter", "Iteraciones máximas", "int", 200, (50, 5000, 50)),
                   _p("alpha", "Regularización L2 (alpha)", "float", 0.0001, (0.0, 1.0, 0.0001))],
        "variaciones": [{}, {"hidden_layer_sizes": "32,16", "max_iter": 500},
                        {"hidden_layer_sizes": "64,32", "max_iter": 500, "alpha": 0.001}],
    },
}
if not HAY_XGB:
    del ALGORITMOS["XGBoost"]
ALGORITMOS_LAB = [a for a, i in ALGORITMOS.items() if i["lab"]]

# opciones de preprocesamiento que acepta cada método (además de los parámetros del estimador)
OPCIONES_PREP = ("nombre", "escalar", "imputacion", "n_atributos")
CRITERIOS = ["F1 macro", "AUC", "Precisión Global", "F1", "Recall", "Precision (VPP)"]


# ======================================================================
class SupervisadoLab(EDA):
    """
    Parte común de los métodos supervisados del laboratorio. Hereda de EDA.
    - Separa predictoras / variable a predecir y hace la partición entrenamiento-prueba
    - Preprocesamiento (imputación, escalado, one-hot, selección de atributos) dentro
      de un Pipeline, para que no haya fuga de información del conjunto de prueba
    - Entrena, evalúa y guarda el historial de experimentos
    Las clases se codifican 0..n-1 internamente (XGBoost lo exige); en binaria la
    clase positiva siempre queda como código 1.
    """

    def __init__(self, datos: pd.DataFrame, variable_predecir, predictoras=None, positiva=None,
                 test_size=0.25, eliminar_duplicados=True, semilla=42):
        datos = datos.dropna(subset=[variable_predecir])
        super().__init__(datos)
        self.__variable_predecir = variable_predecir
        self.__predictoras = list(predictoras or [c for c in datos.columns if c != variable_predecir])
        self.__semilla = int(semilla)
        clases = sorted(datos[variable_predecir].astype(str).unique())
        if len(clases) < 2:
            raise ValueError("La variable a predecir necesita al menos dos clases.")
        if len(clases) == 2:
            positiva = str(positiva) if str(positiva) in clases else clases[1]
            clases = [c for c in clases if c != positiva] + [positiva]
        self.__clases = clases
        self.__resultados = []      # una fila de índices por experimento
        self.__experimentos = {}    # nombre -> pipeline, predicciones y probabilidades
        self.__ultimo = None
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
    def clases(self):
        return self.__clases

    @property
    def binaria(self):
        return len(self.__clases) == 2

    @property
    def positiva(self):
        return self.__clases[1] if self.binaria else None

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
    def _codificar(self, y):
        return y.astype(str).map({c: i for i, c in enumerate(self.__clases)}).astype(int)

    def decodificar(self, codigos):
        return np.asarray(self.__clases, dtype=object)[np.asarray(codigos, dtype=int)]

    def particionar(self, test_size=0.25, eliminar_duplicados=True):
        """Separa X / y y hace la partición entrenamiento-prueba estratificada."""
        datos = self.datos[self.__predictoras + [self.__variable_predecir]]
        if eliminar_duplicados:
            datos = datos.drop_duplicates()
        y = self._codificar(datos[self.__variable_predecir])
        X = datos[self.__predictoras]
        estratos = y if y.value_counts().min() >= 2 else None
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=test_size, stratify=estratos, random_state=self.__semilla)
        self.__resultados, self.__experimentos, self.__ultimo = [], {}, None

    def resumen_particion(self):
        filas = {"Conjunto": ["Entrenamiento", "Prueba"],
                 "Filas": [len(self.__X_train), len(self.__X_test)],
                 "Predictoras": [self.__X_train.shape[1]] * 2}
        for i, c in enumerate(self.__clases):
            filas[f"% clase {c}"] = [round(self.__y_train.eq(i).mean() * 100, 2),
                                     round(self.__y_test.eq(i).mean() * 100, 2)]
        return pd.DataFrame(filas).set_index("Conjunto")

    def distribucion_clases(self):
        y = self.datos[self.__variable_predecir].astype(str)
        return pd.DataFrame({"Cantidad": y.value_counts(),
                             "Porcentaje": (y.value_counts(normalize=True) * 100).round(2)})

    def relacion_negativos_positivos(self):
        # negativos por cada positivo en entrenamiento (sirve para scale_pos_weight)
        if not self.binaria:
            return 1.0
        return round(float((self.__y_train == 0).sum() / max((self.__y_train == 1).sum(), 1)), 2)

    def preprocesador(self, escalar=False, imputacion="median"):
        # numéricas: imputación (+ escalado); categóricas: moda + one-hot
        numericas = self.__X_train.select_dtypes(include="number").columns.tolist()
        categoricas = self.__X_train.select_dtypes(exclude="number").columns.tolist()
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

    def crear_pipeline(self, estimador, escalar=False, imputacion="median", n_atributos=None):
        pasos = [("prep", self.preprocesador(escalar, imputacion))]
        if n_atributos:
            pasos.append(("seleccion", SelectKBest(f_classif, k=n_atributos)))
        pasos.append(("modelo", estimador))
        return Pipeline(pasos)

    # =================================================================
    # Entrenamiento y evaluación
    # =================================================================
    def entrenar(self, nombre, estimador, algoritmo=None, escalar=False, imputacion="median",
                 n_atributos=None, config=None):
        """
        Entrena el estimador (dentro del pipeline) con el conjunto de entrenamiento,
        lo evalúa en el de prueba y lo guarda en el historial.
        Retorna los índices de calidad del modelo.
        """
        if n_atributos:
            total = len(self.preprocesador().fit(self.__X_train).get_feature_names_out())
            n_atributos = min(int(n_atributos), total)
        inicio = time.time()
        pipe = self.crear_pipeline(estimador, escalar, imputacion, n_atributos)
        pipe.fit(self.__X_train, self.__y_train)
        proba = pipe.predict_proba(self.__X_test)
        y_pred = pipe.classes_[proba.argmax(axis=1)]
        indices = self.indices_calidad(self.__y_test, y_pred, proba, pipe.classes_)
        indices["Tiempo (s)"] = round(time.time() - inicio, 2)
        fila = {"Modelo": nombre, "Algoritmo": algoritmo or type(estimador).__name__, **indices}
        # si se repite el nombre se reemplaza el experimento anterior
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre] + [fila]
        self.__experimentos[nombre] = {"pipeline": pipe, "y_pred": y_pred, "proba": proba,
                                       "algoritmo": fila["Algoritmo"], "indices": indices,
                                       "config": dict(config or {}, escalar=escalar, imputacion=imputacion,
                                                      n_atributos=n_atributos)}
        self.__ultimo = nombre
        return pd.Series(indices, name=nombre)

    def indices_calidad(self, y_real, y_pred, proba, codigos):
        """
        Índices de calidad. En binaria la clase positiva es la del código 1 y además se calculan
        los índices del curso (matriz con filas = real, columnas = predicción):
          PP = VP/(VP+FN)   PN = VN/(VN+FP)   AP = VP/(VP+FP)   AN = VN/(VN+FN)
          PFP = FP/(FP+VN)  PFN = FN/(FN+VP)
        En multiclase Precision, Recall y F1 son promedios macro y el AUC es uno-contra-todos.
        """
        mc = confusion_matrix(y_real, y_pred, labels=codigos)
        indices = {"Precisión Global": np.trace(mc) / mc.sum(),
                   "Error Global": 1 - np.trace(mc) / mc.sum()}
        for i, c in enumerate(codigos):
            # precisión por categoría (aciertos de la clase / total real de la clase)
            total = mc[i, :].sum()
            indices[f"Precisión clase {self.__clases[c]}"] = mc[i, i] / total if total else np.nan
        if self.binaria:
            indices["Precision (VPP)"] = precision_score(y_real, y_pred, pos_label=1, zero_division=0)
            indices["Recall"] = recall_score(y_real, y_pred, pos_label=1, zero_division=0)
            indices["F1"] = f1_score(y_real, y_pred, pos_label=1, zero_division=0)
            indices["AUC"] = roc_auc_score(y_real, proba[:, 1]) if len(set(y_real)) == 2 else np.nan
            (vn, fp), (fn, vp) = mc
            div = lambda a, b: a / b if b else np.nan
            indices.update({"PP": div(vp, vp + fn), "PN": div(vn, vn + fp), "AP": div(vp, vp + fp),
                            "AN": div(vn, vn + fn), "PFP": div(fp, fp + vn), "PFN": div(fn, fn + vp)})
        else:
            indices["Precision (VPP)"] = precision_score(y_real, y_pred, average="macro", zero_division=0)
            indices["Recall"] = recall_score(y_real, y_pred, average="macro", zero_division=0)
            indices["F1"] = f1_score(y_real, y_pred, average="macro", zero_division=0)
            try:
                indices["AUC"] = roc_auc_score(y_real, proba, multi_class="ovr", labels=codigos)
            except ValueError:
                indices["AUC"] = np.nan
        indices["F1 macro"] = f1_score(y_real, y_pred, average="macro", zero_division=0)
        return {k: round(float(v), 4) for k, v in indices.items()}

    def validacion_cruzada(self, nombre=None, k=5):
        """Validación cruzada estratificada (sobre entrenamiento) de un experimento ya corrido."""
        pipe = clone(self.__experimentos[nombre or self.__ultimo]["pipeline"])
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.__semilla)
        metricas = {"Precisión Global": "accuracy", "F1 macro": "f1_macro"}
        if self.binaria:
            metricas.update({"F1": "f1", "AUC": "roc_auc"})
        else:
            metricas["AUC"] = "roc_auc_ovr"
        res = cross_validate(pipe, self.__X_train, self.__y_train, cv=cv, scoring=metricas, n_jobs=-1)
        tabla = pd.DataFrame({m: res[f"test_{m}"] for m in metricas})
        tabla.index = [f"Fold {i + 1}" for i in range(k)]
        tabla.loc["Promedio"] = tabla.iloc[:k].mean()
        tabla.loc["Desv. estándar"] = tabla.iloc[:k].std()
        return tabla.round(4)

    def predecir(self, nombre, nuevos: pd.DataFrame):
        faltan = [c for c in self.__predictoras if c not in nuevos.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en los datos nuevos: {faltan}")
        pipe = self.__experimentos[nombre]["pipeline"]
        return self.decodificar(pipe.predict(nuevos[self.__predictoras]))

    # =================================================================
    # Resultados
    # =================================================================
    def conteos(self, nombre=None):
        """VP, VN, FP y FN del experimento (solo binaria)."""
        (vn, fp), (fn, vp) = self.matriz_confusion(nombre).values
        return {"VP": int(vp), "VN": int(vn), "FP": int(fp), "FN": int(fn)}

    def resultados(self, algoritmo=None):
        tabla = pd.DataFrame(self.__resultados)
        if algoritmo is not None and not tabla.empty:
            tabla = tabla[tabla["Algoritmo"] == algoritmo]
        return tabla.reset_index(drop=True)

    def mejor_modelo(self, criterio="F1 macro", algoritmo=None):
        """Nombre del mejor experimento según el criterio (de un algoritmo o de todos)."""
        tabla = self.resultados(algoritmo)
        if tabla.empty or tabla[criterio].isna().all():
            return None
        return tabla.loc[tabla[criterio].idxmax(), "Modelo"]

    def mejores_por_algoritmo(self, criterio="F1 macro"):
        tabla = self.resultados()
        if tabla.empty:
            return tabla
        nombres = [self.mejor_modelo(criterio, a) for a in tabla["Algoritmo"].unique()]
        return (tabla[tabla["Modelo"].isin([n for n in nombres if n])]
                .sort_values(criterio, ascending=False).reset_index(drop=True))

    def borrar_experimento(self, nombre):
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre]
        self.__experimentos.pop(nombre, None)
        if self.__ultimo == nombre:
            self.__ultimo = self.__resultados[-1]["Modelo"] if self.__resultados else None

    def matriz_confusion(self, nombre=None):
        exp = self.__experimentos[nombre or self.__ultimo]
        codigos = list(range(len(self.__clases)))
        mc = confusion_matrix(self.__y_test, exp["y_pred"], labels=codigos)
        return pd.DataFrame(mc, index=[f"Real {c}" for c in self.__clases],
                            columns=[f"Pred {c}" for c in self.__clases])

    # =================================================================
    # Gráficos
    # =================================================================
    def fig_matriz_confusion(self, nombre=None, normalizar=False):
        nombre = nombre or self.__ultimo
        mc = self.matriz_confusion(nombre)
        z = mc.values
        total = z.sum(axis=1, keepdims=True)
        pct = np.divide(z, total, out=np.zeros_like(z, dtype=float), where=total > 0)
        texto = [[f"{100 * pct[i, j]:.1f} %" if normalizar else
                  f"{z[i, j]}<br><span style='font-size:11px'>{100 * pct[i, j]:.1f} %</span>"
                  for j in range(z.shape[1])] for i in range(z.shape[0])]
        fig = go.Figure(go.Heatmap(z=pct, x=list(mc.columns), y=list(mc.index), text=texto, texttemplate="%{text}",
                                   colorscale=g.SECUENCIAL, zmin=0, zmax=1, showscale=False,
                                   hovertemplate="%{y} / %{x}<extra></extra>"))
        fig.update_layout(height=max(340, 60 * len(mc) + 160), yaxis=dict(autorange="reversed"),
                          title=f"Matriz de confusión: {nombre}")
        return fig

    def fig_roc(self, nombres=None, titulo="Curvas ROC"):
        """Curvas ROC (solo clasificación binaria; la positiva es el código 1)."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=g.BORDE),
                                 showlegend=False))
        for i, nombre in enumerate(nombres or list(self.__experimentos)):
            exp = self.__experimentos[nombre]
            fpr, tpr, _ = roc_curve(self.__y_test, exp["proba"][:, 1], pos_label=1)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nombre} (AUC={exp['indices']['AUC']:.3f})",
                                     line=dict(color=g.color_cluster(i), width=2)))
        fig.update_layout(height=480, title=titulo, xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos",
                          legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98, font=dict(size=11)))
        return fig

    def fig_precision_por_clase(self, nombres=None, titulo="Precisión por categoría"):
        tabla = self.resultados().set_index("Modelo")
        if nombres is not None:
            tabla = tabla.loc[nombres]
        fig = go.Figure()
        for i, c in enumerate(self.__clases):
            fig.add_trace(go.Bar(name=str(c), x=tabla.index, y=tabla[f"Precisión clase {c}"],
                                 marker_color=g.color_cluster(i)))
        fig.update_layout(barmode="group", height=460, title=titulo, yaxis=dict(range=[0, 1]))
        return fig

    def fig_comparacion(self, metricas=("Precisión Global", "F1", "F1 macro", "AUC"), algoritmo=None,
                        nombres=None, titulo="Comparación de modelos"):
        tabla = self.resultados(algoritmo).set_index("Modelo")
        if nombres is not None:
            tabla = tabla.loc[nombres]
        fig = go.Figure()
        for i, m in enumerate(metricas):
            fig.add_trace(go.Bar(name=m, y=tabla.index, x=tabla[m], orientation="h",
                                 marker_color=g.color_cluster(i),
                                 hovertemplate="%{y}<br>" + m + ": %{x:.4f}<extra></extra>"))
        fig.update_layout(barmode="group", height=max(360, 30 * len(tabla) * len(metricas) / 2 + 140),
                          title=titulo, xaxis=dict(range=[0, 1]), yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=-0.12))
        return fig


# ======================================================================
class ClasificacionLab(SupervisadoLab):
    """
    Métodos de clasificación del laboratorio. Hereda de SupervisadoLab (y este de EDA).
    - KNN, Árboles de Decisión, Random Forest, XGBoost y AdaBoost
    - Cada método sin parámetros es la configuración estándar (valores por defecto
      de la librería); los parámetros extra (**parametros) van directo al estimador.
    - Opciones de preprocesamiento: escalar, imputacion y n_atributos (selección de atributos).
    """

    @staticmethod
    def nombre_config(algoritmo, parametros, escalar=None, n_atributos=None, imputacion="median"):
        # ej. "KNN (n_neighbors=15, weights=distance)" o "KNN (estándar)"
        partes = [f"{k}={v}" for k, v in parametros.items()]
        if escalar is False:
            partes.append("sin escalar")
        if escalar is True and algoritmo not in ("KNN", "SVM", "Logística", "Red neuronal"):
            partes.append("escalado")
        if imputacion != "median":
            partes.append(f"imputación {imputacion}")
        if n_atributos:
            partes.append(f"{n_atributos} atributos")
        return f"{algoritmo} ({', '.join(partes) if partes else 'estándar'})"

    # =================================================================
    # Algoritmos
    # =================================================================
    def knn(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # KNN usa distancias, por eso escala por defecto
        nombre = nombre or self.nombre_config("KNN", parametros, escalar, n_atributos, imputacion)
        modelo = KNeighborsClassifier(n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "KNN", escalar, imputacion, n_atributos, parametros)

    def arbol(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("DT", parametros, escalar or None, n_atributos, imputacion)
        modelo = DecisionTreeClassifier(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "DT", escalar, imputacion, n_atributos, parametros)

    def bosque(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("RF", parametros, escalar or None, n_atributos, imputacion)
        modelo = RandomForestClassifier(random_state=self.semilla, n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "RF", escalar, imputacion, n_atributos, parametros)

    def xgboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        if not HAY_XGB:
            raise ImportError("Falta la librería xgboost (pip install xgboost).")
        if parametros.get("scale_pos_weight") == "ratio":
            parametros["scale_pos_weight"] = self.relacion_negativos_positivos()
        nombre = nombre or self.nombre_config("XGBoost", parametros, escalar or None, n_atributos, imputacion)
        modelo = XGBClassifier(random_state=self.semilla, n_jobs=-1, verbosity=0,
                               eval_metric="logloss" if self.binaria else "mlogloss", **parametros)
        return self.entrenar(nombre, modelo, "XGBoost", escalar, imputacion, n_atributos, parametros)

    def adaboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None,
                 profundidad_base=1, **parametros):
        # el clasificador base es un árbol; profundidad 1 (stump) es el estándar
        etiqueta = dict(parametros, profundidad_base=profundidad_base) if profundidad_base != 1 else parametros
        nombre = nombre or self.nombre_config("AdaBoost", etiqueta, escalar or None, n_atributos, imputacion)
        base = DecisionTreeClassifier(max_depth=profundidad_base, random_state=self.semilla)
        modelo = AdaBoostClassifier(estimator=base, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "AdaBoost", escalar, imputacion, n_atributos, etiqueta)

    def gradient_boosting(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Gradient Boosting", parametros, escalar or None, n_atributos, imputacion)
        modelo = GradientBoostingClassifier(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "Gradient Boosting", escalar, imputacion, n_atributos, parametros)

    def svm(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # SVM usa distancias, por eso escala por defecto; probability=True para la curva ROC
        nombre = nombre or self.nombre_config("SVM", parametros, escalar, n_atributos, imputacion)
        modelo = SVC(probability=True, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "SVM", escalar, imputacion, n_atributos, parametros)

    def bayes(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Bayes", parametros, escalar or None, n_atributos, imputacion)
        return self.entrenar(nombre, GaussianNB(**parametros), "Bayes", escalar, imputacion, n_atributos, parametros)

    def logistica(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("Logística", parametros, escalar, n_atributos, imputacion)
        modelo = LogisticRegression(max_iter=2000, **parametros)
        return self.entrenar(nombre, modelo, "Logística", escalar, imputacion, n_atributos, parametros)

    def lda(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("LDA", parametros, escalar or None, n_atributos, imputacion)
        modelo = LinearDiscriminantAnalysis(**parametros)
        return self.entrenar(nombre, modelo, "LDA", escalar, imputacion, n_atributos, parametros)

    def qda(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("QDA", parametros, escalar or None, n_atributos, imputacion)
        modelo = QuadraticDiscriminantAnalysis(**parametros)
        return self.entrenar(nombre, modelo, "QDA", escalar, imputacion, n_atributos, parametros)

    def red_neuronal(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # hidden_layer_sizes puede venir como texto "32,16"
        estimador = dict(parametros)
        capas = estimador.get("hidden_layer_sizes")
        if isinstance(capas, str):
            capas = tuple(int(x) for x in capas.replace(" ", "").split(",") if x)
            if not capas:
                raise ValueError("Indique al menos una capa, ej. 32,16")
            estimador["hidden_layer_sizes"] = capas
        nombre = nombre or self.nombre_config("Red neuronal", parametros, escalar, n_atributos, imputacion)
        modelo = MLPClassifier(random_state=self.semilla, **estimador)
        return self.entrenar(nombre, modelo, "Red neuronal", escalar, imputacion, n_atributos, parametros)

    def ejecutar(self, algoritmo, configuracion=None):
        """Corre una configuración ({parámetros + opciones de preprocesamiento}) de un algoritmo."""
        metodo = getattr(self, ALGORITMOS[algoritmo]["metodo"])
        return metodo(**dict(configuracion or {}))

    def comparar(self, algoritmo, configuraciones):
        """
        Corre varias configuraciones de un mismo algoritmo y retorna sus índices.
        algoritmo: una clave de ALGORITMOS ('KNN', 'DT', 'RF', 'XGBoost', 'AdaBoost', 'SVM', ...)
        configuraciones: lista de dicts, ej. [{}, {"n_neighbors": 15}, {"escalar": False}]
        """
        filas = [self.ejecutar(algoritmo, conf) for conf in configuraciones]
        return pd.DataFrame(filas)

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

    def importancia_variables(self, nombre=None):
        """
        Importancia de variables: la propia del modelo si la tiene (árboles y potenciación);
        si no (KNN, SVM, ...), por permutación: cuánto baja el acierto al desordenar cada variable.
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
            r = permutation_importance(pipe, self.X_test.iloc[:muestra], self.y_test.iloc[:muestra],
                                       n_repeats=5, random_state=self.semilla, n_jobs=-1)
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

    def reglas_arbol(self, nombre=None, profundidad=3):
        pipe = self.experimentos[nombre or self.ultimo]["pipeline"]
        return export_text(pipe.named_steps["modelo"], feature_names=self.nombres_atributos(nombre),
                           class_names=[str(c) for c in self.clases], max_depth=profundidad)

    def fig_arbol(self, nombre=None, profundidad=3):
        """Primeros niveles del árbol dibujados con Plotly (nodos con regla, muestras y clase)."""
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
        textos, colores, xs, ys = [], [], [], []
        for n, x, y in nodos:
            valores = arbol.value[n][0]
            clase = int(np.argmax(valores))
            hoja = arbol.children_left[n] == -1
            regla = "hoja" if hoja else f"{atributos[arbol.feature[n]]} ≤ {arbol.threshold[n]:.3f}"
            textos.append(f"<b>{regla}</b><br>n = {arbol.n_node_samples[n]}<br>clase: {self.clases[clase]}"
                          f"<br>pureza: {valores.max() / valores.sum():.2f}")
            colores.append(g.color_cluster(clase))
            xs.append(x)
            ys.append(y)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=textos, textposition="bottom center",
                                 textfont=dict(size=10), hovertext=textos, hoverinfo="text",
                                 marker=dict(size=16, color=colores, line=dict(color="#FFFFFF", width=1.5)),
                                 showlegend=False))
        fig.update_layout(height=170 * (profundidad + 1) + 80, title=f"Primeros {profundidad} niveles del árbol",
                          xaxis=dict(visible=False, range=[-0.02, 1.02]),
                          yaxis=dict(visible=False, range=[-profundidad - 0.6, 0.3]),
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    def curva_k(self, valores_k=range(1, 52, 5), escalar=True, **parametros):
        """Índices según la cantidad de vecinos (no se guarda en el historial)."""
        filas = []
        for k in valores_k:
            pipe = self.crear_pipeline(KNeighborsClassifier(n_neighbors=k, n_jobs=-1, **parametros), escalar)
            pipe.fit(self.X_train, self.y_train)
            proba = pipe.predict_proba(self.X_test)
            ind = self.indices_calidad(self.y_test, pipe.classes_[proba.argmax(axis=1)], proba, pipe.classes_)
            filas.append({"k": k, **{m: ind[m] for m in ["Precisión Global", "F1", "F1 macro", "AUC"]}})
        return pd.DataFrame(filas)

    @staticmethod
    def fig_curva_k(tabla):
        fig = go.Figure()
        for i, m in enumerate(["Precisión Global", "F1", "F1 macro", "AUC"]):
            fig.add_trace(go.Scatter(x=tabla["k"], y=tabla[m], mode="lines+markers", name=m,
                                     line=dict(color=g.color_cluster(i), width=2)))
        fig.update_layout(height=420, title="KNN: índices según k", xaxis_title="k (vecinos)", yaxis_title="Índice")
        return fig
