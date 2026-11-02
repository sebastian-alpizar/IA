"""
clasificacion.py
Métodos de clasificación (Laboratorio 2 + resto del curso). Hereda de Supervisado (y este de EDA):

    EDA -> Supervisado(EDA) -> Clasificacion(Supervisado)

- KNN, Árboles de Decisión, Random Forest, XGBoost y AdaBoost (Laboratorio 2) y además
  Gradient Boosting, SVM, Bayes, regresión logística, LDA, QDA y red neuronal
- Cada método sin parámetros es la configuración estándar (valores por defecto de la librería);
  los parámetros extra (**parametros) van directo al estimador, así se prueban las variaciones.
- Partición estratificada; las clases se codifican 0..n-1 internamente (XGBoost lo exige) y en
  binaria la clase positiva siempre queda como código 1.
Todo devuelve tablas o figuras de Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from shared import graficos as g
from ..supervisado import Supervisado          # clase padre, carpeta supervisado/
from .algoritmos import ALGORITMOS, CRITERIOS, HAY_XGB

if HAY_XGB:
    from xgboost import XGBClassifier


class Clasificacion(Supervisado):
    """
    Métodos de clasificación. Hereda de Supervisado (y este de EDA).
    Opciones de preprocesamiento de cada algoritmo: escalar, imputacion y n_atributos
    (selección de atributos con el F del ANOVA).
    """

    ALGORITMOS = ALGORITMOS
    CRITERIOS = CRITERIOS
    MENOR_ES_MEJOR = ()

    def __init__(self, datos: pd.DataFrame, variable_predecir, predictoras=None, positiva=None,
                 test_size=0.25, eliminar_duplicados=True, semilla=42):
        # las clases se definen antes de particionar (la partición las codifica)
        clases = sorted(datos.dropna(subset=[variable_predecir])[variable_predecir].astype(str).unique())
        if len(clases) < 2:
            raise ValueError("La variable a predecir necesita al menos dos clases.")
        if len(clases) == 2:
            positiva = str(positiva) if str(positiva) in clases else clases[1]
            clases = [c for c in clases if c != positiva] + [positiva]
        self.__clases = clases
        super().__init__(datos, variable_predecir, predictoras, test_size, eliminar_duplicados, semilla)

    # --- Propiedades ---
    @property
    def clases(self):
        return self.__clases

    @property
    def binaria(self):
        return len(self.__clases) == 2

    @property
    def positiva(self):
        return self.__clases[1] if self.binaria else None

    # =================================================================
    # Ganchos de Supervisado para clasificación
    # =================================================================
    def _preparar_y(self, y):
        return self._codificar(y)

    def _estratos(self, y):
        # partición estratificada (si alguna clase tiene menos de 2 casos no se puede)
        return y if y.value_counts().min() >= 2 else None

    def _puntaje_seleccion(self):
        return f_classif

    def _evaluar(self, pipe):
        proba = pipe.predict_proba(self.X_test)
        y_pred = pipe.classes_[proba.argmax(axis=1)]
        indices = self.indices_calidad(self.y_test, y_pred, proba, pipe.classes_)
        return indices, {"y_pred": y_pred, "proba": proba}

    def _particiones_cv(self, k):
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=self.semilla)

    def _metricas_cv(self):
        metricas = {"Precisión Global": ("accuracy", 1), "F1 macro": ("f1_macro", 1)}
        if self.binaria:
            metricas.update({"F1": ("f1", 1), "AUC": ("roc_auc", 1)})
        else:
            metricas["AUC"] = ("roc_auc_ovr", 1)
        return metricas

    def _decodificar_prediccion(self, prediccion):
        return self.decodificar(prediccion)

    # =================================================================
    # Preparación de los datos
    # =================================================================

    def _codificar(self, y):
        return y.astype(str).map({c: i for i, c in enumerate(self.__clases)}).astype(int)

    def decodificar(self, codigos):
        return np.asarray(self.__clases, dtype=object)[np.asarray(codigos, dtype=int)]

    def resumen_particion(self):
        filas = {"Conjunto": ["Entrenamiento", "Prueba"],
                 "Filas": [len(self.X_train), len(self.X_test)],
                 "Predictoras": [self.X_train.shape[1]] * 2}
        for i, c in enumerate(self.__clases):
            filas[f"% clase {c}"] = [round(self.y_train.eq(i).mean() * 100, 2),
                                     round(self.y_test.eq(i).mean() * 100, 2)]
        return pd.DataFrame(filas).set_index("Conjunto")

    def distribucion_clases(self):
        y = self.datos[self.variable_predecir].astype(str)
        return pd.DataFrame({"Cantidad": y.value_counts(),
                             "Porcentaje": (y.value_counts(normalize=True) * 100).round(2)})

    def relacion_negativos_positivos(self):
        # negativos por cada positivo en entrenamiento (sirve para scale_pos_weight)
        if not self.binaria:
            return 1.0
        return round(float((self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)), 2)

    # =================================================================
    # Índices de calidad
    # =================================================================

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

    def conteos(self, nombre=None):
        """VP, VN, FP y FN del experimento (solo binaria)."""
        (vn, fp), (fn, vp) = self.matriz_confusion(nombre).values
        return {"VP": int(vp), "VN": int(vn), "FP": int(fp), "FN": int(fn)}

    def matriz_confusion(self, nombre=None):
        exp = self.experimentos[nombre or self.ultimo]
        codigos = list(range(len(self.__clases)))
        mc = confusion_matrix(self.y_test, exp["y_pred"], labels=codigos)
        return pd.DataFrame(mc, index=[f"Real {c}" for c in self.__clases],
                            columns=[f"Pred {c}" for c in self.__clases])

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

    # =================================================================
    # Análisis del modelo
    # =================================================================

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

    # =================================================================
    # Gráficos
    # =================================================================

    def fig_matriz_confusion(self, nombre=None, normalizar=False):
        nombre = nombre or self.ultimo
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
        for i, nombre in enumerate(nombres or list(self.experimentos)):
            exp = self.experimentos[nombre]
            fpr, tpr, _ = roc_curve(self.y_test, exp["proba"][:, 1], pos_label=1)
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
