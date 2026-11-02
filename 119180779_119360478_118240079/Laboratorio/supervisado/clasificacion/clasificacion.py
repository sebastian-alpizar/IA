import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from supervisado import Supervisado   # clase padre, carpeta supervisado/


class Clasificacion(Supervisado):
    """
    Métodos de clasificación del framework. Hereda de Supervisado (y este de EDA).
    - KNN, Árboles de Decisión, Random Forest, XGBoost y AdaBoost
    - Cada método sin parámetros es la configuración estándar (valores por defecto
      de la librería); los parámetros extra (**parametros) van directo al estimador,
      así se prueban las variaciones de configuración.
    - Opciones de preprocesamiento: escalar, imputacion y n_atributos (selección de atributos).
    - nombre_config, comparar, nombres_atributos e importancia_variables se heredan de Supervisado
      (los comparte con Regresion).
    """

    def __init__(self, dataframe: pd.DataFrame, variable_predecir, test_size=0.25,
                 eliminar_duplicados=True, semilla=42, excluir=None):
        super().__init__(dataframe, variable_predecir, test_size, eliminar_duplicados, semilla, excluir)

    # =================================================================
    # Algoritmos
    # =================================================================
    def knn(self, nombre=None, escalar=True, imputacion="median", n_atributos=None, **parametros):
        # KNN usa distancias, por eso escala por defecto
        nombre = nombre or self.nombre_config("KNN", parametros, escalar, n_atributos, imputacion)
        modelo = KNeighborsClassifier(n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "KNN", escalar, imputacion, n_atributos)

    def arbol(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("DT", parametros, None, n_atributos, imputacion)
        modelo = DecisionTreeClassifier(random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "DT", escalar, imputacion, n_atributos)

    def bosque(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("RF", parametros, None, n_atributos, imputacion)
        modelo = RandomForestClassifier(random_state=self.semilla, n_jobs=-1, **parametros)
        return self.entrenar(nombre, modelo, "RF", escalar, imputacion, n_atributos)

    def xgboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None, **parametros):
        nombre = nombre or self.nombre_config("XGBoost", parametros, None, n_atributos, imputacion)
        modelo = XGBClassifier(random_state=self.semilla, n_jobs=-1, eval_metric="logloss", **parametros)
        return self.entrenar(nombre, modelo, "XGBoost", escalar, imputacion, n_atributos)

    def adaboost(self, nombre=None, escalar=False, imputacion="median", n_atributos=None,
                 profundidad_base=1, **parametros):
        # el clasificador base es un árbol; profundidad 1 (stump) es el estándar
        etiqueta = dict(parametros, profundidad_base=profundidad_base) if profundidad_base != 1 else parametros
        nombre = nombre or self.nombre_config("AdaBoost", etiqueta, None, n_atributos, imputacion)
        base = DecisionTreeClassifier(max_depth=profundidad_base, random_state=self.semilla)
        modelo = AdaBoostClassifier(estimator=base, random_state=self.semilla, **parametros)
        return self.entrenar(nombre, modelo, "AdaBoost", escalar, imputacion, n_atributos)

    # =================================================================
    # Análisis del modelo
    # =================================================================
    def graficar_arbol(self, nombre=None, profundidad=3):
        pipe = (self.experimentos[nombre] if nombre else self.ultimo)["pipeline"]
        modelo = pipe.named_steps["modelo"]
        plt.figure(figsize=(20, 9))
        plot_tree(modelo, max_depth=profundidad, feature_names=self.nombres_atributos(nombre),
                  class_names=[str(c) for c in modelo.classes_], filled=True, rounded=True, fontsize=8)
        plt.title(f"Primeros {profundidad} niveles del árbol: {nombre or 'último modelo'}")

    def curva_k(self, valores_k=range(1, 52, 5), escalar=True, **parametros):
        # precisión, F1 y AUC según la cantidad de vecinos (sin guardar en el historial)
        filas = []
        for k in valores_k:
            pipe = self.crear_pipeline(KNeighborsClassifier(n_neighbors=k, n_jobs=-1, **parametros), escalar)
            pipe.fit(self.X_train, self.y_train)
            proba = pipe.predict_proba(self.X_test)
            y_pred = pipe.classes_[proba.argmax(axis=1)]
            filas.append({"k": k, "Precisión Global": accuracy_score(self.y_test, y_pred),
                          "F1": f1_score(self.y_test, y_pred, pos_label=pipe.classes_[1]),
                          "AUC": roc_auc_score(self.y_test, proba[:, 1])})
        tabla = pd.DataFrame(filas).round(4)
        plt.figure(figsize=(8, 4))
        for m in ["Precisión Global", "F1", "AUC"]:
            plt.plot(tabla["k"], tabla[m], marker="o", label=m)
        plt.xlabel("k (vecinos)")
        plt.title("KNN: índices según k")
        plt.legend()
        return tabla
