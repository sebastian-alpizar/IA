"""
algoritmos.py (clasificación)
Configuración de los algoritmos de clasificación: los 5 del Laboratorio 2 (KNN, Árboles de
Decisión, Random Forest, XGBoost y AdaBoost) y los demás del curso. La interfaz arma los
controles a partir de "params" y el botón de variaciones corre "variaciones".
"""

from importlib.util import find_spec

from ..supervisado import parametro as _p

HAY_XGB = find_spec("xgboost") is not None    # XGBoost es opcional

# ----------------------------------------------------------------------
# Algoritmos
#   lab: True para los cinco que pide el Laboratorio 2
#   metodo: nombre del método de Clasificacion
#   escalar: valor por defecto (KNN usa distancias, por eso escala)
#   params: (nombre, etiqueta, tipo, default de la librería, extra)
#           tipo "int"/"float" -> extra = (min, max, paso); "select" -> extra = opciones; "text" -> texto libre
#           un default None significa "sin límite" / "no usar" (se muestra como 0 o "ninguno")
#   variaciones: configuración estándar + variaciones (en los del Lab 2, las del notebook)
# ----------------------------------------------------------------------
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
CRITERIOS = ["F1 macro", "AUC", "Precisión Global", "F1", "Recall", "Precision (VPP)"]
