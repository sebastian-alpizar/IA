"""
algoritmos.py (regresión)
Configuración de los algoritmos de regresión del Laboratorio 3 por familia (lineales,
núcleos y árboles) y los demás del curso (Gradient Boosting y KNN), los índices de calidad
y los criterios para escoger el mejor modelo.
"""

from importlib.util import find_spec

from ..supervisado import parametro as _p

HAY_XGB = find_spec("xgboost") is not None    # XGBoost es opcional


# ----------------------------------------------------------------------
# Algoritmos (misma estructura que en clasificacion/algoritmos.py)
#   lab: True para los que pide el Laboratorio 3
#   familia: lineal / núcleos / árboles (para comparar dentro y entre familias)
#   metodo: nombre del método de Regresion
#   escalar: valor por defecto (penalizados, SVR y KNN dependen de la escala)
#   params: (nombre, etiqueta, tipo, default de la librería, extra)
#   variaciones: configuración estándar + variaciones (las del notebook del Lab 3)
# ----------------------------------------------------------------------
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
