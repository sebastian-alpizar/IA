"""
supervisado.py
Parte común de los métodos supervisados. Hereda de EDA:

    EDA -> Supervisado(EDA) -> Clasificacion(Supervisado)   (carpeta clasificacion/, Laboratorio 2)
                            -> Regresion(Supervisado)       (carpeta regresion/, Laboratorio 3)

- Separa predictoras / variable a predecir y hace la partición entrenamiento-prueba
- Preprocesamiento dentro de un Pipeline (imputación, escalado, one-hot y selección de
  atributos), para que no haya fuga de información del conjunto de prueba
- Entrena, evalúa y guarda el historial de experimentos; escoge el mejor modelo por
  algoritmo (o por familia); validación cruzada; importancia de variables; predicción
Lo que cambia entre clasificación y regresión (codificar la respuesta, estratificar,
los índices de calidad, la validación cruzada...) está en métodos "gancho" (_preparar_y,
_estratos, _evaluar, _metricas_cv, ...) que cada clase hija redefine.
"""

import time

import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import EDA               # clase base, carpeta eda/
from shared import graficos as g


def parametro(nombre, etiqueta, tipo, default, extra, ayuda=""):
    """
    Describe un parámetro de un algoritmo para armar su control en la interfaz.
    tipo "int"/"float" -> extra = (min, max, paso); "select" -> extra = opciones; "text" -> texto libre.
    Un default None significa "sin límite" / "no usar" (se muestra como 0 o "ninguno").
    """
    return {"nombre": nombre, "etiqueta": etiqueta, "tipo": tipo, "default": default, "extra": extra, "ayuda": ayuda}


class Supervisado(EDA):
    """
    Métodos supervisados del framework. Hereda de EDA, así que también tiene todos los
    métodos de análisis exploratorio. Las clases hijas definen:
      ALGORITMOS      nombre -> configuración del algoritmo (en su algoritmos.py)
      CRITERIOS       índices con los que se puede escoger el mejor modelo
      MENOR_ES_MEJOR  índices de error (el mejor es el de menor valor)
    y los métodos gancho _preparar_y, _estratos, _puntaje_seleccion, _evaluar,
    _particiones_cv y _metricas_cv.
    """

    ALGORITMOS = {}
    CRITERIOS = []
    MENOR_ES_MEJOR = ()
    SCORING_IMPORTANCIA = None    # índice de la importancia por permutación (None = el del estimador)

    def __init__(self, datos: pd.DataFrame, variable_predecir, predictoras=None, test_size=0.25,
                 eliminar_duplicados=True, semilla=42):
        datos = datos.dropna(subset=[variable_predecir])
        super().__init__(datos)
        self.__variable_predecir = variable_predecir
        self.__predictoras = list(predictoras or [c for c in datos.columns if c != variable_predecir])
        self.__semilla = int(semilla)
        self.__resultados = []      # una fila de índices por experimento
        self.__experimentos = {}    # nombre -> pipeline, predicciones, índices y configuración
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
    # Ganchos (los redefine cada clase hija)
    # =================================================================
    def _preparar_y(self, y):
        """Transforma la variable a predecir antes de particionar (codificar clases, pasar a float...)."""
        return y

    def _estratos(self, y):
        """Variable para estratificar la partición (None = partición aleatoria simple)."""
        return None

    def _puntaje_seleccion(self):
        """Estadístico de SelectKBest (F del ANOVA en clasificación, F de la regresión en regresión)."""
        raise NotImplementedError

    def _evaluar(self, pipe):
        """Evalúa el pipeline en el conjunto de prueba: retorna (índices, datos extra del experimento)."""
        raise NotImplementedError

    def _particiones_cv(self, k):
        raise NotImplementedError

    def _metricas_cv(self):
        """Índices de la validación cruzada: nombre -> (scorer de sklearn, signo para la escala natural)."""
        raise NotImplementedError

    def _columnas_extra(self, algoritmo):
        """Columnas adicionales del historial (ej. la familia del algoritmo en regresión)."""
        return {}

    def _decodificar_prediccion(self, prediccion):
        return prediccion

    # =================================================================
    # Preparación de los datos
    # =================================================================
    def particionar(self, test_size=0.25, eliminar_duplicados=True):
        """Separa X / y y hace la partición entrenamiento-prueba. Borra el historial."""
        datos = self.datos[self.__predictoras + [self.__variable_predecir]]
        if eliminar_duplicados:
            datos = datos.drop_duplicates()
        y = self._preparar_y(datos[self.__variable_predecir])
        X = datos[self.__predictoras]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=test_size, stratify=self._estratos(y), random_state=self.__semilla)
        self.__resultados, self.__experimentos, self.__ultimo = [], {}, None

    def preprocesador(self, escalar=False, imputacion="median", columnas=None):
        # numéricas: imputación (+ escalado); categóricas: moda + one-hot
        # columnas: usar solo esas predictoras (ej. regresión lineal simple); None = todas
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
            pasos.append(("seleccion", SelectKBest(self._puntaje_seleccion(), k=n_atributos)))
        pasos.append(("modelo", estimador))
        return Pipeline(pasos)

    # =================================================================
    # Entrenamiento y evaluación
    # =================================================================
    @classmethod
    def nombre_config(cls, algoritmo, parametros, escalar=None, n_atributos=None, imputacion="median"):
        # ej. "KNN (n_neighbors=15, weights=distance)" o "Ridge (estándar)"
        partes = [f"{k}={v}" for k, v in parametros.items()]
        if escalar is False:
            partes.append("sin escalar")
        if escalar is True and not cls.ALGORITMOS.get(algoritmo, {}).get("escalar"):
            partes.append("escalado")
        if imputacion != "median":
            partes.append(f"imputación {imputacion}")
        if n_atributos:
            partes.append(f"{n_atributos} atributos")
        return f"{algoritmo} ({', '.join(partes) if partes else 'estándar'})"

    def entrenar(self, nombre, estimador, algoritmo=None, escalar=False, imputacion="median",
                 n_atributos=None, config=None, columnas=None):
        """
        Entrena el estimador (dentro del pipeline) con el conjunto de entrenamiento,
        lo evalúa en el de prueba y lo guarda en el historial. Retorna los índices de calidad.
        Si se repite el nombre se reemplaza el experimento anterior.
        """
        if n_atributos:
            total = len(self.preprocesador(columnas=columnas).fit(self.__X_train).get_feature_names_out())
            n_atributos = min(int(n_atributos), total)
        inicio = time.time()
        pipe = self.crear_pipeline(estimador, escalar, imputacion, n_atributos, columnas)
        pipe.fit(self.__X_train, self.__y_train)
        indices, extra = self._evaluar(pipe)
        indices["Tiempo (s)"] = round(time.time() - inicio, 2)
        algoritmo = algoritmo or type(estimador).__name__
        fila = {"Modelo": nombre, "Algoritmo": algoritmo, **self._columnas_extra(algoritmo), **indices}
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre] + [fila]
        self.__experimentos[nombre] = {"pipeline": pipe, "algoritmo": algoritmo, "indices": indices,
                                       "config": dict(config or {}, escalar=escalar, imputacion=imputacion,
                                                      n_atributos=n_atributos), **extra}
        self.__ultimo = nombre
        return pd.Series(indices, name=nombre)

    def ejecutar(self, algoritmo, configuracion=None):
        """Corre una configuración ({parámetros + opciones de preprocesamiento}) de un algoritmo."""
        metodo = getattr(self, self.ALGORITMOS[algoritmo]["metodo"])
        return metodo(**dict(configuracion or {}))

    def comparar(self, algoritmo, configuraciones):
        """
        Corre varias configuraciones de un mismo algoritmo y retorna sus índices.
        configuraciones: lista de dicts, ej. [{}, {"n_neighbors": 15}, {"escalar": False}]
        """
        return pd.DataFrame([self.ejecutar(algoritmo, conf) for conf in configuraciones])

    def validacion_cruzada(self, nombre=None, k=5):
        """Validación cruzada (sobre entrenamiento) de un experimento ya corrido."""
        pipe = clone(self.__experimentos[nombre or self.__ultimo]["pipeline"])
        metricas = self._metricas_cv()
        res = cross_validate(pipe, self.__X_train, self.__y_train, cv=self._particiones_cv(k), n_jobs=-1,
                             scoring={m: s for m, (s, _) in metricas.items()})
        tabla = pd.DataFrame({m: res[f"test_{m}"] * signo for m, (_, signo) in metricas.items()})
        tabla.index = [f"Fold {i + 1}" for i in range(k)]
        tabla.loc["Promedio"] = tabla.iloc[:k].mean()
        tabla.loc["Desv. estándar"] = tabla.iloc[:k].std()
        return tabla.round(4)

    def predecir(self, nombre, nuevos: pd.DataFrame):
        faltan = [c for c in self.__predictoras if c not in nuevos.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en los datos nuevos: {faltan}")
        pipe = self.__experimentos[nombre]["pipeline"]
        return self._decodificar_prediccion(pipe.predict(nuevos[self.__predictoras]))

    # =================================================================
    # Resultados
    # =================================================================
    def resultados(self, algoritmo=None, familia=None):
        tabla = pd.DataFrame(self.__resultados)
        if tabla.empty:
            return tabla
        if algoritmo is not None:
            tabla = tabla[tabla["Algoritmo"] == algoritmo]
        if familia is not None and "Familia" in tabla:
            tabla = tabla[tabla["Familia"] == familia]
        return tabla.reset_index(drop=True)

    def mejor_modelo(self, criterio=None, algoritmo=None, familia=None):
        """Nombre del mejor experimento según el criterio (de un algoritmo, de una familia o de todos)."""
        criterio = criterio or self.CRITERIOS[0]
        tabla = self.resultados(algoritmo, familia)
        if tabla.empty or tabla[criterio].isna().all():
            return None
        fila = tabla[criterio].idxmin() if criterio in self.MENOR_ES_MEJOR else tabla[criterio].idxmax()
        return tabla.loc[fila, "Modelo"]

    def mejores(self, criterio=None, por="Algoritmo"):
        """Mejor modelo de cada algoritmo (por='Algoritmo') o de cada familia (por='Familia')."""
        criterio = criterio or self.CRITERIOS[0]
        tabla = self.resultados()
        if tabla.empty:
            return tabla
        filtro = {"Algoritmo": "algoritmo", "Familia": "familia"}[por]
        nombres = [self.mejor_modelo(criterio, **{filtro: v}) for v in tabla[por].unique()]
        return (tabla[tabla["Modelo"].isin([n for n in nombres if n])]
                .sort_values(criterio, ascending=criterio in self.MENOR_ES_MEJOR, kind="stable")
                .reset_index(drop=True))

    def borrar_experimento(self, nombre):
        self.__resultados = [r for r in self.__resultados if r["Modelo"] != nombre]
        self.__experimentos.pop(nombre, None)
        if self.__ultimo == nombre:
            self.__ultimo = self.__resultados[-1]["Modelo"] if self.__resultados else None

    # =================================================================
    # Análisis del modelo
    # =================================================================
    def nombres_atributos(self, nombre=None):
        # nombres de las variables después del one-hot y la selección de atributos
        pipe = self.__experimentos[nombre or self.__ultimo]["pipeline"]
        nombres = pipe.named_steps["prep"].get_feature_names_out()
        if "seleccion" in pipe.named_steps:
            nombres = pipe.named_steps["seleccion"].get_feature_names_out(nombres)
        return list(nombres)

    def importancia_variables(self, nombre=None):
        """
        Importancia de variables: la propia del modelo si la tiene (árboles y potenciación);
        si no, por permutación: cuánto empeora el modelo en prueba al desordenar cada variable.
        """
        exp = self.__experimentos[nombre or self.__ultimo]
        if "importancia" in exp:
            return exp["importancia"]
        pipe = exp["pipeline"]
        modelo = pipe.named_steps["modelo"]
        if hasattr(modelo, "feature_importances_"):
            imp = pd.Series(modelo.feature_importances_, index=self.nombres_atributos(nombre))
        else:
            muestra = min(len(self.__X_test), 1500)
            r = permutation_importance(pipe, self.__X_test.iloc[:muestra], self.__y_test.iloc[:muestra],
                                       n_repeats=5, scoring=self.SCORING_IMPORTANCIA,
                                       random_state=self.__semilla, n_jobs=-1)
            imp = pd.Series(r.importances_mean, index=self.__X_test.columns)
        exp["importancia"] = imp.rename("Importancia").sort_values(ascending=False).round(4)
        return exp["importancia"]

    def importancia_propia(self, nombre=None):
        modelo = self.__experimentos[nombre or self.__ultimo]["pipeline"].named_steps["modelo"]
        return hasattr(modelo, "feature_importances_")

    def fig_importancia(self, nombre=None, top=15):
        imp = self.importancia_variables(nombre).head(top).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=g.AZUL,
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(height=max(320, 26 * len(imp) + 100),
                          title=f"Importancia de variables: {nombre or self.__ultimo}")
        return fig
