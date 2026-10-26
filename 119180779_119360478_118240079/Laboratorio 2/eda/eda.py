import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    """
    Clase base del framework: encapsula un pandas.DataFrame
    y expone los métodos de análisis exploratorio (EDA).
    NoSupervisado (y Supervisado) heredan de esta clase.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.__df = dataframe.copy()

    def __str__(self):
        return str(self.__df.info())

    # --- Métodos básicos ---
    def mostrar(self):
        return self.__df

    def primeras_filas(self, n=10):
        return self.__df.head(n)

    def ultimas_filas(self, n=10):
        return self.__df.tail(n)

    def obtener_fila(self, indice):
        return self.__df.iloc[indice]

    def obtener_columna(self, nombre):
        return self.__df[nombre]

    def dimensiones(self):
        return self.__df.shape

    def nombres_columnas(self):
        return list(self.__df.columns)

    def tipos_datos(self):
        return self.__df.dtypes

    # --- Limpieza ---
    def contar_nulos(self):
        return self.__df.isnull().sum()

    def eliminar_nulos(self):
        return self.__df.dropna()

    def reemplazar_nulos(self, valor=0):
        return self.__df.fillna(valor)

    def detectar_duplicados(self):
        return self.__df[self.__df.duplicated()]

    def eliminar_duplicados(self):
        return self.__df.drop_duplicates()

    # --- Estadísticas y correlación ---
    def estadisticas(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.describe()

    def correlacion(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.corr()

    # --- Visualizaciones ---
    def histogramas(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        datos_num.hist(figsize=(10, 8))
        plt.show()

    def boxplots(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        datos_num.plot(kind="box", figsize=(10, 6))
        plt.show()
        
    def scatterplots(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        sns.pairplot(datos_num)
        plt.show()
        
    def mapa_calor(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(datos_num.corr(), annot=True, cmap="coolwarm")
        plt.show()

    # --- Transformaciones ---
    def normalizar(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return (datos_num - datos_num.min()) / (datos_num.max() - datos_num.min())
        
    def estandarizar(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return (datos_num - datos_num.mean()) / datos_num.std()
        
    def imputar_nulos(self):
        datos_num = self.__df.select_dtypes(include=["float64", "int64"])
        return datos_num.fillna(datos_num.mean())

    # --- Selección y exportación ---
    def seleccionar_variables(self, columnas):
        return self.__df[columnas]

    def guardar_csv(self, ruta="dataframe_guardado.csv"):
        self.__df.to_csv(ruta, index=False)
        return ruta

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        self.__df.describe().to_csv(ruta)
        return ruta

    def obtener_datos_numericos(self):
        return self.__df.select_dtypes(include=["float64", "int64"])

    # --- Métodos adicionales ---
    def ordenar(self, columna):
        return self.__df.sort_values(by=columna)

    def filtrar(self, columna, operador, valor):
        if operador == "==":
            return self.__df[self.__df[columna] == valor]
        elif operador == "!=":
            return self.__df[self.__df[columna] != valor]
        elif operador == ">":
            return self.__df[self.__df[columna] > valor]
        elif operador == ">=":
            return self.__df[self.__df[columna] >= valor]
        elif operador == "<":
            return self.__df[self.__df[columna] < valor]
        elif operador == "<=":
            return self.__df[self.__df[columna] <= valor]

    def agregar_columna(self, nombre, valor):
        self.__df[nombre] = valor
        return self.__df

    def eliminar_columna(self, nombre):
        self.__df = self.__df.drop(columns=[nombre])
        return self.__df

    def visualizar_datos(self):
        print(self.__df.head(10))
        print("\nDimensiones:", self.__df.shape)

    def frecuencias(self):
        freqs = {}
        for col in self.__df.columns:
            freqs[col] = self.__df[col].value_counts().to_frame(name="frecuencia")
        return freqs

    def convertir_tipos(self, conversiones):
        return self.__df.astype(conversiones)

    def distribucion_variables(self):
        return {
            "numericas": self.__df.select_dtypes(include=["float64", "int64"]).describe(),
            "categoricas": self.__df.select_dtypes(include=["object"]).describe()
        }

    def detectar_outliers(self):
        outliers = {}
        for col in self.__df.select_dtypes(include=["float64", "int64"]).columns:
            q1 = self.__df[col].quantile(0.25)
            q3 = self.__df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.__df[col] < (q1 - 1.5 * iqr)) | (self.__df[col] > (q3 + 1.5 * iqr))
            outliers[col] = self.__df[mask]
        return outliers
