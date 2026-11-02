import pandas as pd
from no_supervisado import NoSupervisado   # carpeta no_supervisado/
from clasificacion import Clasificacion    # carpeta clasificacion/
from regresion import Regresion            # carpeta regresion/

class DataFrameModel:
    """
    Modelo que encapsula la lógica de negocio.
    Se conecta con NoSupervisado (que hereda de EDA) y expone métodos para el Controller.
    Si se indica variable_predecir se conecta con Clasificacion o, con tipo="regresion",
    con Regresion (las dos heredan de Supervisado -> EDA).
    """

    def __init__(self, ruta_csv, sep=",", variable_predecir=None, tipo="clasificacion", **kwargs):
        datos = pd.read_csv(ruta_csv, sep=sep)
        datos.columns = datos.columns.str.strip()   # el USCrime trae espacios en los nombres
        if variable_predecir is None:
            self.df = NoSupervisado(datos)
        elif tipo == "regresion":
            self.df = Regresion(datos, variable_predecir, **kwargs)
        else:
            self.df = Clasificacion(datos, variable_predecir, **kwargs)

    # --- Métodos básicos ---
    def mostrar(self): return self.df.mostrar()
    def primeras_filas(self, n=10): return self.df.primeras_filas(n)
    def ultimas_filas(self, n=10): return self.df.ultimas_filas(n)
    def obtener_fila(self, indice): return self.df.obtener_fila(indice)
    def obtener_columna(self, nombre): return self.df.obtener_columna(nombre)
    def dimensiones(self): return self.df.dimensiones()
    def nombres_columnas(self): return self.df.nombres_columnas()
    def tipos_datos(self): return self.df.tipos_datos()

    # --- Limpieza ---
    def contar_nulos(self): return self.df.contar_nulos()
    def eliminar_nulos(self): return self.df.eliminar_nulos()
    def reemplazar_nulos(self, valor=0): return self.df.reemplazar_nulos(valor)
    def detectar_duplicados(self): return self.df.detectar_duplicados()
    def eliminar_duplicados(self): return self.df.eliminar_duplicados()

    # --- Estadísticas y correlación ---
    def estadisticas(self): return self.df.estadisticas()
    def correlacion(self): return self.df.correlacion()

    # --- Visualizaciones ---
    def histogramas(self): self.df.histogramas()
    def boxplots(self): self.df.boxplots()
    def scatterplots(self): self.df.scatterplots()
    def mapa_calor(self): self.df.mapa_calor()

    # --- Transformaciones ---
    def normalizar(self): return self.df.normalizar()
    def estandarizar(self): return self.df.estandarizar()
    def imputar_nulos(self): return self.df.imputar_nulos()

    # --- Selección y exportación ---
    def seleccionar_variables(self, columnas): return self.df.seleccionar_variables(columnas)
    def guardar_csv(self, ruta="dataframe_guardado.csv"): return self.df.guardar_csv(ruta)
    def exportar_resultados(self, ruta="resultados_eda.csv"): return self.df.exportar_resultados(ruta)

    # --- ACP ---
    def acp(self, n_componentes=5, estandarizar=True): return self.df.ACP(n_componentes, estandarizar)
    def plano_principal(self, **kwargs): self.df.plano_principal(**kwargs)
    def circulo(self, **kwargs): self.df.circulo(**kwargs)
    def integrado(self, **kwargs): self.df.integrado(**kwargs)
    def correlacion_var(self): return self.df.correlacion_var
    def coordenadas_ind(self): return self.df.coordenadas_ind
    def var_explicada(self): return self.df.var_explicada

    # --- Agrupamiento / clustering ---
    def kmeans(self, n_clusters=3, init="k-means++", n_acp=None, graficar=True):
        return self.df.kmeans(n_clusters, init, n_acp, graficar)
    def hac(self, n_clusters=3, enlace="ward", n_acp=None, graficar=True):
        return self.df.hac(n_clusters, enlace, n_acp, graficar)
    def codo(self, k_max=10, n_acp=None): return self.df.codo(k_max, n_acp)
    def comparar_kmeans(self, configuraciones): return self.df.comparar_kmeans(configuraciones)
    def comparar_hac(self, configuraciones): return self.df.comparar_hac(configuraciones)
    def perfil_clusters(self, etiquetas, columnas=None): return self.df.perfil_clusters(etiquetas, columnas)

    # --- t-SNE y UMAP ---
    def tsne(self, perplexity=30, colores=None, titulo="t-SNE"):
        return self.df.tsne(perplexity, colores, titulo)
    def umap(self, n_neighbors=15, min_dist=0.1, colores=None, titulo="UMAP"):
        return self.df.umap(n_neighbors, min_dist, colores, titulo)

    # --- Métodos adicionales para Controller ---
    def ordenar(self, columna): return self.df.ordenar(columna)
    def filtrar(self, columna, operador, valor): return self.df.filtrar(columna, operador, valor)
    def agregar_columna(self, nombre, valor): return self.df.agregar_columna(nombre, valor)
    def eliminar_columna(self, nombre): return self.df.eliminar_columna(nombre)
    def visualizar_datos(self): return self.df.visualizar_datos()

    def frecuencias(self): return self.df.frecuencias()
    def convertir_tipos(self, conversiones): return self.df.convertir_tipos(conversiones)
    def distribucion_variables(self): return self.df.distribucion_variables()
    def detectar_outliers(self): return self.df.detectar_outliers()

    # --- Supervisados: clasificación ---
    def particionar(self, test_size=0.25, eliminar_duplicados=True, variables=None):
        return self.df.particionar(test_size, eliminar_duplicados, variables)
    def distribucion_clases(self): return self.df.distribucion_clases()
    def knn(self, **kwargs): return self.df.knn(**kwargs)
    def arbol(self, **kwargs): return self.df.arbol(**kwargs)
    def bosque(self, **kwargs): return self.df.bosque(**kwargs)
    def xgboost(self, **kwargs): return self.df.xgboost(**kwargs)
    def adaboost(self, **kwargs): return self.df.adaboost(**kwargs)
    def comparar(self, algoritmo, configuraciones): return self.df.comparar(algoritmo, configuraciones)
    def curva_k(self, **kwargs): return self.df.curva_k(**kwargs)
    def validacion_cruzada(self, nombre=None, k=5): return self.df.validacion_cruzada(nombre, k)
    def resultados(self, algoritmo=None): return self.df.resultados(algoritmo)
    def mejor_modelo(self, criterio="AUC", algoritmo=None): return self.df.mejor_modelo(criterio, algoritmo)
    def matriz_confusion(self, nombre=None): return self.df.matriz_confusion(nombre)
    def importancia_variables(self, nombre=None): return self.df.importancia_variables(nombre)
    def experimentos(self): return self.df.experimentos
    def graficar_matriz_confusion(self, **kwargs): self.df.graficar_matriz_confusion(**kwargs)
    def graficar_roc(self, **kwargs): self.df.graficar_roc(**kwargs)
    def graficar_comparacion(self, **kwargs): self.df.graficar_comparacion(**kwargs)
    def graficar_importancia(self, **kwargs): self.df.graficar_importancia(**kwargs)
    def graficar_arbol(self, **kwargs): self.df.graficar_arbol(**kwargs)

    # --- Supervisados: regresión ---
    def resumen_variable(self): return self.df.resumen_variable()
    def correlacion_respuesta(self, top=None): return self.df.correlacion_respuesta(top)
    def lineal_simple(self, **kwargs): return self.df.lineal_simple(**kwargs)
    def lineal_multiple(self, **kwargs): return self.df.lineal_multiple(**kwargs)
    def lasso(self, **kwargs): return self.df.lasso(**kwargs)
    def lasso_cv(self, **kwargs): return self.df.lasso_cv(**kwargs)
    def ridge(self, **kwargs): return self.df.ridge(**kwargs)
    def ridge_cv(self, **kwargs): return self.df.ridge_cv(**kwargs)
    def svr(self, **kwargs): return self.df.svr(**kwargs)
    def mejores(self, criterio="RMSE", por="Algoritmo"): return self.df.mejores(criterio, por)
    def parametro_optimo(self, nombre=None): return self.df.parametro_optimo(nombre)
    def coeficientes(self, nombre=None, top=None): return self.df.coeficientes(nombre, top)
    def residuos(self, nombre=None): return self.df.residuos(nombre)
    def curva_parametro(self, **kwargs): return self.df.curva_parametro(**kwargs)
    def buscar_hiperparametros(self, **kwargs): return self.df.buscar_hiperparametros(**kwargs)
    def ruta_regularizacion(self, **kwargs): return self.df.ruta_regularizacion(**kwargs)
    def benchmark(self, **kwargs): return self.df.benchmark(**kwargs)
    def resumen_benchmark(self, **kwargs): return self.df.resumen_benchmark(**kwargs)
    def prueba_estadistica(self, **kwargs): return self.df.prueba_estadistica(**kwargs)
    def graficar_real_vs_predicho(self, **kwargs): self.df.graficar_real_vs_predicho(**kwargs)
    def graficar_residuos(self, **kwargs): self.df.graficar_residuos(**kwargs)
    def graficar_coeficientes(self, **kwargs): self.df.graficar_coeficientes(**kwargs)
    def graficar_benchmark(self, **kwargs): self.df.graficar_benchmark(**kwargs)
