import pandas as pd

class DataFrameController:
    """
    Controller que recibe el Model y el View.
    Gestiona eventos de la interfaz y decide qué método del Model ejecutar.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view

    def iniciar(self):
        # Flujo inicial
        self.view.mostrar_mensaje("Iniciando controlador...")
        self.view.mostrar_mensaje("--- Primeras filas del dataset ---")
        self.view.mostrar(self.model.primeras_filas(10))
        self.view.mostrar_mensaje(f"Dimensiones del dataset: {self.model.dimensiones()}")

    def ejecutar(self, metodo, **kwargs):
        try:
            if metodo == "Información":
                self.view.mostrar_mensaje(self.model.dimensiones())

            elif metodo == "Mostrar":
                self.view.mostrar(self.model.mostrar().head(20))

            elif metodo == "Primeras filas":
                self.view.mostrar(self.model.primeras_filas(10))

            elif metodo == "Últimas filas":
                self.view.mostrar(self.model.ultimas_filas(10))

            elif metodo == "Obtener fila":
                indice = kwargs.get("indice", 0)
                self.view.mostrar(self.model.obtener_fila(indice))

            elif metodo == "Obtener columna":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.obtener_columna(columna).head(20))

            elif metodo == "Estadísticas":
                self.view.mostrar_estadisticas(self.model.estadisticas())

            elif metodo == "Correlación":
                self.view.mostrar_correlacion(self.model.correlacion())

            elif metodo == "Frecuencias":
                self.view.mostrar_frecuencias(self.model.frecuencias())

            elif metodo == "Detectar duplicados":
                self.view.mostrar_duplicados(self.model.detectar_duplicados())

            elif metodo == "Eliminar duplicados":
                self.view.mostrar_eliminados(self.model.eliminar_duplicados())

            elif metodo == "Ordenar":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.ordenar(columna))

            elif metodo == "Filtrar":
                columna = kwargs.get("columna")
                operador = kwargs.get("operador")
                valor = kwargs.get("valor")
                self.view.mostrar(self.model.filtrar(columna, operador, valor))

            elif metodo == "Agregar columna":
                nombre = kwargs.get("nombre", "nueva_columna")
                valor = kwargs.get("valor", 1)
                self.view.mostrar(self.model.agregar_columna(nombre, valor))

            elif metodo == "Eliminar columna":
                columna = kwargs.get("columna")
                self.view.mostrar(self.model.eliminar_columna(columna))

            elif metodo == "Detectar outliers":
                self.view.mostrar_outliers(self.model.detectar_outliers())

            elif metodo == "Distribución variables":
                distribucion = self.model.distribucion_variables()
                self.view.mostrar_distribucion(distribucion)

            # --- Visualizaciones ---
            elif metodo == "Histogramas":
                self.view.mostrar_mensaje("Histograma de variables numéricas")
                self.model.histogramas()

            elif metodo == "Boxplots":
                self.view.mostrar_mensaje("Boxplots de variables numéricas")
                self.model.boxplots()

            elif metodo == "Scatterplots":
                self.view.mostrar_mensaje("Scatterplots (matriz de dispersión)")
                self.model.scatterplots()

            elif metodo == "Mapa calor":
                self.view.mostrar_mensaje("Mapa de calor de correlaciones")
                self.model.mapa_calor()

            # --- Transformaciones ---
            elif metodo == "Normalizar":
                self.view.mostrar_mensaje("Normalización de variables")
                self.view.mostrar(self.model.normalizar())

            elif metodo == "Estandarizar":
                self.view.mostrar_mensaje("Estandarización de variables")
                self.view.mostrar(self.model.estandarizar())

            elif metodo == "Imputar nulos":
                self.view.mostrar_mensaje("Imputación de valores nulos")
                self.view.mostrar(self.model.imputar_nulos())

            # --- ACP ---
            elif metodo == "ACP":
                var_exp = self.model.acp(kwargs.get("n_componentes", 5), kwargs.get("estandarizar", True))
                self.view.mostrar_var_explicada(var_exp)

            elif metodo == "Plano principal":
                self.model.plano_principal(**kwargs)

            elif metodo == "Círculo":
                self.model.circulo(**kwargs)

            elif metodo == "Integrado":
                self.model.integrado(**kwargs)

            # --- Agrupamiento / clustering ---
            elif metodo == "Codo":
                self.view.mostrar(self.model.codo(kwargs.get("k_max", 10), kwargs.get("n_acp")))

            elif metodo == "KMeans":
                etiquetas, score = self.model.kmeans(kwargs.get("n_clusters", 3),
                                                     kwargs.get("init", "k-means++"),
                                                     kwargs.get("n_acp"),
                                                     kwargs.get("graficar", True))
                return etiquetas

            elif metodo == "HAC":
                etiquetas, score = self.model.hac(kwargs.get("n_clusters", 3),
                                                  kwargs.get("enlace", "ward"),
                                                  kwargs.get("n_acp"),
                                                  kwargs.get("graficar", True))
                return etiquetas

            elif metodo == "Comparar KMeans":
                self.view.mostrar(self.model.comparar_kmeans(kwargs.get("configuraciones")))

            elif metodo == "Comparar HAC":
                self.view.mostrar(self.model.comparar_hac(kwargs.get("configuraciones")))

            elif metodo == "Perfil clusters":
                self.view.mostrar(self.model.perfil_clusters(kwargs.get("etiquetas"), kwargs.get("columnas")))

            # --- t-SNE y UMAP ---
            elif metodo == "TSNE":
                return self.model.tsne(kwargs.get("perplexity", 30), kwargs.get("colores"),
                                       kwargs.get("titulo", "t-SNE"))

            elif metodo == "UMAP":
                return self.model.umap(kwargs.get("n_neighbors", 15), kwargs.get("min_dist", 0.1),
                                       kwargs.get("colores"), kwargs.get("titulo", "UMAP"))

            # --- Supervisados: clasificación ---
            elif metodo == "Distribución clases":
                self.view.mostrar(self.model.distribucion_clases())

            elif metodo == "Particionar":
                self.view.mostrar(self.model.particionar(kwargs.get("test_size", 0.25),
                                                         kwargs.get("eliminar_duplicados", True),
                                                         kwargs.get("variables")))

            elif metodo in ("KNN", "Árbol", "Random Forest", "XGBoost", "AdaBoost"):
                algoritmos = {"KNN": "knn", "Árbol": "arbol", "Random Forest": "bosque",
                              "XGBoost": "xgboost", "AdaBoost": "adaboost"}
                indices = getattr(self.model, algoritmos[metodo])(**kwargs)
                self.view.mostrar_indices(indices)
                return indices

            elif metodo == "Comparar":
                tabla = self.model.comparar(kwargs.get("algoritmo"), kwargs.get("configuraciones"))
                self.view.mostrar_indices(tabla)
                return tabla

            elif metodo == "Curva K":
                self.view.mostrar_indices(self.model.curva_k(**kwargs))

            elif metodo == "Validación cruzada":
                self.view.mostrar_indices(self.model.validacion_cruzada(kwargs.get("nombre"), kwargs.get("k", 5)))

            elif metodo == "Resultados":
                self.view.mostrar_indices(self.model.resultados(kwargs.get("algoritmo")))

            elif metodo == "Mejor modelo":
                mejor = self.model.mejor_modelo(kwargs.get("criterio", "AUC"), kwargs.get("algoritmo"))
                self.view.mostrar_mensaje(f"Mejor modelo según {kwargs.get('criterio', 'AUC')}: {mejor['Modelo']}")
                self.view.mostrar_indices(mejor.to_frame().T.infer_objects())
                return mejor["Modelo"]

            elif metodo == "Matriz confusión":
                self.view.mostrar(self.model.matriz_confusion(kwargs.get("nombre")))
                self.model.graficar_matriz_confusion(**kwargs)

            elif metodo == "ROC":
                self.model.graficar_roc(**kwargs)

            elif metodo == "Gráfico comparación":
                self.model.graficar_comparacion(**kwargs)

            elif metodo == "Importancia":
                self.model.graficar_importancia(**kwargs)
                return self.model.importancia_variables(kwargs.get("nombre"))

            elif metodo == "Graficar árbol":
                self.model.graficar_arbol(**kwargs)

            # --- Exportación ---
            elif metodo == "Guardar CSV":
                ruta = kwargs.get("ruta", "resultado_final.csv")
                self.view.mostrar_mensaje(f"Archivo guardado en: {self.model.guardar_csv(ruta)}")

            elif metodo == "Exportar resultados":
                ruta = kwargs.get("ruta", "estadisticas_finales.csv")
                self.view.mostrar_mensaje(f"Resultados exportados en: {self.model.exportar_resultados(ruta)}")

            else:
                self.view.mostrar_mensaje(f"Método '{metodo}' no reconocido.")

        except Exception as e:
            self.view.mostrar_mensaje(f"ERROR: {type(e).__name__}: {e}")
