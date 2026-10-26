import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate
import pandas as pd

class DataFrameView:
    """
    Vista encargada de mostrar resultados y gráficas al usuario.
    """

    def mostrar(self, datos):
        print("\n--- Datos ---")
        if isinstance(datos, pd.DataFrame):
            display(
                datos.style.set_table_styles(
                    [{'selector': 'thead th',
                      'props': [('background-color', '#f0f0f0'),
                                ('color', 'black'),
                                ('font-weight', 'bold')]}]
                ).background_gradient(cmap="Blues").format(precision=2)
            )
        elif isinstance(datos, pd.Series):
            display(datos.to_frame().style.background_gradient(cmap="Blues").format(precision=2))
        elif datos is None:
            print("[INFO]: No hay datos para mostrar.")
        else:
            print(datos)

    def mostrar_mensaje(self, mensaje):
        print(f"\n[INFO]: {mensaje}")

    def mostrar_estadisticas(self, estadisticas):
        print("\n=== Estadísticas descriptivas ===")
        if isinstance(estadisticas, pd.DataFrame):
            display(estadisticas.style.background_gradient(cmap="Greens").format(precision=2))
        else:
            print(estadisticas)

    def mostrar_correlacion(self, correlacion):
        print("\n=== Matriz de correlación ===")
        if isinstance(correlacion, pd.DataFrame):
            datos_num = correlacion.select_dtypes(include=["float64", "int64"])
            display(datos_num.style.background_gradient(cmap="coolwarm").format(precision=2))
        else:
            print(correlacion)

    def mostrar_frecuencias(self, frecuencias):
        print("\n=== Frecuencias ===")
        for col, freq in frecuencias.items():
            print(f"\nColumna: {col}")
            if isinstance(freq, pd.DataFrame):
                display(freq.style.background_gradient(cmap="Purples").format(precision=0))
            else:
                print(tabulate(freq, headers="keys", tablefmt="fancy_grid"))

    def mostrar_outliers(self, outliers):
        print("\n=== Outliers detectados ===")
        for col, df_out in outliers.items():
            if isinstance(df_out, pd.DataFrame):
                print(f"\nColumna: {col} | Filas detectadas: {len(df_out)}")
                display(df_out.style.background_gradient(cmap="Reds").format(precision=2))
            else:
                print(f"\nColumna: {col}")
                print(df_out)

    def mostrar_duplicados(self, duplicados):
        print("\n=== Duplicados detectados ===")
        if isinstance(duplicados, pd.DataFrame):
            print(f"Total duplicados: {len(duplicados)} filas")
            display(duplicados.style.background_gradient(cmap="Oranges").format(precision=2))
        else:
            print(duplicados)

    def mostrar_eliminados(self, df_sin_duplicados):
        print("\n=== Dataset sin duplicados ===")
        if isinstance(df_sin_duplicados, pd.DataFrame):
            print(f"Total filas después de eliminar duplicados: {len(df_sin_duplicados)}")
            display(df_sin_duplicados.head(10).style.background_gradient(cmap="Blues").format(precision=2))
        else:
            print(df_sin_duplicados)

    def mostrar_var_explicada(self, var_explicada):
        print("\n=== Varianza explicada por componentes ===")
        if isinstance(var_explicada, pd.DataFrame):
            display(var_explicada.style.background_gradient(cmap="Greens").format(precision=2))
        else:
            print(var_explicada)

    def mostrar_indices(self, indices):
        # índices de calidad de clasificación (4 decimales para poder comparar modelos)
        print("\n=== Índices de calidad ===")
        if isinstance(indices, pd.Series):
            indices = indices.to_frame().T
        if isinstance(indices, pd.DataFrame):
            display(indices.style.background_gradient(cmap="Greens").format(precision=4))
        else:
            print(indices)

    def mostrar_clusters(self, etiquetas, score):
        print("\n=== Resultados Clustering ===")
        print("Etiquetas:", etiquetas)
        print("Silhouette Score:", score)

    def mostrar_distribucion(self, distribucion):
        print("\n=== Distribución de variables ===")
        if "numericas" in distribucion and isinstance(distribucion["numericas"], pd.DataFrame):
            print("\nVariables numéricas:")
            display(distribucion["numericas"].style.background_gradient(cmap="Blues").format(precision=2))
        if "categoricas" in distribucion and isinstance(distribucion["categoricas"], pd.DataFrame):
            print("\nVariables categóricas:")
            display(distribucion["categoricas"].style.background_gradient(cmap="Purples"))

    def mostrar_grafico(self, grafico_func):
        """
        Recibe una función de graficado y la ejecuta.
        """
        grafico_func()
        plt.show()
