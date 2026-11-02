"""
Controller.py
Controlador que recibe el Model y el View. Atiende las acciones del usuario (cargar un
archivo, preparar los datos, correr un ACP o un clustering, entrenar un modelo, validar,
comparar, predecir...), le pide al modelo que las ejecute y, si algo falla, usa la vista
para mostrar el mensaje. Las páginas (mvc/paginas/) llaman a estos métodos en vez de
modificar el estado directamente.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from eda import EDA
from no_supervisado import NoSupervisado

CARPETA_EJEMPLOS = Path(__file__).resolve().parent.parent / "datos_ejemplo"
EJEMPLOS = {
    "Iris (clasificación, 3 especies)": "iris.csv",
    "Cáncer de mama (clasificación binaria)": "cancer_mama.csv",
    "Potabilidad del agua (Lab 2, clasificación binaria)": "potabilidad.csv",
    "Predicción de diabetes (Lab 2, clasificación binaria, 100 000 filas)": "diabetes_prediccion.csv",
    "USCrime (no supervisado y regresión del Lab 3, 1994 comunidades)": "USCrime.csv",
}


# --- Cálculos pesados en caché (se recalculan solo si cambian los datos o los parámetros) ---
@st.cache_data(show_spinner=False)
def _codo(df, cols, k_max, n_acp):
    return NoSupervisado(df, list(cols)).codo(k_max, n_acp)


@st.cache_data(show_spinner=False)
def _tsne(df, cols, perp):
    return NoSupervisado(df, list(cols)).tsne(perp)


@st.cache_data(show_spinner=False)
def _umap(df, cols, nn, md):
    return NoSupervisado(df, list(cols)).umap(nn, md)


class DataFrameController:

    def __init__(self, model, view, paginas=None):
        self.model = model
        self.view = view
        self.paginas = paginas or {}    # clave -> st.Page (para los botones que cambian de página)

    def _intentar(self, funcion, prefijo="", aviso=False):
        """Ejecuta una acción; si falla muestra el error en la vista y retorna None."""
        try:
            return funcion()
        except Exception as e:  # noqa: BLE001
            (self.view.aviso if aviso else self.view.error)(f"{prefijo}{e}")
            return None

    # =================================================================
    # Navegación
    # =================================================================
    def ir_a(self, clave):
        st.switch_page(self.paginas[clave])

    def requiere_datos(self):
        """True si hay datos cargados; si no, muestra el aviso con el botón para ir a la carga."""
        if self.model.hay_datos:
            return True
        self.view.sin_datos(lambda: self.ir_a("carga"))
        return False

    # =================================================================
    # Carga y preparación de los datos
    # =================================================================
    def cargar_archivo(self, contenido: bytes, nombre, encabezado=True, indice=False):
        """Guarda el CSV, detecta su formato y lo lee. Retorna el formato detectado."""
        formato = self.model.guardar_archivo(contenido, nombre)
        self.leer_archivo(formato["sep"], formato["decimal"], encabezado, indice)
        return formato

    def cargar_ejemplo(self, etiqueta, encabezado=True, indice=False):
        ruta = CARPETA_EJEMPLOS / EJEMPLOS[etiqueta]
        return self.cargar_archivo(ruta.read_bytes(), ruta.name, encabezado, indice)

    def leer_archivo(self, sep, decimal, encabezado=True, indice=False):
        try:
            self.model.leer_archivo(sep, decimal, encabezado, indice)
        except Exception as e:  # noqa: BLE001
            self.model.error_lectura = f"No se pudo leer el archivo: {e}"

    def preparar_datos(self, nulos="Dejar como están", duplicados=False, quitar=(), a_categoricas=()):
        """Aplica la preparación sobre los datos originales (así se puede probar sin perder nada)."""
        df = self.model.datos_originales.drop(columns=list(quitar))
        for c in a_categoricas:
            df[c] = df[c].astype("Int64").astype(str)
        if nulos.startswith("Eliminar"):
            df = df.dropna()
        elif nulos.startswith("Imputar"):
            df = EDA(df).tratar_nulos("imputar")
        if duplicados:
            df = df.drop_duplicates()
        self.model.guardar_datos(df)
        return df

    # =================================================================
    # No supervisado
    # =================================================================
    def acp(self, ns, n_componentes, escalar):
        return ns.acp(n_componentes, escalar)

    def codo(self, columnas, k_max, n_acp):
        tabla = _codo(self.model.datos, tuple(columnas), k_max, n_acp or None)
        return self.model.guardar_resultado("ns", "codo", tabla)

    def kmeans(self, ns, k, init, n_init, n_acp):
        r = ns.kmeans(k, init, n_init, n_acp or None)
        self.model.guardar_cluster(f"K-means (k={k})", pd.Series(r["etiquetas"], index=ns.indice))
        return self.model.guardar_resultado("ns", "kmeans", {**r, "k": k})

    def hac(self, ns, k, enlace, n_acp):
        r = ns.hac(k, enlace, n_acp or None)
        self.model.guardar_cluster(f"HAC {enlace} (k={k})", pd.Series(r["etiquetas"], index=ns.indice))
        self.model.guardar_resultado("ns", "hac_fig", ns.fig_dendrograma(k, enlace, n_acp or None))
        return self.model.guardar_resultado("ns", "hac", {**r, "k": k, "enlace": enlace})

    def proyeccion(self, metodo, columnas, perplexity=30, n_neighbors=15, min_dist=0.1):
        if metodo == "t-SNE":
            xy, kl = _tsne(self.model.datos, tuple(columnas), perplexity)
            titulo = f"t-SNE (perplexity {perplexity}) · KL = {kl:.4f}"
        else:
            xy = _umap(self.model.datos, tuple(columnas), n_neighbors, min_dist)
            titulo = f"UMAP (n_neighbors {n_neighbors}, min_dist {min_dist})"
        return self.model.guardar_resultado("ns", "proyeccion", (xy, titulo))

    # =================================================================
    # Supervisados (clasificación y regresión)
    # =================================================================
    def clasificacion(self, objetivo, predictoras, positiva, prueba, semilla, duplicados):
        return self._intentar(lambda: self.model.clasificacion(objetivo, predictoras, positiva, prueba, semilla,
                                                               duplicados), "No se pudo particionar: ")

    def regresion(self, objetivo, predictoras, prueba, semilla, duplicados):
        return self._intentar(lambda: self.model.regresion(objetivo, predictoras, prueba, semilla, duplicados),
                              "No se pudo particionar: ")

    def entrenar(self, sup, algoritmo, configuracion):
        """Entrena una configuración de un algoritmo y la agrega al historial."""
        return self._intentar(lambda: sup.ejecutar(algoritmo, configuracion), "El modelo no se pudo entrenar: ")

    def correr_plan(self, sup, plan):
        """Corre en lote {algoritmo: [configuraciones]} con barra de progreso."""
        total = sum(len(v) for v in plan.values())
        avanzar, quitar = self.view.barra_progreso()
        hecho = 0
        for algoritmo, variaciones in plan.items():
            for conf in variaciones:
                avanzar(hecho, total, f"{algoritmo}: {conf or 'estándar'}")
                self._intentar(lambda: sup.ejecutar(algoritmo, conf), f"{algoritmo} {conf}: ", aviso=True)
                hecho += 1
        quitar()

    def borrar_experimento(self, sup, nombre):
        sup.borrar_experimento(nombre)

    def validacion_cruzada(self, seccion, sup, nombre, k):
        tabla = self._intentar(lambda: sup.validacion_cruzada(nombre, int(k)))
        if tabla is not None:
            self.model.guardar_resultado(seccion, f"cv_{nombre}_{k}", tabla)
        return tabla

    def leer_nuevos(self, contenido: bytes):
        return self._intentar(lambda: self.model.leer_nuevos(contenido), "No se pudo leer el archivo: ")

    def predecir(self, sup, nombre, nuevos):
        return self._intentar(lambda: sup.predecir(nombre, nuevos))

    # --- solo clasificación ---
    def curva_k(self, sup, valores, escalar, pesos):
        return self.model.guardar_resultado("clas", "curva_k", sup.curva_k(valores, escalar, weights=pesos))

    # --- solo regresión ---
    def ruta_regularizacion(self, sup, tipo, k):
        return self.model.guardar_resultado("reg", f"ruta_{tipo}", sup.ruta_regularizacion(tipo, int(k)))

    def curva_parametro(self, sup, algoritmo, parametro, texto_valores, k):
        """texto_valores: valores separados por coma (ej. '1, 2, 4, 8')."""
        try:
            valores = [float(x) for x in texto_valores.replace(";", ",").split(",") if x.strip()]
            if parametro in ("max_depth", "n_estimators", "n_neighbors"):
                valores = [int(x) for x in valores]
        except ValueError as e:
            self.view.error(f"Revisá los valores: {e}")
            return None
        tabla = sup.curva_parametro(algoritmo, parametro, valores, int(k))
        return self.model.guardar_resultado("reg", "curva", (algoritmo, tabla))

    def benchmark(self, sup, nombres, k, repeticiones):
        avanzar, quitar = self.view.barra_progreso()
        tabla = sup.benchmark(nombres, int(k), int(repeticiones),
                              progreso=lambda i, n, nombre: avanzar(i, n, f"Validando {nombre}"))
        quitar()
        return tabla
