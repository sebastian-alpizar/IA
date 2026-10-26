from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nucleo.eda import EDA
from vistas.comunes import datos, descargar_csv, guardar_datos, tabla
from vistas.estilo import encabezado, metricas

CARPETA_EJEMPLOS = Path(__file__).resolve().parent.parent / "datos_ejemplo"
EJEMPLOS = {
    "Iris (clasificación, 3 especies)": "iris.csv",
    "Cáncer de mama (clasificación binaria)": "cancer_mama.csv",
    "Potabilidad del agua (Lab 2, clasificación binaria)": "potabilidad.csv",
    "Predicción de diabetes (Lab 2, clasificación binaria, 100 000 filas)": "diabetes_prediccion.csv",
    "USCrime (no supervisado, 1994 comunidades)": "USCrime.csv",
}
SEPARADORES = {"Coma": ",", "Punto y coma": ";", "Tab": "\t", "|": "|"}
DECIMALES = {"Punto": ".", "Coma": ","}


def _cargar_bytes(contenido, nombre):
    st.session_state.archivo_bytes = contenido
    st.session_state.archivo_nombre = nombre
    fmt = EDA.detectar_formato(contenido)
    st.session_state.archivo_formato = fmt
    # valores por defecto de los controles según lo detectado
    st.session_state.sel_sep = next(k for k, v in SEPARADORES.items() if v == fmt["sep"])
    st.session_state.sel_dec = next(k for k, v in DECIMALES.items() if v == fmt["decimal"])
    _leer()
    st.rerun()   # para que la barra lateral muestre el dataset nuevo


def _leer():
    try:
        df = EDA.leer_csv(st.session_state.archivo_bytes, sep=SEPARADORES[st.session_state.sel_sep],
                          decimal=DECIMALES[st.session_state.sel_dec],
                          encabezado=st.session_state.get("sel_enc", True),
                          indice=st.session_state.get("sel_idx", False),
                          encoding=st.session_state.archivo_formato["encoding"])
        if df.shape[1] < 2:
            st.session_state.error_lectura = "Se leyó una sola columna: revisá el separador."
        else:
            st.session_state.error_lectura = None
        guardar_datos(df, st.session_state.archivo_nombre, originales=df)
    except Exception as e:  # noqa: BLE001
        st.session_state.error_lectura = f"No se pudo leer el archivo: {e}"


def pagina():
    encabezado("Carga de datos", "Subí un archivo CSV, revisá cómo se leyó y prepará los datos para el análisis.",
               "Paso 1")

    izq, der = st.columns([1, 1.35], gap="medium")
    with izq:
        with st.container(border=True):
            st.markdown("**Archivo**")
            subido = st.file_uploader("Archivo CSV", type=["csv", "txt"], label_visibility="collapsed")
            if subido is not None and subido.name != st.session_state.get("archivo_nombre"):
                _cargar_bytes(subido.getvalue(), subido.name)
            st.markdown("**o usá un dataset de ejemplo**")
            ejemplo = st.selectbox("Ejemplo", list(EJEMPLOS), label_visibility="collapsed")
            if st.button("Cargar ejemplo", width="stretch"):
                ruta = CARPETA_EJEMPLOS / EJEMPLOS[ejemplo]
                _cargar_bytes(ruta.read_bytes(), ruta.name)

    with der:
        with st.container(border=True):
            st.markdown("**Lectura del archivo**")
            hay_archivo = "archivo_bytes" in st.session_state
            c1, c2 = st.columns(2)
            with c1:
                st.segmented_control("Separador de datos", list(SEPARADORES), key="sel_sep",
                                     default="Coma" if "sel_sep" not in st.session_state else None,
                                     disabled=not hay_archivo)
                st.checkbox("La primera fila tiene los nombres", True, key="sel_enc", disabled=not hay_archivo)
            with c2:
                st.segmented_control("Separador decimal", list(DECIMALES), key="sel_dec",
                                     default="Punto" if "sel_dec" not in st.session_state else None,
                                     disabled=not hay_archivo)
                st.checkbox("La primera columna son nombres de fila", False, key="sel_idx", disabled=not hay_archivo)
            if hay_archivo:
                fmt = st.session_state.archivo_formato
                st.caption(f"Detectado automáticamente: separador «{repr(fmt['sep'])[1:-1]}», decimal «{fmt['decimal']}», "
                           f"codificación {fmt['encoding']}.")
            if st.button("Volver a leer con estas opciones", disabled=not hay_archivo, type="primary"):
                if st.session_state.sel_sep and st.session_state.sel_dec:
                    _leer()
                    st.rerun()

    if st.session_state.get("error_lectura"):
        st.error(st.session_state.error_lectura)

    df = datos()
    if df is None:
        return

    e = EDA(df)
    r = e.resumen()
    st.write("")
    metricas([{"et": "Filas", "val": f"{r['filas']:,}".replace(",", ".")},
              {"et": "Columnas", "val": r["columnas"]},
              {"et": "Numéricas / categóricas", "val": f"{r['numericas']} / {r['categoricas']}"},
              {"et": "Celdas vacías", "val": r["nulos"], "tipo": "mala" if r["nulos"] else "buena"},
              {"et": "Filas duplicadas", "val": r["duplicados"], "tipo": "mala" if r["duplicados"] else "buena"}],
             columnas=5)

    t1, t2, t3 = st.tabs(["Vista previa", "Variables", "Preparar datos"])
    with t1:
        st.dataframe(df.head(200), width="stretch", height=380)
        st.caption(f"Primeras {min(200, len(df))} de {len(df)} filas.")
    with t2:
        tabla(e.tipos(), altura=420)
    with t3:
        _preparar()


def _preparar():
    original = st.session_state.datos_originales
    st.markdown("Los cambios se aplican sobre el archivo original, así que podés probar sin perder nada.")
    c1, c2 = st.columns(2)
    with c1:
        nulos = st.radio("Datos ausentes", ["Dejar como están", "Eliminar filas con vacíos",
                                            "Imputar (mediana o moda)"], horizontal=False)
        duplicados = st.checkbox("Eliminar filas duplicadas", value=False)
    with c2:
        quitar = st.multiselect("Columnas que no se usan (ID, códigos...)", list(original.columns))
        candidatas = [c for c in original.select_dtypes(include=np.number).columns
                      if original[c].nunique() <= 15 and c not in quitar]
        a_categoricas = st.multiselect("Numéricas que en realidad son categorías", candidatas,
                                       help="Por ejemplo 0/1 o códigos de grupo.")
    if st.button("Aplicar cambios", type="primary"):
        df = original.drop(columns=quitar)
        for c in a_categoricas:
            df[c] = df[c].astype("Int64").astype(str)
        if nulos.startswith("Eliminar"):
            df = df.dropna()
        elif nulos.startswith("Imputar"):
            df = EDA(df).tratar_nulos("imputar")
        if duplicados:
            df = df.drop_duplicates()
        guardar_datos(df)
        st.success(f"Listo: {df.shape[0]} filas y {df.shape[1]} columnas.")
        st.rerun()
    descargar_csv(datos(), "datos_preparados.csv", "Descargar datos preparados")
