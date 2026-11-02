"""
carga.py (página)
Vista de Carga de datos: subir un CSV o escoger un ejemplo, revisar cómo se leyó y
preparar los datos. Leer y preparar lo hace el controlador; aquí solo se muestra.
"""

import numpy as np
import streamlit as st

from mvc.Controller import EJEMPLOS

SEPARADORES = {"Coma": ",", "Punto y coma": ";", "Tab": "\t", "|": "|"}
DECIMALES = {"Punto": ".", "Coma": ","}


def _opciones():
    """Encabezado e índice según los controles de lectura (por defecto: con nombres, sin índice)."""
    return st.session_state.get("sel_enc", True), st.session_state.get("sel_idx", False)


def _cargar(ctrl, accion):
    """Carga (archivo o ejemplo) y deja los controles de lectura con lo que se detectó."""
    formato = accion()
    st.session_state.sel_sep = next(k for k, sep in SEPARADORES.items() if sep == formato["sep"])
    st.session_state.sel_dec = next(k for k, dec in DECIMALES.items() if dec == formato["decimal"])
    st.rerun()   # para que la barra lateral muestre el dataset nuevo


def pagina(ctrl):
    v = ctrl.view
    m = ctrl.model
    v.encabezado("Carga de datos", "Subí un archivo CSV, revisá cómo se leyó y prepará los datos para el análisis.",
               "Paso 1")

    izq, der = st.columns([1, 1.35], gap="medium")
    with izq:
        with st.container(border=True):
            st.markdown("**Archivo**")
            subido = st.file_uploader("Archivo CSV", type=["csv", "txt"], label_visibility="collapsed")
            if subido is not None and (m.archivo is None or subido.name != m.archivo["nombre"]):
                _cargar(ctrl, lambda: ctrl.cargar_archivo(subido.getvalue(), subido.name, *_opciones()))
            st.markdown("**o usá un dataset de ejemplo**")
            ejemplo = st.selectbox("Ejemplo", list(EJEMPLOS), label_visibility="collapsed")
            if st.button("Cargar ejemplo", width="stretch"):
                _cargar(ctrl, lambda: ctrl.cargar_ejemplo(ejemplo, *_opciones()))

    with der:
        with st.container(border=True):
            st.markdown("**Lectura del archivo**")
            hay_archivo = m.archivo is not None
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
                fmt = m.archivo["formato"]
                st.caption(f"Detectado automáticamente: separador «{repr(fmt['sep'])[1:-1]}», decimal «{fmt['decimal']}», "
                           f"codificación {fmt['encoding']}.")
            if st.button("Volver a leer con estas opciones", disabled=not hay_archivo, type="primary"):
                if st.session_state.sel_sep and st.session_state.sel_dec:
                    ctrl.leer_archivo(SEPARADORES[st.session_state.sel_sep], DECIMALES[st.session_state.sel_dec],
                                      *_opciones())
                    st.rerun()

    if m.error_lectura:
        v.error(m.error_lectura)

    if not m.hay_datos:
        return
    df = m.datos
    e = m.eda()
    r = e.resumen()
    st.write("")
    v.metricas([{"et": "Filas", "val": f"{r['filas']:,}".replace(",", ".")},
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
        v.tabla(e.tipos(), altura=420)
    with t3:
        _preparar(ctrl)


def _preparar(ctrl):
    v = ctrl.view
    original = ctrl.model.datos_originales
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
        df = ctrl.preparar_datos(nulos, duplicados, quitar, a_categoricas)
        v.exito(f"Listo: {df.shape[0]} filas y {df.shape[1]} columnas.")
        st.rerun()
    v.descargar_csv(ctrl.model.datos, "datos_preparados.csv", "Descargar datos preparados")
