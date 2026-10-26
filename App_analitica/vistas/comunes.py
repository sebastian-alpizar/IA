"""
comunes.py
Estado compartido de la app (datos cargados, resultados) y controles reutilizables.
"""

import html

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# Estado
# ----------------------------------------------------------------------
def iniciar_estado():
    valores = {"datos": None, "datos_originales": None, "nombre": None, "version": 0,
               "clas": {}, "reg": {}, "clusters": {}}
    for k, v in valores.items():
        st.session_state.setdefault(k, v)


def guardar_datos(df, nombre=None, originales=None):
    """Guarda un dataset nuevo y borra resultados viejos que ya no aplican."""
    st.session_state.datos = df
    if originales is not None:
        st.session_state.datos_originales = originales
    if nombre:
        st.session_state.nombre = nombre
    st.session_state.version += 1
    st.session_state.clas = {}
    st.session_state.reg = {}
    st.session_state.clusters = {}


def datos():
    return st.session_state.get("datos")


def requiere_datos():
    """Si no hay datos cargados muestra un aviso y detiene la página."""
    if datos() is None:
        st.markdown("<div class='tarjeta'><h3>Todavía no hay datos</h3>"
                    "<p>Cargá un CSV o elegí un dataset de ejemplo para usar esta sección.</p></div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Ir a Carga de datos", type="primary"):
            st.switch_page(st.session_state["_pagina_carga"])
        st.stop()
    return datos()


def tarjeta_dataset_lateral():
    d = datos()
    with st.sidebar:
        if d is None:
            st.markdown("<div class='dataset'><div class='et'>Dataset activo</div>"
                        "<div class='nom'>Ninguno</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='dataset'><div class='et'>Dataset activo</div>"
                        f"<div class='nom'>{html.escape(st.session_state.nombre or 'datos')}</div>"
                        f"<div class='et'>{d.shape[0]:,} filas · {d.shape[1]} columnas</div></div>".replace(",", "."),
                        unsafe_allow_html=True)
        st.caption("Inteligencia Artificial · UNA")


# ----------------------------------------------------------------------
# Controles
# ----------------------------------------------------------------------
def controles_parametros(especificacion, clave, columnas=3):
    """Arma los controles de un modelo a partir de su lista de parámetros."""
    valores = {}
    if not especificacion:
        st.caption("Este modelo no tiene parámetros para ajustar.")
        return valores
    cols = st.columns(columnas)
    for i, p in enumerate(especificacion):
        with cols[i % columnas]:
            k = f"{clave}_{p['nombre']}"
            if p["tipo"] == "int":
                mn, mx, paso = p["extra"]
                valores[p["nombre"]] = st.number_input(p["etiqueta"], int(mn), int(mx), int(p["default"]), int(paso),
                                                       key=k, help=p["ayuda"] or None)
            elif p["tipo"] == "float":
                mn, mx, paso = p["extra"]
                valores[p["nombre"]] = st.number_input(p["etiqueta"], float(mn), float(mx), float(p["default"]),
                                                       float(paso), key=k, format="%.4f", help=p["ayuda"] or None)
            elif p["tipo"] == "select":
                valores[p["nombre"]] = st.selectbox(p["etiqueta"], p["extra"], index=p["extra"].index(p["default"]),
                                                    key=k, help=p["ayuda"] or None)
            else:
                valores[p["nombre"]] = st.text_input(p["etiqueta"], p["default"], key=k, help=p["ayuda"] or None)
    return valores


def descargar_csv(df, nombre, etiqueta="Descargar CSV", clave=None):
    st.download_button(etiqueta, df.to_csv(index=True).encode("utf-8"), file_name=nombre, mime="text/csv",
                       key=clave or f"dl_{nombre}")


def tabla(df, altura=None, formato=None):
    """Muestra un DataFrame con formato numérico uniforme."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    estilo = df.style.format(formato or (lambda v: f"{v:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                         if isinstance(v, float) else v))
    if altura:
        st.dataframe(estilo, width="stretch", height=altura)
    else:
        st.dataframe(estilo, width="stretch")
