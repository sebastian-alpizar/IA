"""
app.py - Aplicación de analítica de datos (EDA, no supervisado y clasificación).
Ejecutar con:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from vistas import carga, clasificacion, exploracion, inicio, no_supervisado
from vistas.comunes import iniciar_estado, tarjeta_dataset_lateral
from vistas.estilo import aplicar_estilo

RAIZ = Path(__file__).resolve().parent

st.set_page_config(page_title="Analítica de datos", page_icon=str(RAIZ / "recursos" / "icono.svg"),
                   layout="wide", initial_sidebar_state="expanded")
aplicar_estilo()
iniciar_estado()
st.logo(str(RAIZ / "recursos" / "logo.svg"), size="large")

# --- Páginas (menú lateral agrupado por secciones) ---
p_inicio = st.Page(inicio.pagina, title="Inicio", icon=":material/home:", default=True, url_path="inicio")
p_carga = st.Page(carga.pagina, title="Carga de datos", icon=":material/upload_file:", url_path="carga")
p_eda = st.Page(exploracion.pagina, title="Exploración", icon=":material/query_stats:", url_path="exploracion")
p_nosup = st.Page(no_supervisado.pagina, title="ACP y clustering", icon=":material/bubble_chart:", url_path="no-supervisado")
p_clas = st.Page(clasificacion.pagina, title="Clasificación", icon=":material/category:", url_path="clasificacion")

# referencias para los botones que cambian de página
st.session_state["_pagina_carga"] = p_carga
st.session_state["_pagina_eda"] = p_eda

navegacion = st.navigation({
    "": [p_inicio],
    "Datos": [p_carga, p_eda],
    "No supervisados": [p_nosup],
    "Supervisados": [p_clas],
})
tarjeta_dataset_lateral()
navegacion.run()
