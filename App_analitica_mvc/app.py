"""
app.py - Aplicación de analítica de datos (EDA, no supervisado, clasificación y regresión).
Ejecutar con:  streamlit run app.py

Solo arranca la app con el patrón MVC (carpeta mvc/):
  - DataFrameModel: los datos y resultados de la sesión (uno por persona, se guarda en la sesión)
  - DataFrameView: componentes visuales y estilo
  - DataFrameController: las acciones del usuario
y arma el menú lateral con una página (vista) por sección (carpeta mvc/paginas/).
"""

from pathlib import Path

import streamlit as st

from mvc import DataFrameController, DataFrameModel, DataFrameView
from mvc.paginas import carga, clasificacion, exploracion, inicio, no_supervisado, regresion

RAIZ = Path(__file__).resolve().parent

st.set_page_config(page_title="Analítica de datos", page_icon=str(RAIZ / "recursos" / "icono.svg"),
                   layout="wide", initial_sidebar_state="expanded")

# --- Inicialización MVC (el modelo vive en la sesión; vista y controlador se crean en cada ejecución) ---
if "modelo" not in st.session_state:
    st.session_state.modelo = DataFrameModel()
model = st.session_state.modelo
view = DataFrameView()
controller = DataFrameController(model, view)

view.aplicar_estilo()
st.logo(str(RAIZ / "recursos" / "logo.svg"), size="large")

# --- Páginas (menú lateral agrupado por secciones); cada vista recibe el controlador ---
paginas = {
    "inicio": st.Page(lambda: inicio.pagina(controller), title="Inicio", icon=":material/home:",
                      url_path="inicio", default=True),
    "carga": st.Page(lambda: carga.pagina(controller), title="Carga de datos", icon=":material/upload_file:",
                     url_path="carga"),
    "exploracion": st.Page(lambda: exploracion.pagina(controller), title="Exploración",
                           icon=":material/query_stats:", url_path="exploracion"),
    "no_supervisado": st.Page(lambda: no_supervisado.pagina(controller), title="ACP y clustering",
                              icon=":material/bubble_chart:", url_path="no-supervisado"),
    "clasificacion": st.Page(lambda: clasificacion.pagina(controller), title="Clasificación",
                             icon=":material/category:", url_path="clasificacion"),
    "regresion": st.Page(lambda: regresion.pagina(controller), title="Regresión", icon=":material/trending_up:",
                         url_path="regresion"),
}
controller.paginas = paginas   # para los botones que cambian de página

navegacion = st.navigation({
    "": [paginas["inicio"]],
    "Datos": [paginas["carga"], paginas["exploracion"]],
    "No supervisados": [paginas["no_supervisado"]],
    "Supervisados": [paginas["clasificacion"], paginas["regresion"]],
})
view.tarjeta_dataset_lateral(model.nombre, model.forma())
navegacion.run()
