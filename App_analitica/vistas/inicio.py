import streamlit as st

from vistas.comunes import datos


def pagina():
    st.markdown("""
    <div class='hero'>
      <div class='chip'>EDA · No supervisado · Clasificación</div>
      <h1 style='margin-top:14px'>Analítica de datos</h1>
      <p>Subí un CSV y hacé todo el análisis en un solo lugar: exploración, ACP y clustering (como discoveR)
      y modelos de clasificación (como predictoR), con comparación de modelos
      y predicción de datos nuevos.</p>
    </div>
    <div class='pasos'>
      <div class='pasoc'><div class='n'>1</div><h3>Cargá los datos</h3>
        <p>CSV con detección automática de separador y decimal, o uno de los datasets de ejemplo.</p></div>
      <div class='pasoc'><div class='n'>2</div><h3>Exploralos</h3>
        <p>Estadísticas, distribuciones, correlaciones, atípicos y normalidad.</p></div>
      <div class='pasoc'><div class='n'>3</div><h3>Buscá estructura</h3>
        <p>ACP con plano, círculo e integrado; K-means, HAC, t-SNE y UMAP.</p></div>
      <div class='pasoc'><div class='n'>4</div><h3>Predecí</h3>
        <p>Entrená, compará modelos (KNN, DT, RF, XGBoost, AdaBoost del Lab 2 y más) y aplicá el mejor a datos nuevos.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Empezar: cargar datos", type="primary", width="stretch"):
            st.switch_page(st.session_state["_pagina_carga"])
    with c2:
        if datos() is not None and st.button("Ir a exploración", width="stretch"):
            st.switch_page(st.session_state["_pagina_eda"])
