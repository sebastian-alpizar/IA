"""
inicio.py (página)
Portada de la app: qué hace y los pasos del análisis.
"""

import streamlit as st


def pagina(ctrl):
    st.markdown("""
    <div class='hero'>
      <div class='chip'>EDA · No supervisado · Clasificación · Regresión</div>
      <h1 style='margin-top:14px'>Analítica de datos</h1>
      <p>Subí un CSV y hacé todo el análisis en un solo lugar: exploración, ACP y clustering (como discoveR)
      y modelos de clasificación y regresión (como predictoR), con comparación y benchmarking de modelos
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
        <p>Clasificación (KNN, DT, RF, XGBoost, AdaBoost del Lab 2) y regresión (lineales, Lasso, Ridge, SVR y árboles del Lab 3); compará y aplicá el mejor a datos nuevos.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("Empezar: cargar datos", type="primary", width="stretch"):
            ctrl.ir_a("carga")
    with c2:
        if ctrl.model.hay_datos and st.button("Ir a exploración", width="stretch"):
            ctrl.ir_a("exploracion")
