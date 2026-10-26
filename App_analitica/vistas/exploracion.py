import numpy as np
import streamlit as st

from nucleo.eda import EDA
from vistas.comunes import descargar_csv, requiere_datos, tabla
from vistas.estilo import encabezado


def pagina():
    df = requiere_datos()
    e = EDA(df)
    encabezado("Exploración", "Estadísticas básicas, distribuciones, relaciones entre variables y valores atípicos.",
               "Paso 2")

    num = list(e.numericas().columns)
    todas = list(df.columns)
    t1, t2, t3, t4, t5 = st.tabs(["Resumen", "Distribución", "Dispersión", "Correlación", "Atípicos y normalidad"])

    with t1:
        est = e.estadisticas()
        if est.empty:
            st.info("No hay variables numéricas.")
        else:
            tabla(est, altura=min(560, 38 + 35 * len(est)))
            descargar_csv(est, "estadisticas.csv", "Descargar estadísticas")
        cat = e.categoricas().columns
        if len(cat):
            st.markdown("**Variables categóricas**")
            cols = st.columns(min(3, len(cat)))
            for i, c in enumerate(cat[:6]):
                with cols[i % len(cols)]:
                    st.caption(c)
                    tabla(e.frecuencias(c).head(10))

    with t2:
        c1, c2 = st.columns([1, 3])
        with c1:
            var = st.selectbox("Variable", todas, key="eda_var")
            s = df[var]
            if np.issubdtype(s.dtype, np.number):
                st.metric("Media", f"{s.mean():.4g}")
                st.metric("Mediana", f"{s.median():.4g}")
                st.metric("Desv. estándar", f"{s.std():.4g}")
                st.metric("Asimetría", f"{s.skew():.3f}",
                          help="> 0: cola a la derecha; < 0: cola a la izquierda")
            else:
                st.metric("Categorías", s.nunique())
                st.metric("Más frecuente", str(s.mode().iloc[0]))
        with c2:
            st.plotly_chart(e.fig_distribucion(var), width="stretch", key="eda1")
        if len(num) > 1:
            st.markdown("**Varias variables a la vez**")
            elegidas = st.multiselect("Diagramas de caja", num, default=num[:min(6, len(num))], key="eda_box")
            if elegidas:
                st.plotly_chart(e.fig_boxplots(elegidas), width="stretch", key="eda2")

    with t3:
        if len(num) < 2:
            st.info("Se necesitan al menos dos variables numéricas.")
        else:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("Eje X", num, 0, key="eda_x")
            y = c2.selectbox("Eje Y", num, 1, key="eda_y")
            color = c3.selectbox("Color", ["(ninguno)"] + todas, key="eda_color")
            st.plotly_chart(e.fig_dispersion(x, y, None if color == "(ninguno)" else color), width="stretch", key="eda3")
            st.caption(f"Correlación de Pearson entre {x} y {y}: {df[x].corr(df[y]):.4f}")

    with t4:
        if len(num) < 2:
            st.info("Se necesitan al menos dos variables numéricas.")
        else:
            c1, c2 = st.columns([1, 3])
            metodo = c1.radio("Método", ["pearson", "spearman", "kendall"], key="eda_met",
                              help="Pearson: relación lineal. Spearman/Kendall: por rangos, más robustos a atípicos.")
            if len(num) > 40:
                elegidas = c2.multiselect("Variables (son muchas, escogé algunas)", num, default=num[:25])
                sub = EDA(df[elegidas]) if len(elegidas) > 1 else e
            else:
                sub = e
            st.plotly_chart(sub.fig_correlacion(metodo), width="stretch", key="eda4")
            corr = e.correlacion(metodo)
            pares = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
                     .rename("r").reset_index().rename(columns={"level_0": "Variable 1", "level_1": "Variable 2"}))
            pares = pares.reindex(pares["r"].abs().sort_values(ascending=False).index).head(15)
            st.markdown("**Pares más correlacionados**")
            tabla(pares.set_index(["Variable 1", "Variable 2"]))

    with t5:
        if not num:
            st.info("No hay variables numéricas.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Atípicos por la regla 1,5 × IQR**")
                tabla(e.outliers(), altura=420)
            with c2:
                st.markdown("**Prueba de normalidad (D'Agostino)**")
                tabla(e.normalidad(), altura=420)
                st.caption("p-valor < 0,05: se rechaza que la variable sea normal.")
