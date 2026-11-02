"""
no_supervisado.py (página)
Vista de Aprendizaje no supervisado (Laboratorio 1): ACP, K-means, HAC, t-SNE y UMAP.
Los cálculos los hace NoSupervisado (carpeta no_supervisado/) y las acciones el controlador.
"""

import numpy as np
import pandas as pd
import streamlit as st

from no_supervisado import HAY_UMAP, NoSupervisado
from shared.formato import num


def pagina(ctrl):
    if not ctrl.requiere_datos():
        return
    v = ctrl.view
    m = ctrl.model
    df = m.datos
    v.encabezado("Aprendizaje no supervisado",
               "Reducción de dimensionalidad (ACP, t-SNE, UMAP) y agrupamiento (K-means, HAC).", "Paso 3")
    numericas = list(df.select_dtypes(include=np.number).columns)
    if len(numericas) < 2:
        st.warning("Se necesitan al menos dos variables numéricas.")
        return

    with st.expander(f"Variables que se usan ({len(numericas)} numéricas disponibles)", expanded=False):
        cols = st.multiselect("Variables", numericas, default=numericas, key="ns_cols",
                              help="Quitá identificadores o la variable respuesta si la hay.")
    cols = st.session_state.get("ns_cols") or numericas
    if len(cols) < 2:
        st.warning("Elegí al menos dos variables.")
        return
    ns = m.no_supervisado(cols)
    if len(ns.indice) < len(df):
        st.caption(f"Se ignoran {len(df) - len(ns.indice)} filas con valores vacíos.")

    t_acp, t_km, t_hac, t_proj, t_clu = st.tabs(["ACP", "K-means", "Jerárquico (HAC)", "t-SNE y UMAP", "Clusters guardados"])

    # ================================================================ ACP
    with t_acp:
        c1, c2, c3, c4 = st.columns(4)
        escalar = c1.toggle("Estandarizar", True, help="Recomendado si las variables están en escalas distintas.")
        n_comp = c2.number_input("Componentes", 2, len(cols), min(len(cols), 10))
        ejes = c3.selectbox("Plano", [f"CP{i+1} y CP{j+1}" for i in range(min(n_comp, 5))
                                      for j in range(i + 1, min(n_comp, 5))])
        e1, e2 = [int(x.replace("CP", "")) - 1 for x in ejes.split(" y ")]
        opciones_color = ["(ninguno)"] + list(m.clusters) + list(df.columns)
        color_sel = c4.selectbox("Colorear individuos por", opciones_color, key="acp_color")
        ctrl.acp(ns, n_comp, escalar)
        color = _color(m, ns, color_sel)

        a1, a2, a3, a4, a5 = st.tabs(["Varianza", "Plano principal", "Círculo de correlación", "Integrado", "Tablas"])
        with a1:
            tv = ns.tabla_varianza()
            n80 = int((tv["Acumulada (%)"] < 80).sum() + 1)
            kaiser = int((tv["Autovalor"] > 1).sum()) if escalar else None
            v.metricas([{"et": "CP1 + CP2", "val": f"{tv['Acumulada (%)'].iloc[1]:.2f} %".replace('.', ','),
                       "nota": "información en el plano principal", "tipo": "destacada"},
                      {"et": f"Con {n_comp} componentes", "val": f"{tv['Acumulada (%)'].iloc[-1]:.2f} %".replace('.', ',')},
                      {"et": "Componentes para 80 %", "val": n80 if tv["Acumulada (%)"].iloc[-1] >= 80 else f"> {n_comp}"},
                      {"et": "Criterio de Kaiser", "val": kaiser if kaiser is not None else "—",
                       "nota": "autovalor > 1"}])
            st.plotly_chart(ns.fig_varianza(), width="stretch", key="ns1")
        with a2:
            st.plotly_chart(ns.fig_plano((e1, e2), color, color_sel, etiquetas=len(ns.indice) <= 60), width="stretch", key="ns2")
            st.caption("Cada punto es un individuo. Individuos cercanos tienen valores parecidos en las variables.")
        with a3:
            top = st.slider("Variables a dibujar (las mejor representadas en el plano)", 2, len(cols),
                            min(15, len(cols)), key="acp_top")
            st.plotly_chart(ns.fig_circulo((e1, e2), top), width="stretch", key="ns3")
            st.caption("Flecha larga = variable bien representada. Flechas en la misma dirección = correlación "
                       "positiva; opuestas = negativa; en ángulo recto = sin relación.")
        with a4:
            top_i = st.slider("Variables a dibujar", 3, len(cols), min(12, len(cols)), key="acp_top_i")
            st.plotly_chart(ns.fig_integrado((e1, e2), top_i, color), width="stretch", key="ns4")
            st.caption("Los individuos del lado hacia el que apunta una flecha tienen valores altos en esa variable.")
        with a5:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Correlación variable - componente**")
                v.tabla(ns.correlacion_variables().iloc[:, :5], altura=420)
            with c2:
                st.markdown("**Calidad de representación (cos²) en el plano**")
                v.tabla(ns.cos2_variables((e1, e2)), altura=420)
            st.markdown(f"**Contribución de los individuos a CP{e1 + 1} y CP{e2 + 1}** (los 15 que más aportan)")
            contrib = ns.contribucion_individuos((e1, e2))
            v.tabla(contrib.loc[contrib.sum(axis=1).sort_values(ascending=False).index[:15]])
            st.caption("Porcentaje de la varianza de cada componente que aporta el individuo; los que más aportan "
                       "son los que definen el eje.")
            d1, d2 = st.columns(2)
            with d1:
                v.descargar_csv(ns.coordenadas(), "coordenadas_acp.csv", "Descargar coordenadas de los individuos")
            with d2:
                v.descargar_csv(contrib, "contribuciones_acp.csv", "Descargar contribuciones",
                                clave="dl_contribuciones")

    # ============================================================ K-means
    with t_km:
        with st.expander("Método del codo: ¿cuántos clusters?", expanded=False):
            c1, c2 = st.columns([1, 3])
            kmax = c1.number_input("k máximo", 3, 15, 10, key="codo_k")
            nacp_codo = c1.number_input("Sobre componentes del ACP (0 = datos completos)", 0, len(cols), 0,
                                        key="codo_acp")
            if c1.button("Calcular codo"):
                with st.spinner("Calculando..."):
                    ctrl.codo(cols, kmax, nacp_codo)
            codo = m.resultado("ns", "codo")
            if codo is not None:
                c2.plotly_chart(NoSupervisado.fig_codo(codo), width="stretch", key="ns_codo")
        c1, c2, c3, c4 = st.columns(4)
        k = c1.number_input("Cantidad de clusters (k)", 2, 20, 3, key="km_k")
        init = c2.selectbox("Inicialización", ["k-means++", "random"], key="km_init")
        n_init = c3.number_input("Arranques", 1, 50, 10, key="km_ninit")
        nacp = c4.number_input("Usar componentes del ACP (0 = no)", 0, len(cols), 0, key="km_acp")
        if st.button("Ejecutar K-means", type="primary"):
            ctrl.kmeans(ns, k, init, n_init, nacp)
        km = m.resultado("ns", "kmeans")
        if km is not None:
            _resultado_cluster(ctrl, ns, km, f"K-means (k={km['k']})",
                               extra={"et": "Inercia", "val": num(km["inercia"], 1)})

    # ================================================================ HAC
    with t_hac:
        c1, c2, c3 = st.columns(3)
        k_h = c1.number_input("Cantidad de clusters (k)", 2, 20, 3, key="hac_k")
        enlace = c2.selectbox("Enlace", ["ward", "complete", "average", "single"], key="hac_enlace",
                              help="Ward: grupos compactos y parejos. Simple: puede formar cadenas.")
        nacp_h = c3.number_input("Usar componentes del ACP (0 = no)", 0, len(cols), 0, key="hac_acp")
        if len(ns.indice) > 6000:
            st.warning("Con más de 6000 filas el jerárquico puede tardar bastante.")
        if st.button("Ejecutar HAC", type="primary"):
            with st.spinner("Construyendo el árbol..."):
                ctrl.hac(ns, k_h, enlace, nacp_h)
        h = m.resultado("ns", "hac")
        if h is not None:
            st.plotly_chart(m.resultado("ns", "hac_fig"), width="stretch", key="ns5")
            _resultado_cluster(ctrl, ns, h, f"HAC {h['enlace']} (k={h['k']})")

    # ===================================================== t-SNE / UMAP
    with t_proj:
        c1, c2, c3, c4 = st.columns(4)
        metodo = c1.radio("Método", ["t-SNE", "UMAP"], horizontal=True, key="pr_met")
        if metodo == "t-SNE":
            perp = c2.slider("Perplexity", 5, 100, 30, key="pr_perp",
                             help="Aproximadamente cuántos vecinos mira cada punto.")
        else:
            nn = c2.slider("Vecinos (n_neighbors)", 2, 100, 15, key="pr_nn")
            md = c3.slider("Distancia mínima (min_dist)", 0.0, 1.0, 0.1, 0.05, key="pr_md")
        color_sel = c4.selectbox("Colorear por", ["(ninguno)"] + list(m.clusters) + list(df.columns),
                                 key="pr_color")
        if metodo == "UMAP" and not HAY_UMAP:
            st.error("Falta instalar umap-learn: pip install umap-learn")
        elif st.button(f"Calcular {metodo}", type="primary"):
            with st.spinner(f"Calculando {metodo}..."):
                if metodo == "t-SNE":
                    ctrl.proyeccion(metodo, cols, perplexity=perp)
                else:
                    ctrl.proyeccion(metodo, cols, n_neighbors=nn, min_dist=md)
        proyeccion = m.resultado("ns", "proyeccion")
        if proyeccion is not None and len(proyeccion[0]) == len(ns.indice):
            xy, titulo = proyeccion
            st.plotly_chart(ns.fig_2d(xy, _color(m, ns, color_sel), color_sel, titulo), width="stretch", key="ns6")
            st.caption("Los ejes no tienen significado: importa qué puntos quedan juntos. "
                       "El tamaño de los grupos y la distancia entre grupos lejanos no se interpretan.")

    # ================================================== Clusters guardados
    with t_clu:
        if not m.clusters:
            st.info("Ejecutá K-means o HAC para guardar agrupamientos aquí.")
        else:
            claves = list(m.clusters)
            st.markdown("**Tamaño de cada cluster**")
            tam = pd.DataFrame({k: e.value_counts().sort_index() for k, e in m.clusters.items()})
            v.tabla(tam.fillna(0).astype(int))
            if len(claves) >= 2:
                a, b = st.columns(2)
                c_a = a.selectbox("Comparar", claves, 0)
                c_b = b.selectbox("con", claves, 1)
                st.markdown("**Tabla cruzada**")
                st.dataframe(pd.crosstab(m.clusters[c_a].rename(c_a),
                                         m.clusters[c_b].rename(c_b)), width="stretch")
            salida = df.copy()
            for nombre, etiquetas in m.clusters.items():
                salida[nombre] = etiquetas
            v.descargar_csv(salida, "datos_con_clusters.csv", "Descargar datos con los clusters")


def _color(m, ns, seleccion):
    if seleccion == "(ninguno)":
        return None
    if seleccion in m.clusters:
        return m.clusters[seleccion].reindex(ns.indice).values
    return m.datos.loc[ns.indice, seleccion].values


def _resultado_cluster(ctrl, ns, r, nombre, extra=None):
    v = ctrl.view
    df = ctrl.model.datos
    items = [{"et": "Silhouette", "val": num(r["Silhouette"]), "nota": "de -1 a 1, más alto mejor", "tipo": "destacada"},
             {"et": "Calinski-Harabasz", "val": num(r["Calinski-Harabasz"], 1), "nota": "más alto mejor"},
             {"et": "Davies-Bouldin", "val": num(r["Davies-Bouldin"]), "nota": "más bajo mejor"}]
    tam = pd.Series(r["etiquetas"]).value_counts().sort_index()
    items.append({"et": "Tamaños", "val": " / ".join(str(n) for n in tam.values),
                  "tipo": "mala" if tam.min() < 0.02 * tam.sum() else ""})
    if extra:
        items.insert(3, extra)
    v.metricas(items, columnas=len(items))
    if tam.min() < 0.02 * tam.sum():
        st.warning("Hay un cluster con menos del 2 % de los datos: probablemente solo separa puntos atípicos.")
    p1, p2, p3, p4 = st.tabs(["Perfil", "Radar", "En el plano del ACP", "Medias por cluster"])
    with p1:
        st.plotly_chart(ns.fig_perfil(r["etiquetas"]), width="stretch", key=f"ns7_{nombre}")
        st.caption("Rojo: el cluster está por debajo del promedio; azul: por encima. Valores en desviaciones estándar.")
    with p2:
        st.plotly_chart(ns.fig_radar(r["etiquetas"]), width="stretch", key=f"ns8_{nombre}")
    with p3:
        ctrl.acp(ns, 2, True)
        st.plotly_chart(ns.fig_plano((0, 1), r["etiquetas"], "cluster"), width="stretch", key=f"ns9_{nombre}")
    with p4:
        perfil = ns.perfil(r["etiquetas"], df.select_dtypes(include=np.number))
        v.tabla(perfil, altura=420)
        v.descargar_csv(perfil, f"perfil_{nombre}.csv", "Descargar perfil", clave=f"dl_perfil_{nombre}")
