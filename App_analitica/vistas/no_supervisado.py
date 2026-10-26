import numpy as np
import pandas as pd
import streamlit as st

from nucleo.no_supervisado import HAY_UMAP, NoSupervisado
from vistas.comunes import descargar_csv, requiere_datos, tabla
from vistas.estilo import encabezado, metricas, num


# --- Cálculos pesados en caché (se recalculan solo si cambian datos o parámetros) ---
@st.cache_data(show_spinner=False)
def _codo(df, cols, k_max, n_acp):
    return NoSupervisado(df, list(cols)).codo(k_max, n_acp)


@st.cache_data(show_spinner=False)
def _tsne(df, cols, perp):
    return NoSupervisado(df, list(cols)).tsne(perp)


@st.cache_data(show_spinner=False)
def _umap(df, cols, nn, md):
    return NoSupervisado(df, list(cols)).umap(nn, md)


def pagina():
    df = requiere_datos()
    encabezado("Aprendizaje no supervisado",
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
    ns = NoSupervisado(df, cols)
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
        opciones_color = ["(ninguno)"] + list(st.session_state.clusters) + list(df.columns)
        color_sel = c4.selectbox("Colorear individuos por", opciones_color, key="acp_color")
        ns.acp(n_comp, escalar)
        color = _color(df, ns, color_sel)

        a1, a2, a3, a4, a5 = st.tabs(["Varianza", "Plano principal", "Círculo de correlación", "Integrado", "Tablas"])
        with a1:
            tv = ns.tabla_varianza()
            n80 = int((tv["Acumulada (%)"] < 80).sum() + 1)
            kaiser = int((tv["Autovalor"] > 1).sum()) if escalar else None
            metricas([{"et": "CP1 + CP2", "val": f"{tv['Acumulada (%)'].iloc[1]:.2f} %".replace('.', ','),
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
            top = st.slider("Variables con nombre (las mejor representadas)", 5, len(cols), min(12, len(cols)),
                            key="acp_top")
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
                tabla(ns.correlacion_variables().iloc[:, :5], altura=420)
            with c2:
                st.markdown("**Calidad de representación (cos²) en el plano**")
                tabla(ns.cos2_variables((e1, e2)), altura=420)
            descargar_csv(ns.coordenadas(), "coordenadas_acp.csv", "Descargar coordenadas de los individuos")

    # ============================================================ K-means
    with t_km:
        with st.expander("Método del codo: ¿cuántos clusters?", expanded=False):
            c1, c2 = st.columns([1, 3])
            kmax = c1.number_input("k máximo", 3, 15, 10, key="codo_k")
            nacp_codo = c1.number_input("Sobre componentes del ACP (0 = datos completos)", 0, len(cols), 0,
                                        key="codo_acp")
            if c1.button("Calcular codo"):
                with st.spinner("Calculando..."):
                    st.session_state.codo = _codo(df, tuple(cols), kmax, nacp_codo or None)
            if "codo" in st.session_state:
                c2.plotly_chart(NoSupervisado.fig_codo(st.session_state.codo), width="stretch", key="ns_codo")
        c1, c2, c3, c4 = st.columns(4)
        k = c1.number_input("Cantidad de clusters (k)", 2, 20, 3, key="km_k")
        init = c2.selectbox("Inicialización", ["k-means++", "random"], key="km_init")
        n_init = c3.number_input("Arranques", 1, 50, 10, key="km_ninit")
        nacp = c4.number_input("Usar componentes del ACP (0 = no)", 0, len(cols), 0, key="km_acp")
        if st.button("Ejecutar K-means", type="primary"):
            r = ns.kmeans(k, init, n_init, nacp or None)
            st.session_state.clusters[f"K-means (k={k})"] = pd.Series(r["etiquetas"], index=ns.indice)
            st.session_state.km_res = {**r, "k": k}
        if "km_res" in st.session_state:
            _resultado_cluster(ns, df, st.session_state.km_res, f"K-means (k={st.session_state.km_res['k']})",
                               extra={"et": "Inercia", "val": num(st.session_state.km_res["inercia"], 1)})

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
                r = ns.hac(k_h, enlace, nacp_h or None)
                st.session_state.clusters[f"HAC {enlace} (k={k_h})"] = pd.Series(r["etiquetas"], index=ns.indice)
                st.session_state.hac_res = {**r, "k": k_h, "enlace": enlace, "nacp": nacp_h or None}
                st.session_state.hac_fig = ns.fig_dendrograma(k_h, enlace, nacp_h or None)
        if "hac_res" in st.session_state:
            h = st.session_state.hac_res
            st.plotly_chart(st.session_state.hac_fig, width="stretch", key="ns5")
            _resultado_cluster(ns, df, h, f"HAC {h['enlace']} (k={h['k']})")

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
        color_sel = c4.selectbox("Colorear por", ["(ninguno)"] + list(st.session_state.clusters) + list(df.columns),
                                 key="pr_color")
        if metodo == "UMAP" and not HAY_UMAP:
            st.error("Falta instalar umap-learn: pip install umap-learn")
        elif st.button(f"Calcular {metodo}", type="primary"):
            with st.spinner(f"Calculando {metodo}..."):
                if metodo == "t-SNE":
                    xy, kl = _tsne(df, tuple(cols), perp)
                    st.session_state.proj = (xy, f"t-SNE (perplexity {perp}) · KL = {kl:.4f}")
                else:
                    xy = _umap(df, tuple(cols), nn, md)
                    st.session_state.proj = (xy, f"UMAP (n_neighbors {nn}, min_dist {md})")
        if "proj" in st.session_state and len(st.session_state.proj[0]) == len(ns.indice):
            xy, titulo = st.session_state.proj
            st.plotly_chart(ns.fig_2d(xy, _color(df, ns, color_sel), color_sel, titulo), width="stretch", key="ns6")
            st.caption("Los ejes no tienen significado: importa qué puntos quedan juntos. "
                       "El tamaño de los grupos y la distancia entre grupos lejanos no se interpretan.")

    # ================================================== Clusters guardados
    with t_clu:
        if not st.session_state.clusters:
            st.info("Ejecutá K-means o HAC para guardar agrupamientos aquí.")
        else:
            claves = list(st.session_state.clusters)
            st.markdown("**Tamaño de cada cluster**")
            tam = pd.DataFrame({k: v.value_counts().sort_index() for k, v in st.session_state.clusters.items()})
            tabla(tam.fillna(0).astype(int))
            if len(claves) >= 2:
                a, b = st.columns(2)
                c_a = a.selectbox("Comparar", claves, 0)
                c_b = b.selectbox("con", claves, 1)
                st.markdown("**Tabla cruzada**")
                st.dataframe(pd.crosstab(st.session_state.clusters[c_a].rename(c_a),
                                         st.session_state.clusters[c_b].rename(c_b)), width="stretch")
            salida = df.copy()
            for k, v in st.session_state.clusters.items():
                salida[k] = v
            descargar_csv(salida, "datos_con_clusters.csv", "Descargar datos con los clusters")


def _color(df, ns, seleccion):
    if seleccion == "(ninguno)":
        return None
    if seleccion in st.session_state.clusters:
        return st.session_state.clusters[seleccion].reindex(ns.indice).values
    return df.loc[ns.indice, seleccion].values


def _resultado_cluster(ns, df, r, nombre, extra=None):
    items = [{"et": "Silhouette", "val": num(r["Silhouette"]), "nota": "de -1 a 1, más alto mejor", "tipo": "destacada"},
             {"et": "Calinski-Harabasz", "val": num(r["Calinski-Harabasz"], 1), "nota": "más alto mejor"},
             {"et": "Davies-Bouldin", "val": num(r["Davies-Bouldin"]), "nota": "más bajo mejor"}]
    tam = pd.Series(r["etiquetas"]).value_counts().sort_index()
    items.append({"et": "Tamaños", "val": " / ".join(str(v) for v in tam.values),
                  "tipo": "mala" if tam.min() < 0.02 * tam.sum() else ""})
    if extra:
        items.insert(3, extra)
    metricas(items, columnas=len(items))
    if tam.min() < 0.02 * tam.sum():
        st.warning("Hay un cluster con menos del 2 % de los datos: probablemente solo separa puntos atípicos.")
    p1, p2, p3, p4 = st.tabs(["Perfil", "Radar", "En el plano del ACP", "Medias por cluster"])
    with p1:
        st.plotly_chart(ns.fig_perfil(r["etiquetas"]), width="stretch", key=f"ns7_{nombre}")
        st.caption("Rojo: el cluster está por debajo del promedio; azul: por encima. Valores en desviaciones estándar.")
    with p2:
        st.plotly_chart(ns.fig_radar(r["etiquetas"]), width="stretch", key=f"ns8_{nombre}")
    with p3:
        ns.acp(2, True)
        st.plotly_chart(ns.fig_plano((0, 1), r["etiquetas"], "cluster"), width="stretch", key=f"ns9_{nombre}")
    with p4:
        perfil = ns.perfil(r["etiquetas"], df.select_dtypes(include=np.number))
        tabla(perfil, altura=420)
        descargar_csv(perfil, f"perfil_{nombre}.csv", "Descargar perfil", clave=f"dl_perfil_{nombre}")
