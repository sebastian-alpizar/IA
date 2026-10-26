"""
no_supervisado.py
NoSupervisado(EDA): ACP (plano, círculo, integrado), HAC, K-means y
proyecciones t-SNE / UMAP. Todo devuelve tablas o figuras de Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from . import graficos as g
from .eda import EDA

try:
    import umap as umap_lib
    HAY_UMAP = True
except ImportError:
    HAY_UMAP = False


class NoSupervisado(EDA):

    def __init__(self, datos: pd.DataFrame, columnas=None):
        super().__init__(datos)
        num = self.numericas().dropna()
        if columnas:
            num = num[[c for c in columnas if c in num.columns]]
        self.__num = num
        self.__acp = None

    @property
    def variables(self):
        return list(self.__num.columns)

    @property
    def indice(self):
        return self.__num.index

    def matriz(self, escalar=True):
        x = self.__num.values
        return StandardScaler().fit_transform(x) if escalar else x

    # ==================================================================
    # ACP
    # ==================================================================
    def acp(self, n_componentes=None, escalar=True):
        x = self.matriz(escalar)
        n = min(n_componentes or x.shape[1], x.shape[1], x.shape[0])
        modelo = PCA(n_components=n, random_state=42).fit(x)
        coords = modelo.transform(x)
        corr = np.array([[np.corrcoef(x[:, j], coords[:, k])[0, 1] for k in range(n)]
                         for j in range(x.shape[1])])
        self.__acp = {
            "modelo": modelo, "coords": coords,
            "var": 100 * modelo.explained_variance_ratio_,
            "corr": pd.DataFrame(corr, index=self.variables, columns=[f"CP{i+1}" for i in range(n)]),
            "escalar": escalar,
        }
        return self.tabla_varianza()

    def tabla_varianza(self):
        v = self.__acp["var"]
        return pd.DataFrame({"Varianza (%)": v.round(2), "Acumulada (%)": np.cumsum(v).round(2),
                             "Autovalor": self.__acp["modelo"].explained_variance_.round(3)},
                            index=[f"CP{i+1}" for i in range(len(v))])

    def coordenadas(self):
        c = self.__acp["coords"]
        return pd.DataFrame(c, index=self.indice, columns=[f"CP{i+1}" for i in range(c.shape[1])])

    def correlacion_variables(self):
        return self.__acp["corr"]

    def cos2_variables(self, ejes=(0, 1)):
        c = self.__acp["corr"].iloc[:, list(ejes)]
        return (c ** 2).sum(axis=1).sort_values(ascending=False).to_frame("cos² en el plano")

    def contribucion_individuos(self, ejes=(0, 1)):
        c = self.__acp["coords"][:, list(ejes)]
        contrib = 100 * (c ** 2) / (c ** 2).sum(axis=0)
        return pd.DataFrame(contrib, index=self.indice,
                            columns=[f"Contribución CP{e+1} (%)" for e in ejes]).round(3)

    def fig_varianza(self):
        v = self.__acp["var"][:30]
        x = [f"CP{i+1}" for i in range(len(v))]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=v, marker_color=g.AZUL, name="Individual",
                             text=[f"{a:.1f}" for a in v], textposition="outside"))
        fig.add_trace(go.Scatter(x=x, y=np.cumsum(v), mode="lines+markers", name="Acumulada",
                                 line=dict(color=g.ROJO)))
        fig.add_hline(y=80, line_dash="dot", line_color=g.TEXTO_SUAVE, annotation_text="80 %")
        fig.update_layout(height=420, title="Varianza explicada por componente", yaxis_title="%")
        return fig

    def _etiqueta_eje(self, e):
        return f"CP{e+1} ({self.__acp['var'][e]:.2f} %)"

    def fig_plano(self, ejes=(0, 1), color=None, nombre_color="", etiquetas=False):
        c = self.__acp["coords"]
        fig = go.Figure()
        if color is None:
            fig.add_trace(go.Scatter(x=c[:, ejes[0]], y=c[:, ejes[1]], mode="markers",
                                     marker=dict(color=g.AZUL, size=7, opacity=0.7),
                                     text=self.indice.astype(str), hovertemplate="%{text}<extra></extra>"))
        elif _es_categorico(color):
            color = pd.Series(color, index=self.indice).astype(str)
            for i, cat in enumerate(sorted(color.unique())):
                m = (color == cat).values
                fig.add_trace(go.Scatter(x=c[m, ejes[0]], y=c[m, ejes[1]], mode="markers", name=str(cat),
                                         marker=dict(color=g.color_cluster(i), size=7, opacity=0.8),
                                         text=self.indice[m].astype(str), hovertemplate="%{text}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=c[:, ejes[0]], y=c[:, ejes[1]], mode="markers",
                                     marker=dict(color=np.asarray(color), colorscale=g.SECUENCIAL, size=7,
                                                 colorbar=dict(title=nombre_color), opacity=0.85),
                                     text=self.indice.astype(str), hovertemplate="%{text}<extra></extra>"))
        if etiquetas:
            fig.add_trace(go.Scatter(x=c[:, ejes[0]], y=c[:, ejes[1]], mode="text",
                                     text=self.indice.astype(str), textposition="top center",
                                     showlegend=False, textfont=dict(size=10, color=g.TEXTO_SUAVE)))
        fig.add_hline(y=0, line_dash="dash", line_color=g.BORDE)
        fig.add_vline(x=0, line_dash="dash", line_color=g.BORDE)
        fig.update_layout(height=520, title="Plano principal", xaxis_title=self._etiqueta_eje(ejes[0]),
                          yaxis_title=self._etiqueta_eje(ejes[1]))
        return fig

    def fig_circulo(self, ejes=(0, 1), top=None):
        corr = self.__acp["corr"].iloc[:, list(ejes)]
        cos2 = (corr ** 2).sum(axis=1)
        mostrar = cos2.sort_values(ascending=False).index[:top] if top else corr.index
        fig = go.Figure()
        t = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=g.AZUL_CLARO),
                                 hoverinfo="skip", showlegend=False))
        for var in corr.index:
            x, y = corr.loc[var].values
            destacado = var in mostrar
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.3,
                               arrowcolor=g.AZUL if destacado else "#C9D2DE")
        # variables con flechas casi iguales se rotulan juntas para que no se encimen
        grupos = []
        for var in mostrar:
            punto = corr.loc[var].values
            for gr in grupos:
                if np.linalg.norm(gr["p"] - punto) < 0.16:
                    gr["n"].append(var)
                    break
            else:
                grupos.append({"p": punto, "n": [var]})
        for gr in grupos:
            x, y = gr["p"] * 1.1
            fig.add_annotation(x=x, y=y, text="<br>".join(gr["n"]), showarrow=False, align="left" if x >= 0 else "right",
                               xanchor="left" if x >= 0 else "right", font=dict(size=10, color=g.AZUL),
                               bgcolor="rgba(255,255,255,0.75)")
        fig.add_trace(go.Scatter(x=corr.iloc[:, 0], y=corr.iloc[:, 1], mode="markers",
                                 marker=dict(size=4, color="rgba(0,0,0,0)"), text=corr.index,
                                 hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>", showlegend=False))
        fig.add_hline(y=0, line_dash="dash", line_color=g.BORDE)
        fig.add_vline(x=0, line_dash="dash", line_color=g.BORDE)
        fig.update_layout(height=560, title="Círculo de correlación",
                          xaxis=dict(range=[-1.2, 1.2], title=self._etiqueta_eje(ejes[0]), scaleanchor="y"),
                          yaxis=dict(range=[-1.2, 1.2], title=self._etiqueta_eje(ejes[1])))
        return fig

    def fig_integrado(self, ejes=(0, 1), top=15, color=None):
        """Plano principal + flechas de las variables mejor representadas (biplot)."""
        fig = self.fig_plano(ejes, color=color)
        c = self.__acp["coords"]
        corr = self.__acp["corr"].iloc[:, list(ejes)]
        escala = 0.8 * min(np.abs(c[:, ejes[0]]).max(), np.abs(c[:, ejes[1]]).max())
        cos2 = (corr ** 2).sum(axis=1).sort_values(ascending=False)
        fig.update_traces(marker_opacity=0.35, selector=dict(mode="markers"))   # puntos suaves, flechas al frente
        grupos = []
        for var in cos2.index[:top]:
            x, y = corr.loc[var].values * escala
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1.8, arrowcolor=g.ROJO)
            punto = corr.loc[var].values
            for gr in grupos:
                if np.linalg.norm(gr["p"] - punto) < 0.16:
                    gr["n"].append(var)
                    break
            else:
                grupos.append({"p": punto, "n": [var]})
        for gr in grupos:
            x, y = gr["p"] * escala * 1.1
            fig.add_annotation(x=x, y=y, text="<br>".join(gr["n"]), showarrow=False,
                               xanchor="left" if x >= 0 else "right", align="left" if x >= 0 else "right",
                               font=dict(size=10, color=g.ROJO), bgcolor="rgba(255,255,255,0.8)")
        fig.update_layout(title="Integrado: plano principal y variables")
        return fig

    # ==================================================================
    # Clustering
    # ==================================================================
    def datos_cluster(self, n_acp=None):
        x = self.matriz(True)
        if n_acp:
            x = PCA(n_components=min(n_acp, x.shape[1]), random_state=42).fit_transform(x)
        return x

    @staticmethod
    def metricas(x, etiquetas):
        if len(np.unique(etiquetas)) < 2:
            return {"Silhouette": np.nan, "Calinski-Harabasz": np.nan, "Davies-Bouldin": np.nan}
        return {"Silhouette": silhouette_score(x, etiquetas),
                "Calinski-Harabasz": calinski_harabasz_score(x, etiquetas),
                "Davies-Bouldin": davies_bouldin_score(x, etiquetas)}

    def kmeans(self, k=3, init="k-means++", n_init=10, n_acp=None):
        x = self.datos_cluster(n_acp)
        modelo = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=42).fit(x)
        res = {"etiquetas": modelo.labels_, "inercia": modelo.inertia_, **self.metricas(x, modelo.labels_)}
        return res

    def codo(self, k_max=10, n_acp=None):
        x = self.datos_cluster(n_acp)
        filas = []
        for k in range(2, k_max + 1):
            m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
            filas.append({"k": k, "Inercia": m.inertia_, "Silhouette": silhouette_score(x, m.labels_)})
        return pd.DataFrame(filas).set_index("k")

    @staticmethod
    def fig_codo(tabla):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Método del codo (inercia)", "Silhouette por k"))
        fig.add_trace(go.Scatter(x=tabla.index, y=tabla["Inercia"], mode="lines+markers",
                                 line=dict(color=g.AZUL), name="Inercia"), 1, 1)
        fig.add_trace(go.Scatter(x=tabla.index, y=tabla["Silhouette"], mode="lines+markers",
                                 line=dict(color=g.ROJO), name="Silhouette"), 1, 2)
        mejor = tabla["Silhouette"].idxmax()
        fig.add_vline(x=mejor, line_dash="dot", line_color=g.TEXTO_SUAVE, row=1, col=2)
        fig.update_layout(height=380, showlegend=False)
        fig.update_xaxes(title_text="k", dtick=1)
        return fig

    def hac(self, k=3, enlace="ward", n_acp=None):
        x = self.datos_cluster(n_acp)
        etiquetas = AgglomerativeClustering(n_clusters=k, linkage=enlace).fit_predict(x)
        return {"etiquetas": etiquetas, **self.metricas(x, etiquetas)}

    def fig_dendrograma(self, k=3, enlace="ward", n_acp=None, hojas=30):
        x = self.datos_cluster(n_acp)
        z = linkage(x, method=enlace)
        corte = (z[-k, 2] + z[-k + 1, 2]) / 2 if k > 1 else None
        d = dendrogram(z, truncate_mode="lastp", p=hojas, no_plot=True, color_threshold=corte)
        fig = go.Figure()
        colores = {}
        for xs, ys, col in zip(d["icoord"], d["dcoord"], d["color_list"]):
            if col not in colores:
                colores[col] = g.color_cluster(len(colores)) if col != "C0" else g.TEXTO_SUAVE
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colores[col], width=1.6),
                                     hoverinfo="skip", showlegend=False))
        if corte:
            fig.add_hline(y=corte, line_dash="dash", line_color=g.ROJO,
                          annotation_text=f"corte en k = {k}", annotation_position="top right")
        fig.update_layout(height=440, title=f"Dendrograma (enlace {enlace}, últimas {hojas} uniones)",
                          xaxis=dict(tickvals=[5 + 10 * i for i in range(len(d["ivl"]))], ticktext=d["ivl"],
                                     tickangle=-60, tickfont=dict(size=9)),
                          yaxis_title="Distancia de fusión")
        return fig

    def perfil(self, etiquetas, original=None):
        """Media de cada variable por cluster (escala original) y tamaño."""
        datos = (original if original is not None else self.__num).loc[self.indice]
        num = datos.select_dtypes(include=np.number)
        t = num.groupby(np.asarray(etiquetas)).mean().T
        t.columns = [f"Cluster {c}" for c in t.columns]
        return t

    def fig_perfil(self, etiquetas, variables=None, maximo=25):
        z = pd.DataFrame(self.matriz(True), index=self.indice, columns=self.variables)
        medias = z.groupby(np.asarray(etiquetas)).mean()
        if variables:
            medias = medias[variables]
        else:
            orden = (medias.max() - medias.min()).sort_values(ascending=False).index[:maximo]
            medias = medias[orden]
        lim = float(np.abs(medias.values).max())
        tam = pd.Series(etiquetas).value_counts().sort_index()
        fig = go.Figure(go.Heatmap(z=medias.T.values, x=[f"Cluster {c} (n={tam[c]})" for c in medias.index],
                                   y=medias.columns, zmin=-lim, zmax=lim, colorscale=g.DIVERGENTE,
                                   text=np.round(medias.T.values, 2), texttemplate="%{text}",
                                   colorbar=dict(title="z")))
        fig.update_layout(height=max(360, 24 * len(medias.columns) + 120),
                          title="Perfil de los clusters (media estandarizada; variables que más los separan)",
                          yaxis=dict(autorange="reversed"))
        return fig

    def fig_radar(self, etiquetas, variables=None, maximo=10):
        z = pd.DataFrame(self.matriz(True), index=self.indice, columns=self.variables)
        medias = z.groupby(np.asarray(etiquetas)).mean()
        if not variables:
            variables = list((medias.max() - medias.min()).sort_values(ascending=False).index[:maximo])
        m = medias[variables]
        rango = (m.max() - m.min()).replace(0, 1)
        m = 100 * (m - m.min()) / rango
        fig = go.Figure()
        for i, c in enumerate(m.index):
            fig.add_trace(go.Scatterpolar(r=list(m.loc[c]) + [m.loc[c].iloc[0]], theta=variables + [variables[0]],
                                          fill="toself", name=f"Cluster {c}", opacity=0.55,
                                          line=dict(color=g.color_cluster(i))))
        fig.update_layout(height=480, title="Radar de los centros (0 = menor, 100 = mayor entre clusters)",
                          polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)))
        return fig

    # ==================================================================
    # Proyecciones 2D
    # ==================================================================
    def tsne(self, perplexity=30):
        x = self.matriz(True)
        modelo = TSNE(n_components=2, perplexity=min(perplexity, len(x) - 1), random_state=42)
        return modelo.fit_transform(x), modelo.kl_divergence_

    def umap(self, n_neighbors=15, min_dist=0.1):
        if not HAY_UMAP:
            raise ImportError("Falta instalar umap-learn")
        x = self.matriz(True)
        return umap_lib.UMAP(n_neighbors=min(n_neighbors, len(x) - 1), min_dist=min_dist,
                             random_state=42).fit_transform(x)

    def fig_2d(self, coords, color=None, nombre_color="", titulo=""):
        fig = go.Figure()
        if color is None:
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                                     marker=dict(color=g.AZUL, size=6, opacity=0.7), text=self.indice.astype(str),
                                     hovertemplate="%{text}<extra></extra>"))
        elif _es_categorico(color):
            color = pd.Series(np.asarray(color)).astype(str)
            for i, cat in enumerate(sorted(color.unique())):
                m = (color == cat).values
                fig.add_trace(go.Scatter(x=coords[m, 0], y=coords[m, 1], mode="markers", name=str(cat),
                                         marker=dict(color=g.color_cluster(i), size=6, opacity=0.8)))
        else:
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                                     marker=dict(color=np.asarray(color), colorscale=g.SECUENCIAL, size=6,
                                                 colorbar=dict(title=nombre_color))))
        fig.update_layout(height=520, title=titulo, xaxis=dict(showticklabels=False, title=""),
                          yaxis=dict(showticklabels=False, title=""))
        return fig


def _es_categorico(valores):
    s = pd.Series(np.asarray(valores))
    return (not pd.api.types.is_numeric_dtype(s)) or (pd.api.types.is_integer_dtype(s) and s.nunique() <= 12)
