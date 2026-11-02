"""
eda.py
Clase base del framework (EDA). NoSupervisado (carpeta no_supervisado/) y
Supervisado (carpeta supervisado/) heredan de ella.
Todos los métodos devuelven tablas (DataFrame) o figuras de Plotly para que
la interfaz solo tenga que mostrarlas.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from shared import graficos as g


class EDA:
    """
    Encapsula un pandas.DataFrame y expone los métodos de análisis exploratorio:
    resúmenes, estadísticas, atípicos, normalidad, correlación, limpieza y gráficos.
    """

    def __init__(self, datos: pd.DataFrame):
        self.__datos = datos.copy()

    # ------------------------------------------------------------------
    # Datos
    # ------------------------------------------------------------------
    @property
    def datos(self):
        return self.__datos

    def numericas(self):
        return self.__datos.select_dtypes(include=np.number)

    def categoricas(self):
        return self.__datos.select_dtypes(exclude=np.number)

    # ------------------------------------------------------------------
    # Resúmenes
    # ------------------------------------------------------------------
    def resumen(self):
        d = self.__datos
        return {"filas": d.shape[0], "columnas": d.shape[1],
                "numericas": self.numericas().shape[1], "categoricas": self.categoricas().shape[1],
                "nulos": int(d.isna().sum().sum()), "duplicados": int(d.duplicated().sum())}

    def tipos(self):
        d = self.__datos
        return pd.DataFrame({
            "Tipo": ["Numérica" if pd.api.types.is_numeric_dtype(d[c]) else "Categórica" for c in d.columns],
            "Valores distintos": d.nunique().values,
            "Nulos": d.isna().sum().values,
            "Ejemplo": [str(d[c].dropna().iloc[0]) if d[c].notna().any() else "" for c in d.columns],
        }, index=d.columns)

    def estadisticas(self):
        num = self.numericas()
        if num.empty:
            return pd.DataFrame()
        t = num.describe().T
        t["asimetría"] = num.skew()
        t["curtosis"] = num.kurt()
        t = t.rename(columns={"count": "n", "mean": "media", "std": "desv. est.", "min": "mínimo",
                              "25%": "Q1", "50%": "mediana", "75%": "Q3", "max": "máximo"})
        return t

    def frecuencias(self, columna):
        conteo = self.__datos[columna].value_counts(dropna=False)
        return pd.DataFrame({"Frecuencia": conteo, "Porcentaje": (100 * conteo / conteo.sum()).round(2)})

    def outliers(self):
        filas = []
        for c in self.numericas().columns:
            s = self.__datos[c].dropna()
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((s < li) | (s > ls)).sum())
            filas.append({"Variable": c, "Límite inferior": li, "Límite superior": ls,
                          "Atípicos": n, "% atípicos": round(100 * n / max(len(s), 1), 2)})
        return pd.DataFrame(filas).set_index("Variable").sort_values("Atípicos", ascending=False)

    def normalidad(self):
        """Prueba de normalidad (D'Agostino) por variable: p < 0.05 indica que no es normal."""
        filas = []
        for c in self.numericas().columns:
            s = self.__datos[c].dropna()
            if len(s) >= 8 and s.nunique() > 2:
                _, p = stats.normaltest(s)
                filas.append({"Variable": c, "Asimetría": s.skew(), "p-valor": p,
                              "¿Normal? (α=0.05)": "Sí" if p >= 0.05 else "No"})
        return pd.DataFrame(filas).set_index("Variable") if filas else pd.DataFrame()

    def correlacion(self, metodo="pearson"):
        return self.numericas().corr(method=metodo)

    # ------------------------------------------------------------------
    # Limpieza (devuelven un DataFrame nuevo, no modifican el objeto)
    # ------------------------------------------------------------------
    def tratar_nulos(self, estrategia="eliminar"):
        d = self.__datos.copy()
        if estrategia == "eliminar":
            return d.dropna()
        if estrategia == "imputar":
            for c in d.columns:
                if d[c].isna().any():
                    if pd.api.types.is_numeric_dtype(d[c]):
                        d[c] = d[c].fillna(d[c].median())
                    else:
                        d[c] = d[c].fillna(d[c].mode().iloc[0])
            return d
        return d

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    def fig_distribucion(self, columna):
        s = self.__datos[columna].dropna()
        if pd.api.types.is_numeric_dtype(s):
            fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2], shared_xaxes=True, vertical_spacing=0.04)
            fig.add_trace(go.Histogram(x=s, nbinsx=30, marker_color=g.AZUL, opacity=0.85, name="Frecuencia"), 1, 1)
            fig.add_trace(go.Box(x=s, marker_color=g.AZUL, name="", boxpoints="outliers", orientation="h"), 2, 1)
            fig.add_vline(x=s.mean(), line_dash="dash", line_color=g.ROJO, row=1, col=1,
                          annotation_text=f"media {s.mean():.3g}", annotation_position="top right")
            fig.update_layout(showlegend=False, height=420, title=f"Distribución de {columna}")
            fig.update_yaxes(showticklabels=False, row=2, col=1)
            return fig
        conteo = s.value_counts().head(30)
        fig = go.Figure(go.Bar(x=conteo.index.astype(str), y=conteo.values, marker_color=g.AZUL,
                               text=conteo.values, textposition="outside"))
        fig.update_layout(height=420, title=f"Frecuencias de {columna}", yaxis_title="Cantidad")
        return fig

    def fig_boxplots(self, columnas):
        fig = go.Figure()
        for i, c in enumerate(columnas):
            fig.add_trace(go.Box(y=self.__datos[c], name=c, marker_color=g.color_cluster(i), boxpoints="outliers"))
        fig.update_layout(height=420, showlegend=False, title="Diagramas de caja")
        return fig

    def fig_dispersion(self, x, y, color=None):
        d = self.__datos
        fig = px.scatter(d, x=x, y=y, color=color, opacity=0.75,
                         color_continuous_scale=g.SECUENCIAL, color_discrete_sequence=g.CATEGORICOS)
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        fig.update_layout(height=460, title=f"{y} contra {x}")
        return fig

    def fig_correlacion(self, metodo="pearson"):
        """Mapa de calor de la matriz de correlación con el valor escrito en cada celda."""
        c = self.correlacion(metodo)
        n = len(c)
        letra = 12 if n <= 12 else 10 if n <= 20 else 9 if n <= 30 else 7
        fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1,
                                   colorscale=g.DIVERGENTE, colorbar=dict(title="r"),
                                   texttemplate="%{z:.2f}", textfont=dict(size=letra),
                                   xgap=1, ygap=1,
                                   hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
        alto = max(420, 30 * n + 160)        # celdas de ~30 px para que el número se lea
        fig.update_layout(height=alto, title=f"Matriz de correlación ({metodo})", plot_bgcolor="#FFFFFF",
                          xaxis=dict(automargin=True, tickangle=-45, showgrid=False),
                          yaxis=dict(autorange="reversed", automargin=True, showgrid=False))
        return fig
