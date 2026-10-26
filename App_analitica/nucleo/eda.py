"""
eda.py
Clase base del framework (EDA). NoSupervisado y Supervisado heredan de ella.
Todos los métodos devuelven tablas (DataFrame) o figuras de Plotly para que
la interfaz solo tenga que mostrarlas.
"""

import csv
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import graficos as g


class EDA:

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
    # Lectura de CSV (detecta separador, decimal y codificación)
    # ------------------------------------------------------------------
    @staticmethod
    def detectar_formato(contenido: bytes):
        """Adivina codificación, separador y decimal a partir de los bytes del archivo."""
        texto, encoding = None, "utf-8"
        for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                texto = contenido.decode(enc)
                encoding = enc
                break
            except UnicodeDecodeError:
                continue
        muestra = texto[:20000]
        primera = muestra.splitlines()[0] if muestra else ""
        conteo = {s: primera.count(s) for s in [",", ";", "\t", "|"]}
        sep = max(conteo, key=conteo.get) if max(conteo.values()) > 0 else ","
        try:
            sniff = csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
            if conteo.get(sniff, 0) >= conteo[sep]:
                sep = sniff
        except csv.Error:
            pass
        # Si el separador no es coma y hay números tipo 3,14 -> decimal coma
        decimal = "."
        if sep != ",":
            cuerpo = "\n".join(muestra.splitlines()[1:50])
            if pd.Series(cuerpo.split(sep)).str.strip().str.match(r"^-?\d+,\d+$").sum() > 3:
                decimal = ","
        return {"encoding": encoding, "sep": sep, "decimal": decimal}

    @staticmethod
    def leer_csv(contenido: bytes, sep=",", decimal=".", encabezado=True, indice=False,
                 encoding="utf-8", na_values=("?", "NA", "na", "")):
        datos = pd.read_csv(io.BytesIO(contenido), sep=sep, decimal=decimal,
                            header=0 if encabezado else None,
                            index_col=0 if indice else None,
                            encoding=encoding, na_values=list(na_values))
        datos.columns = [str(c).strip() for c in datos.columns]   # quita espacios en los nombres
        if not encabezado:
            datos.columns = [f"V{i+1}" for i in range(datos.shape[1])]
        return datos

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
            "Ejemplo": [d[c].dropna().iloc[0] if d[c].notna().any() else None for c in d.columns],
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

    def fig_correlacion(self, metodo="pearson", anotar=True):
        c = self.correlacion(metodo)
        fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1,
                                   colorscale=g.DIVERGENTE, colorbar=dict(title="r"),
                                   text=np.round(c.values, 2) if anotar and len(c) <= 20 else None,
                                   texttemplate="%{text}" if anotar and len(c) <= 20 else None,
                                   hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
        alto = min(900, max(420, 22 * len(c)))
        fig.update_layout(height=alto, title=f"Matriz de correlación ({metodo})",
                          yaxis=dict(autorange="reversed"))
        return fig
