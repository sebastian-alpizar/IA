"""
graficos.py
Paleta y plantilla de Plotly para que todos los gráficos de la app se vean iguales.
Colores tomados del mockup (azul #2B4C7E sobre fondo gris verdoso claro).
"""

import plotly.graph_objects as go
import plotly.io as pio

# --- Paleta base (misma del mockup) ---
AZUL = "#2B4C7E"
AZUL_CLARO = "#8FA3BF"
TEXTO = "#1E2A2A"
TEXTO_SUAVE = "#4A5654"
BORDE = "#D5DAD5"
FONDO = "#F5F6F3"
VERDE = "#1F5A30"
VERDE_FONDO = "#E1EEE4"
ROJO = "#8E1F18"
ROJO_FONDO = "#F6E1DF"

# Colores para categorías o clusters (se distinguen también por luminosidad)
CATEGORICOS = ["#2B4C7E", "#D08C3A", "#3E8E7E", "#B5474B", "#7A5C99",
               "#8FA3BF", "#6B7F3A", "#C9A227", "#4F6D7A", "#A0522D"]

# Escala continua (claro -> azul oscuro) y divergente (rojo - blanco - azul)
SECUENCIAL = [[0, "#EEF2F7"], [0.5, "#8FA3BF"], [1, "#1C3558"]]
DIVERGENTE = [[0, "#8E1F18"], [0.5, "#F7F7F5"], [1, "#2B4C7E"]]

pio.templates["analitica"] = go.layout.Template(
    layout=dict(
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=TEXTO),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        colorway=CATEGORICOS,
        margin=dict(l=48, r=24, t=48, b=40),
        title=dict(font=dict(size=15, color=TEXTO), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#ECEEEB", linecolor=BORDE, zerolinecolor=BORDE),
        yaxis=dict(gridcolor="#ECEEEB", linecolor=BORDE, zerolinecolor=BORDE),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=BORDE, borderwidth=0),
        hoverlabel=dict(font=dict(family="IBM Plex Sans, sans-serif")),
    )
)
pio.templates.default = "analitica"


def color_cluster(i):
    """Color fijo para el cluster o categoría número i."""
    return CATEGORICOS[i % len(CATEGORICOS)]
