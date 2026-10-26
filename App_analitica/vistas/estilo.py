"""
estilo.py
CSS y componentes visuales reutilizables (encabezados, tarjetas, métricas).
Paleta y tipografía del mockup: IBM Plex Sans, azul #2B4C7E, fondo #F5F6F3.
"""

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.stApp { background: #F5F6F3; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }

/* Barra lateral */
section[data-testid="stSidebar"] { background: #E4E8E4; border-right: 1px solid #D5DAD5; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] { border-radius: 8px; min-height: 40px; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: #FFFFFF; box-shadow: 0 0 0 1px #D5DAD5;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span { color: #2B4C7E; font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
    text-transform: uppercase; letter-spacing: .06em; font-size: 11px; color: #4A5654; font-weight: 600;
}

/* Encabezado de página */
.enc { display: flex; flex-direction: column; gap: 6px; margin-bottom: 18px; }
.enc .paso { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #2B4C7E; }
.enc h1 { margin: 0; padding: 0; font-size: 30px; font-weight: 600; letter-spacing: -0.015em; color: #1E2A2A; }
.enc p { margin: 0; font-size: 15px; color: #4A5654; max-width: 760px; }

/* Tarjetas */
.tarjeta { background: #FFFFFF; border: 1px solid #D5DAD5; border-radius: 12px; padding: 18px 22px; }
.tarjeta h3 { margin: 0 0 6px 0; font-size: 16px; font-weight: 600; color: #1E2A2A; }
.tarjeta p { margin: 0; font-size: 14px; color: #4A5654; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 12px !important; border-color: #D5DAD5 !important; }

/* Métricas */
.metricas { display: grid; gap: 10px; margin: 4px 0 14px 0; }
.metrica { background: #FFFFFF; border: 1px solid #E1E5E1; border-radius: 10px; padding: 12px 14px; }
.metrica .et { font-size: 12.5px; color: #4A5654; }
.metrica .val { font-size: 24px; font-weight: 700; color: #1E2A2A; font-variant-numeric: tabular-nums; line-height: 1.25; }
.metrica .nota { font-size: 12px; color: #4A5654; }
.metrica.destacada { border-color: #2B4C7E; box-shadow: inset 3px 0 0 #2B4C7E; }
.metrica.buena .val { color: #1F5A30; }
.metrica.mala .val { color: #8E1F18; }

/* Chip de estado */
.chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
        background: #E7EDF5; color: #2B4C7E; margin-right: 6px; }
.chip.gris { background: #EEF0ED; color: #4A5654; }

/* Tarjeta del dataset en la barra lateral */
.dataset { background: #FFFFFF; border-radius: 10px; padding: 12px 14px; border: 1px solid #D5DAD5; }
.dataset .et { font-size: 12px; color: #4A5654; }
.dataset .nom { font-size: 15px; font-weight: 600; color: #1E2A2A; word-break: break-all; }

/* Portada */
.hero { background: #FFFFFF; border: 1px solid #D5DAD5; border-radius: 16px; padding: 34px 38px; margin-bottom: 18px; }
.hero h1 { margin: 0 0 8px 0; font-size: 34px; font-weight: 600; letter-spacing: -0.02em; color: #1E2A2A; }
.hero p { margin: 0; font-size: 16px; color: #4A5654; max-width: 760px; }
.pasos { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.pasoc { background: #FFFFFF; border: 1px solid #D5DAD5; border-radius: 12px; padding: 18px; }
.pasoc .n { width: 28px; height: 28px; border-radius: 14px; background: #2B4C7E; color: #fff; display: flex;
            align-items: center; justify-content: center; font-weight: 600; font-size: 13px; margin-bottom: 10px; }
.pasoc h3 { margin: 0 0 4px 0; font-size: 15.5px; font-weight: 600; color: #1E2A2A; }
.pasoc p { margin: 0; font-size: 13.5px; color: #4A5654; }

/* Pestañas */
button[data-baseweb="tab"] { font-size: 14.5px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2B4C7E; }
div[data-baseweb="tab-highlight"] { background-color: #2B4C7E; }

/* Botones */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 8px; min-height: 42px; font-weight: 600; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: #2B4C7E; border-color: #2B4C7E; }
.stButton > button[kind="primary"]:hover { background: #1C3558; border-color: #1C3558; }

/* Tablas */
div[data-testid="stDataFrame"] { border: 1px solid #E1E5E1; border-radius: 10px; }
.formula { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: #4A5654; }
</style>
"""


def aplicar_estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(titulo, descripcion="", paso=""):
    st.markdown(
        f"<div class='enc'>{f'<div class=paso>{html.escape(paso)}</div>' if paso else ''}"
        f"<h1>{html.escape(titulo)}</h1>{f'<p>{html.escape(descripcion)}</p>' if descripcion else ''}</div>",
        unsafe_allow_html=True)


def metricas(items, columnas=None):
    """items: lista de dicts {et, val, nota?, tipo?}  tipo: destacada / buena / mala"""
    columnas = columnas or min(len(items), 4)
    celdas = "".join(
        f"<div class='metrica {i.get('tipo', '')}'><div class='et'>{html.escape(i['et'])}</div>"
        f"<div class='val'>{html.escape(str(i['val']))}</div>"
        f"{'<div class=nota>' + html.escape(i['nota']) + '</div>' if i.get('nota') else ''}</div>"
        for i in items)
    st.markdown(f"<div class='metricas' style='grid-template-columns: repeat({columnas}, minmax(0,1fr))'>{celdas}</div>",
                unsafe_allow_html=True)


def tarjeta(titulo, texto):
    st.markdown(f"<div class='tarjeta'><h3>{html.escape(titulo)}</h3><p>{html.escape(texto)}</p></div>",
                unsafe_allow_html=True)


def formula(texto):
    st.markdown(f"<div class='formula'>{html.escape(texto)}</div>", unsafe_allow_html=True)


def num(x, dec=4, pct=False):
    """Formato numérico con coma decimal (estilo del curso)."""
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if pct:
        return f"{100 * x:.2f} %".replace(".", ",")
    return f"{x:.{dec}f}".replace(".", ",")
