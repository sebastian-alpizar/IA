"""
view.py
Vista encargada de mostrar resultados al usuario: componentes visuales reutilizables
(encabezados, tarjetas, métricas, tablas, fórmulas, mensajes, controles de parámetros)
y el estilo de la app (estilo.css). Cada sección del menú es una página en mvc/paginas/
que usa esta vista para mostrar y el controlador para las acciones del usuario.
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

CSS = Path(__file__).with_name("estilo.css")


class DataFrameView:
    """Vista: componentes visuales de la app (no hace cálculos ni guarda estado)."""

    # =================================================================
    # Estilo y estructura
    # =================================================================
    @staticmethod
    def aplicar_estilo():
        st.markdown(f"<style>\n{CSS.read_text(encoding='utf-8')}\n</style>", unsafe_allow_html=True)

    @staticmethod
    def encabezado(titulo, descripcion="", paso=""):
        st.markdown(
            f"<div class='enc'>{f'<div class=paso>{html.escape(paso)}</div>' if paso else ''}"
            f"<h1>{html.escape(titulo)}</h1>{f'<p>{html.escape(descripcion)}</p>' if descripcion else ''}</div>",
            unsafe_allow_html=True)

    @staticmethod
    def metricas(items, columnas=None):
        """items: lista de dicts {et, val, nota?, tipo?}  tipo: destacada / buena / mala"""
        columnas = columnas or min(len(items), 4)
        celdas = "".join(
            f"<div class='metrica {i.get('tipo', '')}'><div class='et'>{html.escape(i['et'])}</div>"
            f"<div class='val'>{html.escape(str(i['val']))}</div>"
            f"{'<div class=nota>' + html.escape(i['nota']) + '</div>' if i.get('nota') else ''}</div>"
            for i in items)
        st.markdown(f"<div class='metricas' style='grid-template-columns: repeat({columnas}, minmax(0,1fr))'>"
                    f"{celdas}</div>", unsafe_allow_html=True)

    @staticmethod
    def formula(texto):
        st.markdown(f"<div class='formula'>{html.escape(texto)}</div>", unsafe_allow_html=True)

    @staticmethod
    def chips(items, contenedor=st, margen=0):
        """Etiquetas de estado. items: lista de (texto, gris?)"""
        contenedor.markdown(f"<div style='margin-top:{margen}px'>" + "".join(f"<span class='chip{' gris' if gris else ''}'>{html.escape(str(t))}</span>"
                            for t, gris in items) + "</div>", unsafe_allow_html=True)

    # =================================================================
    # Datos
    # =================================================================
    @staticmethod
    def tabla(df, altura=None, formato=None):
        """Muestra un DataFrame con formato numérico uniforme (coma decimal)."""
        if isinstance(df, pd.Series):
            df = df.to_frame()
        estilo = df.style.format(formato or (lambda v: f"{v:,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
                                             if isinstance(v, float) else v))
        if altura:
            st.dataframe(estilo, width="stretch", height=altura)
        else:
            st.dataframe(estilo, width="stretch")

    @staticmethod
    def descargar_csv(df, nombre, etiqueta="Descargar CSV", clave=None):
        st.download_button(etiqueta, df.to_csv(index=True).encode("utf-8"), file_name=nombre, mime="text/csv",
                           key=clave or f"dl_{nombre}")

    # =================================================================
    # Mensajes
    # =================================================================
    @staticmethod
    def mensaje(texto):
        st.info(texto)

    @staticmethod
    def aviso(texto):
        st.warning(texto)

    @staticmethod
    def error(texto):
        st.error(texto)

    @staticmethod
    def exito(texto):
        st.success(texto)

    @staticmethod
    def sin_datos(ir_a_carga):
        """Aviso de que no hay datos cargados, con un botón para ir a la carga."""
        st.markdown("<div class='tarjeta'><h3>Todavía no hay datos</h3>"
                    "<p>Cargá un CSV o elegí un dataset de ejemplo para usar esta sección.</p></div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Ir a Carga de datos", type="primary"):
            ir_a_carga()

    @staticmethod
    def tarjeta_dataset_lateral(nombre, forma):
        """Tarjeta de la barra lateral con el dataset activo. forma: (filas, columnas) o None."""
        with st.sidebar:
            if forma is None:
                st.markdown("<div class='dataset'><div class='et'>Dataset activo</div>"
                            "<div class='nom'>Ninguno</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='dataset'><div class='et'>Dataset activo</div>"
                            f"<div class='nom'>{html.escape(nombre or 'datos')}</div>"
                            f"<div class='et'>{forma[0]:,} filas · {forma[1]} columnas</div></div>".replace(",", "."),
                            unsafe_allow_html=True)
            st.caption("Inteligencia Artificial · UNA")

    # =================================================================
    # Controles
    # =================================================================
    @staticmethod
    def control_parametro(p, clave):
        """
        Un control para un parámetro de un algoritmo (ver supervisado.parametro).
        Devuelve el valor (None = default 'sin límite' / 'ninguno').
        """
        k = f"{clave}_{p['nombre']}"
        ayuda = p["ayuda"] or None
        if p["tipo"] == "int":
            mn, mx, paso = p["extra"]
            valor = st.number_input(p["etiqueta"], int(mn), int(mx), int(p["default"] or 0), int(paso), key=k,
                                    help=ayuda)
            return None if (p["default"] is None and valor == 0) else int(valor)
        if p["tipo"] == "float":
            mn, mx, paso = p["extra"]
            return float(st.number_input(p["etiqueta"], float(mn), float(mx), float(p["default"]), float(paso),
                                         key=k, format="%.4f", help=ayuda))
        if p["tipo"] == "text":
            return st.text_input(p["etiqueta"], p["default"], key=k, help=ayuda).replace(" ", "")
        return st.selectbox(p["etiqueta"], p["extra"], index=p["extra"].index(p["default"]), key=k, help=ayuda,
                            format_func=lambda v: "ninguno (todas)" if v is None and p["nombre"] == "max_features"
                            else ("ninguno" if v is None else str(v)))

    @staticmethod
    def barra_progreso():
        """Barra de progreso: retorna (función para avanzar(i, total, texto), función para quitarla)."""
        barra = st.progress(0.0)
        return (lambda i, total, texto="": barra.progress(i / max(total, 1), text=texto)), barra.empty
