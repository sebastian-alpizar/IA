"""
Página de Clasificación (como predictoR + Laboratorio 2): los cinco algoritmos del Lab 2
(KNN, DT, RF, XGBoost, AdaBoost) y los demás del curso, con configuración estándar y
variaciones, historial de experimentos, selección del mejor modelo por algoritmo,
análisis del modelo elegido y predicción de datos nuevos.
"""

import pandas as pd
import streamlit as st

from nucleo.clasificacion import ALGORITMOS, ALGORITMOS_LAB, CRITERIOS, ClasificacionLab
from nucleo.eda import EDA
from vistas.comunes import descargar_csv, requiere_datos, tabla
from vistas.estilo import encabezado, formula, metricas, num

IMPUTACIONES = {"Mediana": "median", "Media": "mean", "Moda": "most_frequent"}
NOMBRES = {"DT": "Árbol de decisión (DT)", "RF": "Random Forest (RF)"}
LIMITE_LENTOS = 20000   # con más filas de entrenamiento, "Entrenar todos" se salta los algoritmos lentos (SVM)


def _etiqueta(algoritmo):
    return NOMBRES.get(algoritmo, algoritmo) + (" · Lab 2" if algoritmo in ALGORITMOS_LAB else "")


def pagina():
    df = requiere_datos()
    encabezado("Clasificación",
               "Predecí una variable categórica: probá la configuración estándar y variaciones de cada algoritmo, "
               "elegí el mejor modelo de cada uno, analizalo y aplicalo a datos nuevos.",
               "Paso 4 · Supervisado")
    candidatas = [c for c in df.columns if 2 <= df[c].nunique() <= 20]
    if not candidatas:
        st.warning("No hay variables categóricas (entre 2 y 20 clases) para predecir en los datos cargados.")
        return

    # ------------------------------------------------ configuración
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        objetivo = c1.selectbox("Variable a predecir", candidatas, index=len(candidatas) - 1, key="lab_obj",
                                help="Por defecto se propone la última columna.")
        clases = sorted(df[objetivo].dropna().astype(str).unique())
        positiva = None
        if len(clases) == 2:
            positiva = c2.selectbox("Clase positiva", clases, index=1, key=f"lab_pos_{objetivo}",
                                    help="La clase que interesa detectar (potable, diabético...). "
                                         "Se usa para Precision, Recall, F1 y la curva ROC.")
        else:
            c2.markdown(f"<div style='margin-top:34px'><span class='chip gris'>{len(clases)} clases · "
                        f"índices macro</span></div>", unsafe_allow_html=True)
        prueba = c3.slider("% para prueba", 10, 50, 25, 5, key="lab_test")
        semilla = c4.number_input("Semilla", 0, 99999, 42, key="lab_seed")
        c5, c6 = st.columns([3.1, 1])
        disponibles = [c for c in df.columns if c != objetivo]
        with c5:
            predictoras = st.multiselect("Variables predictoras", disponibles, default=disponibles,
                                         key=f"lab_pred_{objetivo}")
        duplicados = c6.toggle("Eliminar duplicados", True, key="lab_dup",
                               help="Quita filas repetidas antes de particionar.")
    if not predictoras:
        st.warning("Elegí al menos una variable predictora.")
        return

    firma = (st.session_state.version, objetivo, positiva, tuple(predictoras), prueba, semilla, duplicados)
    estado = st.session_state.clas
    if estado.get("firma") != firma:
        try:
            modelo = ClasificacionLab(df, objetivo, predictoras, positiva, prueba / 100, duplicados, int(semilla))
        except ValueError as e:
            st.error(f"No se pudo particionar: {e}")
            return
        estado.clear()
        estado.update({"firma": firma, "modelo": modelo})
    modelo = estado["modelo"]

    info = (f"<span class='chip'>Entrenamiento: {len(modelo.X_train)}</span>"
            f"<span class='chip'>Prueba: {len(modelo.X_test)}</span>")
    info += "".join(f"<span class='chip gris'>{c}: {p:.1f} %</span>"
                    for c, p in modelo.distribucion_clases()["Porcentaje"].items())
    st.markdown(info, unsafe_allow_html=True)
    with st.expander("Partición entrenamiento-prueba (estratificada)"):
        tabla(modelo.resumen_particion())
        if modelo.binaria:
            st.caption(f"Relación negativos/positivos en entrenamiento: {modelo.relacion_negativos_positivos()} "
                       "(es el valor que se usa en XGBoost como scale_pos_weight = «ratio»).")
    st.write("")

    t1, t2, t3, t4, t5 = st.tabs(["Experimentar", "Resultados y mejores modelos", "Analizar un modelo",
                                  "Curva k (KNN)", "Predecir datos nuevos"])
    with t1:
        _experimentar(modelo)
    with t2:
        _resultados(modelo)
    with t3:
        _analizar(modelo)
    with t4:
        _curva_k(modelo)
    with t5:
        _predecir(modelo, objetivo)


# ======================================================================
# Experimentar
# ======================================================================
def _control(p, clave):
    """Un control para un parámetro; devuelve el valor (None = default 'sin límite'/'ninguno')."""
    k = f"{clave}_{p['nombre']}"
    ayuda = p["ayuda"] or None
    if p["tipo"] == "int":
        mn, mx, paso = p["extra"]
        valor = st.number_input(p["etiqueta"], int(mn), int(mx), int(p["default"] or 0), int(paso), key=k, help=ayuda)
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


def _experimentar(modelo):
    izq, der = st.columns([1, 2.2], gap="medium")
    with izq:
        with st.container(border=True):
            algoritmo = st.selectbox("Algoritmo", list(ALGORITMOS), key="lab_alg", format_func=_etiqueta)
            info = ALGORITMOS[algoritmo]
            st.caption(info["descripcion"])
            if info.get("lento") and len(modelo.X_train) > LIMITE_LENTOS:
                st.warning(f"Con {len(modelo.X_train):,} filas de entrenamiento puede tardar varios minutos."
                           .replace(",", "."))
            if info["params"]:
                st.markdown("**Parámetros** · los valores iniciales son la configuración estándar")
            config = {}
            for p in info["params"]:
                if p["nombre"] == "scale_pos_weight" and not modelo.binaria:
                    continue
                valor = _control(p, f"lab_{algoritmo}")
                if valor != p["default"]:
                    config[p["nombre"]] = valor
            st.markdown("**Preprocesamiento**")
            escalar = st.toggle("Estandarizar numéricas", info["escalar"], key=f"lab_{algoritmo}_esc",
                                help="KNN usa distancias, por eso escala por defecto; los árboles no lo necesitan.")
            imputacion = st.selectbox("Imputación de vacíos (numéricas)", list(IMPUTACIONES),
                                      key=f"lab_{algoritmo}_imp")
            seleccionar = st.toggle("Selección de atributos (SelectKBest, ANOVA F)", False,
                                    key=f"lab_{algoritmo}_sel")
            n_atributos = None
            if seleccionar:
                n_atributos = st.number_input("Cantidad de atributos", 1, 500, 5, key=f"lab_{algoritmo}_nat")
            if escalar != info["escalar"]:
                config["escalar"] = escalar
            if IMPUTACIONES[imputacion] != "median":
                config["imputacion"] = IMPUTACIONES[imputacion]
            if n_atributos:
                config["n_atributos"] = int(n_atributos)

            if st.button("Entrenar esta configuración", type="primary", width="stretch", key="lab_btn"):
                with st.spinner(f"Entrenando {algoritmo}..."):
                    try:
                        modelo.ejecutar(algoritmo, config)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"El modelo no se pudo entrenar: {e}")
            if st.button("Correr las variaciones de este algoritmo", width="stretch", key="lab_var",
                         help="Corre la configuración estándar y sus variaciones (en los del Lab 2, "
                              "las mismas que se usaron en el notebook)."):
                _correr(modelo, {algoritmo: info["variaciones"]})

        with st.container(border=True):
            st.markdown("**Correr en lote**")
            if st.button("Variaciones de los 5 algoritmos del Lab 2", type="primary", width="stretch",
                         key="lab_var_todos", help="KNN, DT, RF, XGBoost y AdaBoost con todas sus variaciones."):
                _correr(modelo, {a: ALGORITMOS[a]["variaciones"] for a in ALGORITMOS_LAB})
            if st.button("Todos los algoritmos, configuración estándar", width="stretch", key="lab_todos"):
                plan = {a: [{}] for a in ALGORITMOS}
                if len(modelo.X_train) > LIMITE_LENTOS:
                    lentos = [a for a in plan if ALGORITMOS[a].get("lento")]
                    for a in lentos:
                        del plan[a]
                    st.info(f"Se omitió {', '.join(lentos)}: con {len(modelo.X_train):,} filas de entrenamiento "
                            "tardaría mucho. Podés entrenarlo aparte.".replace(",", "."))
                _correr(modelo, plan)

    with der:
        criterio = st.session_state.get("lab_criterio", CRITERIOS[0])
        res = modelo.resultados(algoritmo)
        if res.empty:
            st.markdown("<div class='tarjeta'><h3>Sin experimentos de este algoritmo</h3>"
                        "<p>Ajustá los parámetros y presioná Entrenar, o corré las variaciones del laboratorio. "
                        "Sin cambiar nada se entrena la configuración estándar.</p></div>", unsafe_allow_html=True)
            with st.expander("Variaciones que se van a correr"):
                st.code("\n".join(str(v) for v in info["variaciones"]), language="python")
            return
        mejor = modelo.mejor_modelo(criterio, algoritmo)
        fila = res.set_index("Modelo").loc[mejor]
        metricas([{"et": f"Mejor {algoritmo} según {criterio}", "val": num(fila[criterio]), "nota": mejor,
                   "tipo": "destacada"},
                  {"et": "Precisión global", "val": num(fila["Precisión Global"], pct=True)},
                  {"et": "AUC", "val": num(fila["AUC"], 3)},
                  {"et": "Configuraciones probadas", "val": len(res)}])
        tabla(res.drop(columns="Algoritmo").set_index("Modelo"))
        st.plotly_chart(modelo.fig_comparacion(algoritmo=algoritmo, titulo=f"{algoritmo}: comparación de configuraciones"),
                        width="stretch", key="lab_fig_alg")
        st.caption("El criterio para elegir el mejor se cambia en la pestaña «Resultados y mejores modelos».")


def _correr(modelo, plan):
    total = sum(len(v) for v in plan.values())
    barra = st.progress(0.0)
    hecho = 0
    for algoritmo, variaciones in plan.items():
        for conf in variaciones:
            barra.progress(hecho / total, text=f"{algoritmo}: {conf or 'estándar'}")
            try:
                modelo.ejecutar(algoritmo, conf)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{algoritmo} {conf}: {e}")
            hecho += 1
    barra.empty()


# ======================================================================
# Resultados
# ======================================================================
def _resultados(modelo):
    res = modelo.resultados()
    if res.empty:
        st.info("Todavía no hay experimentos. Entrená configuraciones en la pestaña «Experimentar».")
        return
    c1, c2 = st.columns([1, 3])
    criterio = c1.selectbox("Criterio para elegir el mejor modelo", CRITERIOS, key="lab_criterio",
                            help="F1 macro equilibra el acierto de ambas clases; útil con clases desbalanceadas.")
    mejores = modelo.mejores_por_algoritmo(criterio)
    mejor = mejores.iloc[0]
    metricas([{"et": f"Mejor modelo según {criterio}", "val": num(mejor[criterio]), "nota": mejor["Modelo"],
               "tipo": "destacada"},
              {"et": "Precisión global", "val": num(mejor["Precisión Global"], pct=True)},
              {"et": "Experimentos", "val": len(res)},
              {"et": "Algoritmos", "val": res["Algoritmo"].nunique()}])

    st.markdown("**Mejor modelo de cada algoritmo**")
    tabla(mejores.set_index("Modelo"))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(modelo.fig_comparacion(nombres=list(mejores["Modelo"]), titulo="Comparación de los mejores"),
                        width="stretch", key="lab_fig_mejores")
    with c2:
        if modelo.binaria:
            st.plotly_chart(modelo.fig_roc(list(mejores["Modelo"]), "Curvas ROC de los mejores"),
                            width="stretch", key="lab_roc_mejores")
        else:
            st.plotly_chart(modelo.fig_precision_por_clase(list(mejores["Modelo"]),
                                                           "Precisión por categoría de los mejores"),
                            width="stretch", key="lab_pc_mejores")

    st.markdown("**Historial completo de experimentos**")
    tabla(res.set_index("Modelo"), altura=min(600, 38 * len(res) + 40))
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c1:
        descargar_csv(res.set_index("Modelo"), "resultados_clasificacion.csv", "Descargar resultados",
                      clave="lab_dl_res")
    borrar = c2.selectbox("Experimento", ["—"] + list(res["Modelo"]), key="lab_borrar",
                          label_visibility="collapsed")
    if c3.button("Borrar experimento", width="stretch", disabled=borrar == "—", key="lab_btn_borrar"):
        modelo.borrar_experimento(borrar)
        st.rerun()


# ======================================================================
# Analizar
# ======================================================================
def _analizar(modelo):
    res = modelo.resultados()
    if res.empty:
        st.info("Todavía no hay experimentos para analizar.")
        return
    criterio = st.session_state.get("lab_criterio", CRITERIOS[0])
    nombres = list(res["Modelo"])
    mejor = modelo.mejor_modelo(criterio)
    nombre = st.selectbox("Modelo a analizar", nombres, index=nombres.index(mejor), key="lab_analizar",
                          help=f"Por defecto el mejor según {criterio}.")
    exp = modelo.experimentos[nombre]
    ind = exp["indices"]
    items = [{"et": "Precisión global (PG)", "val": num(ind["Precisión Global"], pct=True), "tipo": "destacada",
              "nota": "aciertos / total"},
             {"et": "Error global (EG)", "val": num(ind["Error Global"], pct=True), "nota": "1 − PG"},
             {"et": "Área bajo la curva (AUC)", "val": num(ind["AUC"], 3)},
             {"et": "F1 macro", "val": num(ind["F1 macro"], 3), "nota": "promedio del F1 de cada clase"}]
    if modelo.binaria:
        pos, neg = modelo.positiva, modelo.clases[0]
        n = modelo.conteos(nombre)
        items += [{"et": "Precisión positiva (PP)", "val": num(ind["PP"], 3),
                   "nota": f"{n['VP']} de {n['VP'] + n['FN']} «{pos}» detectados (= Recall)"},
                  {"et": "Precisión negativa (PN)", "val": num(ind["PN"], 3),
                   "nota": f"{n['VN']} de {n['VN'] + n['FP']} «{neg}» detectados"},
                  {"et": "Asertividad positiva (AP)", "val": num(ind["AP"], 3),
                   "nota": f"de los predichos «{pos}», cuántos lo eran (= Precision)"},
                  {"et": "Asertividad negativa (AN)", "val": num(ind["AN"], 3),
                   "nota": f"de los predichos «{neg}», cuántos lo eran"},
                  {"et": "Falsos positivos (PFP)", "val": num(ind["PFP"], 3), "tipo": "mala" if ind["PFP"] > 0.2 else ""},
                  {"et": "Falsos negativos (PFN)", "val": num(ind["PFN"], 3), "tipo": "mala" if ind["PFN"] > 0.2 else ""},
                  {"et": f"F1 de «{pos}»", "val": num(ind["F1"], 3)},
                  {"et": "Tiempo", "val": f"{ind['Tiempo (s)']:.2f} s", "nota": exp["algoritmo"]}]
    else:
        items += [{"et": "Precision (macro)", "val": num(ind["Precision (VPP)"], 3)},
                  {"et": "Recall (macro)", "val": num(ind["Recall"], 3)},
                  {"et": "Clases", "val": len(modelo.clases)},
                  {"et": "Tiempo", "val": f"{ind['Tiempo (s)']:.2f} s", "nota": exp["algoritmo"]}]
    metricas(items, columnas=4)

    pestanas = ["Matriz de confusión", "Precisión por categoría"]
    if modelo.binaria:
        pestanas.append("Curva ROC")
    pestanas.append("Importancia de variables")
    if exp["algoritmo"] == "DT":
        pestanas.append("Árbol")
    pestanas += ["Validación cruzada", "Configuración"]
    tabs = dict(zip(pestanas, st.tabs(pestanas)))

    with tabs["Matriz de confusión"]:
        normalizar = st.toggle("Normalizar por fila (porcentaje de cada clase real)", key="lab_norm")
        st.plotly_chart(modelo.fig_matriz_confusion(nombre, normalizar), width="stretch", key="lab_mc")
        st.caption("Filas = valor real, columnas = predicción.")
        if modelo.binaria:
            formula("PP = VP/(VP+FN)   PN = VN/(VN+FP)   AP = VP/(VP+FP)   AN = VN/(VN+FN)   "
                    "PFP = FP/(FP+VN)   PFN = FN/(FN+VP)")
    with tabs["Precisión por categoría"]:
        pc = pd.Series({c: ind[f"Precisión clase {c}"] for c in modelo.clases}, name="Precisión").to_frame()
        pc["Casos reales en prueba"] = modelo.matriz_confusion(nombre).sum(axis=1).values
        tabla(pc)
        st.caption("Precisión de la categoría = casos de esa categoría bien clasificados / total real de esa categoría.")
    if "Curva ROC" in tabs:
        with tabs["Curva ROC"]:
            st.plotly_chart(modelo.fig_roc([nombre], "Curva ROC"), width="stretch", key="lab_roc")
    with tabs["Importancia de variables"]:
        top = st.slider("Variables a mostrar", 5, 40, 15, key="lab_top")
        with st.spinner("Calculando importancia..."):
            st.plotly_chart(modelo.fig_importancia(nombre, top), width="stretch", key="lab_imp")
        if modelo.importancia_propia(nombre):
            st.caption("Importancia propia del modelo (cuánto aporta cada variable a las divisiones de los árboles).")
        else:
            st.caption("Este modelo no trae importancia propia: se calcula por permutación, es decir, cuánto baja "
                       "el acierto en prueba al desordenar esa variable.")
    if "Árbol" in tabs:
        with tabs["Árbol"]:
            prof = st.slider("Niveles a mostrar", 1, 5, 3, key="lab_prof")
            st.plotly_chart(modelo.fig_arbol(nombre, prof), width="stretch", key="lab_arbol")
            with st.expander("Reglas del árbol en texto"):
                st.code(modelo.reglas_arbol(nombre, max(prof, 6)), language="text")
    with tabs["Validación cruzada"]:
        c1, c2 = st.columns([1, 3])
        k = c1.number_input("Folds (k)", 2, 20, 5, key="lab_k")
        c2.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
        clave = f"cv_{nombre}_{k}"
        if c2.button("Correr validación cruzada estratificada (sobre entrenamiento)", key="lab_cv"):
            with st.spinner("Validando..."):
                try:
                    st.session_state.clas[clave] = modelo.validacion_cruzada(nombre, int(k))
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
        if clave in st.session_state.clas:
            tabla(st.session_state.clas[clave])
            st.caption("Una desviación estándar baja indica que el modelo es estable entre particiones.")
    with tabs["Configuración"]:
        st.json({k: v for k, v in exp["config"].items() if v is not None} or {"configuración": "estándar"})
        st.caption("Parámetros del estimador y opciones de preprocesamiento usadas en este experimento.")


# ======================================================================
# Curva k
# ======================================================================
def _curva_k(modelo):
    st.markdown("Cómo cambian los índices de KNN según la cantidad de vecinos (no se guarda en el historial).")
    c1, c2, c3, c4, c5 = st.columns(5)
    desde = c1.number_input("k desde", 1, 500, 1, key="lab_kd")
    hasta = c2.number_input("k hasta", 1, 500, 51, key="lab_kh")
    paso = c3.number_input("Paso", 1, 100, 5, key="lab_kp")
    pesos = c4.selectbox("Pesos", ["uniform", "distance"], key="lab_kw")
    escalar = c5.toggle("Estandarizar", True, key="lab_kesc")
    if st.button("Calcular curva", type="primary", key="lab_kbtn"):
        valores = range(int(desde), int(hasta) + 1, int(paso))
        valores = [k for k in valores if k <= len(modelo.X_train)]
        with st.spinner(f"Entrenando KNN con {len(valores)} valores de k..."):
            st.session_state.clas["curva_k"] = modelo.curva_k(valores, escalar, weights=pesos)
    curva = st.session_state.clas.get("curva_k")
    if curva is not None:
        st.plotly_chart(ClasificacionLab.fig_curva_k(curva), width="stretch", key="lab_kfig")
        mejor = curva.loc[curva["F1 macro"].idxmax()]
        st.caption(f"Mejor k según F1 macro: {int(mejor['k'])} (F1 macro = {mejor['F1 macro']:.4f}).")
        tabla(curva.set_index("k"))


# ======================================================================
# Predecir
# ======================================================================
def _predecir(modelo, objetivo):
    res = modelo.resultados()
    if res.empty:
        st.info("Entrená al menos un modelo para poder predecir.")
        return
    criterio = st.session_state.get("lab_criterio", CRITERIOS[0])
    nombres = list(res["Modelo"])
    c1, c2 = st.columns([1, 2])
    elegido = c1.selectbox("Modelo a usar", nombres, index=nombres.index(modelo.mejor_modelo(criterio)),
                           key="lab_pmod")
    archivo = c2.file_uploader("CSV con los datos nuevos (mismas columnas predictoras)", type=["csv", "txt"],
                               key="lab_nuevos")
    usar_prueba = c1.checkbox("O usar los datos de prueba", key="lab_usarprueba")
    nuevos = None
    if archivo is not None:
        contenido = archivo.getvalue()
        nuevos = EDA.leer_csv(contenido, **EDA.detectar_formato(contenido))
    elif usar_prueba:
        nuevos = modelo.X_test.copy()
    if nuevos is None:
        return
    try:
        pred = modelo.predecir(elegido, nuevos)
        salida = nuevos.copy()
        salida[f"Predicción {objetivo}"] = pred
        conteo = pd.Series(pred).value_counts()
        metricas([{"et": f"Predichos como {c}", "val": v} for c, v in conteo.items()][:4])
        st.dataframe(salida.head(300), width="stretch", height=380)
        descargar_csv(salida, "predicciones.csv", "Descargar predicciones", clave="lab_dlpred")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
