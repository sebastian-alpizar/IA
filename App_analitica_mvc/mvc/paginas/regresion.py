"""
regresion.py (página)
Vista de Regresión (Laboratorio 3): modelos lineales (simple, múltiple, Lasso, LassoCV,
Ridge, RidgeCV), SVR y árboles (DT, Random Forest, XGBoost), más Gradient Boosting y KNN.
Configuración estándar y variaciones, historial de experimentos, mejor modelo por algoritmo
y por familia, análisis del modelo elegido, curvas de hiperparámetros y regularización,
benchmarking con validación cruzada repetida y predicción de datos nuevos.
Las acciones (entrenar, validar, comparar, predecir...) las hace el controlador; aquí solo se muestra.
"""

import pandas as pd
import streamlit as st

from shared.formato import num
from supervisado.regresion import ALGORITMOS, ALGORITMOS_LAB, CRITERIOS, Regresion

IMPUTACIONES = {"Mediana": "median", "Media": "mean", "Moda": "most_frequent"}
NOMBRES = {"DT": "Árbol de regresión (DT)", "RF": "Random Forest (RF)", "KNN": "KNN (vecinos)"}
NO_PREDICTORAS = {"id", "fold", "state", "index", "indice", "índice"}   # se desmarcan por defecto
LIMITE_LENTOS = 20000   # con más filas de entrenamiento, los lotes se saltan los algoritmos lentos (SVR)


def _etiqueta(algoritmo):
    info = ALGORITMOS[algoritmo]
    return f"{NOMBRES.get(algoritmo, algoritmo)} · {info['familia']}" + (" · Lab 3" if info["lab"] else "")


def _criterio():
    return st.session_state.get("reg_criterio", CRITERIOS[0])


def pagina(ctrl):
    if not ctrl.requiere_datos():
        return
    v = ctrl.view
    df = ctrl.model.datos
    v.encabezado("Regresión",
               "Predecí una variable numérica continua: probá la configuración estándar y variaciones de cada "
               "algoritmo, compará dentro de cada familia y entre familias, y aplicá el mejor modelo a datos nuevos.",
               "Paso 4 · Supervisado")
    numericas = df.select_dtypes(include="number").columns
    candidatas = [c for c in numericas if df[c].nunique() > 10]
    if not candidatas:
        st.warning("No hay variables numéricas continuas (con más de 10 valores distintos) para predecir.")
        return

    # ------------------------------------------------ configuración
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        objetivo = c1.selectbox("Variable a predecir", candidatas, index=len(candidatas) - 1, key="reg_obj",
                                help="Solo numéricas con más de 10 valores distintos. Por defecto la última.")
        prueba = c2.slider("% para prueba", 10, 50, 25, 5, key="reg_test")
        semilla = c3.number_input("Semilla", 0, 99999, 42, key="reg_seed")
        c4.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)
        duplicados = c4.toggle("Eliminar duplicados", True, key="reg_dup",
                               help="Quita filas repetidas antes de particionar.")
        disponibles = [c for c in df.columns if c != objetivo]
        por_defecto = [c for c in disponibles if c.strip().lower() not in NO_PREDICTORAS]
        predictoras = st.multiselect("Variables predictoras", disponibles, default=por_defecto,
                                     key=f"reg_pred_{objetivo}",
                                     help="Las columnas que son identificadores o particiones (id, fold, state...) "
                                          "se desmarcan por defecto porque no describen a los individuos.")
    if not predictoras:
        st.warning("Elegí al menos una variable predictora.")
        return

    reg = ctrl.regresion(objetivo, predictoras, prueba, semilla, duplicados)
    if reg is None:
        return

    y = reg.datos[objetivo]
    v.chips([(f"Entrenamiento: {len(reg.X_train)}", False), (f"Prueba: {len(reg.X_test)}", False),
             (f"Media de {objetivo}: {num(y.mean(), 3)}", True), (f"Desv. estándar: {num(y.std(), 3)}", True)])
    with st.expander("Variable a predecir y partición entrenamiento-prueba"):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(reg.fig_distribucion_variable(), width="stretch", key="reg_fig_y")
            v.tabla(reg.resumen_variable())
        with c2:
            st.plotly_chart(reg.fig_correlacion_respuesta(15), width="stretch", key="reg_fig_corr")
        st.markdown("**Partición (aleatoria simple: la variable es continua, no se estratifica)**")
        v.tabla(reg.resumen_particion())
    st.write("")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Experimentar", "Resultados y mejores modelos", "Analizar un modelo",
                                      "Curvas y regularización", "Benchmarking", "Predecir datos nuevos"])
    with t1:
        _experimentar(ctrl, reg)
    with t2:
        _resultados(ctrl, reg)
    with t3:
        _analizar(ctrl, reg)
    with t4:
        _curvas(ctrl, reg)
    with t5:
        _benchmark(ctrl, reg)
    with t6:
        _predecir(ctrl, reg, objetivo)


# ======================================================================
# Experimentar
# ======================================================================
def _experimentar(ctrl, reg):
    v = ctrl.view
    izq, der = st.columns([1, 2.2], gap="medium")
    with izq:
        with st.container(border=True):
            algoritmo = st.selectbox("Algoritmo", list(ALGORITMOS), key="reg_alg", format_func=_etiqueta)
            info = ALGORITMOS[algoritmo]
            st.caption(info["descripcion"])
            if info.get("lento") and len(reg.X_train) > LIMITE_LENTOS:
                st.warning(f"Con {len(reg.X_train):,} filas de entrenamiento puede tardar varios minutos."
                           .replace(",", "."))
            config = {}
            if algoritmo == "Lineal simple":
                ranking = reg.correlacion_respuesta()
                if ranking.empty:
                    st.error("Se necesita al menos una predictora numérica.")
                    return
                variable = st.selectbox("Predictora", list(ranking.index), key="reg_simple_var",
                                        format_func=lambda c: f"{c} (r = {ranking[c]:.3f})",
                                        help="Ordenadas por la magnitud de su correlación con la respuesta.")
                if variable != ranking.index[0]:
                    config["variable"] = variable
            if info["params"]:
                st.markdown("**Parámetros** · los valores iniciales son la configuración estándar")
            for p in info["params"]:
                valor = v.control_parametro(p, f"reg_{algoritmo}")
                if valor != p["default"]:
                    config[p["nombre"]] = valor
            if algoritmo != "Lineal simple":
                st.markdown("**Preprocesamiento**")
                escalar = st.toggle("Estandarizar numéricas", info["escalar"], key=f"reg_{algoritmo}_esc",
                                    help="Lasso, Ridge, SVR y KNN dependen de la escala de las variables; "
                                         "mínimos cuadrados y los árboles no.")
                imputacion = st.selectbox("Imputación de vacíos (numéricas)", list(IMPUTACIONES),
                                          key=f"reg_{algoritmo}_imp")
                seleccionar = st.toggle("Selección de atributos (SelectKBest, F de la regresión)", False,
                                        key=f"reg_{algoritmo}_sel")
                n_atributos = None
                if seleccionar:
                    n_atributos = st.number_input("Cantidad de atributos", 1, 500, 20, key=f"reg_{algoritmo}_nat")
                if escalar != info["escalar"]:
                    config["escalar"] = escalar
                if IMPUTACIONES[imputacion] != "median":
                    config["imputacion"] = IMPUTACIONES[imputacion]
                if n_atributos:
                    config["n_atributos"] = int(n_atributos)

            if st.button("Entrenar esta configuración", type="primary", width="stretch", key="reg_btn"):
                with st.spinner(f"Entrenando {algoritmo}..."):
                    ctrl.entrenar(reg, algoritmo, config)
            if st.button("Correr las variaciones de este algoritmo", width="stretch", key="reg_var",
                         help="Corre la configuración estándar y sus variaciones (las del notebook del Lab 3)."):
                ctrl.correr_plan(reg, {algoritmo: info["variaciones"]})

        with st.container(border=True):
            st.markdown("**Correr en lote**")
            if st.button("Variaciones de los algoritmos del Lab 3", type="primary", width="stretch",
                         key="reg_var_todos", help="Lineales, Lasso/Ridge (y sus CV), SVR, DT, RF y XGBoost "
                                                   "con todas sus variaciones."):
                ctrl.correr_plan(reg, _sin_lentos(v, reg, {a: ALGORITMOS[a]["variaciones"] for a in ALGORITMOS_LAB}))
            if st.button("Todos los algoritmos, configuración estándar", width="stretch", key="reg_todos"):
                ctrl.correr_plan(reg, _sin_lentos(v, reg, {a: [{}] for a in ALGORITMOS}))

    with der:
        criterio = _criterio()
        res = reg.resultados(algoritmo)
        if res.empty:
            st.markdown("<div class='tarjeta'><h3>Sin experimentos de este algoritmo</h3>"
                        "<p>Ajustá los parámetros y presioná Entrenar, o corré las variaciones del laboratorio. "
                        "Sin cambiar nada se entrena la configuración estándar.</p></div>", unsafe_allow_html=True)
            with st.expander("Variaciones que se van a correr"):
                st.code("\n".join(str(var) for var in info["variaciones"]), language="python")
            return
        mejor = reg.mejor_modelo(criterio, algoritmo)
        fila = res.set_index("Modelo").loc[mejor]
        v.metricas([{"et": f"Mejor {algoritmo} según {criterio}", "val": num(fila[criterio]), "nota": mejor,
                   "tipo": "destacada"},
                  {"et": "RMSE", "val": num(fila["RMSE"])},
                  {"et": "R²", "val": num(fila["R²"])},
                  {"et": "Configuraciones probadas", "val": len(res)}])
        v.tabla(res.drop(columns=["Algoritmo", "Familia"]).set_index("Modelo"))
        st.plotly_chart(reg.fig_comparacion(algoritmo=algoritmo,
                                               titulo=f"{algoritmo}: comparación de configuraciones"),
                        width="stretch", key="reg_fig_alg")
        st.caption("El criterio para elegir el mejor se cambia en la pestaña «Resultados y mejores modelos». "
                   "En RMSE, MAE y ER el mejor es el menor; en R² el mayor.")


def _sin_lentos(v, reg, plan):
    if len(reg.X_train) > LIMITE_LENTOS:
        lentos = [a for a in plan if ALGORITMOS[a].get("lento")]
        for a in lentos:
            del plan[a]
        if lentos:
            v.mensaje(f"Se omitió {', '.join(lentos)}: con {len(reg.X_train):,} filas de entrenamiento "
                    "tardaría mucho. Podés entrenarlo aparte.".replace(",", "."))
    return plan


# ======================================================================
# Resultados
# ======================================================================
def _resultados(ctrl, reg):
    v = ctrl.view
    res = reg.resultados()
    if res.empty:
        st.info("Todavía no hay experimentos. Entrená configuraciones en la pestaña «Experimentar».")
        return
    c1, _ = st.columns([1, 3])
    criterio = c1.selectbox("Criterio para elegir el mejor modelo", CRITERIOS, key="reg_criterio",
                            help="RMSE castiga más los errores grandes; MAE es más robusto a valores atípicos; "
                                 "ER es el error relativo; R² es la proporción de varianza explicada.")
    por_alg = reg.mejores(criterio, "Algoritmo")
    por_fam = reg.mejores(criterio, "Familia")
    mejor = por_alg.iloc[0]
    v.metricas([{"et": f"Mejor modelo según {criterio}", "val": num(mejor[criterio]), "nota": mejor["Modelo"],
               "tipo": "destacada"},
              {"et": "R² del mejor", "val": num(mejor["R²"])},
              {"et": "Experimentos", "val": len(res)},
              {"et": "Algoritmos · familias", "val": f"{res['Algoritmo'].nunique()} · {res['Familia'].nunique()}"}])

    st.markdown("**Mejor modelo de cada familia** (comparación entre familias)")
    v.tabla(por_fam.set_index("Modelo"))
    st.markdown("**Mejor modelo de cada algoritmo** (comparación dentro de cada familia)")
    v.tabla(por_alg.set_index("Modelo"))
    st.plotly_chart(reg.fig_comparacion(nombres=list(por_alg["Modelo"]),
                                           titulo="Mejor modelo de cada algoritmo (conjunto de prueba)"),
                    width="stretch", key="reg_fig_mejores")
    st.plotly_chart(reg.fig_real_vs_predicho(list(por_fam["Modelo"]), columnas=min(4, len(por_fam))),
                    width="stretch", key="reg_fig_rvp_fam")
    st.caption("Un punto sobre la línea roja es una predicción perfecta. Los puntos bajo la línea en los valores "
               "altos indican que el modelo subestima los casos extremos.")

    st.markdown("**Historial completo de experimentos**")
    v.tabla(res.set_index("Modelo"), altura=min(600, 38 * len(res) + 40))
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c1:
        v.descargar_csv(res.set_index("Modelo"), "resultados_regresion.csv", "Descargar resultados", clave="reg_dl_res")
    borrar = c2.selectbox("Experimento", ["—"] + list(res["Modelo"]), key="reg_borrar", label_visibility="collapsed")
    if c3.button("Borrar experimento", width="stretch", disabled=borrar == "—", key="reg_btn_borrar"):
        ctrl.borrar_experimento(reg, borrar)
        st.rerun()


# ======================================================================
# Analizar
# ======================================================================
def _analizar(ctrl, reg):
    v = ctrl.view
    res = reg.resultados()
    if res.empty:
        st.info("Todavía no hay experimentos para analizar.")
        return
    criterio = _criterio()
    nombres = list(res["Modelo"])
    mejor = reg.mejor_modelo(criterio)
    nombre = st.selectbox("Modelo a analizar", nombres, index=nombres.index(mejor), key="reg_analizar",
                          help=f"Por defecto el mejor según {criterio}.")
    exp = reg.experimentos[nombre]
    ind = exp["indices"]
    r = reg.residuos(nombre)
    v.metricas([{"et": "RMSE", "val": num(ind["RMSE"]), "tipo": "destacada", "nota": "raíz del error cuadrático medio"},
              {"et": "MAE", "val": num(ind["MAE"]), "nota": "error absoluto medio"},
              {"et": "Error relativo (ER)", "val": num(ind["ER"], pct=True), "nota": "Σ|error| / Σ|real|"},
              {"et": "R²", "val": num(ind["R²"]), "nota": "varianza explicada",
               "tipo": "mala" if ind["R²"] < 0 else ""},
              {"et": "Error máximo", "val": num(r["Residuo"].abs().max())},
              {"et": "Residuo promedio", "val": num(r["Residuo"].mean()), "nota": "cerca de 0 = sin sesgo"},
              {"et": "Predicciones negativas", "val": int((r["Predicción"] < 0).sum()),
               "nota": "imposibles si la variable es ≥ 0"},
              {"et": "Tiempo", "val": f"{ind['Tiempo (s)']:.2f} s", "nota": f"{exp['algoritmo']} · "
                                                                        f"{ALGORITMOS[exp['algoritmo']]['familia']}"}],
             columnas=4)
    v.formula("RMSE = √(Σ(y−ŷ)²/n)   MAE = Σ|y−ŷ|/n   ER = Σ|y−ŷ| / Σ|y|   R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²")
    st.write("")

    pestanas = ["Real vs predicción", "Residuos"]
    if reg.es_lineal(nombre):
        pestanas.append("Coeficientes")
    pestanas.append("Importancia de variables")
    if exp["algoritmo"] == "DT":
        pestanas.append("Árbol")
    pestanas += ["Validación cruzada", "Configuración"]
    tabs = dict(zip(pestanas, st.tabs(pestanas)))

    with tabs["Real vs predicción"]:
        st.plotly_chart(reg.fig_real_vs_predicho([nombre], 1), width="stretch", key="reg_rvp")
    with tabs["Residuos"]:
        st.plotly_chart(reg.fig_residuos(nombre), width="stretch", key="reg_res")
        st.caption("Si el modelo está bien especificado los residuos se ven como una nube sin forma alrededor de 0. "
                   "Un abanico que se abre indica heterocedasticidad; una tendencia, que falta captar algo.")
    if "Coeficientes" in tabs:
        with tabs["Coeficientes"]:
            coef = reg.coeficientes(nombre)
            optimo = reg.parametro_optimo(nombre)
            items = [{"et": "Intercepto (w0)", "val": num(reg.intercepto(nombre))},
                     {"et": "Coeficientes distintos de 0", "val": f"{int((coef.abs() > 1e-10).sum())} de {len(coef)}"}]
            if optimo:
                items.append({"et": "alpha escogido por validación cruzada", "val": f"{optimo['alpha']:.4g}".replace(".", ","),
                              "tipo": "destacada"})
            v.metricas(items, columnas=3)
            top = st.slider("Coeficientes a mostrar", 5, 60, 20, key="reg_top_coef")
            st.plotly_chart(reg.fig_coeficientes(nombre, top), width="stretch", key="reg_coef")
            escalado = exp["config"].get("escalar")
            st.caption("Variables estandarizadas: los coeficientes se pueden comparar entre sí." if escalado else
                       "Variables en su escala original: un coeficiente indica cuánto cambia la predicción por "
                       "unidad de esa variable.")
            with st.expander("Todos los coeficientes"):
                v.tabla(coef.to_frame())
    with tabs["Importancia de variables"]:
        top = st.slider("Variables a mostrar", 5, 40, 15, key="reg_top")
        with st.spinner("Calculando importancia..."):
            st.plotly_chart(reg.fig_importancia(nombre, top), width="stretch", key="reg_imp")
        if reg.importancia_propia(nombre):
            st.caption("Importancia propia del modelo (cuánto reduce cada variable el error en las divisiones de los "
                       "árboles).")
        else:
            st.caption("Este modelo no trae importancia propia: se calcula por permutación, es decir, cuánto sube el "
                       "RMSE en prueba al desordenar esa variable.")
    if "Árbol" in tabs:
        with tabs["Árbol"]:
            prof = st.slider("Niveles a mostrar", 1, 5, 3, key="reg_prof")
            st.plotly_chart(reg.fig_arbol(nombre, prof), width="stretch", key="reg_arbol")
            with st.expander("Reglas del árbol en texto"):
                st.code(reg.reglas_arbol(nombre, max(prof, 6)), language="text")
    with tabs["Validación cruzada"]:
        c1, c2 = st.columns([1, 3])
        k = c1.number_input("Folds (k)", 2, 20, 5, key="reg_k")
        c2.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
        clave = f"cv_{nombre}_{k}"
        if c2.button("Correr validación cruzada (sobre entrenamiento)", key="reg_cv"):
            with st.spinner("Validando..."):
                ctrl.validacion_cruzada("reg", reg, nombre, k)
        cv = ctrl.model.resultado("reg", clave)
        if cv is not None:
            v.tabla(cv)
            st.caption("Una desviación estándar baja indica que el modelo es estable entre particiones.")
    with tabs["Configuración"]:
        st.json({k: val for k, val in exp["config"].items() if val is not None} or {"configuración": "estándar"})
        st.caption("Parámetros del estimador y opciones de preprocesamiento usadas en este experimento.")


# ======================================================================
# Curvas de hiperparámetros y regularización
# ======================================================================
def _curvas(ctrl, reg):
    v = ctrl.view
    st.markdown("Estos cálculos usan validación cruzada sobre el entrenamiento y **no se guardan en el historial**.")
    with st.container(border=True):
        st.markdown("**Ruta de regularización (Lasso y Ridge)**")
        c1, c2, c3 = st.columns([1.2, 1, 1.6])
        tipo = c1.segmented_control("Modelo", ["Lasso", "Ridge"], default="Lasso", key="reg_ruta_tipo") or "Lasso"
        k = c2.number_input("Folds (k)", 2, 20, 10, key="reg_ruta_k")
        c3.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if c3.button("Calcular ruta", type="primary", key="reg_ruta_btn"):
            with st.spinner(f"Ajustando {tipo} con 40 valores de alpha..."):
                ctrl.ruta_regularizacion(reg, tipo, k)
        ruta = ctrl.model.resultado("reg", f"ruta_{tipo}")
        if ruta is not None:
            coefs, resumen = ruta
            st.plotly_chart(Regresion.fig_ruta_regularizacion(coefs, resumen, tipo), width="stretch",
                            key="reg_ruta_fig")
            mejor = resumen.loc[resumen["RMSE validación"].idxmin()]
            st.caption(f"Mejor alpha = {mejor['alpha']:.4g} (RMSE de validación {mejor['RMSE validación']:.4f}, "
                       f"{int(mejor['Coeficientes ≠ 0'])} coeficientes distintos de 0). En color, las variables con "
                       "mayor coeficiente en ese alpha. Lasso va anulando coeficientes al subir alpha; Ridge los "
                       "encoge sin anularlos.")
            with st.expander("Tabla de la ruta"):
                v.tabla(resumen.set_index("alpha"))

    with st.container(border=True):
        st.markdown("**Curva de un hiperparámetro** · RMSE de entrenamiento contra validación (muestra el sobreajuste)")
        c1, c2, c3, c4 = st.columns([1.2, 2, 0.8, 1])
        algoritmo = c1.selectbox("Algoritmo", list(Regresion.CURVAS), key="reg_curva_alg",
                                 format_func=lambda a: NOMBRES.get(a, a))
        parametro, valores = Regresion.CURVAS[algoritmo]
        texto = c2.text_input(f"Valores de {parametro} (separados por coma)",
                              ", ".join(f"{x:g}" for x in valores), key=f"reg_curva_val_{algoritmo}")
        k = c3.number_input("Folds", 2, 20, 5, key="reg_curva_k")
        c4.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if c4.button("Calcular curva", type="primary", key="reg_curva_btn"):
            with st.spinner(f"Validando los valores de {parametro}..."):
                ctrl.curva_parametro(reg, algoritmo, parametro, texto, k)
        resultado = ctrl.model.resultado("reg", "curva")
        if resultado is not None:
            alg, curva = resultado
            st.plotly_chart(Regresion.fig_curva_parametro(curva, f"{alg}: RMSE según {curva.columns[0]}"),
                            width="stretch", key="reg_curva_fig")
            st.caption("Cuando el error de entrenamiento sigue bajando pero el de validación sube, el modelo está "
                       "sobreajustando. El resto de parámetros queda en su valor estándar.")
            v.tabla(curva.set_index(curva.columns[0]))


# ======================================================================
# Benchmarking
# ======================================================================
def _benchmark(ctrl, reg):
    v = ctrl.view
    res = reg.resultados()
    if len(res) < 2:
        st.info("Entrená al menos dos modelos para compararlos con validación cruzada repetida.")
        return
    st.markdown("Un solo conjunto de prueba puede favorecer a un modelo por azar. Aquí cada modelo se evalúa con "
                "**validación cruzada repetida sobre el entrenamiento, con las mismas particiones para todos**, y se "
                "aplican las pruebas de Friedman y de Wilcoxon (con corrección de Holm) para ver si las diferencias "
                "son significativas.")
    criterio = _criterio()
    with st.container(border=True):
        nombres = list(res["Modelo"])
        defecto = list(reg.mejores(criterio, "Algoritmo")["Modelo"])
        elegidos = st.multiselect("Modelos a comparar", nombres, default=defecto, key="reg_bench_mod",
                                  help="Por defecto el mejor de cada algoritmo.")
        c1, c2, c3, c4 = st.columns(4)
        k = c1.number_input("Particiones (k)", 2, 20, 10, key="reg_bench_k")
        rep = c2.number_input("Repeticiones", 1, 10, 3, key="reg_bench_rep")
        crit_b = c3.selectbox("Índice a comparar", ["RMSE", "MAE", "R²"], key="reg_bench_crit")
        c4.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        correr = c4.button("Correr benchmarking", type="primary", width="stretch", key="reg_bench_btn",
                           disabled=len(elegidos) < 2)
        st.caption(f"{len(elegidos)} modelos × {int(k) * int(rep)} particiones = "
                   f"{len(elegidos) * int(k) * int(rep)} entrenamientos. Con muchos árboles o filas puede tardar "
                   "unos minutos.")
    if correr:
        ctrl.benchmark(reg, elegidos, k, rep)
    if reg.benchmark_tabla is None:
        return

    resumen = reg.resumen_benchmark(crit_b)
    friedman, comp = reg.prueba_estadistica(crit_b)
    mejor = resumen.iloc[0]
    items = [{"et": f"Mejor según {crit_b} promedio", "val": num(mejor[f"{crit_b} promedio"]),
              "nota": mejor["Modelo"], "tipo": "destacada"},
             {"et": "Rango promedio del mejor", "val": num(mejor["Rango promedio"], 2), "nota": "1 = mejor en todas"}]
    if friedman:
        items.append({"et": "Prueba de Friedman (p)", "val": f"{friedman['p']:.2g}".replace(".", ","),
                      "nota": "p < 0,05: los modelos no son equivalentes" if friedman["p"] < 0.05
                      else "no hay diferencias significativas",
                      "tipo": "buena" if friedman["p"] < 0.05 else ""})
    empatados = comp.loc[comp["Significativa"] == "no", "Contra"].tolist() if not comp.empty else []
    items.append({"et": "Empatados con el mejor", "val": len(empatados), "nota": "Wilcoxon + Holm, α = 0,05"})
    v.metricas(items)

    st.markdown("**Resumen de la validación cruzada repetida**")
    v.tabla(resumen.set_index("Modelo"))
    st.plotly_chart(reg.fig_benchmark(crit_b), width="stretch", key="reg_bench_fig")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"**Wilcoxon: el mejor modelo contra cada uno** ({crit_b})")
        if not comp.empty:
            v.tabla(comp.set_index("Contra").drop(columns="Mejor"),
                  formato=lambda v: (f"{v:.2e}" if abs(v) < 0.001 and v != 0 else f"{v:.4f}")
                  if isinstance(v, float) else v)
    with c2:
        st.markdown("**Mejor modelo de cada familia en el benchmarking**")
        v.tabla(resumen.groupby("Familia", sort=False).head(1).set_index("Modelo")[["Familia", f"{crit_b} promedio",
                                                                                    "Rango promedio"]])
    if empatados:
        st.caption(f"No hay evidencia de que el mejor supere a: {', '.join(empatados)}. Las particiones de una "
                   "validación cruzada comparten individuos, así que estas pruebas tienden a ser optimistas: "
                   "tomalas como apoyo, no como prueba definitiva.")
    v.descargar_csv(reg.benchmark_tabla.set_index("Modelo"), "benchmark_regresion.csv",
                  "Descargar resultados por partición", clave="reg_dl_bench")


# ======================================================================
# Predecir
# ======================================================================
def _predecir(ctrl, reg, objetivo):
    v = ctrl.view
    res = reg.resultados()
    if res.empty:
        st.info("Entrená al menos un modelo para poder predecir.")
        return
    nombres = list(res["Modelo"])
    c1, c2 = st.columns([1, 2])
    elegido = c1.selectbox("Modelo a usar", nombres, index=nombres.index(reg.mejor_modelo(_criterio())),
                           key="reg_pmod")
    archivo = c2.file_uploader("CSV con los datos nuevos (mismas columnas predictoras)", type=["csv", "txt"],
                               key="reg_nuevos")
    usar_prueba = c1.checkbox("O usar los datos de prueba", key="reg_usarprueba")
    nuevos = None
    if archivo is not None:
        nuevos = ctrl.leer_nuevos(archivo.getvalue())
    elif usar_prueba:
        nuevos = reg.X_test.copy()
    if nuevos is None:
        return
    pred = ctrl.predecir(reg, elegido, nuevos)
    if pred is None:
        return
    pred = pd.Series(pred, index=nuevos.index)
    salida = nuevos.copy()
    salida[f"Predicción {objetivo}"] = pred
    v.metricas([{"et": "Filas predichas", "val": len(pred)},
                {"et": "Predicción promedio", "val": num(pred.mean())},
                {"et": "Mínimo", "val": num(pred.min())},
                {"et": "Máximo", "val": num(pred.max())}])
    if usar_prueba and archivo is None:
        st.caption(f"Con los datos de prueba: RMSE = {num(reg.experimentos[elegido]['indices']['RMSE'])}.")
    st.dataframe(salida.head(300), width="stretch", height=380)
    v.descargar_csv(salida, "predicciones_regresion.csv", "Descargar predicciones", clave="reg_dlpred")
