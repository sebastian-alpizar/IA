"""
verificar_arquitectura.py
Revisa automáticamente que la app respete la arquitectura (MVC + framework por herencia)
y que las funciones y clases se usen. Correr desde la carpeta App_analitica:

    python verificar_arquitectura.py

Lee el código con el módulo ast (no ejecuta la app). Cada regla imprime OK o los lugares
exactos (archivo:línea) donde no se cumple.
"""

import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NUCLEO = ["shared", "eda", "no_supervisado", "supervisado"]          # framework (sin interfaz)
PAGINAS = sorted((RAIZ / "mvc" / "paginas").glob("[a-z]*.py"))

# Acciones que cambian el estado: solo las puede llamar el controlador
MUTADORES_NUCLEO = {"ejecutar", "entrenar", "comparar", "particionar", "borrar_experimento", "validacion_cruzada",
                    "benchmark", "curva_k", "curva_parametro", "ruta_regularizacion", "predecir", "acp", "kmeans",
                    "hac", "codo", "tsne", "umap"}
MUTADORES_MODELO = {"guardar_datos", "guardar_archivo", "leer_archivo", "guardar_resultado", "guardar_cluster",
                    "clasificacion", "regresion"}
CLASES_NUCLEO = {"EDA", "NoSupervisado", "Supervisado", "Clasificacion", "Regresion"}
WIDGETS = {"button", "selectbox", "multiselect", "slider", "number_input", "text_input", "checkbox", "toggle",
           "radio", "file_uploader", "tabs", "columns", "markdown", "dataframe", "plotly_chart", "metric",
           "info", "warning", "error", "success", "progress", "download_button"}

fallas = 0


def arbol(ruta):
    return ast.parse(ruta.read_text(encoding="utf-8"))


def archivos(*carpetas):
    for c in carpetas:
        p = RAIZ / c
        yield from ([p] if p.is_file() else sorted(p.rglob("*.py")))


def importa(ruta):
    """Módulos que importa un archivo (raíz del nombre)."""
    mods = set()
    for n in ast.walk(arbol(ruta)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def regla(titulo, problemas):
    global fallas
    if problemas:
        fallas += len(problemas)
        print(f"  FALLA  {titulo}")
        for p in problemas:
            print(f"           - {p}")
    else:
        print(f"  OK     {titulo}")


def rel(ruta, n=None):
    return f"{ruta.relative_to(RAIZ)}" + (f":{n.lineno}" if n is not None else "")


def llamadas(ruta):
    """(nodo, nombre del método, texto del objeto) de cada llamada obj.metodo(...)."""
    for n in ast.walk(arbol(ruta)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            yield n, n.func.attr, ast.unparse(n.func.value)


print("\n1. Capas separadas (quién puede importar a quién)")
regla("El framework (shared, eda, no_supervisado, supervisado) no conoce la interfaz: no importa streamlit ni mvc",
      [f"{rel(f)} importa {m}" for f in archivos(*NUCLEO) for m in importa(f) & {"streamlit", "mvc"}])
regla("El Model no importa streamlit (no dibuja nada)",
      [rel(RAIZ / "mvc/model.py")] if "streamlit" in importa(RAIZ / "mvc/model.py") else [])
regla("La View no importa el framework (solo muestra lo que le pasan)",
      [f"importa {m}" for m in importa(RAIZ / "mvc/view.py") & set(NUCLEO + ["mvc"])])

print("\n2. Las páginas (vistas) no cambian el estado: todo pasa por el Controller")
p_instancia, p_mutan_nucleo, p_mutan_modelo, p_asignan, p_sesion = [], [], [], [], []
for f in PAGINAS:
    for n in ast.walk(arbol(f)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in CLASES_NUCLEO:
            p_instancia.append(f"{rel(f, n)} crea {n.func.id}(...)")
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                texto = ast.unparse(t)
                if isinstance(t, ast.Attribute) and (texto.startswith("ctrl.model.") or texto.startswith("m.")):
                    p_asignan.append(f"{rel(f, n)} asigna {texto}")
                if texto.startswith("st.session_state.") and texto not in ("st.session_state.sel_sep",
                                                                            "st.session_state.sel_dec"):
                    p_sesion.append(f"{rel(f, n)} escribe {texto}")
    for n, metodo, obj in llamadas(f):
        if obj in ("ctrl", "ctrl.view", "v"):
            continue
        if metodo in MUTADORES_NUCLEO:
            p_mutan_nucleo.append(f"{rel(f, n)} llama {obj}.{metodo}()")
        if metodo in MUTADORES_MODELO and obj in ("ctrl.model", "m"):
            p_mutan_modelo.append(f"{rel(f, n)} llama {obj}.{metodo}()")
regla("Las páginas no crean objetos del framework (los crea el Model)", p_instancia)
regla("Las páginas no entrenan, validan, predicen ni agrupan directamente (lo pide el Controller)", p_mutan_nucleo)
regla("Las páginas no llaman métodos que modifican el Model", p_mutan_modelo)
regla("Las páginas no asignan atributos del Model", p_asignan)
regla("Las páginas no guardan datos en st.session_state (solo el estado de los controles de lectura)", p_sesion)

print("\n3. El Controller no dibuja: usa la View para mostrar mensajes")
ctrl = RAIZ / "mvc/Controller.py"
regla("Controller.py no crea controles ni muestra contenido con st.*",
      [f"{rel(ctrl, n)} usa st.{m}()" for n, m, o in llamadas(ctrl) if o == "st" and m in WIDGETS])

print("\n4. Herencia del framework (igual que en el Laboratorio)")
sys.path.insert(0, str(RAIZ))
from eda import EDA                                    # noqa: E402
from no_supervisado import NoSupervisado               # noqa: E402
from supervisado import Clasificacion, Regresion, Supervisado  # noqa: E402
esperado = {NoSupervisado: EDA, Supervisado: EDA, Clasificacion: Supervisado, Regresion: Supervisado}
regla("EDA -> NoSupervisado; EDA -> Supervisado -> Clasificacion y Regresion",
      [f"{c.__name__} hereda de {c.__bases__[0].__name__}, no de {p.__name__}"
       for c, p in esperado.items() if c.__bases__[0] is not p])
ganchos = ["_puntaje_seleccion", "_evaluar", "_particiones_cv", "_metricas_cv"]
regla("Clasificacion y Regresion implementan los métodos gancho de Supervisado",
      [f"{c.__name__} no redefine {g}" for c in (Clasificacion, Regresion) for g in ganchos
       if getattr(c, g) is getattr(Supervisado, g)])
repetidos = [f"{c.__name__}.{m}" for c in (Clasificacion, Regresion) for m in
             ("entrenar", "particionar", "preprocesador", "crear_pipeline", "resultados", "mejor_modelo", "mejores",
              "validacion_cruzada", "importancia_variables", "nombres_atributos", "ejecutar", "borrar_experimento")
             if m in vars(c) and not (c is Regresion and m == "particionar")]
regla("Lo común no está repetido en las hijas (se hereda de Supervisado)", repetidos)

print("\n5. Funciones y métodos que no se usan en ningún lado")
codigo = {f: f.read_text(encoding="utf-8") for f in RAIZ.rglob("*.py") if f.name != Path(__file__).name}
usados = set()
for f, texto in codigo.items():
    for n in ast.walk(ast.parse(texto)):
        if isinstance(n, ast.Attribute):
            usados.add(n.attr)
        elif isinstance(n, ast.Name):
            usados.add(n.id)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            usados.add(n.value)          # métodos llamados por nombre: ALGORITMOS[...]["metodo"]
sin_usar = {"interfaz": [], "framework": []}
for f, texto in codigo.items():
    capa = "interfaz" if rel(f).startswith(("mvc", "app.py")) else "framework"
    for n in ast.walk(ast.parse(texto)):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("__") and n.name not in usados:
            sin_usar[capa].append(f"{rel(f, n)} {n.name}")
regla("Toda función y clase de la interfaz (mvc/, app.py) se usa", sin_usar["interfaz"])
# el framework es una librería, como en el Laboratorio: puede tener métodos que la interfaz no usa
# (por ejemplo para usarlos desde un notebook); se informan pero no son falla
print("  INFO   Métodos del framework que la interfaz no usa (API para notebooks):",
      ", ".join(x.split(" ")[-1] for x in sin_usar["framework"]) or "ninguno")

print("\n6. Cada acción del Controller la usa alguna página")
metodos = [n.name for n in ast.walk(arbol(ctrl)) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
texto_paginas = "\n".join(f.read_text(encoding="utf-8") for f in PAGINAS)
regla("Métodos públicos de DataFrameController usados desde mvc/paginas/",
      [m for m in metodos if f"ctrl.{m}(" not in texto_paginas and m != "ir_a" or
       (m == "ir_a" and "ctrl.ir_a(" not in texto_paginas)])

print(f"\n{'Todo en orden' if not fallas else f'{fallas} problema(s) encontrados'}.")
sys.exit(1 if fallas else 0)
