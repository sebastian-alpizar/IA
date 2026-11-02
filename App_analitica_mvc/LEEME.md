# Analítica de datos (Streamlit)

App web para subir un CSV y hacerle todo el análisis del curso de Inteligencia Artificial:

| Sección | Qué hace | Equivale a |
|---|---|---|
| Carga de datos | CSV con detección de separador, decimal y codificación; ejemplos incluidos; limpieza (vacíos, duplicados, columnas, tipos) | predictoR: Datos |
| Exploración | Estadísticas, distribuciones, dispersión, correlación, atípicos (IQR), normalidad | Estadísticas básicas |
| No supervisados → ACP y clustering | ACP (varianza, plano principal, círculo, integrado, cos²), K-means (codo, perfil, radar), HAC (dendrograma), t-SNE, UMAP | discoveR |
| Supervisados → Clasificación | Los 5 del Laboratorio 2 (KNN, Árboles de Decisión, Random Forest, XGBoost, AdaBoost) y además Gradient Boosting, SVM, Bayes, logística, LDA, QDA y red neuronal. Configuración estándar y variaciones (las del notebook del Lab 2 con un clic), imputación, escalado y selección de atributos (SelectKBest); historial de experimentos y mejor modelo por algoritmo (F1 macro, AUC, ...); PG/EG, PP/PN/AP/AN/PFP/PFN, Precision, Recall, F1, matriz de confusión, ROC, importancia de variables, árbol, validación cruzada, curva k; predicción de datos nuevos | predictoR + Laboratorio 2 |
| Supervisados → Regresión | Los del Laboratorio 3 por familia: lineales (simple, múltiple, Lasso, LassoCV, Ridge, RidgeCV), núcleos (SVR rbf, lineal, polinomial) y árboles (Árbol de regresión, Random Forest, XGBoost), además Gradient Boosting y KNN. Configuración estándar y variaciones (las del notebook del Lab 3 con un clic), imputación, escalado y selección de atributos; RMSE, MAE, ER y R²; mejor modelo por algoritmo y por familia; real vs predicción, residuos, coeficientes, alpha óptimo, importancia de variables, árbol; ruta de regularización y curvas de hiperparámetros; benchmarking con validación cruzada repetida y pruebas de Friedman y Wilcoxon (Holm); predicción de datos nuevos | predictoR + Laboratorio 3 |

## Cómo correrla (Windows)

Abrí **Anaconda Prompt** (o la terminal de VS Code), andá a esta carpeta y corré:

```
pip install -r requirements.txt
streamlit run app.py
```

Se abre sola en el navegador (http://localhost:8501). Para cerrarla: `Ctrl + C` en la terminal.

También podés hacer doble clic en `iniciar.bat`.

## Estructura

Tiene la misma organización que el framework del Laboratorio: una carpeta por rama del
framework (cada clase hereda de la anterior) y la interfaz con el patrón MVC.

```
app.py                     entrada: crea el Model, la View y el Controller y arma el menú
shared/                    lo que usan todas las capas
  graficos.py              paleta y plantilla de los gráficos (Plotly)
  lectura.py               lectura de CSV (detecta separador, decimal y codificación)
  formato.py               números con coma decimal
eda/
  eda.py                   EDA: clase base
no_supervisado/
  no_supervisado.py        NoSupervisado(EDA): ACP, K-means, HAC, t-SNE, UMAP (Laboratorio 1)
supervisado/
  supervisado.py           Supervisado(EDA): partición, pipeline, historial, validación cruzada, ...
  clasificacion/
    algoritmos.py          configuración de los algoritmos (parámetros y variaciones del Lab 2)
    clasificacion.py       Clasificacion(Supervisado) (Laboratorio 2 + resto del curso)
  regresion/
    algoritmos.py          configuración de los algoritmos (parámetros y variaciones del Lab 3)
    regresion.py           Regresion(Supervisado) (Laboratorio 3 + resto del curso)
mvc/
  model.py                 DataFrameModel: datos y resultados de la sesión, crea los objetos de análisis
  view.py                  DataFrameView: componentes visuales (encabezados, métricas, tablas, mensajes)
  estilo.css               colores y tipografía de la interfaz
  Controller.py            DataFrameController: acciones del usuario (cargar, entrenar, validar, predecir...)
  paginas/                 una vista por sección del menú (inicio, carga, exploración,
                           no supervisado, clasificación y regresión)
datos_ejemplo/             iris, cáncer de mama, USCrime (Lab 1 y Lab 3), potabilidad y diabetes (Lab 2)
recursos/                  logo e ícono
.streamlit/config.toml     colores del tema
```

Jerarquía de clases:

```
EDA ─┬─ NoSupervisado
     └─ Supervisado ─┬─ Clasificacion
                     └─ Regresion
```

Flujo MVC: cada página (vista) muestra los datos del `DataFrameModel` y, cuando la persona
hace algo (cargar un archivo, entrenar, validar...), llama al `DataFrameController`, que le
pide al modelo que lo haga y usa la `DataFrameView` para mostrar mensajes y errores.

Convención de la matriz de confusión: filas = valor real, columnas = predicción.
PP = VP/(VP+FN), PN = VN/(VN+FP), AP = VP/(VP+FP), AN = VN/(VN+FN), PFP = FP/(FP+VN), PFN = FN/(FN+VP).

Índices de regresión: RMSE = √(Σ(y−ŷ)²/n), MAE = Σ|y−ŷ|/n, ER = Σ|y−ŷ|/Σ|y| (error relativo, como en el código guía), R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)². En RMSE, MAE y ER el mejor modelo es el de menor valor; en R², el de mayor.
