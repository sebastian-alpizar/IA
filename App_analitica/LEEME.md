# Analítica de datos (Streamlit)

App web para subir un CSV y hacerle todo el análisis del curso de Inteligencia Artificial:

| Sección | Qué hace | Equivale a |
|---|---|---|
| Carga de datos | CSV con detección de separador, decimal y codificación; ejemplos incluidos; limpieza (vacíos, duplicados, columnas, tipos) | predictoR: Datos |
| Exploración | Estadísticas, distribuciones, dispersión, correlación, atípicos (IQR), normalidad | Estadísticas básicas |
| No supervisados → ACP y clustering | ACP (varianza, plano principal, círculo, integrado, cos²), K-means (codo, perfil, radar), HAC (dendrograma), t-SNE, UMAP | discoveR |
| Supervisados → Clasificación | Los 5 del Laboratorio 2 (KNN, Árboles de Decisión, Random Forest, XGBoost, AdaBoost) y además Gradient Boosting, SVM, Bayes, logística, LDA, QDA y red neuronal. Configuración estándar y variaciones (las del notebook del Lab 2 con un clic), imputación, escalado y selección de atributos (SelectKBest); historial de experimentos y mejor modelo por algoritmo (F1 macro, AUC, ...); PG/EG, PP/PN/AP/AN/PFP/PFN, Precision, Recall, F1, matriz de confusión, ROC, importancia de variables, árbol, validación cruzada, curva k; predicción de datos nuevos | predictoR + Laboratorio 2 |

## Cómo correrla (Windows)

Abrí **Anaconda Prompt** (o la terminal de VS Code), andá a esta carpeta y corré:

```
pip install -r requirements.txt
streamlit run app.py
```

Se abre sola en el navegador (http://localhost:8501). Para cerrarla: `Ctrl + C` en la terminal.

También podés hacer doble clic en `iniciar.bat`.

## Estructura

```
app.py                  entrada: menú lateral y páginas
nucleo/                 el framework de la clase (sin interfaz)
  eda.py                EDA: clase base
  no_supervisado.py     NoSupervisado(EDA): ACP, K-means, HAC, t-SNE, UMAP
  clasificacion.py      SupervisadoLab(EDA) -> ClasificacionLab (Laboratorio 2 + resto del curso)
  graficos.py           paleta y plantilla de los gráficos
vistas/                 una página por sección del menú
datos_ejemplo/          iris, cáncer de mama, USCrime, potabilidad y predicción de diabetes (Lab 2)
.streamlit/config.toml  colores del tema
```

Convención de la matriz de confusión: filas = valor real, columnas = predicción.
PP = VP/(VP+FN), PN = VN/(VN+FP), AP = VP/(VP+FP), AN = VN/(VN+FN), PFP = FP/(FP+VN), PFN = FN/(FN+VP).
