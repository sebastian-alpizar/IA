"""
model.py
Modelo que encapsula el estado de la aplicación y se conecta con el framework:
  - el dataset activo (y el original, para poder deshacer la preparación)
  - el archivo cargado y cómo se leyó
  - los objetos de análisis: EDA, NoSupervisado, Clasificacion y Regresion
  - los resultados que se guardan entre interacciones (clusters, validaciones, curvas...)
Hay un solo DataFrameModel por sesión (app.py lo guarda en st.session_state), así que
sus atributos se conservan mientras la persona usa la app.
"""

from eda import EDA                              # carpeta eda/
from no_supervisado import NoSupervisado         # carpeta no_supervisado/
from shared.lectura import detectar_formato, leer_csv
from supervisado import Clasificacion, Regresion  # carpeta supervisado/


class DataFrameModel:

    SECCIONES = ("clas", "reg", "ns")   # clasificación, regresión, no supervisado

    def __init__(self):
        self.datos = None               # DataFrame activo (ya preparado)
        self.datos_originales = None    # como se leyó del archivo
        self.nombre = None              # nombre del archivo
        self.version = 0                # sube cada vez que cambian los datos
        self.archivo = None             # {"bytes", "nombre", "formato"} del CSV cargado
        self.error_lectura = None
        self.clusters = {}              # nombre -> etiquetas (pd.Series) de K-means / HAC
        self._resultados = {s: {} for s in self.SECCIONES}

    # =================================================================
    # Datos
    # =================================================================
    @property
    def hay_datos(self):
        return self.datos is not None

    def forma(self):
        return None if self.datos is None else self.datos.shape

    def guardar_datos(self, df, nombre=None, originales=None):
        """Guarda un dataset nuevo y borra los resultados viejos que ya no aplican."""
        self.datos = df
        if originales is not None:
            self.datos_originales = originales
        if nombre:
            self.nombre = nombre
        self.version += 1
        self.clusters = {}
        self._resultados = {s: {} for s in self.SECCIONES}

    def guardar_archivo(self, contenido: bytes, nombre):
        """Guarda los bytes del CSV y detecta su formato (separador, decimal y codificación)."""
        self.archivo = {"bytes": contenido, "nombre": nombre, "formato": detectar_formato(contenido)}
        return self.archivo["formato"]

    def leer_archivo(self, sep, decimal, encabezado=True, indice=False):
        """Lee el CSV guardado con las opciones dadas y lo deja como dataset activo."""
        df = leer_csv(self.archivo["bytes"], sep=sep, decimal=decimal, encabezado=encabezado, indice=indice,
                      encoding=self.archivo["formato"]["encoding"])
        self.error_lectura = "Se leyó una sola columna: revisá el separador." if df.shape[1] < 2 else None
        self.guardar_datos(df, self.archivo["nombre"], originales=df)
        return df

    @staticmethod
    def leer_nuevos(contenido: bytes):
        """Lee un CSV de datos nuevos (para predecir) detectando su formato."""
        return leer_csv(contenido, **detectar_formato(contenido))

    # =================================================================
    # Objetos de análisis
    # =================================================================
    def eda(self, columnas=None):
        return EDA(self.datos if columnas is None else self.datos[columnas])

    def no_supervisado(self, columnas):
        return NoSupervisado(self.datos, columnas)

    def _supervisado(self, seccion, clase, firma, **kwargs):
        """Crea el objeto (Clasificacion o Regresion) o reutiliza el de la sesión si la configuración no cambió."""
        estado = self._resultados[seccion]
        firma = (self.version,) + firma
        if estado.get("firma") != firma:
            objeto = clase(self.datos, **kwargs)          # puede lanzar ValueError
            estado.clear()
            estado.update({"firma": firma, "objeto": objeto})
        return estado["objeto"]

    def clasificacion(self, objetivo, predictoras, positiva, prueba, semilla, duplicados):
        return self._supervisado("clas", Clasificacion, (objetivo, positiva, tuple(predictoras), prueba, semilla,
                                                         duplicados),
                                 variable_predecir=objetivo, predictoras=predictoras, positiva=positiva,
                                 test_size=prueba / 100, eliminar_duplicados=duplicados, semilla=int(semilla))

    def regresion(self, objetivo, predictoras, prueba, semilla, duplicados):
        return self._supervisado("reg", Regresion, (objetivo, tuple(predictoras), prueba, semilla, duplicados),
                                 variable_predecir=objetivo, predictoras=predictoras, test_size=prueba / 100,
                                 eliminar_duplicados=duplicados, semilla=int(semilla))

    # =================================================================
    # Resultados guardados entre interacciones
    # =================================================================
    def resultado(self, seccion, clave, defecto=None):
        return self._resultados[seccion].get(clave, defecto)

    def guardar_resultado(self, seccion, clave, valor):
        self._resultados[seccion][clave] = valor
        return valor

    def guardar_cluster(self, nombre, etiquetas):
        self.clusters[nombre] = etiquetas
