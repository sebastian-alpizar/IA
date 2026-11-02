"""
lectura.py (shared)
Lectura de archivos CSV: detecta codificación, separador y decimal, y lee el archivo
quitando los espacios de los nombres de las columnas. La usan el modelo (carga de datos)
y la predicción de datos nuevos.
"""

import csv
import io

import pandas as pd


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
