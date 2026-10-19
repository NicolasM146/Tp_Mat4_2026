import csv
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def cargar_datos(carpeta):
    """Lee day.csv o el ZIP, siempre desde la carpeta de este programa."""
    archivo_csv = carpeta / "day.csv"
    archivo_zip = carpeta / "bike+sharing+dataset.zip"
    if archivo_csv.is_file():
        with archivo_csv.open(encoding="utf-8-sig", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
        fuente = "day.csv"
    elif archivo_zip.is_file():
        with ZipFile(archivo_zip) as comprimido:
            with comprimido.open("day.csv") as archivo:
                with io.TextIOWrapper(archivo, encoding="utf-8-sig", newline="") as texto:
                    filas = list(csv.DictReader(texto))
        fuente = "day.csv dentro de bike+sharing+dataset.zip"
    else:
        raise FileNotFoundError(
            "Colocá day.csv o bike+sharing+dataset.zip junto a EjercicioA2.py."
        )

    x = np.array([float(fila["temp"]) for fila in filas])
    y = np.array([float(fila["cnt"]) for fila in filas])
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Los datos deben contener al menos dos filas y valores finitos.")
    if np.ptp(x) == 0:
        raise ValueError("temp debe tener valores distintos para ajustar una recta.")
    return x, y, fuente


def numero(valor, decimales=12):
    return f"{valor:.{decimales}f}".replace(".", ",")


def main():
    carpeta = Path(__file__).resolve().parent
    x, y, fuente = cargar_datos(carpeta)
    n = len(x)
    media_x = x.mean()
    media_y = y.mean()
    suma_x = x.sum()
    suma_y = y.sum()
    suma_x2 = np.sum(x ** 2)
    suma_xy = np.sum(x * y)

    Sxx = np.sum((x - media_x) ** 2)
    Sxy = np.sum((x - media_x) * (y - media_y))
    b1 = Sxy / Sxx
    b0 = media_y - b1 * media_x

    # polyfit devuelve primero la pendiente y luego la ordenada al origen.
    b1_herramienta, b0_herramienta = np.polyfit(x, y, 1)
    diferencia_b0 = abs(b0 - b0_herramienta)
    diferencia_b1 = abs(b1 - b1_herramienta)
    coinciden = np.allclose(
        [b0, b1], [b0_herramienta, b1_herramienta], rtol=0, atol=1e-8
    )
    if not coinciden:
        raise ArithmeticError("Los dos métodos no coinciden: revisar el cálculo.")

    informe = f"""RESULTADOS A.2
==============
Fuente: {fuente} | n = {n}
x = temp (normalizada); y = cnt (alquileres diarios).

RECTA FINAL
  cnt_estimado = {numero(b0, 4)} + {numero(b1, 4)}*temp

COEFICIENTES Y COMPROBACIÓN
  Método              b0 (ordenada)      b1 (pendiente)
  Fórmulas            {numero(b0, 8)}      {numero(b1, 8)}
  NumPy polyfit       {numero(b0_herramienta, 8)}      {numero(b1_herramienta, 8)}
  Diferencia absoluta {diferencia_b0:.2e}            {diferencia_b1:.2e}
  Resultado: COINCIDEN (tolerancia absoluta: 1e-8).

VALORES CALCULADOS
  Media de x = {numero(media_x, 8)}
  Media de y = {numero(media_y, 8)}
  Sxx        = {numero(Sxx, 8)}
  Sxy        = {numero(Sxy, 8)}

  b1 = Sxy/Sxx = {numero(Sxy, 8)}/{numero(Sxx, 8)} = {numero(b1, 8)}
  b0 = media_y - b1*media_x = {numero(b0, 8)}

DERIVADAS PARCIALES
  sum indica suma de i=1 a n; df/db0 y df/db1 son parciales.
  f(b0,b1) = sum [yi - b0 - b1*xi]^2

  df/db0 = sum 2*(yi-b0-b1*xi)*(-1)
         = -2*(sum yi - n*b0 - b1*sum xi)
  df/db1 = sum 2*(yi-b0-b1*xi)*(-xi)
         = -2*(sum xi*yi - b0*sum xi - b1*sum xi^2)

ECUACIONES NORMALES (derivadas = 0)
  n*b0 + (sum xi)*b1 = sum yi
  (sum xi)*b0 + (sum xi^2)*b1 = sum xi*yi

  Con los datos:
  {n}*b0 + {numero(suma_x, 8)}*b1 = {numero(suma_y, 0)}
  {numero(suma_x, 8)}*b0 + {numero(suma_x2, 8)}*b1 = {numero(suma_xy, 8)}

DESPEJE
  b0 = (sum yi)/n - b1*(sum xi)/n = media_y - b1*media_x
  (media_y - b1*media_x)*sum xi + b1*sum xi^2 = sum xi*yi
  b1*(sum xi^2 - n*media_x^2) = sum xi*yi - n*media_x*media_y

  Sxx = sum (xi-media_x)^2 = sum xi^2 - n*media_x^2
  Sxy = sum (xi-media_x)*(yi-media_y) = sum xi*yi - n*media_x*media_y
  b1 = Sxy/Sxx; b0 = media_y - b1*media_x.

Valores mostrados redondeados; cálculos con precisión completa.
"""

    salida = carpeta / "Resultados A.2.txt"
    # UTF-8 con BOM permite reconocer los acentos en Bloc de notas.
    salida.write_text(informe, encoding="utf-8-sig")
    print(f"Archivo generado: {salida}")
    print("Resultados del A.2 guardados y verificados.")


if __name__ == "__main__":
    main()
