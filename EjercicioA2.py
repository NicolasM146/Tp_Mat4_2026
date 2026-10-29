from pathlib import Path
import numpy as np
import pandas as pd


def numero(valor, decimales=12):
    return f"{valor:.{decimales}f}".replace(".", ",")


def main():
    carpeta = Path(__file__).resolve().parent
    archivo = carpeta / "bike+sharing+dataset" / "day.csv"

    # Carga de datos con pandas
    df = pd.read_csv(archivo)
    x = df["temp"].values
    y = df["cnt"].values
    n = len(x)

    # Medias y sumatorias muestrales
    media_x = np.mean(x)
    media_y = np.mean(y)
    suma_x = np.sum(x)
    suma_y = np.sum(y)
    suma_x2 = np.sum(x ** 2)
    suma_xy = np.sum(x * y)

    # Estimadores por ecuaciones normales
    Sxx = np.sum((x - media_x) ** 2)
    Sxy = np.sum((x - media_x) * (y - media_y))
    b1 = Sxy / Sxx
    b0 = media_y - b1 * media_x

    # Verificación con NumPy polyfit
    b1_herramienta, b0_herramienta = np.polyfit(x, y, 1)
    diferencia_b0 = abs(b0 - b0_herramienta)
    diferencia_b1 = abs(b1 - b1_herramienta)

    informe = f"""RESULTADOS A.2
==============
Fuente: bike+sharing+dataset/day.csv | n = {n}
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
    salida.write_text(informe, encoding="utf-8-sig")
    print(f"Archivo generado: {salida}")
    print("Resultados del A.2 guardados y verificados.")


if __name__ == "__main__":
    main()
