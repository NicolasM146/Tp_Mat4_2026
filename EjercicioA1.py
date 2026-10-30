"""
A.1 - Ajuste del modelo cuadrático de cnt en función de temp.

Ajusta ŷ(x) = b0 + b1·x + b2·x² y calcula el rango observado de temp.
Muestra los resultados y los guarda en "Resultados A.1.txt" en la carpeta
del script. La derivada y el punto crítico se hacen a mano en el informe.
"""
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/danwild/bike-share-prediction/"
    "master/Bike-Sharing-Dataset/day.csv"
)
FACTOR_CELSIUS = 41   # temp * 41 = grados Celsius (según el enunciado)


def cargar_datos(carpeta):
    """Lee bike+sharing+dataset/day.csv junto al script; si no está, usa la URL."""
    local = carpeta / "bike+sharing+dataset" / "day.csv"
    try:
        return pd.read_csv(local if local.exists() else URL)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise SystemExit(1)


def main():
    carpeta = Path(__file__).resolve().parent
    lineas = []

    def mostrar(texto=""):
        """Imprime en pantalla y guarda la misma línea para el archivo."""
        print(texto)
        lineas.append(str(texto))

    df = cargar_datos(carpeta)
    x = df["temp"].to_numpy()
    y = df["cnt"].to_numpy()

    mostrar("RESULTADOS A.1")
    mostrar("==============")
    mostrar(f"Fuente: day.csv | n = {len(x)} días.\n")

    # 1) Ajuste cuadrático (polyfit devuelve de mayor a menor grado)
    b2, b1, b0 = np.polyfit(x, y, 2)
    mostrar("AJUSTE CUADRÁTICO")
    mostrar(f"  b0 = {b0}")
    mostrar(f"  b1 = {b1}")
    mostrar(f"  b2 = {b2}")
    mostrar(f"  ŷ(x) = {b0:.4f} + {b1:.4f}·x + ({b2:.4f})·x²\n")

    # 2) Rango observado de temp
    mostrar("RANGO OBSERVADO DE temp")
    mostrar(f"  mínimo = {x.min()}  ->  {x.min() * FACTOR_CELSIUS:.4f} °C")
    mostrar(f"  máximo = {x.max()}  ->  {x.max() * FACTOR_CELSIUS:.4f} °C")

    salida = carpeta / "Resultados A.1.txt"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8-sig")
    print(f"\nArchivo generado: {salida}")


if __name__ == "__main__":
    main()