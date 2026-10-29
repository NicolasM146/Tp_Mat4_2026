"""
A.1 - Modelo cuadrático de cnt en función de temp y derivación.

Ajusta y_estimado(x) = b0 + b1*x + b2*x^2, lo verifica con las ecuaciones normales,
evalúa la derivada en dos puntos, halla y clasifica el punto crítico y guarda
todo en "Resultados A.1.txt" en la carpeta del script.
Las cuentas del informe se hacen a mano; esto sirve para verificarlas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/danwild/bike-share-prediction/"
    "master/Bike-Sharing-Dataset/day.csv"
)
FACTOR_CELSIUS = 41            # temp * 41 = grados Celsius (según el enunciado)
PUNTOS = (0.25, 0.8)           # valores de x donde se evalúa la derivada


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
    x_min, x_max = x.min(), x.max()

    mostrar("RESULTADOS A.1")
    mostrar("==============")
    mostrar(f"Fuente: day.csv | n = {len(x)} días.")
    mostrar(f"Rango observado de temp: [{x_min:.4f}; {x_max:.4f}] "
            f"-> [{x_min * FACTOR_CELSIUS:.2f} °C; {x_max * FACTOR_CELSIUS:.2f} °C]\n")

    # 1) Ajuste cuadrático (polyfit devuelve de mayor a menor grado)
    b2, b1, b0 = np.polyfit(x, y, 2)

    # 2) Verificación con las ecuaciones normales: (XᵀX) b = Xᵀy
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    coinciden = np.allclose([b0, b1, b2], beta)

    mostrar("MODELO AJUSTADO")
    mostrar(f"  polyfit:            b0 = {b0:.4f}   b1 = {b1:.4f}   b2 = {b2:.4f}")
    mostrar(f"  ecuaciones normales: b0 = {beta[0]:.4f}   b1 = {beta[1]:.4f}   b2 = {beta[2]:.4f}")
    mostrar(f"  ¿Coinciden? {coinciden}")
    mostrar(f"  y_estimado(x) = {b0:.4f} + {b1:.4f}*x + ({b2:.4f})*x^2\n")

    # 3) Derivada y evaluación en dos puntos
    mostrar("DERIVADA")
    mostrar(f"  dy/dx = b1 + 2*b2*x = {b1:.4f} + ({2 * b2:.4f})*x")
    mostrar("  Unidades: alquileres/día por unidad de temp; dividir por "
            f"{FACTOR_CELSIUS} para obtener alquileres/día por °C.\n")
    for xv in PUNTOS:
        d = b1 + 2 * b2 * xv
        mostrar(f"  x = {xv} ({xv * FACTOR_CELSIUS:.2f} °C, "
                f"{'dentro' if x_min <= xv <= x_max else 'FUERA'} del rango observado)")
        mostrar(f"    dy/dx = {b1:.4f} + ({2 * b2:.4f})*{xv} = {d:.4f}")
        mostrar(f"    por °C: {d:.4f} / {FACTOR_CELSIUS} = {d / FACTOR_CELSIUS:.4f}")
    mostrar()

    # 4) Punto crítico
    xc = -b1 / (2 * b2)
    d2 = 2 * b2
    tipo = "máximo" if d2 < 0 else "mínimo"
    yc = b0 + b1 * xc + b2 * xc ** 2
    mostrar("PUNTO CRÍTICO")
    mostrar(f"  x* = -b1 / (2*b2) = {xc:.6f}  ->  {xc * FACTOR_CELSIUS:.4f} °C")
    mostrar(f"  d2y/dx2 = 2*b2 = {d2:.4f}  ->  {tipo}")
    mostrar(f"  ¿Dentro del rango observado? {x_min <= xc <= x_max} "
            f"([{x_min * FACTOR_CELSIUS:.2f}; {x_max * FACTOR_CELSIUS:.2f}] °C)")
    mostrar(f"  Demanda esperada en x*: y_estimado(x*) = {yc:.2f} alquileres/día")

    # Se guarda el informe
    salida = carpeta / "Resultados A.1.txt"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8-sig")
    print(f"\nArchivo generado: {salida}")


if __name__ == "__main__":
    main()