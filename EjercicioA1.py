import numpy as np
import pandas as pd

df = pd.read_csv("bike+sharing+dataset/day.csv")   # o la URL directa si preferís no commitear el CSV
x = df["temp"].to_numpy()
y = df["cnt"].to_numpy()

# 1) Ajuste cuadrático (polyfit devuelve de mayor a menor grado)
b2, b1, b0 = np.polyfit(x, y, 2)
print("polyfit:", b0, b1, b2)

# 2) Verificación con ecuaciones normales
X = np.column_stack([np.ones_like(x), x, x**2])
beta = np.linalg.solve(X.T @ X, X.T @ y)
print("normales:", beta)

# 3) Punto crítico y rango observado
xc = -b1 / (2 * b2)
print("x* =", xc, "->", xc * 41, "°C")
print("rango temp:", x.min(), x.max(), "->", x.min() * 41, x.max() * 41, "°C")