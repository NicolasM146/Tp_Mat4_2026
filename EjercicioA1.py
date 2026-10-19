import numpy as np
import pandas as pd

df = pd.read_csv("bike+sharing+dataset/day.csv")
x = df["temp"].to_numpy()
y = df["cnt"].to_numpy()

# 1) Ajuste cuadrático
b2, b1, b0 = np.polyfit(x, y, 2)
print("Cuadrático:", b0, b1, b2)

# 2) rango observado
print("rango temp:", x.min(), x.max(), "->", x.min() * 41, x.max() * 41, "°C")

# 3) Ajuste lineal por mínimos cuadrados
b1, b0 = np.polyfit(x, y, 1)
print("lineal:", b0, b1)