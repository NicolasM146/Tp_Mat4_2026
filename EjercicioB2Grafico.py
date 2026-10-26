import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/danwild/bike-share-prediction/master/Bike-Sharing-Dataset/day.csv"
df = pd.read_csv(url)

# Definir x (atemp) y y (cnt)
x = df["atemp"]
y = df["cnt"]

n = len(df)
x_bar, y_bar = x.mean(), y.mean()

sxx = np.sum((x - x_bar) ** 2)
sxy = np.sum((x - x_bar) * (y - y_bar))

b1 = sxy / sxx
b0 = y_bar - b1 * x_bar

plt.figure(figsize=(9, 6))

plt.scatter(x, y, alpha=0.4, color="#177c78", label="Días observados")

x_linea = np.linspace(x.min(), x.max(), 100)
y_linea = b0 + b1 * x_linea
plt.plot(
    x_linea,
    y_linea,
    color="#cb4545",
    linewidth=2.5,
    label=f"Recta ajustada: ŷ = {b1:.2f} · atemp + {b0:.2f}",
)

plt.title(
    "Diagrama de Dispersión: cnt y atemp (Sensación Térmica)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Sensación Térmica Normalizada (atemp)", fontsize=11)
plt.ylabel("Cantidad de Alquileres Diarios (cnt)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

plt.show()
