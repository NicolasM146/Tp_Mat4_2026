"""
B.2 - Regresión lineal simple de cnt contra temp, atemp, hum y windspeed.

Para cada variable se calcula el cuadro de B.2 de dos maneras:
  1) con las fórmulas vistas en clase (Sxx, Sxy, SCE, ...)
  2) con statsmodels (la herramienta)
y se verifica que coincidan. Muestra todo por pantalla y lo guarda en
"Resultados B.2.txt" (junto con el gráfico) en la carpeta del script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

URL = (
    "https://raw.githubusercontent.com/danwild/bike-share-prediction/"
    "master/Bike-Sharing-Dataset/day.csv"
)
VARIABLES = ["temp", "atemp", "hum", "windspeed"]
X_ASTERISCO = 0.65   # x* de B.4, en unidades normalizadas
ALFA = 0.05          # intervalos al 95 %


def cargar_datos():
    """Lee data/day.csv si existe junto al script; si no, lo baja de la URL."""
    local = Path(__file__).resolve().parent / "data" / "day.csv"
    try:
        return pd.read_csv(local if local.exists() else URL)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise SystemExit(1)


def ajuste_por_formulas(x, y, x0):
    """Todo el cuadro de B.2 calculado con las fórmulas."""
    n = len(x)
    x_bar, y_bar = x.mean(), y.mean()

    sxx = np.sum((x - x_bar) ** 2)
    sxy = np.sum((x - x_bar) * (y - y_bar))
    syy = np.sum((y - y_bar) ** 2)

    b1 = sxy / sxx
    b0 = y_bar - b1 * x_bar

    sce = syy - sxy ** 2 / sxx          # = suma de (yi - yi_sombrero)^2
    sigma2 = sce / (n - 2)
    r2 = 1 - sce / syy
    r = sxy / np.sqrt(sxx * syy)        # conserva el signo de la pendiente

    t_crit = stats.t.ppf(1 - ALFA / 2, n - 2)

    # IC de los parámetros
    ee_b1 = np.sqrt(sigma2 / sxx)
    ee_b0 = np.sqrt(sigma2 * (1 / n + x_bar ** 2 / sxx))
    ic_b1 = (b1 - t_crit * ee_b1, b1 + t_crit * ee_b1)
    ic_b0 = (b0 - t_crit * ee_b0, b0 + t_crit * ee_b0)

    # ICM (respuesta media) e IP (predicción) en x*
    y0 = b0 + b1 * x0
    h = 1 / n + (x0 - x_bar) ** 2 / sxx
    mitad_icm = t_crit * np.sqrt(sigma2 * h)
    mitad_ip = t_crit * np.sqrt(sigma2 * (1 + h))

    return {
        "b0": b0, "b1": b1, "sigma2": sigma2, "R2": r2, "r": r,
        "ic_b1": ic_b1, "ic_b0": ic_b0, "y0": y0,
        "icm": (y0 - mitad_icm, y0 + mitad_icm),
        "ip": (y0 - mitad_ip, y0 + mitad_ip),
        "sxx": sxx, "sxy": sxy, "sce": sce, "syy": syy,
    }


def ajuste_por_statsmodels(nombre, x, y, x0):
    """Los mismos resultados, pero con la herramienta."""
    modelo = sm.OLS(y, sm.add_constant(x)).fit()
    nuevo = pd.DataFrame({"const": [1.0], nombre: [x0]})
    pred = modelo.get_prediction(nuevo).summary_frame(alpha=ALFA)
    ci = modelo.conf_int(alpha=ALFA)
    return {
        "b0": modelo.params.iloc[0], "b1": modelo.params.iloc[1],
        "sigma2": modelo.mse_resid, "R2": modelo.rsquared,
        "ic_b0": tuple(ci.iloc[0]), "ic_b1": tuple(ci.iloc[1]),
        "icm": (pred["mean_ci_lower"].iloc[0], pred["mean_ci_upper"].iloc[0]),
        "ip": (pred["obs_ci_lower"].iloc[0], pred["obs_ci_upper"].iloc[0]),
    }


def coinciden(a, b):
    """Compara los resultados de las dos vías."""
    claves = ["b0", "b1", "sigma2", "R2"]
    ok = all(np.isclose(a[k], b[k]) for k in claves)
    for k in ["ic_b0", "ic_b1", "icm", "ip"]:
        ok = ok and np.allclose(a[k], b[k])
    return ok


def graficar(df, nombre, b0, b1, r2, carpeta):
    """Dispersión de la variable elegida contra cnt, con la recta ajustada."""
    x, y = df[nombre], df["cnt"]
    x_linea = np.linspace(x.min(), x.max(), 100)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.4, color="#177c78", label="Días observados")
    plt.plot(
        x_linea, b0 + b1 * x_linea, color="#cb4545", linewidth=2.5,
        label=f"Recta ajustada: ŷ = {b0:.2f} + {b1:.2f} · {nombre}  (R² = {r2:.4f})",
    )
    plt.title(f"Diagrama de dispersión: cnt contra {nombre}",
              fontsize=13, fontweight="bold")
    plt.xlabel(f"{nombre} (normalizada)", fontsize=11)
    plt.ylabel("Cantidad de alquileres diarios (cnt)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    archivo = carpeta / "Grafico B.2.png"
    plt.savefig(archivo, dpi=200, bbox_inches="tight")
    print(f"Gráfico guardado en {archivo}")
    plt.show()


def main():
    carpeta = Path(__file__).resolve().parent
    lineas = []   # acá se va juntando el informe

    def mostrar(texto=""):
        """Imprime en pantalla y guarda la misma línea para el archivo."""
        print(texto)
        lineas.append(str(texto))

    df = cargar_datos()
    y = df["cnt"]
    mostrar("RESULTADOS B.2")
    mostrar("==============")
    mostrar(f"Fuente: day.csv | n = {len(df)} días.")
    mostrar(f"x* = {X_ASTERISCO} (escala normalizada), intervalos al "
            f"{int((1 - ALFA) * 100)} %.\n")

    filas = []
    ajustes = {}
    for nombre in VARIABLES:
        x = df[nombre]
        f = ajuste_por_formulas(x, y, X_ASTERISCO)
        h = ajuste_por_statsmodels(nombre, x, y, X_ASTERISCO)

        ajustes[nombre] = f
        fuera = not (x.min() <= X_ASTERISCO <= x.max())
        mostrar(f"--- {nombre} ---")
        mostrar(f"Rango observado: [{x.min():.4f}; {x.max():.4f}]"
                + ("   <-- x* queda FUERA (extrapolación)" if fuera else ""))
        mostrar(f"Sxx = {f['sxx']:.4f}   Sxy = {f['sxy']:.2f}   "
                f"SCE = {f['sce']:.2f}   Syy = {f['syy']:.2f}")
        mostrar(f"Fórmulas:    b0 = {f['b0']:.4f}   b1 = {f['b1']:.4f}")
        mostrar(f"statsmodels: b0 = {h['b0']:.4f}   b1 = {h['b1']:.4f}")
        mostrar(f"¿Coinciden las dos vías? {coinciden(f, h)}\n")

        filas.append({
            "Xi": nombre,
            "b0": f["b0"], "b1": f["b1"],
            "sigma2": f["sigma2"], "R2": f["R2"], "r": f["r"],
            "IC(b1)": f"[{f['ic_b1'][0]:.2f}; {f['ic_b1'][1]:.2f}]",
            "IC(b0)": f"[{f['ic_b0'][0]:.2f}; {f['ic_b0'][1]:.2f}]",
            "ICM(Y|x*)": f"[{f['icm'][0]:.2f}; {f['icm'][1]:.2f}]",
            "IP(Y|x*)": f"[{f['ip'][0]:.2f}; {f['ip'][1]:.2f}]",
            "x* en rango": "no" if fuera else "sí",
        })

    cuadro = pd.DataFrame(filas).set_index("Xi")
    mostrar("=== Cuadro de B.2 ===")
    mostrar(cuadro.round({"b0": 4, "b1": 4, "sigma2": 2, "R2": 4, "r": 4})
            .to_string())

    # Selección de la variable: mayor R2 (equivale a menor sigma2 y mayor |r|)
    mejor = cuadro["R2"].idxmax()
    mostrar(f"\nVariable elegida: {mejor}")
    mostrar(f"  R2 = {cuadro.loc[mejor, 'R2']:.4f}   "
            f"sigma2 = {cuadro.loc[mejor, 'sigma2']:.2f}   "
            f"|r| = {abs(cuadro.loc[mejor, 'r']):.4f}")
    ranking = cuadro["R2"].sort_values(ascending=False)
    mostrar("  Ranking por R2: "
            + ", ".join(f"{v} ({r:.4f})" for v, r in ranking.items()))

    # Se guarda el informe ANTES de graficar, porque plt.show() frena el script
    salida = carpeta / "Resultados B.2.txt"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8-sig")
    print(f"\nArchivo generado: {salida}")

    a = ajustes[mejor]
    graficar(df, mejor, a["b0"], a["b1"], a["R2"], carpeta)


if __name__ == "__main__":
    main()