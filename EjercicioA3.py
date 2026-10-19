"""Resuelve A.3 y guarda un TXT de resultados y el gráfico solicitado."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIABLES = ("temp", "atemp", "hum", "windspeed")
TASAS = {"Pequeña": 1e-8, "Adecuada": 5e-4, "Grande": 1e-3}
MAX_ITERACIONES = 150_000
TOLERANCIA_GRADIENTE = 1e-5
FACTOR_DIVERGENCIA = 100


def cargar_datos(carpeta):
    archivo = carpeta / "bike+sharing+dataset" / "day.csv"
    with archivo.open(encoding="utf-8-sig", newline="") as entrada:
        filas = list(csv.DictReader(entrada))
    X = np.array([[1.0] + [float(fila[v]) for v in VARIABLES] for fila in filas])
    y = np.array([float(fila["cnt"]) for fila in filas])
    if X.ndim != 2 or len(y) <= X.shape[1]:
        raise ValueError("Se necesitan más observaciones que parámetros.")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Los datos deben ser numéricos y finitos.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Las columnas del modelo deben ser independientes.")
    return X, y


def costo(X, y, b):
    residuos = X @ b - y
    return float(residuos @ residuos)


def descenso_gradiente(X, y, H, c, alpha):
    """f = ||Xb-y||²; gradiente = H b - c, H=2 XᵀX, c=2 Xᵀy."""
    b = np.zeros(X.shape[1])
    historial = [costo(X, y, b)]
    limite = FACTOR_DIVERGENCIA * historial[0]
    estado = "Máximo de iteraciones"

    for _ in range(MAX_ITERACIONES):
        # Es la misma derivada 2*X.T@(X@b-y), precomputada para ahorrar trabajo.
        gradiente = H @ b - c
        if np.linalg.norm(gradiente) <= TOLERANCIA_GRADIENTE:
            estado = "Convergió"
            break
        b = b - alpha * gradiente
        historial.append(costo(X, y, b))
        if not np.isfinite(historial[-1]) or historial[-1] > limite:
            estado = "Divergió"
            break

    # Se evalúa también el último punto, incluso si se alcanzó el máximo.
    norma_gradiente = float(np.linalg.norm(2 * X.T @ (X @ b - y)))
    if estado != "Divergió" and norma_gradiente <= TOLERANCIA_GRADIENTE:
        estado = "Convergió"
    return {
        "b": b,
        "historial": np.array(historial),
        "iteraciones": len(historial) - 1,
        "estado": estado,
        "norma_gradiente": norma_gradiente,
        "alpha": alpha,
    }


def numero(valor, decimales=6):
    return f"{valor:.{decimales}f}".replace(".", ",")


def graficar(resultados, f_minimo, archivo):
    colores = {"Pequeña": "#d08819", "Adecuada": "#177c78", "Grande": "#cb4545"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
        for nombre, resultado in resultados.items():
            valores = resultado["historial"]
            # Conserva el comienzo y muestrea el resto para aligerar el gráfico.
            indices = np.unique(np.concatenate((
                np.arange(min(len(valores), 101)),
                np.linspace(0, len(valores) - 1, min(len(valores), 2000), dtype=int),
            )))
            ax.plot(indices, valores[indices], color=colores[nombre], linewidth=2,
                    label=f"{nombre}: α = {resultado['alpha']:g}")
            ax.scatter([len(valores) - 1], [valores[-1]], color=colores[nombre], s=30, zorder=5)
        ax.axhline(f_minimo, color="#44546a", linestyle="--", linewidth=1.2,
                   label="Mínimo por mínimos cuadrados")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("log")
        ax.set_xlim(0, 1.15 * max(r["iteraciones"] for r in resultados.values()))
        ax.set_xlabel("Iteración k (escala lineal de 0 a 10; logarítmica después)")
        ax.set_ylabel("f(b) = suma de errores cuadrados (escala logarítmica)")
        ax.set_title("A.3 · Descenso del gradiente con tres tasas", loc="left", fontweight="bold")
        ax.grid(True, which="major", alpha=0.2)
        ax.legend(loc="upper right", framealpha=0.95)
        fig.savefig(archivo, dpi=160)
        plt.close(fig)


def main():
    carpeta = Path(__file__).resolve().parent
    X, y = cargar_datos(carpeta)
    H = 2 * X.T @ X
    c = 2 * X.T @ y

    # Solución de referencia del modelo de B.5, independiente de A.1 y A.2.
    # Resuelve las ecuaciones normales sin calcular la inversa explícitamente.
    b_exacto = np.linalg.solve(H, c)
    b_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    if not np.allclose(b_exacto, b_lstsq, rtol=0, atol=1e-6):
        raise ArithmeticError("Las dos soluciones de referencia no coinciden.")
    f_minimo = costo(X, y, b_exacto)
    limite_alpha = 2 / np.linalg.eigvalsh(H)[-1]

    resultados = {
        nombre: descenso_gradiente(X, y, H, c, alpha)
        for nombre, alpha in TASAS.items()
    }
    adecuado = resultados["Adecuada"]
    if adecuado["estado"] != "Convergió":
        raise ArithmeticError("La tasa adecuada no alcanzó la tolerancia indicada.")
    if resultados["Grande"]["estado"] != "Divergió":
        raise ArithmeticError("La tasa grande no produjo la divergencia esperada.")
    diferencia = np.abs(adecuado["b"] - b_exacto)
    if diferencia.max() > 1e-4:
        raise ArithmeticError("El descenso no se acercó suficientemente a mínimos cuadrados.")

    tabla_coeficientes = "\n".join(
        f"  {nombre:<12}{numero(bg):>17}{numero(be):>21}{d:>14.2e}"
        for nombre, bg, be, d in zip(
            ("b0", "b1 temp", "b2 atemp", "b3 hum", "b4 windspeed"),
            adecuado["b"], b_exacto, diferencia,
        )
    )
    tabla_tasas = "\n".join(
        f"  {nombre:<10}{r['alpha']:<11g}{r['iteraciones']:>8}"
        f"  {numero(r['historial'][-1], 2):>19}  {r['estado']}"
        for nombre, r in resultados.items()
    )
    b = adecuado["b"]
    grafico = carpeta / "Grafico A.3.png"
    informe = f"""RESULTADOS A.3
==============
Fuente: bike+sharing+dataset/day.csv | n = {len(y)} días.
Variables: temp, atemp, hum, windspeed (escala original normalizada).

MODELO OBTENIDO POR DESCENSO DEL GRADIENTE
  cnt_estimado = {numero(b[0], 4)} + {numero(b[1], 4)}*temp
                + {numero(b[2], 4)}*atemp {numero(b[3], 4)}*hum
                {numero(b[4], 4)}*windspeed

COMPARACIÓN CON LA SOLUCIÓN DE MÍNIMOS CUADRADOS (B.5)
  Parámetro            Gradiente    Mínimos cuadrados    Diferencia
{tabla_coeficientes}
  Mayor diferencia absoluta: {diferencia.max():.3e}.
  f mínimo cuadrados = {numero(f_minimo, 6)}
  f gradiente        = {numero(adecuado['historial'][-1], 6)}

TRES TASAS DE APRENDIZAJE (mismo inicio: b = [0, 0, 0, 0, 0])
  Tasa      Alpha       Iteraciones             f final  Resultado
{tabla_tasas}
  f inicial = {numero(float(y @ y), 2)}

  Pequeña: el error baja lentamente; no converge en {MAX_ITERACIONES} pasos.
  Adecuada: el error desciende hasta el mínimo y alcanza la tolerancia.
  Grande: el error crece; los pasos excesivos provocan divergencia.
  Para esta función cuadrática, la estabilidad exige 0 < alpha < {limite_alpha:.10f}.

FUNCIÓN, GRADIENTE Y ACTUALIZACIÓN
  X: columnas [1, temp, atemp, hum, windspeed]; y = cnt.
  f(b) = sum (yi - Xi*b)^2 = ||X*b - y||^2
  grad f(b) = 2*X^T*(X*b - y)
  b(k+1) = b(k) - alpha*grad f(b(k))
  Referencia: (X^T*X)*b = X^T*y, resuelta con numpy.linalg.solve.

POR QUÉ SE RESTA EL GRADIENTE (2 líneas)
  El gradiente apunta hacia el mayor crecimiento local de f.
  En sentido opuesto, la derivada direccional es -||grad f|| < 0 fuera del mínimo.

CRITERIO DE PARADA
  Convergencia: norma euclídea del gradiente <= {TOLERANCIA_GRADIENTE:g}.
  Norma final con tasa adecuada: {adecuado['norma_gradiente']:.3e}.
  Límite: {MAX_ITERACIONES} iteraciones; alcanzarlo no significa convergencia.
  Divergencia: detener si f > {FACTOR_DIVERGENCIA}*f_inicial o deja de ser finito.

GRÁFICO
  {grafico.name}: las tres curvas f contra iteración en los mismos ejes.
  Las escalas indicadas permiten ver la divergencia y la convergencia juntas.
"""
    graficar(resultados, f_minimo, grafico)
    salida = carpeta / "Resultados A.3.txt"
    salida.write_text(informe, encoding="utf-8-sig")
    print(f"Archivo generado: {salida}")
    print(f"Gráfico generado: {grafico}")
    print(f"Verificado: diferencia máxima frente a mínimos cuadrados = {diferencia.max():.3e}")


if __name__ == "__main__":
    main()
