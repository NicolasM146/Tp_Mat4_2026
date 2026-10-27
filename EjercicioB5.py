"""B.5: regresión múltiple por ecuaciones normales y comparación con B.2.

Como en B.2, calcula por fórmulas y contrasta con statsmodels.
Usa la notación STC, SCR y SCE de B.1, y compara con el mejor modelo de B.2.
Genera Resultados B.5.txt y Texto B.5 para informe.txt junto al programa.
No necesita ejecutar los demás scripts ni utiliza resultados de B.3 o B.4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


VARIABLES = ("temp", "atemp", "hum", "windspeed")


def cargar_datos(carpeta):
    archivo = carpeta / "bike+sharing+dataset" / "day.csv"
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró el dataset local: {archivo}")
    df = pd.read_csv(archivo)
    predictores = df[list(VARIABLES)].to_numpy(dtype=float)
    y = df["cnt"].to_numpy(dtype=float)
    if predictores.ndim != 2 or len(y) <= len(VARIABLES) + 1:
        raise ValueError("Se necesitan más observaciones que parámetros.")
    if not (np.isfinite(predictores).all() and np.isfinite(y).all()):
        raise ValueError("Los datos deben contener valores numéricos finitos.")
    return predictores, y


def ajuste_por_formulas(X, y):
    """Resuelve (X.T X)b = X.T y, equivalente a b = (X.T X)^-1 X.T y."""
    n, p = X.shape
    k = p - 1  # Número de predictores, sin contar el intercepto.
    if np.linalg.matrix_rank(X) != p:
        raise ValueError("El modelo no tiene una solución única: X no tiene rango completo.")

    xtx = X.T @ X
    xty = X.T @ y
    b = np.linalg.solve(xtx, xty)
    residuos = y - X @ b
    sce = float(residuos @ residuos)
    scr = float(np.sum((X @ b - y.mean()) ** 2))
    stc = float(np.sum((y - y.mean()) ** 2))
    if stc <= 0:
        raise ValueError("La variable respuesta debe tener variabilidad.")
    r2 = 1 - sce / stc

    # Se informan ambas fórmulas porque el PDF usa (n-k) en el numerador.
    # En ambas, k cuenta solo los predictores: 4 en múltiple, 1 en simple.
    r2_aj_consigna = 1 - (1 - r2) * (n - k) / (n - k - 1)
    r2_aj_habitual = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    return {
        "b": b,
        "xtx": xtx,
        "xty": xty,
        "sce": sce,
        "scr": scr,
        "stc": stc,
        "r2": r2,
        "r2_aj_consigna": r2_aj_consigna,
        "r2_aj_habitual": r2_aj_habitual,
    }


def ajuste_por_statsmodels(X, y):
    """Verificación independiente con la misma herramienta utilizada en B.2."""
    modelo = sm.OLS(y, X).fit()
    n, p = X.shape
    k = p - 1
    return {
        "b": np.asarray(modelo.params),
        "sce": float(modelo.ssr),
        "scr": float(modelo.ess),
        "stc": float(modelo.centered_tss),
        "r2": float(modelo.rsquared),
        # rsquared_adj implementa la fórmula estándar, NO la literal del PDF.
        "r2_aj_habitual": float(modelo.rsquared_adj),
        "r2_aj_consigna": 1 - (1 - modelo.rsquared) * (n - k) / (n - k - 1),
    }


def coinciden(formulas, herramienta):
    coeficientes_ok = np.allclose(formulas["b"], herramienta["b"], rtol=0, atol=1e-6)
    claves = ("sce", "scr", "stc", "r2", "r2_aj_habitual", "r2_aj_consigna")
    medidas_ok = all(
        np.isclose(formulas[c], herramienta[c], rtol=1e-10, atol=1e-10)
        for c in claves
    )
    return bool(coeficientes_ok and medidas_ok)


def numero(valor, decimales=6):
    return f"{valor:.{decimales}f}".replace(".", ",")


def main():
    carpeta = Path(__file__).resolve().parent
    predictores, y = cargar_datos(carpeta)
    n = len(y)
    X = np.column_stack((np.ones(n), predictores))
    multiple = ajuste_por_formulas(X, y)
    herramienta = ajuste_por_statsmodels(X, y)
    if not coinciden(multiple, herramienta):
        raise ArithmeticError("Las fórmulas del modelo múltiple no coinciden con statsmodels.")
    simples = {}
    for j, nombre in enumerate(VARIABLES):
        X_simple = np.column_stack((np.ones(n), predictores[:, j]))
        simples[nombre] = ajuste_por_formulas(X_simple, y)
        if not coinciden(simples[nombre], ajuste_por_statsmodels(X_simple, y)):
            raise ArithmeticError(f"Las fórmulas del modelo de {nombre} no coinciden con statsmodels.")
    mejor = max(simples, key=lambda nombre: simples[nombre]["r2"])
    simple = simples[mejor]
    b = multiple["b"]

    diferencia = np.abs(b - herramienta["b"])
    etiquetas = ("b0 (intercepto)", "b1 (temp)", "b2 (atemp)", "b3 (hum)", "b4 (windspeed)")
    coeficientes = "\n".join(
        f"  {etiqueta:<17}{numero(valor, 6):>17}{numero(valor_sm, 6):>17}{dif:>15.2e}"
        for etiqueta, valor, valor_sm, dif in zip(etiquetas, b, herramienta["b"], diferencia)
    )
    comparacion = "\n".join(
        f"  {nombre:<22}{numero(a['r2']):>12}"
        f"{numero(a['r2_aj_consigna']):>20}{numero(a['r2_aj_habitual']):>20}"
        for nombre, a in ((f"Simple ({mejor})", simple), ("Múltiple", multiple))
    )
    matriz = "\n".join(
        "  [" + "  ".join(f"{numero(v, 6):>13}" for v in fila) + "]"
        for fila in multiple["xtx"]
    )
    vector = "\n".join(f"  [{numero(v, 6):>20}]" for v in multiple["xty"])
    mejora = multiple["r2_aj_consigna"] > simple["r2_aj_consigna"]
    conclusion = "supera" if mejora else "no supera"

    informe = f"""RESULTADOS B.5 - REGRESIÓN MÚLTIPLE
==================================
Fuente: bike+sharing+dataset/day.csv | n = {n} días.
Respuesta: cnt. Predictores: temp, atemp, hum y windspeed normalizados.

COEFICIENTES: FÓRMULAS Y HERRAMIENTA, COMO EN B.2
  Parámetro                 Fórmulas      statsmodels     Diferencia
{coeficientes}
  ¿Coinciden las dos vías? Sí (diferencia absoluta menor que 1e-6).

MODELO FINAL
  cnt_estimado = {numero(b[0], 4)} + {numero(b[1], 4)}*temp
                + {numero(b[2], 4)}*atemp {numero(b[3], 4)}*hum
                {numero(b[4], 4)}*windspeed

COMPARACIÓN CON EL MEJOR MODELO SIMPLE DEL B.2
  Modelo                          R2    R2 aj. consigna     R2 aj. habitual
{comparacion}
  Mejor simple: {mejor}, elegido por el mayor R2 entre las cuatro variables.
  Recta simple: cnt_estimado = {numero(simple['b'][0], 4)} + {numero(simple['b'][1], 4)}*{mejor}

  El modelo múltiple explica el {numero(100 * multiple['r2'], 2)}% de la variabilidad observada,
  frente al {numero(100 * simple['r2'], 2)}% del modelo simple. También {conclusion} su R2 ajustado.
  La comparación describe el ajuste a estos datos; no evalúa predicciones futuras.

FÓRMULAS DE R2 AJUSTADO
  Consigna: R2_aj = 1 - (1-R2)*(n-k)/(n-k-1).
  Habitual: R2_aj = 1 - (1-R2)*(n-1)/(n-k-1).
  k = número de predictores: 4 en el múltiple y 1 en el simple.
  El PDF usa n-k donde la fórmula habitual usa n-1; se informan ambas.
  Para seguir literalmente la consigna, usar la columna "R2 aj. consigna".

RESOLUCIÓN MATRICIAL
  X tiene columnas [1, temp, atemp, hum, windspeed] y dimensión {n} x 5.
  Se resuelve (X^T*X)*b = X^T*y con numpy.linalg.solve.
  Equivale a b = (X^T*X)^(-1)*X^T*y, sin calcular la inversa explícitamente.

  X^T*X =
{matriz}
  X^T*y =
{vector}

VERIFICACIÓN
  Mayor diferencia de coeficientes frente a statsmodels: {diferencia.max():.2e}.
  SCR múltiple = {numero(multiple['scr'], 6)}
  SCE múltiple = {numero(multiple['sce'], 6)}
  STC = {numero(multiple['stc'], 6)}
  SCR + SCE = {numero(multiple['scr'] + multiple['sce'], 6)}
  R2 = 1 - SCE/STC = {numero(multiple['r2'])}
  R2 de statsmodels = {numero(herramienta['r2'])}
  R2 ajustado estándar: fórmulas = {numero(multiple['r2_aj_habitual'])}; statsmodels = {numero(herramienta['r2_aj_habitual'])}.
  El R2 ajustado literal de la consigna se calcula aparte: {numero(multiple['r2_aj_consigna'])}.
  Cálculos sin redondeo intermedio; valores mostrados redondeados.
"""
    # Texto independiente para pegar en el informe, siguiendo A.2, A.3 y B.2.
    tabla_informe = "\n".join(
        f"{etiqueta}: fórmulas = {numero(valor, 4)}; statsmodels = {numero(valor_sm, 4)}."
        for etiqueta, valor, valor_sm in zip(etiquetas, b, herramienta["b"])
    )
    texto_informe = f"""B.5. Regresión múltiple

Se ajustó un modelo de regresión múltiple utilizando los {n} registros de day.csv. La variable respuesta fue cnt y las variables explicativas fueron temp, atemp, hum y windspeed, manteniendo la escala normalizada del dataset.

Estimación por mínimos cuadrados

Se construyó la matriz X de dimensión {n} x 5, con una columna de unos y las cuatro variables explicativas. Se resolvieron las ecuaciones normales:

(XᵀX)β̂ = Xᵀy, equivalentes a β̂ = (XᵀX)⁻¹Xᵀy.

La resolución se realizó mediante numpy.linalg.solve. Como en B.2, los resultados se contrastaron con statsmodels:

{tabla_informe}

Los coeficientes coinciden dentro de la precisión numérica, con una diferencia máxima de {diferencia.max():.2e}. También coinciden, al redondear a cuatro decimales, con la solución de mínimos cuadrados usada como referencia en A.3.

Modelo final

cnt_estimado = {numero(b[0], 4)} + {numero(b[1], 4)}·temp + {numero(b[2], 4)}·atemp {numero(b[3], 4)}·hum {numero(b[4], 4)}·windspeed.

Cálculo del coeficiente de determinación

Usando la notación de B.1:
SCE = sum(yi - ŷi)² = {numero(multiple['sce'], 4)}.
STC = sum(yi - ȳ)² = {numero(multiple['stc'], 4)}.
R² = 1 - SCE/STC = {numero(multiple['r2'])}.

Comparación con el modelo simple de B.2

En B.2 se seleccionó {mejor}, con el modelo cnt_estimado = {numero(simple['b'][0], 4)} + {numero(simple['b'][1], 4)}·{mejor}. Para comparar se aplicó literalmente la fórmula de R² ajustado del enunciado:

R² ajustado = 1 - (1 - R²)·(n - k)/(n - k - 1).

Aquí n = {n} y k es el número de variables explicativas, sin contar el intercepto. Para el modelo simple k = 1 y para el múltiple k = 4.

Simple ({mejor}): R² = {numero(simple['r2'])}; R² ajustado según consigna = {numero(simple['r2_aj_consigna'])}.
Múltiple: R² = {numero(multiple['r2'])}; R² ajustado según consigna = {numero(multiple['r2_aj_consigna'])}.

Conclusión

El modelo múltiple explica el {numero(100 * multiple['r2'], 2)}% de la variabilidad observada, frente al {numero(100 * simple['r2'], 2)}% del modelo simple, una diferencia de {numero(100 * (multiple['r2'] - simple['r2']), 2)} puntos porcentuales. También {conclusion} su R² ajustado. Por lo tanto, incluir conjuntamente las cuatro variables mejora el ajuste a estos datos. Esto no implica causalidad ni garantiza una mejora predictiva en días nuevos.

Nota sobre R² ajustado: la fórmula estándar utiliza n-1 en el numerador, en lugar de n-k. Con ella se obtiene {numero(multiple['r2_aj_habitual'])} para el múltiple y {numero(simple['r2_aj_habitual'])} para el simple. Estos valores coinciden con rsquared_adj de statsmodels y la conclusión se mantiene. Los resultados anteriores se identifican como "según consigna" para distinguir ambas fórmulas.
"""
    salida = carpeta / "Resultados B.5.txt"
    salida.write_text(informe, encoding="utf-8-sig")
    salida_informe = carpeta / "Texto B.5 para informe.txt"
    salida_informe.write_text(texto_informe, encoding="utf-8-sig")
    print(informe)
    print(f"Archivo generado: {salida}")
    print(f"Texto para el PDF: {salida_informe}")


if __name__ == "__main__":
    main()
