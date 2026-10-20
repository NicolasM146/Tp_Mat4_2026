from pathlib import Path
import numpy as np
import pandas as pd


def numero(valor, decimales=4):
    """Formatea un número flotante usando coma como separador decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def main():
    # Define la carpeta actual y la ruta relativa al archivo dentro del repositorio
    carpeta = Path(__file__).resolve().parent
    archivo = carpeta / "bike+sharing+dataset" / "day.csv"

    # 1. Cargar datos desde el archivo local de la carpeta
    try:
        df = pd.read_csv(archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {archivo}")
        return

    # Variables del modelo B.1
    x = df["temp"].values
    y = df["cnt"].values
    n = len(y)

    # 2. Promedios y sumatorias (cálculos manuales vectorizados)
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    sxx = np.sum((x - x_bar) ** 2)
    syy = np.sum((y - y_bar) ** 2)
    sxy = np.sum((x - x_bar) * (y - y_bar))

    # Coeficientes de la recta de regresión
    b1 = sxy / sxx
    b0 = y_bar - b1 * x_bar

    # Valores predichos por la recta
    y_pred = b0 + b1 * x

    # 3. Descomposición de la variabilidad
    stc = syy  # Suma Total de Cuadrados (Syy)
    scr = np.sum((y_pred - y_bar) ** 2)  # Suma de Cuadrados de la Regresión (Sxy^2 / Sxx)
    sce = np.sum((y - y_pred) ** 2)  # Suma de Cuadrados del Error (Syy - SCR)

    # Grados de libertad
    gl_reg = 1
    gl_err = n - 2
    gl_tot = n - 1

    # Cuadrados medios y métricas
    cmr = scr / gl_reg
    sigma2 = sce / gl_err
    sigma = np.sqrt(sigma2)
    r2 = 1 - (sce / stc)

    # 4. Redacción e informe formateado
    informe = f"""RESULTADOS B.1 · VARIANZA
========================
Fuente: bike+sharing+dataset/day.csv | n = {n} días.
Modelo simple: cnt en función de temp (cnt = b0 + b1 * temp)

RECTA DE REGRESIÓN AJUSTADA
  b0 (intercepto) = {numero(b0, 4)}
  b1 (pendiente)  = {numero(b1, 4)}
  Ecuación: cnt_estimado = {numero(b0, 2)} + {numero(b1, 2)} * temp

DESCOMPOSICIÓN DE LA VARIABILIDAD (TABLA ANOVA)
  Fuente de Variación     Suma Cuadrados (SC)      gl       Cuadrado Medio (CM)
  -----------------------------------------------------------------------------
  Regresión (Explicada)   {numero(scr, 2):>19}     {gl_reg:>3}   {numero(cmr, 2):>19}
  Error (No explicada)    {numero(sce, 2):>19}   {gl_err:>5}   {numero(sigma2, 2):>19}
  -----------------------------------------------------------------------------
  Total                   {numero(stc, 2):>19}   {gl_tot:>5}

VERIFICACIÓN NUMÉRICA
  SCR + SCE = {numero(scr, 2)} + {numero(sce, 2)}
            = {numero(scr + sce, 2)}
  STC       = {numero(stc, 2)}
  ¿Coinciden exactamente?: {"Sí" if np.isclose(scr + sce, stc) else "No"}

ESTIMACIÓN DE LA VARIANZA DEL MODELO (sigma^2)
  gl = n - 2 = {n} - 2 = {gl_err}
  sigma^2 = SCE / (n - 2) = {numero(sigma2, 4)} (bicicletas)^2
  Desvío estándar residual (sigma) = {numero(sigma, 2)} bicicletas

COEFICIENTE DE DETERMINACIÓN (R^2)
  R^2 = 1 - (SCE / STC) = {numero(r2, 4)} ({numero(r2 * 100, 2)}%)

INTERPRETACIÓN DE R^2
  - Proporción explicada: El {numero(r2 * 100, 2)}% de la variabilidad total observada en la
    cantidad diaria de alquileres de bicicletas (cnt) es explicada linealmente por la
    temperatura (temp). El {numero((1 - r2) * 100, 2)}% restante corresponde a variabilidad
    no explicada (otros factores como lluvia, humedad, si es laboral/feriado o azar).

  - Qué NO significa:
    1. No implica Causalidad: Una asociación estadística del {numero(r2 * 100, 2)}% no prueba
       que la temperatura sea la causa directa del alquiler. Si fuera causa-efecto pura, a mayor
       temperatura siempre habría más alquileres, lo cual no ocurre con calor extremo.
    2. No demuestra linealidad óptima: Mide la fuerza de una relación lineal, pero no descarta
       que una estructura no lineal (ej. cuadrática) ajuste mejor al fenómeno.
    3. No indica que el modelo sea inadecuado: Explicar casi un 40% de la demanda urbana con una sola
       variable representa un nivel de explicación significativo.
"""

    salida = carpeta / "Resultados B.1.txt"
    salida.write_text(informe, encoding="utf-8-sig")

    print(informe)
    print(f"\nArchivo generado: {salida}")


if __name__ == "__main__":
    main()
