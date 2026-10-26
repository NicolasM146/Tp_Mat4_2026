import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm


# 1. Cargar el dataset directamente desde la URL
url = "https://raw.githubusercontent.com/danwild/bike-share-prediction/master/Bike-Sharing-Dataset/day.csv"

try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    exit()

print(f"Hay {len(df)} elementos en el DataFrame.\n")
# X variable predictora

x_predictora = df['windspeed']

#Sacamos el promedio
promedio_predictora = x_predictora.mean()
print(f"El promedio de la variable predictora es: {promedio_predictora:.4f}")

#Sacamos el promedio calidad
promedio_respuesta = df['cnt'].mean()
print(f"El promedio de calidad es: {promedio_respuesta:.4f}")

#Sacamos Σ (xi)
suma_predictora = x_predictora.sum()
print(f"Σ (xi) es: {suma_predictora:.2f}")

#Sacamos Σ (yi)
suma_respuesta= df['cnt'].sum()
print(f"Σ (yi) es: {suma_respuesta:.2f}")


#Calculamos Sumatoria Σ (xiyI)
df['XiYi'] = x_predictora * df['cnt']
sumatoria_XiYi = df['XiYi'].sum()
print(f"Σ (xiyI) es : {sumatoria_XiYi:.2f}")

#Caculamos (Σxi) .(Σyi)
sumatoria_XiconYi = suma_predictora * suma_respuesta
print(f"(Σxi) .(Σyi) es: {sumatoria_XiconYi:.2f}")

#calculamos Σ (xi²)
df['XiCuadrado'] = x_predictora ** 2
sumatoria_XiCuadrado = df['XiCuadrado'].sum()
print(f" Σ (xi²): {sumatoria_XiCuadrado:.2f}")


#calculamos Σ (xi)²
sumatoria_xitodocuadrado = suma_predictora ** 2
print(f"Σ (xi)² es: {sumatoria_xitodocuadrado:.2f}")


#calculamos Σ (yi²)
df['YiCuadrado'] = df['cnt'] ** 2
sumatoria_YiCuadrado = df['YiCuadrado'].sum()
print(f"Σ (yi²) es: {sumatoria_YiCuadrado:.2f}")

#calculamos Σ (yi)²
sumatoria_yitodocuadrado = suma_respuesta ** 2
print(f"Σ (yi)² es: {sumatoria_yitodocuadrado:.2f}")


#calculamos Sxx
division  = sumatoria_xitodocuadrado / len(df)
sxx = (sumatoria_XiCuadrado - division)
print(f"sxx es: {sxx:.2f}")

#calculamos sxy
sxy = (sumatoria_XiYi - (sumatoria_XiconYi / len(df)))
print(f"sxy es: {sxy:.2f}")


#Primero para sacar SCE vamos a usar la formula de ( syy - ((sxy)2 / sxx))
#Para sacar Syy usadamos la SUmatoria de (Yi - YiPromedio)2
Syy = ((df['cnt'] - promedio_respuesta)**2).sum()
print(f"Syy es : {Syy:.3f}")

syy_aux = (sumatoria_YiCuadrado - ( sumatoria_yitodocuadrado / len(df) ) )
print(f"syy_aux es : {syy_aux:.4f}")

SCE_aux = (Syy - ((sxy**2) / sxx))
print(f"SCE auxiliar es : {SCE_aux:.4f}")


estimador = SCE_aux / (len(df) - 2 )
print(f"Estimador es σ̂ 2 : {estimador:.4f}")

coeficiente_determinacion = 1 - (SCE_aux / syy_aux)
print(f"Coeficiente de determinacion es : {coeficiente_determinacion:.4f}")

corr = x_predictora.corr(df['cnt'])
print("Correlación:", corr)

# ------------- Por si quieren despues chequearlo a mano -------------------


# ----------------------- Esta parte lo hace LinearRegression ------------

# Seleccionar las variables para la regresión lineal simple
# X debe ser un array 2D
X = df[['windspeed']].values
# Y es la variable dependiente
Y = df['cnt'].values

# Crear y entrenar el modelo de regresión
model = LinearRegression()
model.fit(X, Y)


Y_pred = model.predict(X)
Y_mean = df['cnt'].mean()
sce_formula1 = np.sum((df['cnt'] - Y_pred)**2)

print(f"SCE (Fórmula 1 - Con valores predichos): {sce_formula1:.4f}")

# Obtener los coeficientes
# La pendiente es el primer y único coeficiente
pendiente = model.coef_[0]
# La intercepción es el valor de corte con el eje Y
intercepcion = model.intercept_


# IC
print("----- IC --------")
X = sm.add_constant(x_predictora)  # agrega intercepto
y = df['cnt']

model = sm.OLS(y, X).fit()
print(model.summary())
print("----- IC --------")


x_predictora = df["windspeed"]
y_respuesta = df["cnt"]

# 2. Ajustar el modelo OLS
X = sm.add_constant(x_predictora)  # Agrega la columna 'const' para b0
model = sm.OLS(y_respuesta, X).fit()

# 3. Definir x* = 0.65 fija (con la columna 'const' en 1)
x0 = 0.65
nuevo = pd.DataFrame({"const": [1], "temp": [x0]})

# 4. Obtener las predicciones e intervalos al 95% de confianza (alpha=0.05)
pred = model.get_prediction(nuevo)
df_pred = pred.summary_frame(alpha=0.05)

# 5. Extraer los límites exactos de ICM e IP
icm_inferior = df_pred["mean_ci_lower"].values[0]
icm_superior = df_pred["mean_ci_upper"].values[0]

ip_inferior = df_pred["obs_ci_lower"].values[0]
ip_superior = df_pred["obs_ci_upper"].values[0]

print(f"--- RESULTADOS EN x* = {x0} ---")
print(
    f"ICM (Respuesta Media 95%) : [{icm_inferior:.2f} ; {icm_superior:.2f}]"
)
print(f"IP  (Predicción Indiv. 95%) : [{ip_inferior:.2f} ; {ip_superior:.2f}]")


print(f"--- Fórmula de Regresión ---")
print(f"El valor de la pendiente (β1) es: {pendiente:.4f}")
print(f"El valor de la intercepción (β0) es: {intercepcion:.4f}")
