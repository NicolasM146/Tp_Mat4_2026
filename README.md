# T.P. Grupal Matematica 4 - 2026
## Integrantes
### Nicolas agustin Muñoz 19365/6
### Facundo Nicolás Mendoza 19656/5
### Carlos Adrián Sánchez Nicolai 16789/5
### Leonardo Dos Santos - 20806/6

##
## Requisitos para ejecutar el código

- Python 3
- Las librerías de `requirements.txt`

## Instalación y ejecución

Desde la raíz del repositorio:

```powershell
py -m pip install -r requirements.txt
```

Cada script se ejecuta de la misma forma (`py NombreDelScript.py`) y guarda sus resultados en la carpeta donde está el script.

**Importante:** después de cada `git pull`, volver a ejecutar `py -m pip install -r requirements.txt`, porque `git pull` trae el archivo pero no instala las librerías nuevas.

## Datos

Los scripts leen el archivo `day.csv` de la carpeta `bike+sharing+dataset/`, ubicada en la raíz del repositorio. Si no la encuentran, lo descargan de:

https://raw.githubusercontent.com/danwild/bike-share-prediction/master/Bike-Sharing-Dataset/day.csv

Fuente original: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Reproducibilidad

Todos los números del informe salen de ejecutar estos scripts sobre `day.csv`. Los resultados de cada ejecución quedan guardados en los archivos `Resultados *.txt` y en las imágenes `.png`.

## Uso de inteligencia artificial

El uso de asistentes de IA está documentado en la bitácora del Anexo I del informe.