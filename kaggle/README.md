# House Prices - Advanced Regression Techniques

Proyecto de Machine Learning para la competicion de Kaggle
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

El objetivo es predecir el precio de venta de viviendas (`SalePrice`) usando datos tabulares del dataset Ames Housing.

## Resultado

- **Score publico en Kaggle:** `0.12975 RMSLE`
- **Mejor score local de validacion cruzada:** `0.12944 RMSLE`

La metrica oficial es RMSE sobre el logaritmo del precio, normalmente interpretada como RMSLE.

## Que incluye el proyecto

- Limpieza de valores faltantes.
- Ingenieria de variables para superficie total, banos, antiguedad, garaje, sotano y chimenea.
- Preprocesamiento de variables numericas y categoricas.
- Modelos Ridge, Lasso, ElasticNet y Gradient Boosting.
- Validacion cruzada con `KFold`.
- Ensamble ponderado de los mejores modelos.
- Generacion de `submission.csv` para Kaggle.

## Estructura

```text
.
+-- data/
|   +-- raw/
+-- notebooks/
|   +-- 01_house_prices.ipynb
+-- output/
+-- scripts/
|   +-- download_data.ps1
+-- src/
|   +-- house_prices_pipeline.py
+-- INFORME.md
+-- requirements.txt
+-- requirements-extra.txt
+-- README.md
```

Los datos originales de Kaggle y los archivos generados no se suben al repositorio.

## Instalacion

```powershell
pip install -r requirements.txt
```

Para usar notebook, graficos o Kaggle CLI:

```powershell
pip install -r requirements-extra.txt
```

## Datos

Descarga los datos desde Kaggle y coloca estos archivos en `data/raw/`:

```text
train.csv
test.csv
sample_submission.csv
```

## Ejecucion

```powershell
python .\src\house_prices_pipeline.py
```

El script genera:

```text
output\cv_results.csv
output\submission.csv
```

El archivo `output\submission.csv` es el que se sube a Kaggle.

## Tecnologias

- Python
- pandas
- NumPy
- scikit-learn
- Kaggle
