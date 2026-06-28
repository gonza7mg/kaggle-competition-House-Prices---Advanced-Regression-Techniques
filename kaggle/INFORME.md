# Informe del Proyecto Kaggle: House Prices

## 1. Objetivo

El objetivo de este proyecto es predecir el precio de venta de viviendas (`SalePrice`) usando el dataset de la competicion de Kaggle **House Prices - Advanced Regression Techniques**.

La competicion proporciona informacion de casas de Ames, Iowa, con variables sobre superficie, calidad de materiales, antiguedad, barrio, garaje, sotano, banos y otras caracteristicas relacionadas con el valor de una vivienda.

## 2. Metrica de Evaluacion

Kaggle evalua las predicciones usando **RMSE sobre el logaritmo del precio**:

```text
RMSE(log(SalePrice real), log(SalePrice predicho))
```

Esta metrica se suele interpretar como RMSLE. Penaliza los errores relativos, por lo que resulta adecuada cuando existen viviendas de rangos de precio muy distintos.

Por este motivo, el pipeline entrena los modelos aplicando `log1p` al precio y transforma las predicciones de vuelta con `expm1`.

## 3. Resultado

El proyecto obtuvo los siguientes resultados:

| Resultado | Score |
| --- | ---: |
| Score publico de Kaggle | 0.12975 RMSLE |
| Mejor score local de validacion cruzada | 0.12944 RMSLE |

El score local y el score publico de Kaggle son muy parecidos, lo que indica que la validacion local representa bien el comportamiento del modelo en datos no vistos.

## 4. Estructura del Proyecto

```text
.
├── data/
│   └── raw/
├── notebooks/
│   └── 01_house_prices.ipynb
├── output/
├── reports/
│   └── model_scores.csv
├── scripts/
│   └── download_data.ps1
├── src/
│   └── house_prices_pipeline.py
├── requirements.txt
├── requirements-extra.txt
├── README.md
├── PORTFOLIO.md
└── INFORME.md
```

Los datos originales y los archivos generados no se suben a GitHub para evitar publicar archivos de competicion o resultados locales pesados.

## 5. Preparacion de Datos

El pipeline realiza una limpieza inicial de valores faltantes siguiendo la logica del dataset:

- Algunas variables categoricas tienen valores ausentes que significan ausencia real de una caracteristica, por ejemplo sin garaje, sin sotano, sin piscina o sin chimenea. Esos valores se rellenan con `"None"`.
- Algunas variables numericas ausentes significan cantidad cero, por ejemplo area de garaje, banos en sotano o superficie de sotano. Esas variables se rellenan con `0`.
- Variables categoricas con pocos valores perdidos se rellenan con la moda.
- `LotFrontage` se rellena usando la mediana por `Neighborhood`, porque la fachada del lote suele depender del barrio.

Esta limpieza evita que el modelo interprete como informacion perdida lo que realmente significa ausencia de una instalacion.

## 6. Ingenieria de Variables

El script crea variables derivadas para capturar mejor la relacion entre caracteristicas y precio:

- `TotalSF`: suma de superficie de sotano, primera planta y segunda planta.
- `TotalBathrooms`: total ponderado de banos completos y medios banos.
- `TotalPorchSF`: superficie total de porches.
- `HouseAge`: antiguedad de la vivienda al momento de venta.
- `RemodAge`: tiempo desde la ultima remodelacion.
- `GarageAge`: antiguedad del garaje.
- `HasPool`: indica si tiene piscina.
- `HasGarage`: indica si tiene garaje.
- `HasBasement`: indica si tiene sotano.
- `HasFireplace`: indica si tiene chimenea.
- `Has2ndFloor`: indica si tiene segunda planta.
- `IsRemodeled`: indica si la vivienda fue remodelada.
- `OverallQualSquared`: version cuadratica de la calidad general.

Tambien se tratan algunas variables numericas como categoricas:

```text
MSSubClass, MoSold, YrSold
```

Aunque contienen numeros, representan clases o periodos, no cantidades continuas.

## 7. Preprocesamiento

El pipeline separa las variables en dos grupos:

- Variables numericas.
- Variables categoricas.

Para las numericas:

- imputacion con la mediana;
- escalado con `StandardScaler` en los modelos lineales.

Para las categoricas:

- imputacion con la moda;
- codificacion con `OneHotEncoder`.

Todo el preprocesamiento se integra en un `ColumnTransformer`, lo que permite aplicar transformaciones diferentes a cada tipo de columna de forma reproducible.

## 8. Modelos Utilizados

El proyecto entrena cuatro modelos:

```text
Ridge
Lasso
ElasticNet
GradientBoostingRegressor
```

Los modelos lineales funcionan bien en este tipo de problema porque el dataset contiene muchas variables categoricas transformadas mediante one-hot encoding.

`GradientBoostingRegressor` aporta capacidad para capturar relaciones no lineales entre variables como calidad, superficie, barrio y antiguedad.

## 9. Validacion

La validacion se hace con `KFold`:

```text
5 folds
shuffle=True
random_state=42
```

En cada fold:

1. Se entrena el modelo con una parte de los datos.
2. Se predice sobre la parte de validacion.
3. Se calcula RMSLE.

Resumen de validacion:

| Modelo | RMSLE medio | Desviacion |
| --- | ---: | ---: |
| Gradient Boosting | 0.12944 | 0.02223 |
| Ridge | 0.14392 | 0.04377 |
| ElasticNet | 0.14666 | 0.04428 |
| Lasso | 0.14754 | 0.04320 |

## 10. Ensamble

Despues de validar todos los modelos, el script selecciona los tres mejores segun su media de RMSLE.

El ensamble final combina:

```text
GradientBoostingRegressor
Ridge
ElasticNet
```

Los pesos se calculan dando mas importancia a los modelos con menor error de validacion.

## 11. Archivos de Salida

Cuando se ejecuta correctamente, el proyecto genera:

```text
output/cv_results.csv
output/submission.csv
```

El archivo `submission.csv` tiene el formato requerido por Kaggle:

```text
Id,SalePrice
```

## 12. Como Ejecutar

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Colocar los datos en:

```text
data/raw/
```

Ejecutar:

```powershell
python .\src\house_prices_pipeline.py
```

## 13. Conclusion

El proyecto demuestra un flujo completo de ciencia de datos aplicado a un problema de regresion tabular: limpieza de datos, ingenieria de variables, validacion, comparacion de modelos y generacion de una entrega compatible con Kaggle.

El resultado oficial de 0.12975 RMSLE es consistente con la validacion local y muestra que el pipeline generaliza de forma razonable.
