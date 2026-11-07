🇬🇧 [Read in English](README.md)
# Predicción de fallas del sistema de aire (APS) en camiones Scania

Clasificación con clases desbalanceadas y **costos asimétricos**: ¿qué camiones conviene mandar a revisión para minimizar el costo total?

**Autor:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Resumen

| | |
|---|---|
| **Problema** | Detectar camiones con falla inminente del sistema de aire (APS) antes de que ocurra una avería |
| **Datos** | [APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/414/ida2016challenge) (UCI): 60.000 camiones en train (1,7% con falla) y 16.000 en test (2,3%), con 170 variables anonimizadas |
| **Costos (oficiales)** | Revisión innecesaria: 10 · Falla no detectada: 500 |
| **Modelo** | LightGBM, con el umbral de decisión elegido por costo |
| **Resultado en test** | Costo total medio de **8.580** (rango 7.930-9.360 en 5 semillas), contra 156.250 revisando todos los camiones y 187.500 sin revisar ninguno |

## Contexto

Revisar un camión que no tiene falla cuesta poco (10), pero no detectar una falla puede terminar en una avería y cuesta mucho más (500). Por eso la métrica a minimizar no es la exactitud, sino el costo total:

```
costo total = 10 × falsos positivos + 500 × falsos negativos
```

Es la métrica oficial del desafío IDA 2016 en el que se publicó este dataset.

## Datos

- **Origen:** datos de camiones Scania en uso diario. La clase positiva son fallas de un componente específico del sistema de aire (APS); la negativa, camiones con fallas de otros componentes.
- **Tamaño:** 60.000 camiones en train (1.000 con falla) y 16.000 en test (375 con falla).
- **Variables:** 170, anonimizadas por razones de propiedad. Los valores faltantes vienen escritos como `na`.
- **Licencia:** CC BY 4.0.

## Enfoque

1. **Modelo simple.** LightGBM con parámetros razonables y sin ajuste de hiperparámetros. Los valores faltantes se dejan como nulos, ya que LightGBM los maneja de forma nativa.
2. **Umbral por costo.** Se elige el umbral que minimiza `10 × FP + 500 × FN`. Con solo 1.000 fallas en train, apartar un 20% como validación dejaría unas 200, muy pocas para elegir un umbral con estabilidad. Por eso se usan **predicciones fuera de muestra con validación cruzada** de 5 pliegues, que aprovechan todas las fallas.
3. **Grilla de umbrales en escala logarítmica.** Una primera grilla lineal dejó el óptimo en su borde inferior; con la escala logarítmica el óptimo queda en el interior.
4. **Test intacto.** El test oficial se usa una sola vez por corrida, al final, con el umbral ya fijado.
5. **Estabilidad.** Todo el procedimiento (validación cruzada, elección del umbral y evaluación en test) se repite con 5 semillas, para reportar un rango y no un único número.

## Resultados

### Costo en test

| Estrategia | Costo total |
|---|---|
| No revisar ninguno | 187.500 |
| Revisar todos | 156.250 |
| **Modelo (media de 5 semillas)** | **8.580** (desv. 558; rango 7.930-9.360) |
| Referencia: mejor resultado del desafío IDA 2016 | 9.920 |

El costo medio es ~95% menor que el de revisar todos los camiones o el de no revisar ninguno.

### Estabilidad entre semillas

| Semilla | Umbral | Revisiones innecesarias | Fallas no detectadas | Costo en test |
|---|---|---|---|---|
| 42 | 0,00126 | 524 | 6 | 8.240 |
| 1 | 0,00190 | 449 | 8 | 8.490 |
| 2 | 0,00129 | 543 | 5 | 7.930 |
| 3 | 0,00194 | 436 | 10 | 9.360 |
| 4 | 0,00194 | 438 | 9 | 8.880 |

- **Recall:** entre 97,3% y 98,7% (de 375 fallas, se pierden entre 5 y 10).
- **Precisión:** entre 40,5% y 45,6%.
- **Qué cambia entre semillas:** el balance entre revisiones innecesarias y fallas no detectadas. El costo total queda parecido porque una falla no detectada (500) equivale a 50 revisiones innecesarias (10 cada una).

### Corrida principal (semilla 42)

| | Predicho: sin falla | Predicho: con falla |
|---|---|---|
| **Real: sin falla** | 15.101 | 524 |
| **Real: con falla** | 6 | 369 |

- **Validación cruzada (train):** AUC-PR 0,8975 (tasa base 1,7%) · AUC-ROC 0,989.
- **Test:** AUC-PR 0,934 (tasa base 2,3%) · AUC-ROC 0,997. No son comparables de forma directa con los de validación cruzada, porque la proporción de fallas es distinta.

![Costo según el umbral](reports/figures/cost_curve.png)

### Cómo leer la comparación con el desafío

En las cinco corridas el costo fue menor a 9.920 (máximo 9.360), por lo que el resultado es del mismo orden o algo mejor que el mejor del desafío. Pero el rango cubre solo la variación por el azar del entrenamiento (semilla del modelo y pliegues de la validación cruzada); no cubre la variación por el conjunto de test, que es fijo y tiene solo 375 fallas, ni las condiciones distintas del desafío de 2016. Por eso se presenta como una referencia y no como una superación.

## Observaciones

- **Los valores faltantes parecen informativos.** Los camiones con falla tienen en promedio 29,2 valores faltantes por fila, contra 13,9 en los que no fallan. Es una asociación; no se investigó la causa. Respalda la decisión de no imputar.
- **El umbral óptimo es muy bajo** (entre 0,0013 y 0,0019). Con estos costos, el punto de equilibrio de un modelo bien calibrado estaría cerca de `10 / (10 + 500) ≈ 0,02`. Que el óptimo sea mucho menor sugiere que las probabilidades de LightGBM no están calibradas (hipótesis, no verificada).
- **El test tiene más fallas que el train** (2,34% frente a 1,67%), por lo que los dos conjuntos no tienen la misma distribución. Podría favorecer un umbral aún más bajo en test (hipótesis, no verificada).

## Limitaciones

- **Variables anonimizadas:** no se puede interpretar qué mide cada una.
- **Sin fechas:** las filas no tienen marca temporal, por lo que no se pudo hacer validación temporal.
- **Un único tipo de modelo sin ajuste de hiperparámetros.**
- **Probabilidades no calibradas.**
- **El rango entre semillas** no incluye la variación por el conjunto de test (fijo, con 375 fallas).
- **Reproducibilidad:** los resultados pueden variar levemente con la versión de LightGBM y con el número de hilos.

## Próximos pasos

- Ajustar hiperparámetros.
- Calibrar las probabilidades y comparar el umbral resultante con el óptimo por costo.
- Comparar con otros algoritmos (XGBoost, CatBoost, regresión logística).
- Probar ponderación por clase y analizar si mejora el costo.

## Estructura del repositorio

```
project_failures_scania/
├── notebooks/
│   └── fallas_scania.ipynb
├── reports/
│   └── figures/
│       └── cost_curve.png
├── requirements.txt
├── .gitignore
└── README.md
```

La carpeta `data/` no se incluye en el repositorio.

## Cómo reproducirlo

1. Instalar dependencias: `pip install -r requirements.txt`
2. Descargar `aps_failure_training_set.csv` y `aps_failure_test_set.csv` desde [UCI](https://archive.ics.uci.edu/dataset/414/ida2016challenge) y copiarlos en una carpeta `data/` en la raíz del proyecto.
3. Ejecutar `notebooks/fallas_scania.ipynb` de arriba hacia abajo. Genera la figura en `reports/figures/`. La sección 6 repite el procedimiento con 5 semillas y tarda unos minutos.

## Fuente de los datos

Scania CV AB. *APS Failure at Scania Trucks* [Dataset]. 2016. UCI Machine Learning Repository. https://doi.org/10.24432/C51S51. Licencia CC BY 4.0.

## Tecnologías

Python · pandas · LightGBM · scikit-learn · matplotlib
