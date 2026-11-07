🇪🇸 [Leer en español](README.es.md)

# Air Pressure System (APS) Failure Prediction in Scania Trucks

Classification with imbalanced classes and **asymmetric costs**: which trucks should be sent for inspection to minimize total cost?

**Author:** Carlos Adolfo Peralta · [LinkedIn](https://www.linkedin.com/in/carlosperalta1993/) · [GitHub](https://github.com/CarlosAdolfoPeralta)

## Summary

| | |
|---|---|
| **Problem** | Detect trucks with an imminent air pressure system (APS) failure before a breakdown occurs |
| **Data** | [APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/414/ida2016challenge) (UCI): 60,000 trucks in train (1.7% with failure) and 16,000 in test (2.3%), with 170 anonymized features |
| **Costs (official)** | Unnecessary inspection: 10 · Missed failure: 500 |
| **Model** | LightGBM, with the decision threshold chosen by cost |
| **Test result** | Mean total cost of **8,580** (range 7,930-9,360 across 5 seeds), vs. 156,250 inspecting every truck and 187,500 inspecting none |

## Context

Inspecting a truck that has no failure is cheap (10), but missing a failure can end in a breakdown and costs far more (500). For that reason the metric to minimize is not accuracy but total cost:

```
total cost = 10 × false positives + 500 × false negatives
```

This is the official metric of the IDA 2016 challenge in which this dataset was published.

## Data

- **Origin:** data from Scania trucks in daily use. The positive class is failures of a specific component of the air pressure system (APS); the negative class is trucks with failures in other components.
- **Size:** 60,000 trucks in train (1,000 with failure) and 16,000 in test (375 with failure).
- **Features:** 170, anonymized for proprietary reasons. Missing values are written as `na`.
- **License:** CC BY 4.0.

## Approach

1. **Simple model.** LightGBM with reasonable parameters and no hyperparameter tuning. Missing values are left as nulls, since LightGBM handles them natively.
2. **Cost-based threshold.** The threshold that minimizes `10 × FP + 500 × FN` is chosen. With only 1,000 failures in train, setting aside 20% for validation would leave about 200, too few to choose a stable threshold. That is why **out-of-fold predictions from 5-fold cross-validation** are used, which make use of all the failures.
3. **Log-scale threshold grid.** A first linear grid left the optimum at its lower edge; with a log scale the optimum falls in the interior.
4. **Untouched test set.** The official test set is used only once per run, at the end, with the threshold already fixed.
5. **Stability.** The whole procedure (cross-validation, threshold selection and test evaluation) is repeated with 5 seeds, so as to report a range rather than a single number.

## Results

### Cost on test

| Strategy | Total cost |
|---|---|
| Inspect none | 187,500 |
| Inspect all | 156,250 |
| **Model (mean of 5 seeds)** | **8,580** (std. 558; range 7,930-9,360) |
| Reference: best result of the IDA 2016 challenge | 9,920 |

The mean cost is ~95% lower than inspecting every truck or inspecting none.

### Stability across seeds

| Seed | Threshold | Unnecessary inspections | Missed failures | Test cost |
|---|---|---|---|---|
| 42 | 0.00126 | 524 | 6 | 8,240 |
| 1 | 0.00190 | 449 | 8 | 8,490 |
| 2 | 0.00129 | 543 | 5 | 7,930 |
| 3 | 0.00194 | 436 | 10 | 9,360 |
| 4 | 0.00194 | 438 | 9 | 8,880 |

- **Recall:** between 97.3% and 98.7% (out of 375 failures, between 5 and 10 are missed).
- **Precision:** between 40.5% and 45.6%.
- **What changes across seeds:** the balance between unnecessary inspections and missed failures. The total cost stays similar because one missed failure (500) is equivalent to 50 unnecessary inspections (10 each).

### Main run (seed 42)

| | Predicted: no failure | Predicted: failure |
|---|---|---|
| **Actual: no failure** | 15,101 | 524 |
| **Actual: failure** | 6 | 369 |

- **Cross-validation (train):** AUC-PR 0.8975 (base rate 1.7%) · AUC-ROC 0.989.
- **Test:** AUC-PR 0.934 (base rate 2.3%) · AUC-ROC 0.997. They are not directly comparable with the cross-validation figures, because the proportion of failures differs.

![Cost by threshold](reports/figures/cost_curve.png)

### How to read the comparison with the challenge

In all five runs the cost was below 9,920 (maximum 9,360), so the result is of the same order as, or somewhat better than, the best challenge result. But the range covers only the variation due to training randomness (model seed and cross-validation folds); it does not cover the variation due to the test set, which is fixed and has only 375 failures, nor the different conditions of the 2016 challenge. That is why it is presented as a reference and not as having surpassed it.

## Observations

- **Missing values appear to be informative.** Trucks with a failure have on average 29.2 missing values per row, vs. 13.9 for those without. This is an association; the cause was not investigated. It supports the decision not to impute.
- **The optimal threshold is very low** (between 0.0013 and 0.0019). With these costs, the break-even point of a well-calibrated model would be near `10 / (10 + 500) ≈ 0.02`. That the optimum is much lower suggests that LightGBM's probabilities are not calibrated (hypothesis, not verified).
- **The test set has more failures than train** (2.34% vs. 1.67%), so the two sets do not share the same distribution. This could favor an even lower threshold on test (hypothesis, not verified).

## Limitations

- **Anonymized features:** it is not possible to interpret what each one measures.
- **No dates:** the rows have no timestamp, so temporal validation was not possible.
- **A single model type with no hyperparameter tuning.**
- **Uncalibrated probabilities.**
- **The range across seeds** does not include the variation due to the test set (fixed, with 375 failures).
- **Reproducibility:** results may vary slightly with the LightGBM version and the number of threads.

## Next steps

- Tune hyperparameters.
- Calibrate the probabilities and compare the resulting threshold with the cost-optimal one.
- Compare with other algorithms (XGBoost, CatBoost, logistic regression).
- Try class weighting and analyze whether it improves the cost.

## Repository structure

```
proyecto_fallas_scania/
├── notebooks/
│   └── fallas_scania.ipynb
├── reports/
│   └── figures/
│       └── cost_curve.png
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` folder is not included in the repository.

## How to reproduce

1. Install dependencies: `pip install -r requirements.txt`
2. Download `aps_failure_training_set.csv` and `aps_failure_test_set.csv` from [UCI](https://archive.ics.uci.edu/dataset/414/ida2016challenge) and copy them into a `data/` folder at the project root.
3. Run `notebooks/fallas_scania.ipynb` from top to bottom. It generates the figure in `reports/figures/`. Section 6 repeats the procedure with 5 seeds and takes a few minutes.

## Data source

Scania CV AB. *APS Failure at Scania Trucks* [Dataset]. 2016. UCI Machine Learning Repository. https://doi.org/10.24432/C51S51. License CC BY 4.0.

## Tech stack

Python · pandas · LightGBM · scikit-learn · matplotlib
