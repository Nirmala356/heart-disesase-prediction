# Experiment log

## Environment

- Python: 3.10.1, existing workspace `venv`
- pandas: 2.3.3
- numpy: 2.2.6
- matplotlib: 3.10.9
- seaborn: 0.13.2
- scikit-learn: 1.7.2
- joblib: 1.6.0
- streamlit: 1.65.0
- requests: 2.34.2

These versions were observed in the configured environment on 2026-10-25. No packages were installed.

## Data

- Source: UCI Heart Disease, dataset ID 45, processed Cleveland subset
- Repository page: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- Download URL: <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>
- Archive: `data/raw/heart_disease.zip`; processed CSV: `data/processed/cleveland.csv`
- DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)
- Dataset-specific license: not stated in the UCI catalog metadata retrieved; check current terms before reuse/redistribution.
- Observed shape: 303 rows × 14 fields
- Raw `num` counts: 0: 164, 1: 55, 2: 36, 3: 35, 4: 13
- Binary counts (`num == 0` -> 0; `num > 0` -> 1): 164 class 0, 139 class 1
- Missing feature cells: `ca`: 4, `thal`: 2; target: 0
- Source missing marker `?` becomes `NaN`; no rows were dropped or imputed during loading/EDA.

## Experiment settings

- Split: stratified 80/20 train/test, random state 42 (242 train, 61 test)
- CV: 5-fold `StratifiedKFold`, shuffle enabled, random state 42
- Scoring: accuracy, precision, recall, F1, ROC-AUC; GridSearchCV refits on F1
- Selection: highest mean training CV F1; ties broken by recall then ROC-AUC
- Logistic Regression grid: `C` in 0.1, 1.0, 10.0; `max_iter=2000`
- Random Forest grid: 200 trees; `max_depth` None or 5; `min_samples_leaf` 1 or 2
- Gradient Boosting grid: 100 estimators; learning rate 0.05 or 0.1; max depth 1 or 2
- Seed: 42 for split and estimators
- Preprocessing: fold-fitted median numeric imputation; standard scaling for Logistic Regression; most-frequent categorical imputation and one-hot encoding for all models

## Measured results

### Mean cross-validation scores (training folds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best parameters |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.8554 | 0.8781 | 0.8016 | 0.8363 | 0.8995 | `C=10.0` |
| Random Forest | 0.8139 | 0.8129 | 0.7739 | 0.7900 | 0.8924 | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` |
| Gradient Boosting | 0.8469 | 0.8777 | 0.7830 | 0.8234 | 0.8958 | `learning_rate=0.05`, `max_depth=1`, `n_estimators=100` |

Selected model: Logistic Regression, by the training-only CV F1 rule.

### Held-out test metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9632 | 28 | 5 | 3 | 25 |
| Random Forest | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9432 | 27 | 6 | 3 | 25 |
| Gradient Boosting | 0.8689 | 0.8571 | 0.8571 | 0.8571 | 0.9443 | 29 | 4 | 4 | 24 |

Held-out confusion-matrix rows are actual labels `[0, 1]`, columns predicted labels `[0, 1]`. The selected model therefore produced 3 false negatives and 5 false positives on this 61-row test partition. These scores and counts are not clinical safety estimates.

Generated figures in `reports/figures/`: `target_distribution.png`, `numeric_distributions.png`, `features_by_target.png`, `missing_values.png`, `correlation_heatmap.png`, `model_comparison.png`, `confusion_matrix_logistic_regression.png`, `confusion_matrix_random_forest.png`, `confusion_matrix_gradient_boosting.png`, `roc_curves.png`, `precision_recall_curves.png`, `tree_feature_importance.png`, and `permutation_importance.png`.

The reusable model bundle is `models/heart_disease_pipeline.joblib`; structured measured scores are in generated `reports/evaluation_results.json`. A different random state or package version may produce different metrics. These outputs are educational and do not prove causation or clinical utility.
