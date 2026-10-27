# Heart Disease Prediction Using Machine Learning

## Project overview

This portfolio project implements an end-to-end binary classification workflow using the processed Cleveland subset of the UCI Heart Disease dataset. It covers data acquisition and validation, exploratory data analysis, fold-fitted preprocessing, model tuning and comparison, held-out evaluation, offline tests, and an interactive Streamlit application.

The target is derived from the source field `num`: records with `num == 0` map to class 0, and records with `num > 0` map to class 1. The source subset contains 303 records and 13 input features. Under this mapping, 164 records are class 0 and 139 are class 1.

Logistic Regression was selected using training-only cross-validation F1. On the fixed 61-record held-out test partition, it achieved accuracy 0.8689, precision 0.8333, recall 0.8929, F1 0.8621, and ROC-AUC 0.9632. These are estimates from one small split, not evidence of clinical effectiveness.

> **Non-clinical prototype:** This project is not clinically validated. It must not be used to diagnose, treat, or rule out heart disease, or as a substitute for professional medical advice.

## Objective and scope

The objective is to demonstrate a reproducible tabular machine-learning workflow and compare three scikit-learn classifiers on the stated binary task. The workflow emphasizes explicit input validation, prevention of preprocessing leakage, multiple evaluation metrics, reproducible splitting and cross-validation, and clear reporting of uncertainty.

The implementation uses only the processed Cleveland subset, one stratified train/test split, training-only cross-validation for model selection, a fitted pipeline, and a local Streamlit interface. It does not include external or prospective validation, clinical workflow integration, threshold optimization, probability calibration, or clinical deployment.

## Dataset and attribution

- **Source:** UCI Machine Learning Repository, [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), dataset ID 45.
- **Subset used:** processed Cleveland subset, distributed in the [UCI ZIP archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip).
- **DOI:** [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- **Observed shape:** 303 records × 14 columns (13 predictors and the original target).
- **Target counts before mapping:** `num` 0: 164, 1: 55, 2: 36, 3: 35, 4: 13.
- **Target counts after mapping:** class 0: 164; class 1: 139.
- **Missing values:** four missing `ca` values and two missing `thal` values; no missing target values.

The project loader assigns column names according to the UCI data description, parses `?` as missing, and validates schema, numeric values, and raw target labels. No source rows are dropped. The processed CSV and downloaded ZIP are generated or stored locally and are excluded from the repository. The UCI catalog metadata reviewed for dataset ID 45 does not state a dataset-specific license; check UCI's current terms before reuse or redistribution.

See the [data dictionary](data_dictionary.md) for field meanings, codes, observed ranges, and missing-value details.

## Technical stack

| Component | Version or implementation |
|---|---|
| Python | 3.10.1 was observed in the configured project environment |
| Data handling | pandas 2.3.3, NumPy 2.2.6 |
| Machine learning | scikit-learn 1.7.2, joblib 1.6.0 |
| Visualization | Matplotlib 3.10.9, seaborn 0.13.2; Altair for dashboard charts |
| Application and data acquisition | Streamlit 1.65.0, requests 2.34.2 |
| Tests | Python standard-library `unittest` |

Pinned dependencies are listed in [`requirements.txt`](../requirements.txt). Altair is available through the installed Streamlit dependency; it is not separately pinned in that file.

## Data processing and preprocessing

The original target is converted with `num > 0` to produce the positive class. Thirteen source features are retained. The five numeric fields are `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`; the remaining fields are treated as categorical codes.

Preprocessing is part of each estimator pipeline and is fitted only on the training partition or training fold:

- Numeric fields are median-imputed.
- Categorical fields are most-frequent-imputed and one-hot encoded.
- Logistic Regression additionally standardizes numeric fields.
- Unknown categorical values are ignored by the one-hot encoder.

This pipeline design keeps imputation, encoding, and scaling within cross-validation fitting, rather than estimating transformations from the full dataset.

## Training and model comparison

The dataset is split once with a stratified 80/20 train/test split using random state 42 (242 training records, 61 test records). Model selection uses five-fold shuffled `StratifiedKFold` cross-validation on the training partition only, also with random state 42.

Each candidate is tuned with `GridSearchCV`, scoring accuracy, precision, recall, F1, and ROC-AUC. Parameter selection/refitting uses F1. The selected candidate is the one with the highest mean training-fold F1; ties are broken by recall and then ROC-AUC. The held-out test partition is evaluated after selection and is not used to select the model.

| Candidate | Parameters searched |
|---|---|
| Logistic Regression | `C`: 0.1, 1.0, 10.0; `max_iter=2000` |
| Random Forest | 200 trees; `max_depth`: `None` or 5; `min_samples_leaf`: 1 or 2 |
| Gradient Boosting | 100 estimators; `learning_rate`: 0.05 or 0.1; `max_depth`: 1 or 2 |

### Mean cross-validation results

Scores below are means across training folds for each candidate's selected parameters. Logistic Regression was selected because its mean CV F1 was highest.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best parameters |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.8554 | 0.8781 | 0.8016 | 0.8363 | 0.8995 | `C=10.0` |
| Random Forest | 0.8139 | 0.8129 | 0.7739 | 0.7900 | 0.8924 | `max_depth=None`, `min_samples_leaf=1`, `n_estimators=200` |
| Gradient Boosting | 0.8469 | 0.8777 | 0.7830 | 0.8234 | 0.8958 | `learning_rate=0.05`, `max_depth=1`, `n_estimators=100` |

### Held-out test results

Confusion-matrix counts use actual labels as rows and predicted labels as columns.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9632 | 28 | 5 | 3 | 25 |
| Random Forest | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9432 | 27 | 6 | 3 | 25 |
| Gradient Boosting | 0.8689 | 0.8571 | 0.8571 | 0.8571 | 0.9443 | 29 | 4 | 4 | 24 |

For the selected Logistic Regression model on this split, the 61-record holdout produced 28 true negatives, 25 true positives, 5 false positives, and 3 false negatives. These counts describe this test partition only; they do not estimate clinical harms or establish a safe decision threshold.

The model bundle stores the selected fitted pipeline, feature order, target definition, and input ranges/categories used by the app. JSON evaluation results and figures are generated locally by the EDA and training commands and are excluded from the repository.

## Exploratory analysis and generated outputs

The EDA workflow summarizes class balance, missingness, observed values and ranges, and feature distributions. It generates a data summary and figures for target distribution, numeric distributions, feature comparisons, missing values, and continuous-feature correlations.

The training/evaluation workflow generates model comparison and confusion-matrix figures, ROC and precision-recall curves, and tree-based and permutation feature-importance figures. These files are local outputs under `reports/` and are not committed. Feature importance and correlation are descriptive associations, not causal explanations.

## Streamlit application

Run the app with `streamlit run app.py` after completing data preparation and training. The dashboard reads the processed dataset and generated results, figures, and fitted model; it does not retrain the model.

The application provides five views:

- **Overview:** cohort totals, class balance, missing-value summary, and selected-model held-out metrics.
- **Model comparison:** cross-validation and held-out scores, selectable metric views, confusion matrices, and saved ROC/precision-recall plots.
- **Explore data:** interactive distributions and target comparisons, continuous-feature correlations, and missing-value summaries.
- **Explainability:** saved tree and permutation importance images, with cautions about interpretation.
- **Predict:** a form for all 13 model features, using the persisted pipeline and input validation for fields, numeric ranges, and categorical values.

The app displays the predicted binary class and an uncalibrated model probability estimate. It warns that predictions are not a diagnosis and cannot rule out disease. The application does not persist submitted prediction fields by default; this statement does not cover hosting-provider, browser, or infrastructure logging.

## Tests

The project contains five offline `unittest` cases in [`tests/test_pipeline.py`](../tests/test_pipeline.py). They cover invalid target labels, unexpected non-numeric feature text, local ZIP processing, preprocessing and feature order, and a training-artifact-inference round trip with input validation.

Run the suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The previously recorded test run completed with five tests passing. The training-artifact test uses a deterministic synthetic 60-row fixture; its results are test mechanics, not evidence about the UCI dataset. The tests do not cover browser interaction, deployment security, clinical safety, external generalization, calibration, or production monitoring.

## Limitations and responsible use

- The Cleveland subset is small, historical, and may not represent current practice or other populations.
- The target mapping groups four original positive categories into one binary class and is not a validated clinical endpoint.
- Results come from one stratified split and a training-only CV comparison; external and prospective validation are not documented.
- The 61-record holdout produces uncertain estimates. No confidence bounds or clinical safety estimates are claimed.
- The displayed model probability is uncalibrated and is not an individual's clinical risk estimate.
- Subgroup fairness and representativeness have not been established.
- Feature importance is model-specific and non-causal.
- The prototype must not replace professional medical advice, evaluation, or emergency care.

## Reproducibility

From the project root in Windows PowerShell, create the documented Python 3.10 environment and install the pinned dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the complete workflow:

```powershell
python -m src.data_loader
python -m src.eda
python -m src.train
python -m unittest discover -s tests -v
streamlit run app.py
```

If automatic download is unavailable, save the official UCI archive as `data/raw/heart_disease.zip` and process it with:

```powershell
python -m src.data_loader --from-archive
```

The exact experiment configuration and results are also documented in the [experiment log](experiment_log.md).

## Project structure

```text
README.md
LICENSE
requirements.txt
app.py
src/                     Data, preprocessing, EDA, training, evaluation, prediction
tests/                   Offline unit tests
docs/                    Data dictionary, experiment log, project report
data/                    Downloaded/processed data; local and ignored
models/                  Fitted model; generated locally and ignored
reports/                 Generated JSON and figures; local and ignored
```

## License

Project code is distributed under the [MIT License](../LICENSE), Copyright (c) 2026 Nirmala Ravuri. Dataset terms are separate; consult UCI's current terms before reuse or redistribution.
