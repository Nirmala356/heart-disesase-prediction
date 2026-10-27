# Heart Disease Prediction Using Machine Learning

An end-to-end machine-learning portfolio project using the Cleveland subset of the UCI Heart Disease dataset. The project includes validated data loading, exploratory analysis, leakage-safe model comparison, offline tests, and an interactive Streamlit dashboard.

> **Non-clinical prototype:** This project is not clinically validated and must not be used to diagnose, treat, or rule out heart disease.

## What the project does

- Downloads and validates the UCI data archive, or processes a local copy.
- Explores the 303-record Cleveland subset and summarizes missingness and feature distributions.
- Compares Logistic Regression, Random Forest, and Gradient Boosting using a stratified holdout and training-only cross-validation.
- Provides a Streamlit dashboard for cohort exploration, model comparisons, model-importance visualizations, and demonstration predictions.
- Includes offline `unittest` coverage for data validation, preprocessing, training artifact persistence, and inference.

## Dataset

The data source is the [UCI Machine Learning Repository Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), dataset ID 45, [DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X). This project uses its processed Cleveland subset (303 rows, 13 input features). The original target `num` is mapped to binary labels as `num == 0 -> 0` and `num > 0 -> 1`; the observed binary counts are 164 class 0 and 139 class 1.

The acquired file has four missing `ca` values and two missing `thal` values. Raw and processed data are downloaded or generated locally and are not committed. The UCI catalog metadata reviewed for this dataset does not state a dataset-specific license; check UCI's current terms before reuse or redistribution. See the [data dictionary](docs/data_dictionary.md) for field definitions and notes.

## Results

On the fixed stratified 80/20 split (242 training records, 61 held-out records), five-fold stratified cross-validation selected Logistic Regression by mean F1 (0.8363), with recall and ROC-AUC as tie-breakers. Its held-out results were:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9632 |

On this holdout, Logistic Regression produced 28 true negatives, 5 false positives, 3 false negatives, and 25 true positives. These are small-sample estimates from one split, not a performance guarantee or evidence of clinical utility. Detailed model comparisons and experiment settings are in the [experiment log](docs/experiment_log.md) and [project report](docs/project_report.md).

## Setup and run

From the project root in Windows PowerShell, create a Python 3.10 environment and install the pinned dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and validate the dataset, generate EDA and training outputs, run the offline tests, then start the app:

```powershell
python -m src.data_loader
python -m src.eda
python -m src.train
python -m unittest discover -s tests -v
streamlit run app.py
```

If automatic download is unavailable, download the [official UCI ZIP archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip), save it as `data/raw/heart_disease.zip`, and run:

```powershell
python -m src.data_loader --from-archive
```

The local model artifact and generated reports/figures are required by the dashboard and are recreated by the EDA and training commands. The app does not retrain the model. Its displayed probability is an uncalibrated model estimate, and submitted inputs are not persisted by the app by default.

## Repository contents

- `app.py` — Streamlit dashboard
- `src/` — data, preprocessing, EDA, training, evaluation, and prediction modules
- `tests/` — offline unit tests
- `docs/` — data dictionary, experiment log, and detailed project report
- `requirements.txt` — pinned Python dependencies

## Limitations and responsible use

The Cleveland dataset is small, historical, and may not represent current practice or other populations. Results use one split and do not establish external generalization, calibration, fairness, or clinical validity. Feature importance is not causal evidence. This prototype is not a diagnostic tool and cannot safely rule out disease; seek qualified medical care for health concerns.

## License

Project code is provided under the [MIT License](LICENSE), Copyright (c) 2026 Nirmala Ravuri. Dataset terms are separate; consult UCI's current terms before reuse or redistribution.
