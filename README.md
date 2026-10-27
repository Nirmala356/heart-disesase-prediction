# Heart Disease Prediction Using Machine Learning

An end-to-end machine-learning portfolio project built around the UCI Heart Disease Cleveland dataset. It demonstrates data acquisition and validation, exploratory analysis, leakage-aware model comparison, evaluation, automated tests, and an interactive Streamlit application.

> **Non-clinical prototype.** This project is not a clinically validated diagnostic tool. Its predictions must not be used to diagnose, treat, or rule out heart disease.

## Project status

The pipeline has been implemented and run against the official UCI Cleveland file; the Streamlit app and offline tests are implemented. Figures and metrics below were generated from that downloaded file. See [the experiment log](docs/experiment_log.md) for settings and measured results.

## Environment and setup

Use Python 3.10 and run these commands from the project root in Windows PowerShell to create an environment and install the dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency versions are pinned in `requirements.txt`. The commands below download the dataset, generate the local analysis/model/report artifacts used by the dashboard, run the offline tests, and launch the app.

## Dataset

The source is the official UCI Machine Learning Repository Heart Disease dataset, Cleveland subset (dataset ID 45): <https://archive.ics.uci.edu/dataset/45/heart+disease>. The reproducible download URL is <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>. The original dataset documentation describes four databases; this project uses the 303-row Cleveland processed data file in the archive.

The observed original target values are 0–4. For this binary classification task, `num == 0` maps to class 0 and `num > 0` maps to class 1. The acquired file has 4 missing `ca` and 2 missing `thal` values; they are imputed inside each training fold. Binary target counts are 164 class 0 and 139 class 1. See [the data dictionary](docs/data_dictionary.md) for code definitions and limitations.

The dataset is identified with DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). The UCI catalog metadata available for this dataset does not state a dataset-specific license; consult UCI's current terms and the cited source before reuse or redistribution. Raw data are not committed.

## Run the project

From the activated environment and project root:

```powershell
python -m src.data_loader
python -m src.eda
python -m src.train
python -m unittest discover -s tests -v
streamlit run app.py
```

The loader downloads and validates the archive; EDA and training then use the processed local CSV. Training performs a seeded, stratified 80/20 split and selects settings using five-fold CV on the training partition only. The test partition is used for final evaluation after selection. The test command runs the offline test suite.

If automatic download is unavailable, download the official ZIP from <https://archive.ics.uci.edu/static/public/45/heart+disease.zip> (or from the UCI dataset page), save it as `data/raw/heart_disease.zip`, then run:

```powershell
python -m src.data_loader --from-archive
```

The script extracts and validates `processed.cleveland.data` and writes the same processed CSV used by the normal workflow. Do not rename, edit, or fabricate source values.

## Models and evaluation

The classifiers are Logistic Regression (scaled baseline), Random Forest, and Gradient Boosting. A `ColumnTransformer` and estimator-specific `Pipeline` keep imputation, encoding, and scaling inside cross-validation. Selection uses highest mean CV F1, with recall and ROC-AUC as tie-breakers. On this fixed split Logistic Regression was selected (CV F1 0.836). Its held-out metrics were accuracy 0.869, precision 0.833, recall 0.893, F1 0.862, and ROC-AUC 0.963. These are small-sample estimates, not a performance guarantee or clinical validation.

Accuracy alone is not the selection criterion. On the selected model's held-out set, there were 3 false negatives and 5 false positives. Both error types matter in health contexts; these results do not establish clinical utility. Full model comparison and settings are in [the experiment log](docs/experiment_log.md) and [the project report](docs/project_report.md).

## Visualizations

Generated plots are saved to `reports/figures/`: class balance, numerical distributions, target comparisons, missingness, continuous-feature correlation, model metrics, per-model confusion matrices, ROC/precision-recall curves, tree importance, and permutation importance. Feature importance is an association within a fitted model, not evidence of causation.

## Healthcare analytics dashboard

Run the app from the project root with `streamlit run app.py`. Its responsive sidebar separates five views:

- **Overview:** actual cohort totals, binary class balance, missing-value summary, and selected-model held-out metrics.
- **Model comparison:** saved cross-validation and held-out scores, selectable metric comparison, confusion matrices, and the saved ROC and precision-recall figures.
- **Explore data:** interactive distributions and target comparisons from the validated local CSV, plus a continuous-feature correlation heatmap and missing-value table.
- **Explainability:** saved tree and permutation importance figures, with a clear warning that importance is not causation.
- **Predict:** the original interactive form using the persisted fitted pipeline, exact feature order, validated ranges/categories, and visible medical disclaimer.

The charts use Altair, already available through the existing Streamlit environment; Plotly is not installed and no additional package is required. Dashboard results and figures are loaded from `reports/` and are not regenerated by the app. The app does not retrain or modify the model. Its probability display is an uncalibrated model estimate. Submitted inputs are not persisted by default; the app clearly states that this is not a diagnosis and cannot safely rule out disease.

## Limitations

The Cleveland dataset is small, historical, and not representative of all populations or present-day clinical practice. Its labels and measurements are not a substitute for expert review. Split and cross-validation estimates can be uncertain; model scores are not clinical validation, and feature importance is not causal explanation.

## Repository layout

```text
data/                 Downloaded and processed data (not committed)
docs/                 Data dictionary, project report, experiment log
models/               Locally generated fitted pipeline (not committed)
reports/figures/      Locally generated charts (not committed)
src/                  Data, EDA, training, evaluation, prediction modules
tests/                Automated tests (stdlib unittest)
app.py                Streamlit interface
```
