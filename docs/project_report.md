# Heart Disease Prediction Using Machine Learning

**Portfolio project report**

This report documents the implementation, measured results, tests, and limitations of the project. It is a portfolio demonstration, not a clinical decision system.

---

## Project overview

This project builds and evaluates a reproducible binary classification workflow using the Cleveland subset of the UCI Heart Disease dataset. The task distinguishes records whose original target `num` is zero from records whose original `num` is one through four. The processed dataset contains 303 records and 13 input features; the binary labels comprise 164 negative and 139 positive records.

The implemented workflow compares Logistic Regression, Random Forest, and Gradient Boosting. Data are split into a stratified 80% training partition (242 records) and 20% held-out test partition (61 records), with random state 42. Five-fold stratified cross-validation and grid search on the training partition select the model by mean F1, with recall and ROC-AUC as tie-breakers. Preprocessing—including imputation, one-hot encoding, and scaling where appropriate—is fitted inside the model pipelines and cross-validation folds.

Logistic Regression was selected with mean cross-validation F1 of 0.8363. On the saved 61-record holdout it achieved accuracy 0.8689, precision 0.8333, recall 0.8929, F1 0.8621, and ROC-AUC 0.9632. Its confusion matrix contains 28 true negatives, 5 false positives, 3 false negatives, and 25 true positives. These are estimates from one small split, not evidence of clinical effectiveness or validation. The project also provides exploratory and evaluation figures, a persisted fitted pipeline, offline unit tests, and a Streamlit dashboard with a demonstration prediction form.

## Project context

### Context

This project applies classification methods to a public dataset of clinical measurements as a machine-learning portfolio exercise. A dataset-derived pattern is not a diagnosis, causal explanation, or substitute for clinical assessment.

### Engineering approach

The implementation emphasizes input validation, reproducibility, leakage-aware preprocessing, multiple evaluation metrics, saved artifacts, visual inspection, and clear communication of limitations.

### Scope

The scope is limited to the processed Cleveland subset, a binary target derived from its original five-level target, three scikit-learn classifiers, one fixed stratified train/test split, training-only cross-validation for model selection, a saved pipeline, and a local Streamlit demonstration. It does not include clinical deployment, external validation, clinical workflow integration, or prospective evaluation.

## Classification task

Given the 13 feature values recorded for a row in the Cleveland dataset, predict whether its original `num` target is zero (`0`) or greater than zero (`1`). The target conversion is `num == 0 -> 0` and `num in {1, 2, 3, 4} -> 1`. The goal is to compare classification pipelines and report their measured behavior on a held-out partition; it is not to establish a clinical diagnostic rule.

## Project goals

1. Acquire or load and validate the Cleveland processed dataset.
2. Summarize class balance, feature distributions, missing values, and continuous-feature correlations.
3. Compare Logistic Regression, Random Forest, and Gradient Boosting using fold-fitted preprocessing and training-only cross-validation.
4. Select a candidate using the documented mean-F1 rule and evaluate it and the other candidates on one held-out test partition.
5. Save evaluation results, figures, and the selected fitted preprocessing/model pipeline.
6. Demonstrate inference through a Streamlit dashboard with input checks and safety messaging.
7. Document test evidence, uncertainty, dataset limitations, and ethical constraints.

## Implementation overview

The workflow downloads and validates the official processed Cleveland file or reads the local processed CSV; performs exploratory analysis; splits records before fitting preprocessing; evaluates three classifiers using cross-validation on training data; selects and evaluates a pipeline; persists the chosen pipeline and results; generates figures; and presents saved metrics, interactive data exploration, feature-importance plots, and predictions through Streamlit.

## Runtime and dependencies

### Software versions observed in the project environment

| Component | Project evidence |
|---|---|
| Python | Experiment log documents Python 3.10.1 in the existing project environment. |
| Development environment | Visual Studio Code is documented in the project files. |
| Core data/ML | pandas 2.3.3, NumPy 2.2.6, scikit-learn 1.7.2, joblib 1.6.0. |
| Visualization | Matplotlib 3.10.9 and seaborn 0.13.2 for saved plots; Altair is used by the Streamlit dashboard. |
| App/data acquisition | Streamlit 1.65.0 and requests 2.34.2. |
| Tests | Python standard-library `unittest`; no separate test framework is required by the tests inspected. |

The exact requirements pins in [`requirements.txt`](../requirements.txt) are pandas 2.3.3, NumPy 2.2.6, Matplotlib 3.10.9, seaborn 0.13.2, scikit-learn 1.7.2, joblib 1.6.0, Streamlit 1.65.0, and requests 2.34.2.

### Hardware and operating system

The project files do not specify benchmarked hardware or minimum processor, memory, or storage requirements. No measured hardware requirement or performance benchmark is claimed. A supported operating system is not explicitly declared by a package manifest; the README documents Windows PowerShell commands. Use Python compatible with the pinned package versions.

## Dataset and features

### Source and observed data

- **Source:** UCI Machine Learning Repository, Heart Disease, dataset ID 45, processed Cleveland subset.
- **Dataset page:** <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **Download archive URL used in code:** <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>
- **DOI:** <https://doi.org/10.24432/C52P4X>
- **Observed file:** `data/processed/cleveland.csv`, 303 rows × 14 columns (13 predictors plus original target; locally generated/downloaded and excluded from the repository).
- **Version/date:** Project files identify the UCI dataset and archive but do not record an upstream release/version date or archive checksum. The exact source-file version cannot be established beyond the stated dataset and observed 303-row file.
- **Target counts:** Original `num`: 0 = 164, 1 = 55, 2 = 36, 3 = 35, 4 = 13. Binary mapping: class 0 = 164 and class 1 = 139.
- **Missingness:** `ca` has 4 missing cells; `thal` has 2; no target values are missing. The loader parses `?` as missing, validates schema and target labels, and does not impute or drop rows at load time.

### Features and encoding

| Feature | Description and dataset encoding | Implemented preprocessing |
|---|---|---|
| `age` | Age, years | Numeric; median-imputed; standardized for Logistic Regression |
| `sex` | 0 = female; 1 = male | Categorical one-hot encoding |
| `cp` | Chest-pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic | Categorical one-hot encoding |
| `trestbps` | Resting blood pressure on admission, mm Hg | Numeric; median-imputed; standardized for Logistic Regression |
| `chol` | Serum cholesterol, mg/dl | Numeric; median-imputed; standardized for Logistic Regression |
| `fbs` | Fasting blood sugar > 120 mg/dl: 0 false, 1 true | Categorical one-hot encoding |
| `restecg` | Resting ECG: 0 normal, 1 ST-T wave abnormality, 2 probable/definite LV hypertrophy | Categorical one-hot encoding |
| `thalach` | Maximum heart rate achieved, beats per minute | Numeric; median-imputed; standardized for Logistic Regression |
| `exang` | Exercise-induced angina: 0 no, 1 yes | Categorical one-hot encoding |
| `oldpeak` | ST depression induced by exercise relative to rest | Numeric; median-imputed; standardized for Logistic Regression |
| `slope` | Peak exercise ST-segment slope: 1 upsloping, 2 flat, 3 downsloping | Categorical one-hot encoding |
| `ca` | Number of major vessels colored by fluoroscopy, 0–3 | Categorical; most-frequent-imputed; one-hot encoded |
| `thal` | Thalassemia result: 3 normal, 6 fixed defect, 7 reversible defect | Categorical; most-frequent-imputed; one-hot encoded |
| `num` (target) | Original target values 0–4; 0 represents absence and 1–4 are grouped as presence for this project's binary task | Converted to 0/1; missing target values rejected |

The observed numeric ranges are age 29–77, resting blood pressure 94–200, cholesterol 126–564, maximum heart rate 71–202, and oldpeak 0–6.2. Observed categorical codes match those listed above. Preprocessing is fitted within training data/folds, not on the complete dataset.

### Missing-value handling, limitations, and terms

Source missing markers (`?`) become missing values. There are four missing `ca` values and two missing `thal` values, with no missing targets. Numeric fields are median-imputed; categorical fields are most-frequent-imputed, using training-fold fitted transformers. Target missingness is rejected rather than imputed.

The dataset is small, historical, and limited to the Cleveland subset of a broader historical collection. Its source population, measurements, coding, and labels may not represent current practice or other populations. The binary target compresses four positive source categories into one. The project data dictionary and experiment log state that the UCI catalog metadata retrieved for dataset ID 45 did not state a dataset-specific license; consult UCI's current terms before reuse or redistribution. No independent license interpretation is made here.

## Architecture

```text
Official UCI archive / processed local CSV
                 |
      schema + value validation
                 |
       exploratory data analysis
       |                     |
 data summary + EDA plots    stratified train/test split
                             |
                  training-only 5-fold CV/grid search
                             |
                 fit preprocessing within pipelines
                             |
              select by mean CV F1 (ties: recall, ROC-AUC)
                             |
          evaluate all candidates on the 61-row holdout
             |                         |
      evaluation JSON + plots    selected joblib bundle
                                        |
                            validated Streamlit prediction
```

The saved fitted artifact is referenced by the evaluation JSON as `models/heart_disease_pipeline.joblib`. The app reads this model bundle and the generated JSON/figures; it does not retrain models.

## Implementation workflow

1. **Load and validate:** `src/data_loader.py` downloads the UCI ZIP or processes a local archive, assigns the documented 14 names, parses `?` as missing, checks numeric fields and target labels, and writes the processed CSV.
2. **Explore:** `src/eda.py` computes counts, missingness, observed ranges and categories, and saves EDA visualizations plus `reports/data_summary.json`.
3. **Prepare target and features:** `src/preprocessing.py` validates input data, keeps 13 input fields, and maps `num == 0` to 0 and `num > 0` to 1.
4. **Split:** Training code uses `train_test_split` with 20% holdout, stratification, and random state 42: 242 training and 61 test records.
5. **Cross-validate and tune:** `GridSearchCV` with five-fold shuffled `StratifiedKFold` and random state 42 evaluates accuracy, precision, recall, F1, and ROC-AUC on training folds. Grid-search refitting and parameter selection use F1. Candidate parameter grids are in [`docs/experiment_log.md`](experiment_log.md).
6. **Preprocess within folds:** Numeric fields use median imputation; categorical fields use most-frequent imputation and one-hot encoding with unknown categories ignored. Logistic Regression also scales numeric inputs. Pipelines keep transformation fitting inside each training fold.
7. **Select and evaluate:** The candidate with the highest mean training-CV F1 is selected; ties are broken by recall and ROC-AUC. All three tuned candidates are then scored on the held-out partition, which is not used for model selection.
8. **Persist outputs:** The selected fitted pipeline, feature order, UI ranges/categories, target definition, and selected name are saved in a joblib bundle; measured results are written to JSON and evaluation plots to `reports/figures/`.
9. **Predict:** The Streamlit form gathers all 13 inputs. `src/predict.py` checks missing/extra fields, validates numeric values against training ranges and categories against training categories, invokes the saved pipeline, and reports the predicted class and positive-class model probability estimate.

## Candidate models

### Logistic Regression

Logistic Regression models the log-odds of a binary outcome as a linear function of input features and converts this score to a class probability. It is a relatively simple, efficient, interpretable baseline and can perform well when class separation is approximately linear in the transformed feature space. It may underfit nonlinear relationships and can be sensitive to feature scaling and correlated predictors. Here, numeric features are standardized, categorical features are one-hot encoded, and `C` is tuned over 0.1, 1.0, and 10.0. It was selected by the CV F1 rule.

### Random Forest Classifier

Random Forest trains an ensemble of decision trees on bootstrap samples and feature subsets, combining their predictions. It captures nonlinear patterns and interactions and usually needs no numeric scaling. It can be less transparent than a single model, may overfit on small datasets, and impurity-based feature importance can be biased. Here, 200 estimators are used and maximum depth and minimum leaf size are searched over the documented grid. It is an evaluated comparison model, not the selected model.

### Gradient Boosting Classifier

Gradient Boosting builds an additive sequence of decision trees, each attempting to improve on the current ensemble's errors according to a loss function. It can model nonlinear patterns with relatively shallow trees, but performance depends on hyperparameters and boosting may overfit. Here, 100 estimators are used and learning rate and maximum depth are searched over the documented grid. It is an evaluated comparison model, not the selected model.

## Exploratory and evaluation figures

The figures listed below are generated in `reports/figures/` by the EDA and training commands. These outputs are local and excluded from the repository; the listed paths identify the files produced when the workflow is run. Interpretations describe the observed data or saved evaluation results, not clinical associations.

### Figure 1. Binary target-class distribution

**Generated figure:** `reports/figures/target_distribution.png` (created locally; excluded from the repository).

**Interpretation:** The binary cohort contains 164 class-0 and 139 class-1 records. The classes are not identical in size, although the observed counts are relatively close; stratification preserves their proportions in the train/test partition.

### Figure 2. Continuous-feature distributions

**Generated figure:** `reports/figures/numeric_distributions.png` (created locally; excluded from the repository).

**Interpretation:** The figure shows age, resting blood pressure, cholesterol, maximum heart rate, and oldpeak. Shapes and tails differ across inputs; cholesterol has a high-value tail, and oldpeak is concentrated near zero with a right tail. No distribution alone establishes predictive or medical relevance.

### Figure 3. Observed measurements by target group

**Generated figure:** `reports/figures/features_by_target.png` (created locally; excluded from the repository).

**Interpretation:** Boxplots compare age, maximum heart rate, cholesterol, and oldpeak between the observed binary groups. In this cohort, group medians visibly differ for maximum heart rate and oldpeak; age and cholesterol distributions overlap. This is descriptive, unadjusted comparison and not causal evidence.

### Figure 4. Missing values by field

**Generated figure:** `reports/figures/missing_values.png` (created locally; excluded from the repository).

**Interpretation:** Missingness is confined to `ca` (4 records) and `thal` (2 records), for 6 missing feature cells in total. The project imputes these values inside fitted training pipelines.

### Figure 5. Continuous-feature correlation heatmap

**Generated figure:** `reports/figures/correlation_heatmap.png` (created locally; excluded from the repository).

**Interpretation:** The heatmap includes the five continuous features and the derived binary target; categorical code fields are excluded. In the observed data, Pearson correlations with the target are approximately −0.42 for `thalach`, +0.42 for `oldpeak`, +0.22 for age, +0.15 for `trestbps`, and +0.09 for `chol`. Correlation is descriptive, may reflect confounding, and does not imply causation.

### Figure 6. Held-out model comparison

**Generated figure:** `reports/figures/model_comparison.png` (created locally; excluded from the repository).

**Interpretation:** This compares all five saved held-out metrics for each classifier. Logistic Regression has the highest recall and ROC-AUC; Gradient Boosting has the same accuracy and higher precision but lower recall. The scores use only 61 test records and can change substantially with another sample.

### Figure 7. Logistic Regression confusion matrix

**Generated figure:** `reports/figures/confusion_matrix_logistic_regression.png` (created locally; excluded from the repository).

**Interpretation:** With actual classes on rows and predictions on columns, the selected model has TN=28, FP=5, FN=3, TP=25. Thus, 3 of 28 actual positive test records were predicted negative and 5 of 33 actual negative records were predicted positive.

### Figure 8. Random Forest confusion matrix

**Generated figure:** `reports/figures/confusion_matrix_random_forest.png` (created locally; excluded from the repository).

**Interpretation:** Counts are TN=27, FP=6, FN=3, TP=25. The model identifies the same 25 positive test records as Logistic Regression, while it produces one additional false positive.

### Figure 9. Gradient Boosting confusion matrix

**Generated figure:** `reports/figures/confusion_matrix_gradient_boosting.png` (created locally; excluded from the repository).

**Interpretation:** Counts are TN=29, FP=4, FN=4, TP=24. Relative to the other two candidates on this split, it has fewer false positives and more false negatives.

### Figure 10. Held-out ROC curves

**Generated figure:** `reports/figures/roc_curves.png` (created locally; excluded from the repository).

**Interpretation:** Saved ROC-AUC values are 0.9632 for Logistic Regression, 0.9432 for Random Forest, and 0.9443 for Gradient Boosting. ROC-AUC summarizes ranking across thresholds; it does not establish calibration or safe operating characteristics.

### Figure 11. Held-out precision-recall curves

**Generated figure:** `reports/figures/precision_recall_curves.png` (created locally; excluded from the repository).

**Interpretation:** The curves show the precision/recall trade-off across thresholds for the same held-out set. With only 28 positive test examples, each observation can alter the curve noticeably; the selected model's thresholded precision and recall are reported in the results table.

### Figure 12. Tree-model feature importance

**Generated figure:** `reports/figures/tree_feature_importance.png` (created locally; excluded from the repository).

**Interpretation:** The saved tree-based rankings differ: displayed high entries include `thal_3.0` and `thalach` for Random Forest and `thal_3.0`, `ca_0.0`, and `cp_4.0` for Gradient Boosting. These fitted-model importance values depend on this dataset, encoding, and model; they are not causal effects or clinical explanations.

### Figure 13. Held-out permutation importance

**Generated figure:** `reports/figures/permutation_importance.png` (created locally; excluded from the repository).

**Interpretation:** For selected Logistic Regression, the largest positive mean F1 decreases after permutation are shown for `ca` (about 0.097) and `cp` (about 0.051), while several features are near zero or negative. Error bars show variation over 10 permutations. Importance estimates from a 61-record holdout are noisy and should not be read as causal or clinically meaningful.

## Results

### Experimental setup

The values below are cross-checked against the saved evaluation results and [experiment log](experiment_log.md). The dataset has 303 records; the fixed stratified split contains 242 training records and 61 test records. Five-fold cross-validation was run only within training data, and the model-selection criterion was mean CV F1 with recall and ROC-AUC tie-breakers. The JSON results are generated locally and excluded from the repository.

### Mean training cross-validation scores

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 SD |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8554 | 0.8781 | 0.8016 | 0.8363 | 0.8995 | 0.0127 |
| Random Forest | 0.8139 | 0.8129 | 0.7739 | 0.7900 | 0.8924 | 0.0413 |
| Gradient Boosting | 0.8469 | 0.8777 | 0.7830 | 0.8234 | 0.8958 | 0.0347 |

Logistic Regression is selected because it has the highest mean training-fold F1. Its best recorded parameter is `C=10.0`; selection is not based on test-set scores.

### Held-out test metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.8689 | 0.8333 | 0.8929 | 0.8621 | 0.9632 | 28 | 5 | 3 | 25 |
| Random Forest | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9432 | 27 | 6 | 3 | 25 |
| Gradient Boosting | 0.8689 | 0.8571 | 0.8571 | 0.8571 | 0.9443 | 29 | 4 | 4 | 24 |

### Error analysis and metric limitations

For selected Logistic Regression, recall is 25/28 positive test records (89.3%). The three false negatives are positive-labeled records predicted as class 0; the five false positives are negative-labeled records predicted as class 1. Observed specificity is 28/33 (84.8%). False negatives and false positives have different potential consequences in health contexts; this test set cannot quantify real-world clinical harms or define an acceptable threshold.

Accuracy alone is insufficient: it treats errors symmetrically, does not show which class is missed, and can obscure performance under class imbalance. Precision and recall make positive-class error trade-offs more visible; F1 combines precision and recall at the chosen classification threshold; ROC-AUC describes ranking over thresholds. None alone evaluates calibration, clinical utility, representativeness, or decision consequences. The displayed probability is an uncalibrated model output and is not an individual's clinical risk estimate.

The holdout contains 61 records (28 positive and 33 negative). This is a small sample from one dataset subset and one random split; estimates are uncertain and are not confidence bounds. Repeated selection/evaluation on this split, external validation, temporal validation, subgroup robustness, and prospective clinical testing are not documented. The saved tests use deterministic synthetic sample records for pipeline checks; their scores are not empirical results for the Cleveland dataset and are not included in these tables.

## Streamlit application

The `app.py` application is named Heartwise and offers five dashboard views:

- **Overview:** source-cohort totals, binary class balance, missing-value summary, selected-model held-out scores, and the saved target distribution.
- **Model comparison:** saved CV or held-out metrics, selectable metric comparison, selectable confusion matrix, and saved ROC and precision-recall figures.
- **Explore data:** interactive distributions and target comparisons for all 13 dataset features, continuous-variable correlation view, and a missing-value table. Category codes are excluded from its continuous correlation view.
- **Explainability:** saved tree and permutation importance images, with an explicit warning that feature importance is not causation or a clinical explanation.
- **Predict:** a form for all 13 model features. Numeric controls use training-subset ranges and categorical controls use observed training categories. Inputs pass through the persisted fitted pipeline in training feature order.

Prediction output shows the binary class and an **uncalibrated** positive-class probability estimate, with a warning that the system is not a diagnosis, a negative prediction cannot rule out disease, and health concerns require qualified medical advice. The app states that submitted form values are used for that session's prediction and are not saved or persisted by the app by default. This describes application behavior; it is not a claim about hosting-provider, browser, or infrastructure logging.

## Tests

The project includes five `unittest` cases in [`tests/test_pipeline.py`](../tests/test_pipeline.py), covering rejection of unknown target labels and unexpected non-numeric feature text, processing of a local ZIP archive, preprocessing and feature order, and a training/artifact/inference round trip including input-field validation.

**Command run from the project root:**

```powershell
python -m unittest discover -s tests -v
```

**Observed result:** 5 tests ran in 53.705 seconds; all 5 passed. The training-artifact test trains on its deterministic, locally generated 60-row synthetic fixture with a two-fold CV setting. Its perfect scores in that fixture are software-test mechanics only, not evidence about the Cleveland dataset. The test suite is offline and does not test live UCI availability, browser rendering or user flows, deployment security, clinical safety, external generalization, probability calibration, or production monitoring.

## Strengths and limitations

### Advantages

- The loader validates schema, numeric conversion, and allowed raw targets.
- Preprocessing is encapsulated in estimator pipelines and fitted inside cross-validation folds, reducing leakage risk.
- Model selection is defined in code and separated from held-out evaluation.
- Multiple metrics and confusion matrices expose different error types.
- The saved artifact includes fitted transformations, model, feature order, target mapping, and input constraints.
- The dashboard communicates prototype status, uncertainty, probability interpretation, and session-only input handling.
- Offline unit tests cover key data, preprocessing, persistence, and inference behaviors.

### Limitations

- A small, historical, single-subset dataset and one 61-row holdout provide uncertain performance estimates.
- The binary mapping compresses multiple target levels and does not represent a validated clinical endpoint.
- One train/test split and CV experiment do not establish robust external generalization.
- No independent external validation, clinical validation, threshold optimization, or probability calibration is documented.
- The reported tests are software tests, not clinical validation; no app browser automation is present in the inspected tests.
- Feature-importance plots are model-specific, noisy, and non-causal.
- Dataset representativeness and subgroup fairness are not established.

## Responsible use, privacy, and safety

1. **Non-clinical prototype:** The app and report must not be presented as clinical decision support or a diagnostic product.
2. **No clinical validation:** The model has not been validated with clinicians, external institutions, prospective cohorts, or patient outcomes.
3. **Dataset size and representativeness:** 303 historical Cleveland records cannot support assumptions about all ages, populations, health systems, or contemporary measurements. Subgroup performance and bias have not been evaluated.
4. **Error consequences:** False-negative and false-positive predictions can have different consequences. The small holdout does not establish a safe error rate or decision threshold.
5. **Privacy:** The application code does not persist submitted prediction fields by default. A deployed environment may have additional logging or retention behavior; users should not submit sensitive personal health information to an unreviewed deployment.
6. **Probability interpretation:** The displayed probability is not calibrated as a clinical risk estimate and must not be used for healthcare decisions.
7. **Professional care:** A prediction must never replace professional medical advice, evaluation, or emergency care.

## Potential next steps

These are recommendations, not implemented claims:

- Evaluate on suitable independent datasets and cohorts, with documented data governance and compatible feature definitions.
- Estimate uncertainty and confidence intervals using appropriate resampling or validation designs.
- Investigate probability calibration on a sufficiently large, separate calibration dataset and assess calibration in relevant subgroups.
- Improve interpretability with methods suited to the model and evaluation design, while separating explanation from causality.
- Improve accessibility, form guidance, responsive behavior, and keyboard/screen-reader support in the UI.
- If deployed, add authentication, transport and storage protections, privacy review, audit controls, model/version tracking, monitoring, and documented change/incident processes.
- Expand automated tests to include Streamlit/browser interactions, data-contract checks, and reproducible end-to-end smoke tests.

## Summary

The project implements validated data loading, exploratory analysis, fold-fitted preprocessing, CV-based comparison of three classifiers, a saved model bundle, evaluation outputs, offline tests, and a Streamlit demonstration for the UCI Cleveland Heart Disease subset. Logistic Regression was selected under the documented training-only CV F1 rule and achieved held-out accuracy 0.8689, precision 0.8333, recall 0.8929, F1 0.8621, and ROC-AUC 0.9632 on 61 records. These numbers describe this single split only. The project is not clinically validated, its probability output is uncalibrated, and it must not be used to diagnose disease or replace medical advice.

## References

1. UCI Machine Learning Repository. **Heart Disease** (dataset ID 45), including the processed Cleveland subset. <https://archive.ics.uci.edu/dataset/45/heart+disease>. DOI: <https://doi.org/10.24432/C52P4X>.
2. UCI Machine Learning Repository. Dataset archive used by the project loader. <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>.
3. scikit-learn Developers. **scikit-learn documentation** (pipelines, preprocessing, model selection, classifiers, and metrics). <https://scikit-learn.org/stable/>.
4. pandas Development Team. **pandas documentation**. <https://pandas.pydata.org/docs/>.
5. Streamlit, Inc. **Streamlit documentation**. <https://docs.streamlit.io/>.
6. Project implementation: [`README.md`](../README.md), [`src/data_loader.py`](../src/data_loader.py), [`src/preprocessing.py`](../src/preprocessing.py), [`src/eda.py`](../src/eda.py), [`src/train.py`](../src/train.py), [`src/evaluate.py`](../src/evaluate.py), [`src/predict.py`](../src/predict.py), and [`docs/experiment_log.md`](experiment_log.md). Generated JSON results and figures are recreated locally and excluded from the repository.

## Repository structure

```text
heart-disesase-prediction/
├── app.py
├── README.md
├── requirements.txt
├── docs/
│   ├── data_dictionary.md
│   ├── experiment_log.md
│   └── project_report.md
├── data/                 # Downloaded/processed files are local and ignored
├── models/               # Trained model artifact is local and ignored
├── reports/              # Generated JSON and figures are local and ignored
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/
    └── test_pipeline.py
```

The data, fitted model, generated JSON, and figures are local reproducible outputs and are excluded by `.gitignore`.

## Reproducing the project

From the project root in Windows PowerShell, create and activate a Python 3.10 environment and install dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and validate UCI data, create EDA outputs, train/evaluate and save the pipeline, run offline tests, and start Streamlit:

```powershell
python -m src.data_loader
python -m src.eda
python -m src.train
python -m unittest discover -s tests -v
streamlit run app.py
```

If network download is unavailable, place the official ZIP at `data/raw/heart_disease.zip` and run:

```powershell
python -m src.data_loader --from-archive
```

The commands reproduce the local artifacts used by the dashboard. The project tests use deterministic synthetic fixtures and run without network access.
