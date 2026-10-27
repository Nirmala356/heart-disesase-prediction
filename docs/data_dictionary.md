# Data dictionary

## Source and target

The source is the Cleveland subset of the UCI Machine Learning Repository Heart Disease dataset (dataset ID 45; DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)). Source page: <https://archive.ics.uci.edu/dataset/45/heart+disease>. The official ZIP archive contains a processed Cleveland file with no header; its 14 fields are assigned the names below based on UCI documentation and validated on download. The UCI catalog metadata retrieved for dataset ID 45 does not state a dataset-specific license; consult UCI's current terms before reuse or redistribution.

`num` is the original angiographic disease-status target, observed with values `0`–`4`. UCI describes 0 as less than 50% diameter narrowing and its Heart Disease documentation treats values 1–4 as disease presence for the commonly used binary distinction. This project maps `num == 0` to 0 (absence) and `num > 0` to 1 (presence). This is a deliberate educational simplification, not a clinical diagnostic threshold. Missing target values are rejected, not imputed.

In the downloaded Cleveland file, the raw target counts were 0: 164, 1: 55, 2: 36, 3: 35, and 4: 13. After binary mapping, counts were 0: 164 and 1: 139.

## Feature fields

| Field | Description | Dataset code / unit | Planned preprocessing |
|---|---|---|---|
| `age` | Age | years | Numeric median imputation; scaled for Logistic Regression |
| `sex` | Sex | 1 = male; 0 = female | Categorical one-hot encoding |
| `cp` | Chest-pain type | 1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic | Categorical one-hot encoding |
| `trestbps` | Resting blood pressure on admission to hospital | mm Hg | Numeric median imputation; scaled for Logistic Regression |
| `chol` | Serum cholesterol | mg/dl | Numeric median imputation; scaled for Logistic Regression |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true; 0 = false | Categorical one-hot encoding |
| `restecg` | Resting electrocardiographic result | 0 = normal; 1 = ST-T wave abnormality; 2 = probable/definite left ventricular hypertrophy | Categorical one-hot encoding |
| `thalach` | Maximum heart rate achieved | beats per minute | Numeric median imputation; scaled for Logistic Regression |
| `exang` | Exercise-induced angina | 1 = yes; 0 = no | Categorical one-hot encoding |
| `oldpeak` | ST depression induced by exercise relative to rest | ST depression | Numeric median imputation; scaled for Logistic Regression |
| `slope` | Slope of the peak exercise ST segment | 1 = upsloping; 2 = flat; 3 = downsloping | Categorical one-hot encoding |
| `ca` | Number of major vessels colored by fluoroscopy | 0–3 | Categorical one-hot encoding |
| `thal` | Thalassemia test result | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical one-hot encoding |

## Missingness, categories, and limitations

The source encodes missing feature values with `?`, parsed as missing. The downloaded Cleveland file has 4 missing values in `ca` and 2 in `thal`, with no missing targets. No missing feature values are filled before splitting. Numeric missing values are median-imputed and categorical missing values most-frequent-imputed by transformers fitted only on training folds. Category values are dataset codes, not continuous quantities, and are one-hot encoded as categorical inputs.

Observed numeric ranges: age 29–77, resting blood pressure 94–200 mm Hg, cholesterol 126–564 mg/dl, maximum heart rate 71–202 bpm, and oldpeak 0–6.2. Observed categorical values match the documented codes in the table above; `ca` values are 0–3 and `thal` values are 3, 6, and 7. UCI documentation is the authority for code meanings. The data are historical, small, and may not represent current populations or care; feature codes and associations do not imply causation or general clinical validity.
