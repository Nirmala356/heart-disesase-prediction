"""Leakage-safe feature preparation and estimator pipelines."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import COLUMNS, validate_dataset

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [column for column in COLUMNS[:-1] if column not in NUMERIC_FEATURES]
FEATURE_COLUMNS = COLUMNS[:-1]
RANDOM_STATE = 42
ModelName = Literal["Logistic Regression", "Random Forest", "Gradient Boosting"]


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate source data and convert the five-level outcome to binary."""
    validated = validate_dataset(data)
    features = validated[FEATURE_COLUMNS].copy()
    target = validated["num"].gt(0).astype("int64").rename("target")
    return features, target


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Create imputers/encoders fitted only on the data passed to ``fit``."""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model_name: ModelName) -> Pipeline:
    """Build one model and its training-fold-fitted preprocessing pipeline."""
    estimators = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            False,
        ),
    }
    try:
        estimator, scale_numeric = estimators[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model name: {model_name!r}") from error

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric)),
            ("classifier", estimator),
        ]
    )
