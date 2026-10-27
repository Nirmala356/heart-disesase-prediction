"""Validated inference helpers for the persisted fitted model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the fitted pipeline and validate the artifact's expected structure."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python -m src.train` first."
        )
    bundle = joblib.load(model_path)
    required = {
        "model",
        "selected_model",
        "feature_columns",
        "numeric_ranges",
        "categorical_values",
        "target_definition",
    }
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Model artifact is missing required keys: {sorted(required)}.")
    if not callable(getattr(bundle["model"], "predict", None)) or not callable(
        getattr(bundle["model"], "predict_proba", None)
    ):
        raise ValueError("Model artifact does not contain a probability-capable model.")
    return bundle


def predict_from_values(
    values: Mapping[str, Any], bundle: dict[str, Any]
) -> dict[str, int | float]:
    """Validate one ordered feature record and return class and model estimate."""
    feature_columns = bundle["feature_columns"]
    missing = set(feature_columns) - set(values)
    unexpected = set(values) - set(feature_columns)
    if missing or unexpected:
        raise ValueError(
            f"Input fields mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    row = pd.DataFrame([{column: values[column] for column in feature_columns}])
    for column in bundle["numeric_ranges"]:
        try:
            row[column] = pd.to_numeric(row[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature {column!r} must be numeric.") from error
        number = float(row.at[0, column])
        bounds = bundle["numeric_ranges"][column]
        if not np.isfinite(number) or not bounds["min"] <= number <= bounds["max"]:
            raise ValueError(
                f"Feature {column!r} must be between {bounds['min']} and "
                f"{bounds['max']} (the training data range)."
            )

    for column, allowed in bundle["categorical_values"].items():
        value = row.at[0, column]
        if value not in allowed:
            raise ValueError(
                f"Feature {column!r} must be one of the training categories {allowed}."
            )

    prediction = int(bundle["model"].predict(row)[0])
    model_classes = list(bundle["model"].classes_)
    if 1 not in model_classes:
        raise ValueError("The fitted model does not contain positive class 1.")
    probability = float(
        bundle["model"].predict_proba(row)[0, model_classes.index(1)]
    )
    return {"prediction": prediction, "probability": probability}
