"""Cross-validate, select, evaluate, and persist Cleveland classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data_loader import PROJECT_ROOT, load_dataset, validate_dataset
from src.evaluate import calculate_metrics, create_evaluation_figures, positive_probabilities
from src.preprocessing import FEATURE_COLUMNS, ModelName, make_pipeline, split_features_target

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
RANDOM_STATE = 42
MODEL_NAMES: tuple[ModelName, ...] = (
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
)
PARAMETER_GRIDS: dict[ModelName, dict[str, list]] = {
    "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
    "Random Forest": {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [None, 5],
        "classifier__min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [1, 2],
    },
}
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def train_models(
    data: pd.DataFrame | None = None,
    model_path: Path = MODEL_PATH,
    results_path: Path = RESULTS_PATH,
    figures_dir: Path | None = None,
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Select an estimator using training-only CV and evaluate once on the holdout."""
    source_data = load_dataset() if data is None else validate_dataset(data)
    features, target = split_features_target(source_data)
    if target.nunique() != 2:
        raise ValueError("Binary training requires both target classes.")
    if cv_splits < 2 or target.value_counts().min() < cv_splits:
        raise ValueError(
            f"Each class must have at least {cv_splits} rows for stratified CV."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    if y_test.nunique() != 2:
        raise ValueError(
            "The held-out partition must contain both classes to calculate ROC-AUC."
        )
    if y_train.value_counts().min() < cv_splits:
        raise ValueError("Training partition is too small for the requested CV folds.")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    fitted_models: dict[ModelName, object] = {}
    cv_results: dict[str, dict] = {}
    for name in MODEL_NAMES:
        search = GridSearchCV(
            estimator=make_pipeline(name),
            param_grid=PARAMETER_GRIDS[name],
            scoring=SCORING,
            refit="f1",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        index = int(search.best_index_)
        cv_results[name] = {
            metric: float(search.cv_results_[f"mean_test_{metric}"][index])
            for metric in SCORING
        }
        cv_results[name]["std_f1"] = float(search.cv_results_["std_test_f1"][index])
        cv_results[name]["best_params"] = search.best_params_

    selected_name = max(
        MODEL_NAMES,
        key=lambda name: (
            cv_results[name]["f1"],
            cv_results[name]["recall"],
            cv_results[name]["roc_auc"],
        ),
    )

    test_results = {}
    for name, fitted_model in fitted_models.items():
        prediction = fitted_model.predict(x_test)
        probability = positive_probabilities(fitted_model, x_test)
        test_results[name] = calculate_metrics(y_test, prediction, probability)

    evaluated = {name: fitted_models[name] for name in MODEL_NAMES}
    figures_path = figures_dir or PROJECT_ROOT / "reports" / "figures"
    create_evaluation_figures(
        evaluated,
        y_test,
        x_test,
        test_results,
        selected_name,
        figures_path,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_ranges = {
        column: {
            "min": float(x_train[column].min()),
            "max": float(x_train[column].max()),
            "default": float(x_train[column].median()),
        }
        for column in ["age", "trestbps", "chol", "thalach", "oldpeak"]
    }
    categorical_values = {
        column: sorted(int(value) for value in x_train[column].dropna().unique())
        for column in FEATURE_COLUMNS
        if column not in numeric_ranges
    }
    bundle = {
        "model": fitted_models[selected_name],
        "selected_model": selected_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_ranges": numeric_ranges,
        "categorical_values": categorical_values,
        "target_definition": "num == 0 -> 0; num in 1..4 -> 1",
        "random_state": random_state,
    }
    joblib.dump(bundle, model_path)

    results = {
        "dataset": {
            "source": "UCI Heart Disease dataset ID 45, processed Cleveland subset",
            "rows": int(len(source_data)),
            "binary_target_counts": {
                str(int(label)): int(count)
                for label, count in target.value_counts().sort_index().items()
            },
            "split": {
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "test_size": test_size,
                "random_state": random_state,
                "stratified": True,
            },
        },
        "cross_validation": {
            "folds": cv_splits,
            "scoring": list(SCORING),
            "selection_rule": "Highest mean CV F1; ties broken by recall, then ROC-AUC.",
            "models": cv_results,
        },
        "selected_model": selected_name,
        "held_out_test_metrics": test_results,
        "artifact": str(model_path),
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Selected using training-only CV: {selected_name}")
    print(f"Held-out metrics:\n{pd.DataFrame(test_results).T.to_string()}")
    print(f"Saved model: {model_path}")
    print(f"Saved results: {results_path}")
    return results


def main() -> None:
    """Run the reproducible experiment from the project root."""
    train_models()


if __name__ == "__main__":
    main()
