"""Held-out metrics and evaluation figures for fitted classification pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import ModelName


def positive_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return probabilities for class 1, independent of estimator class order."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted classifier does not contain positive class 1.")
    positive_index = classes.index(1)
    return model.predict_proba(features)[:, positive_index]


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict:
    """Compute binary classification metrics and explicit-label confusion counts."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [0, 1],
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_probability))
        if len(np.unique(y_true)) == 2
        else None
    )
    return metrics


def create_evaluation_figures(
    models: dict[ModelName, Pipeline],
    y_test: pd.Series,
    features_test: pd.DataFrame,
    metrics: dict[str, dict],
    selected_model_name: ModelName,
    output_dir: Path,
) -> None:
    """Save per-model confusion matrices and comparison/curve/importance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    probabilities = {
        name: positive_probabilities(model, features_test)
        for name, model in models.items()
    }

    for name, model_metrics in metrics.items():
        matrix = np.asarray(model_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax,
        )
        ax.set(title=f"{name}: held-out confusion matrix", xlabel="Prediction", ylabel="Actual")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison = pd.DataFrame(
        {
            name: [metrics[name].get(metric) for metric in metric_names]
            for name in models
        },
        index=metric_names,
    ).T
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.plot(kind="bar", ax=ax, color=["#2878B5", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"])
    ax.set(title="Held-out model comparison", xlabel="Model", ylabel="Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probability in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
        auc = metrics[name]["roc_auc"]
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(title="Held-out ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ax.plot(recall, precision, label=name)
    ax.set(title="Held-out precision-recall curves", xlabel="Recall", ylabel="Precision")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    tree_names = ["Random Forest", "Gradient Boosting"]
    importance_frames = []
    for name in tree_names:
        pipeline = models[name]
        classifier = pipeline.named_steps["classifier"]
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance_frames.append(
            pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_, "model": name})
        )
    importances = pd.concat(importance_frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, name in zip(axes, tree_names):
        top = importances[importances["model"] == name].nlargest(12, "importance")
        sns.barplot(data=top, x="importance", y="feature", ax=ax, color="#59A14F")
        ax.set(title=f"{name}: fitted feature importance", xlabel="Importance", ylabel="Encoded feature")
    fig.suptitle("Tree-model feature importance (not causal evidence)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tree_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    selected = models[selected_model_name]
    permutation = permutation_importance(
        selected,
        features_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    permutation_frame = pd.DataFrame(
        {
            "feature": features_test.columns,
            "importance": permutation.importances_mean,
            "std": permutation.importances_std,
        }
    ).nlargest(len(features_test.columns), "importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        permutation_frame["feature"],
        permutation_frame["importance"],
        xerr=permutation_frame["std"],
        color="#2878B5",
        alpha=0.85,
    )
    ax.invert_yaxis()
    ax.set(title=f"Permutation importance on held-out set: {selected_model_name}", xlabel="Mean F1 decrease after permutation", ylabel="Input feature")
    fig.tight_layout()
    fig.savefig(output_dir / "permutation_importance.png", dpi=160)
    plt.close(fig)
