"""Generate exploratory plots and a data summary from the validated CSV."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import PROJECT_ROOT, load_dataset
from src.preprocessing import NUMERIC_FEATURES, split_features_target

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "data_summary.json"


def make_eda_figures(data: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> dict:
    """Create readable EDA plots and return measured dataset summary values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    features, target = split_features_target(data)
    labeled = features.copy()
    labeled["target"] = target.map({0: "No disease (0)", 1: "Disease (1-4)"})
    summary = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "raw_target_counts": {
            str(int(label)): int(count)
            for label, count in data["num"].value_counts().sort_index().items()
        },
        "binary_target_counts": {
            str(int(label)): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
        "missing_values": {
            str(column): int(count)
            for column, count in data.isna().sum().items()
        },
        "numeric_ranges": {
            column: {
                "min": float(features[column].min()),
                "max": float(features[column].max()),
            }
            for column in NUMERIC_FEATURES
        },
        "categorical_values": {
            column: sorted(
                int(value) for value in features[column].dropna().unique()
            )
            for column in features.columns
            if column not in NUMERIC_FEATURES
        },
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=target.map({0: "No disease (0)", 1: "Disease (1-4)"}), ax=ax)
    ax.set(title="Cleveland cohort: binary target distribution", xlabel="Target", ylabel="Patients")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, column in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=data, x=column, bins=20, kde=True, ax=ax, color="#2878B5")
        ax.set(title=f"{column} distribution", xlabel=column, ylabel="Patients")
    axes.flat[-1].set_visible(False)
    fig.suptitle("Continuous feature distributions", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    comparison_features = ["age", "thalach", "chol", "oldpeak"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, column in zip(axes.flat, comparison_features):
        sns.boxplot(data=labeled, x="target", y=column, ax=ax, hue="target", legend=False)
        ax.set(title=f"{column} by binary target", xlabel="Target", ylabel=column)
    fig.suptitle("Observed measurements by target group", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "features_by_target.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    missing = data.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#D9534F")
        ax.set(title="Missing values by field", xlabel="Feature", ylabel="Missing records")
    fig.tight_layout()
    fig.savefig(output_dir / "missing_values.png", dpi=160)
    plt.close(fig)

    corr_data = data[NUMERIC_FEATURES].copy()
    corr_data["binary_target"] = target
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_data.corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set(title="Correlation: continuous measurements and binary target")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Load validated local data and write EDA artifacts."""
    data = load_dataset()
    summary = make_eda_figures(data)
    print(f"EDA completed for {summary['rows']} records and {summary['columns']} fields.")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Measured data summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
