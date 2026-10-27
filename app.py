"""Interactive educational healthcare analytics dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import DATA_PATH, PROJECT_ROOT, load_dataset
from src.predict import load_model_bundle, predict_from_values

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_PATH = REPORTS_DIR / "evaluation_results.json"
DATA_SUMMARY_PATH = REPORTS_DIR / "data_summary.json"

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest-pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Peak exercise ST-segment slope",
    "ca": "Major vessels colored by fluoroscopy",
    "thal": "Thalassemia test result",
}
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "False", 1: "True"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Probable/definite LV hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}
MODEL_COLORS = {
    "Logistic Regression": "#087F8C",
    "Random Forest": "#3867A6",
    "Gradient Boosting": "#E28A51",
}

st.set_page_config(
    page_title="Heartwise | Healthcare Analytics",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #16363c;
      --muted: #61777a;
      --teal: #087f8c;
      --mint: #e9f5f3;
      --paper: #f5f8f7;
      --line: #dce8e5;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Aptos', sans-serif; }
    h1, h2, h3 { font-family: 'Segoe UI', 'Aptos', sans-serif !important; color: var(--ink); }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #102f38; }
    [data-testid="stSidebar"] * { color: #e9f5f3; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .32rem 0; }
    [data-testid="stRadio"] input { accent-color: var(--teal); }
    [data-testid="stMetric"] {
      background: #fff; border: 1px solid var(--line); padding: 1rem 1.15rem;
      border-radius: 14px; box-shadow: 0 4px 14px rgba(22,54,60,.04);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    .hero {
      background: linear-gradient(120deg, #123a43 0%, #087f8c 100%);
      padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
      margin: .25rem 0 1.35rem; box-shadow: 0 10px 26px rgba(8,127,140,.15);
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2rem; }
    .hero p { color: #d8eeea; margin: .45rem 0 0; }
    .section-kicker { color: var(--teal); text-transform: uppercase; letter-spacing: .1em;
      font-size: .74rem; font-weight: 700; margin-bottom: .25rem; }
    .note-card {
      background: white; border: 1px solid var(--line); border-left: 4px solid var(--teal);
      padding: .9rem 1rem; border-radius: 10px; color: var(--muted); margin: .5rem 0 1rem;
    }
    div[data-testid="stForm"] { background: white; padding: 1.2rem; border: 1px solid var(--line); border-radius: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Load the validated local dataset for in-memory dashboard exploration."""
    return load_dataset(DATA_PATH)


@st.cache_data
def get_json(path_string: str) -> dict[str, Any]:
    """Load one generated project report without changing its measured values."""
    with Path(path_string).open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict):
        raise ValueError(f"Expected a JSON object in {path_string}.")
    return report


def report_or_stop(path: Path, label: str) -> dict[str, Any]:
    """Load a required report and render an actionable dashboard error."""
    try:
        return get_json(str(path))
    except FileNotFoundError:
        st.error(f"{label} is missing: `{path}`. Run the documented project pipeline first.")
        st.stop()
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read {label}: {error}")
        st.stop()
    raise RuntimeError("Streamlit stop should have terminated report loading.")


def data_or_stop() -> pd.DataFrame:
    """Load source data, reporting missing or invalid local files explicitly."""
    try:
        return get_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not load the validated Cleveland dataset: {error}")
        st.stop()
    raise RuntimeError("Streamlit stop should have terminated dataset loading.")


def image_or_stop(filename: str, caption: str) -> None:
    """Render a saved project figure or explain how to regenerate it."""
    image_path = FIGURES_DIR / filename
    if not image_path.is_file():
        st.error(
            f"Saved figure `{filename}` is missing. Run `python -m src.train` "
            "to regenerate evaluation figures."
        )
        return
    st.image(str(image_path), caption=caption, width="stretch")


def make_metric_frame(results: dict[str, Any], split: str) -> pd.DataFrame:
    """Shape saved CV or held-out metrics into chart-ready rows."""
    if split == "Held-out test":
        model_metrics = results["held_out_test_metrics"]
    else:
        model_metrics = results["cross_validation"]["models"]
    records = [
        {"Model": name, "Metric": METRIC_LABELS[metric], "Score": values[metric]}
        for name, values in model_metrics.items()
        for metric in METRICS
        if values.get(metric) is not None
    ]
    return pd.DataFrame(records)


def section_header(kicker: str, title: str, description: str) -> None:
    """Render a consistent title block for dashboard sections."""
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.header(title)
    st.caption(description)


def overview_page() -> None:
    summary = report_or_stop(DATA_SUMMARY_PATH, "Data summary")
    results = report_or_stop(EVALUATION_PATH, "Evaluation results")
    selected = results["selected_model"]
    selected_metrics = results["held_out_test_metrics"][selected]
    class_counts = summary["binary_target_counts"]
    total_missing = sum(summary["missing_values"].values())

    section_header(
        "PROJECT OVERVIEW",
        "A closer look at the Cleveland cohort",
        "A reproducible, educational classification project — not a clinical decision tool.",
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Records", f"{summary['rows']:,}")
    metric_columns[1].metric("Dataset fields", summary["columns"])
    metric_columns[2].metric("Positive class", f"{int(class_counts['1']):,}")
    metric_columns[3].metric("Missing feature values", total_missing)
    metric_columns[4].metric("Selected model", selected)

    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.subheader("Cohort balance")
        class_frame = pd.DataFrame(
            [
                {"Outcome": "No disease (0)", "Patients": int(class_counts["0"])},
                {"Outcome": "Disease (1–4)", "Patients": int(class_counts["1"])},
            ]
        )
        class_chart = (
            alt.Chart(class_frame)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                x=alt.X("Patients:Q", title="Patients"),
                y=alt.Y("Outcome:N", sort="-x", title=None),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(
                        domain=["No disease (0)", "Disease (1–4)"],
                        range=["#71b7ae", "#087f8c"],
                    ),
                    legend=None,
                ),
                tooltip=["Outcome:N", "Patients:Q"],
            )
            .properties(height=190)
        )
        st.altair_chart(class_chart, width="stretch")
        st.caption("Binary target mapping: original `num == 0` versus `num in 1–4`.")

    with right:
        st.subheader("Missing-value profile")
        missing_frame = pd.DataFrame(
            [
                {"Feature": feature, "Missing": int(count)}
                for feature, count in summary["missing_values"].items()
                if count > 0
            ]
        )
        if missing_frame.empty:
            st.success("No missing values found in the validated source data.")
        else:
            missing_chart = (
                alt.Chart(missing_frame)
                .mark_bar(color="#e28a51", cornerRadiusEnd=6)
                .encode(
                    x=alt.X("Missing:Q", title="Missing records"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    tooltip=["Feature:N", "Missing:Q"],
                )
                .properties(height=190)
            )
            st.altair_chart(missing_chart, width="stretch")
        st.caption(f"{total_missing} missing feature values are imputed inside model pipelines.")

    st.subheader("Selected model · held-out test")
    metric_columns = st.columns(5)
    for column, metric in zip(metric_columns, METRICS):
        column.metric(METRIC_LABELS[metric], f"{selected_metrics[metric]:.3f}")

    split = results["dataset"]["split"]
    st.markdown(
        f'<div class="note-card">Selected by training-only cross-validation ('
        f'{results["cross_validation"]["folds"]} stratified folds). '
        f'The held-out set contains {split["test_rows"]} records; results are '
        "small-sample estimates, not a performance guarantee.</div>",
        unsafe_allow_html=True,
    )
    image_or_stop("target_distribution.png", "Saved target distribution from the validated source dataset.")


def model_comparison_page() -> None:
    results = report_or_stop(EVALUATION_PATH, "Evaluation results")
    section_header(
        "MODEL EVALUATION",
        "Compare model behavior",
        "All reported scores below come from the saved cross-validation and held-out evaluation results.",
    )
    selected = results["selected_model"]
    st.info(
        f"**Selected model: {selected}.** Selection used mean training-fold F1; "
        "ties are broken by recall and then ROC-AUC. The test set was reserved "
        "for final evaluation."
    )
    split = st.radio(
        "Evaluation view",
        ["Held-out test", "Training cross-validation"],
        horizontal=True,
        help="Cross-validation values are training-fold means; held-out values are final test metrics.",
    )
    metric_frame = make_metric_frame(results, split)
    metric_choice = st.selectbox(
        "Compare metric",
        [METRIC_LABELS[metric] for metric in METRICS],
        index=3,
    )
    selected_frame = metric_frame[metric_frame["Metric"] == metric_choice]
    bars = (
        alt.Chart(selected_frame)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title=metric_choice),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=list(MODEL_COLORS),
                    range=list(MODEL_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=["Model:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=290)
    )
    st.altair_chart(bars, width="stretch")

    comparison = metric_frame.pivot(index="Model", columns="Metric", values="Score")
    st.dataframe(
        comparison.rename(columns={label: label for label in METRIC_LABELS.values()}).style.format("{:.3f}"),
        width="stretch",
    )

    st.subheader("Confusion matrix")
    model_names = list(results["held_out_test_metrics"])
    selected_model = st.selectbox(
        "Model for confusion matrix",
        model_names,
        index=model_names.index(selected),
    )
    matrix = results["held_out_test_metrics"][selected_model]["confusion_matrix"]
    confusion_frame = pd.DataFrame(
        [
            {"Actual": actual, "Predicted": predicted, "Patients": int(matrix[actual][predicted])}
            for actual in range(2)
            for predicted in range(2)
        ]
    )
    cells = (
        alt.Chart(confusion_frame)
        .mark_rect(cornerRadius=5)
        .encode(
            x=alt.X(
                "Predicted:O",
                sort=[0, 1],
                axis=alt.Axis(labelExpr="datum.label == '0' ? 'No disease' : 'Disease'"),
            ),
            y=alt.Y(
                "Actual:O",
                sort=[0, 1],
                axis=alt.Axis(labelExpr="datum.label == '0' ? 'No disease' : 'Disease'"),
            ),
            color=alt.Color("Patients:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["Actual:O", "Predicted:O", "Patients:Q"],
        )
        .properties(width=320, height=260)
    )
    labels = cells.mark_text(fontSize=22, fontWeight="bold", color="#16363c").encode(
        text="Patients:Q"
    )
    st.altair_chart(cells + labels, width="content")
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    st.caption(f"TN {tn} · FP {fp} · FN {fn} · TP {tp} — false negatives and false positives both matter.")

    st.subheader("Threshold-independent views")
    st.caption(
        "These saved curves were generated from the held-out test set and the "
        "same fitted candidate models reported above."
    )
    roc_column, pr_column = st.columns(2, gap="large")
    with roc_column:
        image_or_stop("roc_curves.png", "Held-out ROC curves.")
    with pr_column:
        image_or_stop("precision_recall_curves.png", "Held-out precision-recall curves.")


def explore_data_page() -> None:
    summary = report_or_stop(DATA_SUMMARY_PATH, "Data summary")
    data = data_or_stop()
    section_header(
        "EXPLORATORY ANALYSIS",
        "Explore the source data",
        "Interactive charts are built from the validated local Cleveland CSV; values are not sampled or fabricated.",
    )
    data = data.copy()
    data["target_label"] = data["num"].gt(0).map(
        {False: "No disease (0)", True: "Disease (1–4)"}
    )
    st.caption(f"{summary['rows']} records · {len(data.columns) - 1} source fields plus the derived target label")

    features = list(FEATURE_LABELS)
    feature = st.selectbox(
        "Feature to explore",
        features,
        format_func=lambda name: FEATURE_LABELS[name],
    )
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("Feature distribution")
        if feature in NUMERIC_FEATURES:
            distribution = (
                alt.Chart(data)
                .transform_filter(alt.datum[feature] != None)
                .mark_bar(opacity=0.76)
                .encode(
                    x=alt.X(
                        f"{feature}:Q",
                        bin=alt.Bin(maxbins=24),
                        title=FEATURE_LABELS[feature],
                    ),
                    y=alt.Y("count():Q", title="Records"),
                    color=alt.Color(
                        "target_label:N",
                        title="Observed target",
                        scale=alt.Scale(
                            domain=["No disease (0)", "Disease (1–4)"],
                            range=["#71b7ae", "#087f8c"],
                        ),
                    ),
                    tooltip=["target_label:N", "count():Q"],
                )
                .properties(height=300)
            )
        else:
            feature_data = data.dropna(subset=[feature])
            feature_data = feature_data.assign(
                category=feature_data[feature].map(
                    lambda value: CATEGORY_LABELS.get(feature, {}).get(
                        int(value), str(value)
                    )
                )
            )
            distribution = (
                alt.Chart(feature_data)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("count():Q", title="Records"),
                    y=alt.Y("category:N", sort="-x", title=FEATURE_LABELS[feature]),
                    color=alt.Color(
                        "target_label:N",
                        title="Observed target",
                        scale=alt.Scale(
                            domain=["No disease (0)", "Disease (1–4)"],
                            range=["#71b7ae", "#087f8c"],
                        ),
                    ),
                    tooltip=["category:N", "target_label:N", "count():Q"],
                )
                .properties(height=300)
            )
        st.altair_chart(distribution, width="stretch")
    with right:
        st.subheader("Feature by target")
        if feature in NUMERIC_FEATURES:
            comparison = (
                alt.Chart(data)
                .mark_boxplot(extent="min-max")
                .encode(
                    x=alt.X("target_label:N", title="Observed target"),
                    y=alt.Y(f"{feature}:Q", title=FEATURE_LABELS[feature]),
                    color=alt.Color(
                        "target_label:N",
                        legend=None,
                        scale=alt.Scale(
                            domain=["No disease (0)", "Disease (1–4)"],
                            range=["#71b7ae", "#087f8c"],
                        ),
                    ),
                    tooltip=["target_label:N"],
                )
                .properties(height=300)
            )
        else:
            category_data = data.copy()
            category_data["category"] = category_data[feature].map(
                lambda value: (
                    "Missing"
                    if pd.isna(value)
                    else CATEGORY_LABELS.get(feature, {}).get(int(value), str(value))
                )
            )
            comparison = (
                alt.Chart(category_data)
                .mark_bar()
                .encode(
                    x=alt.X("target_label:N", title="Observed target"),
                    y=alt.Y("count():Q", title="Records"),
                    color=alt.Color(
                        "category:N",
                        title=FEATURE_LABELS[feature],
                        scale=alt.Scale(scheme="tableau10"),
                    ),
                    tooltip=["target_label:N", "category:N", "count():Q"],
                )
                .properties(height=300)
            )
        st.altair_chart(comparison, width="stretch")

    st.subheader("Continuous-feature correlation")
    st.caption(
        "Category codes are excluded from this correlation view to avoid treating "
        "nominal categories as continuous measurements."
    )
    numeric_data = data[NUMERIC_FEATURES].copy()
    numeric_data["Binary target"] = data["num"].gt(0).astype(int)
    correlation = (
        numeric_data.corr()
        .rename_axis("Feature")
        .reset_index()
        .melt(id_vars="Feature", var_name="Compared with", value_name="Correlation")
    )
    heatmap = (
        alt.Chart(correlation)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Compared with:N", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Feature:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="Pearson r",
            ),
            tooltip=[
                "Feature:N",
                "Compared with:N",
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(height=310)
    )
    heatmap_text = heatmap.mark_text(fontSize=11).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.Correlation) > 0.55",
            alt.value("white"),
            alt.value("#16363c"),
        ),
    )
    st.altair_chart(heatmap + heatmap_text, width="stretch")
    st.subheader("Missing values")
    missing = pd.DataFrame(
        [
            {"Feature": name, "Missing records": int(count)}
            for name, count in summary["missing_values"].items()
        ]
    ).sort_values("Missing records", ascending=False)
    st.dataframe(missing, width="stretch", hide_index=True)


def explainability_page() -> None:
    section_header(
        "MODEL INTERPRETATION",
        "What the models learned",
        "Saved feature-importance figures provide descriptive model insights, not clinical explanations.",
    )
    st.warning(
        "**Feature importance does not imply causation.** Importance describes "
        "how a fitted model used these variables on this dataset; it is not a "
        "causal effect, medical explanation, or evidence of clinical relevance."
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Tree-based model importance")
        image_or_stop(
            "tree_feature_importance.png",
            "Saved feature importance from the fitted Random Forest and Gradient Boosting models.",
        )
    with right:
        st.subheader("Permutation importance")
        image_or_stop(
            "permutation_importance.png",
            "Saved held-out permutation importance for the selected model.",
        )
    st.caption(
        "These are existing evaluation artifacts from the recorded experiment. "
        "The dashboard does not refit or modify any model."
    )


def prediction_page() -> None:
    section_header(
        "EDUCATIONAL DEMO",
        "Try the saved model",
        "Inputs are processed by the same fitted preprocessing pipeline and in the same feature order used at training time.",
    )
    st.warning(
        "**Educational use only — not a medical diagnosis.** This model is not "
        "clinically validated, cannot safely rule out heart disease, and must "
        "not guide care. Seek qualified medical advice for health concerns."
    )
    st.caption(
        "Submitted inputs are used only for this session's prediction. "
        "The app does not save or persist form values by default."
    )
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return
    st.caption(
        f"Selected model: **{bundle['selected_model']}**. Any displayed "
        "probability is an uncalibrated model estimate, not clinical risk."
    )
    with st.form("prediction_form"):
        st.subheader("Enter dataset features")
        st.caption(
            "Numeric controls are restricted to the training-subset range. "
            "Category choices are codes observed in model training data."
        )
        inputs: dict[str, Any] = {}
        left, right = st.columns(2)
        for index, feature_name in enumerate(bundle["feature_columns"]):
            column = left if index % 2 == 0 else right
            with column:
                label = FEATURE_LABELS[feature_name]
                if feature_name in bundle["categorical_values"]:
                    options = bundle["categorical_values"][feature_name]
                    inputs[feature_name] = st.selectbox(
                        label,
                        options=options,
                        format_func=lambda value, name=feature_name: (
                            f"{value} — {CATEGORY_LABELS.get(name, {}).get(value, str(value))}"
                        ),
                    )
                else:
                    bounds = bundle["numeric_ranges"][feature_name]
                    step = 0.1 if feature_name == "oldpeak" else 1.0
                    inputs[feature_name] = st.number_input(
                        label,
                        min_value=float(bounds["min"]),
                        max_value=float(bounds["max"]),
                        value=float(bounds["default"]),
                        step=step,
                        format="%.1f" if feature_name == "oldpeak" else "%.0f",
                    )
        submitted = st.form_submit_button("Run educational prediction", type="primary")

    if submitted:
        try:
            result = predict_from_values(inputs, bundle)
        except ValueError as error:
            st.error(f"Invalid input: {error}")
            return
        if result["prediction"] == 1:
            st.error("Model output: disease-positive class (original target 1–4).")
        else:
            st.info("Model output: disease-negative class (original target 0).")
        st.metric(
            "Uncalibrated model estimate for the positive class",
            f"{result['probability']:.1%}",
        )
        st.warning(
            "This output is not a diagnosis. A negative prediction does not rule "
            "out disease; consult a qualified clinician for medical concerns."
        )


st.sidebar.markdown(
    "<div style='font-family:Segoe UI,sans-serif;font-size:1.35rem;font-weight:800;'>"
    "♥ &nbsp; HEARTWISE</div>"
    "<div style='font-size:.8rem;opacity:.72;margin:.15rem 0 1.25rem;'>"
    "Healthcare analytics · student project</div>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "DASHBOARD",
    ["Overview", "Model comparison", "Explore data", "Explainability", "Predict"],
    label_visibility="visible",
)
st.sidebar.divider()
st.sidebar.caption("UCI Cleveland · 303 records")
st.sidebar.caption("Educational use only · Not a diagnostic tool")

st.markdown(
    '<div class="hero"><h1>Heart disease analytics</h1>'
    "<p>Transparent machine learning, evaluated on the Cleveland cohort.</p></div>",
    unsafe_allow_html=True,
)

if page == "Overview":
    overview_page()
elif page == "Model comparison":
    model_comparison_page()
elif page == "Explore data":
    explore_data_page()
elif page == "Explainability":
    explainability_page()
else:
    prediction_page()

st.divider()
st.caption(
    "UCI Heart Disease dataset · DOI 10.24432/C52P4X · "
    "Historical small-cohort analysis; metrics are not clinical validation."
)
