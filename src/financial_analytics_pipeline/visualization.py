from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiments import ExperimentResults


def generate_visualizations(
    raw_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    experiment_results: ExperimentResults,
    outputs_dir: Path,
) -> dict[str, str]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dashboard": str(_build_dashboard(feature_frame, experiment_results, outputs_dir)),
        "amount_distribution": str(_build_amount_distribution(raw_frame, cleaned_frame, outputs_dir)),
        "feature_importance": str(_build_feature_importance_plot(experiment_results, outputs_dir)),
        "regression_comparison": str(_build_regression_comparison(experiment_results, outputs_dir)),
        "confusion_matrix": str(_build_confusion_matrix(experiment_results, outputs_dir)),
    }
    return paths


def _build_dashboard(
    feature_frame: pd.DataFrame,
    experiment_results: ExperimentResults,
    outputs_dir: Path,
) -> Path:
    class_counts = feature_frame["class"].value_counts().sort_index()
    classification_frame = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                "f1": result["test"]["f1"],
                "recall": result["test"]["recall"],
                "precision": result["test"]["precision"],
                "roc_auc": result["test"]["roc_auc"],
            }
            for result in experiment_results.classification_results
        ]
    )
    best_classifier = experiment_results.best_classifier
    roc_curve = best_classifier["test"].get("roc_curve", {})
    cluster_frame = experiment_results.cluster_frame

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Class Balance",
            "Classifier F1 Scores",
            "Best ROC Curve",
            "PCA Cluster View",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
    )
    figure.add_trace(
        go.Bar(x=["Non-Fraud", "Fraud"], y=class_counts.tolist(), marker_color=["#5b8ff9", "#e8684a"]),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=classification_frame["model_name"],
            y=classification_frame["f1"],
            marker_color="#4d9f70",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=roc_curve.get("fpr", [0, 1]),
            y=roc_curve.get("tpr", [0, 1]),
            mode="lines",
            line=dict(color="#c44e52", width=3),
            name=best_classifier["model_name"],
        ),
        row=2,
        col=1,
    )
    for cluster_id in sorted(cluster_frame["kmeans_cluster"].unique()):
        cluster_slice = cluster_frame[cluster_frame["kmeans_cluster"] == cluster_id]
        figure.add_trace(
            go.Scatter(
                x=cluster_slice["pca_1"],
                y=cluster_slice["pca_2"],
                mode="markers",
                marker=dict(size=7, opacity=0.75),
                name=f"Cluster {cluster_id}",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    figure.update_layout(
        title="Credit Card Fraud Analytics Dashboard",
        template="plotly_white",
        height=900,
        width=1300,
    )
    destination = outputs_dir / "dashboard.html"
    figure.write_html(destination)
    return destination


def _build_amount_distribution(
    raw_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    outputs_dir: Path,
) -> Path:
    raw_display = raw_frame.assign(stage="raw", amount_display=raw_frame["amount"].clip(upper=raw_frame["amount"].quantile(0.995)))
    cleaned_display = cleaned_frame.assign(
        stage="cleaned",
        amount_display=cleaned_frame["amount"].clip(upper=cleaned_frame["amount"].quantile(0.995)),
    )
    combined = pd.concat([raw_display, cleaned_display], ignore_index=True)
    figure = px.histogram(
        combined,
        x="amount_display",
        color="stage",
        barmode="overlay",
        nbins=80,
        title="Transaction Amount Distribution Before and After Cleaning",
        template="plotly_white",
    )
    destination = outputs_dir / "amount_distribution.html"
    figure.write_html(destination)
    return destination


def _build_feature_importance_plot(
    experiment_results: ExperimentResults,
    outputs_dir: Path,
) -> Path:
    importance_frame = experiment_results.feature_importance_frame
    if importance_frame.empty:
        figure = go.Figure()
        figure.add_annotation(text="Best classifier does not expose feature importances.", showarrow=False)
    else:
        figure = px.bar(
            importance_frame.sort_values("importance"),
            x="importance",
            y="feature_name",
            orientation="h",
            title="Top Feature Importances",
            template="plotly_white",
        )
    destination = outputs_dir / "feature_importance.html"
    figure.write_html(destination)
    return destination


def _build_regression_comparison(
    experiment_results: ExperimentResults,
    outputs_dir: Path,
) -> Path:
    regression_frame = pd.DataFrame(
        [
            {
                "model_name": result["model_name"],
                "rmse": result["test"]["rmse"],
                "r2": result["test"]["r2"],
            }
            for result in experiment_results.regression_results
        ]
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Regression RMSE", "Regression R2"),
    )
    figure.add_trace(
        go.Bar(x=regression_frame["model_name"], y=regression_frame["rmse"], marker_color="#dd8452"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=regression_frame["model_name"], y=regression_frame["r2"], marker_color="#55a868"),
        row=1,
        col=2,
    )
    figure.update_layout(title="Regression Baselines for Log Transaction Amount", template="plotly_white")
    destination = outputs_dir / "regression_comparison.html"
    figure.write_html(destination)
    return destination


def _build_confusion_matrix(
    experiment_results: ExperimentResults,
    outputs_dir: Path,
) -> Path:
    confusion = experiment_results.best_classifier["test"]["confusion_matrix"]
    figure = px.imshow(
        confusion,
        text_auto=True,
        x=["Predicted Non-Fraud", "Predicted Fraud"],
        y=["Actual Non-Fraud", "Actual Fraud"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {experiment_results.best_classifier['model_name']}",
    )
    destination = outputs_dir / "confusion_matrix.html"
    figure.write_html(destination)
    return destination
