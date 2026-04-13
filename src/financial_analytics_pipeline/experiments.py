from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_pipeline import PCA_COLUMNS
from .models import (
    AdaBoostClassifierScratch,
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    GradientBoostingRegressorScratch,
    HierarchicalClusteringScratch,
    KMeansScratch,
    KNNClassifierScratch,
    LinearRegressionScratch,
    LinearSVMClassifierScratch,
    LogisticRegressionScratch,
    PCAScratch,
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
    StandardScalerScratch,
    binary_classification_metrics,
    regression_metrics,
)
from .models.utils import expanding_window_split

CLASSIFICATION_FEATURE_COLUMNS = [
    "time_seconds",
    "amount",
    "log_amount",
    "amount_zscore",
    "amount_clipped",
    "hour_of_day",
    "hour_bucket",
    "hour_sin",
    "hour_cos",
    "amount_hour_mean_ratio",
    "amount_rolling_mean_50",
    "amount_rolling_std_50",
    "amount_velocity",
    "time_delta",
    "v_magnitude",
    "pca_abs_mean",
    "pca_positive_share",
    *PCA_COLUMNS,
]

REGRESSION_FEATURE_COLUMNS = [
    "time_seconds",
    "hour_of_day",
    "hour_bucket",
    "hour_sin",
    "hour_cos",
    "time_delta",
    "v_magnitude",
    "pca_abs_mean",
    "pca_positive_share",
    *PCA_COLUMNS,
]


@dataclass
class ExperimentResults:
    classification_results: list[dict[str, Any]]
    regression_results: list[dict[str, Any]]
    best_classifier: dict[str, Any]
    feature_importance_frame: pd.DataFrame
    cluster_frame: pd.DataFrame


def run_experiments(feature_frame: pd.DataFrame, config: PipelineConfig) -> ExperimentResults:
    classification_results = _run_classification_experiments(feature_frame, config)
    regression_results = _run_regression_experiments(feature_frame, config)
    best_classifier = max(classification_results, key=lambda result: result["test"]["f1"])
    feature_importance_frame = _extract_feature_importance(best_classifier)
    cluster_frame = _run_unsupervised_analysis(feature_frame, config)
    return ExperimentResults(
        classification_results=classification_results,
        regression_results=regression_results,
        best_classifier=best_classifier,
        feature_importance_frame=feature_importance_frame,
        cluster_frame=cluster_frame,
    )


def _run_classification_experiments(
    feature_frame: pd.DataFrame,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    train_frame, test_frame = _time_split(feature_frame, config.test_size)
    train_frame = _undersample_majority_class(train_frame, config)

    x_train = train_frame.loc[:, CLASSIFICATION_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_frame["class"].to_numpy(dtype=int)
    x_test = test_frame.loc[:, CLASSIFICATION_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_frame["class"].to_numpy(dtype=int)

    scaler = StandardScalerScratch()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    jobs = [
        (
            model_name,
            x_train_scaled,
            y_train,
            x_test_scaled,
            y_test,
            config,
        )
        for model_name in _classification_model_names()
    ]
    max_workers = _resolve_worker_count(config.max_parallel_workers, len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_classification_worker, jobs))


def _run_regression_experiments(
    feature_frame: pd.DataFrame,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    working = feature_frame.sort_values("time_seconds").reset_index(drop=True)
    max_rows = min(len(working), 40_000)
    working = working.iloc[:max_rows].copy()
    train_frame, test_frame = _time_split(working, config.test_size)

    x_train = train_frame.loc[:, REGRESSION_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_frame["log_amount"].to_numpy(dtype=float)
    x_test = test_frame.loc[:, REGRESSION_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_frame["log_amount"].to_numpy(dtype=float)

    scaler = StandardScalerScratch()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    jobs = [
        (
            model_name,
            x_train_scaled,
            y_train,
            x_test_scaled,
            y_test,
            config,
        )
        for model_name in _regression_model_names()
    ]
    max_workers = _resolve_worker_count(config.max_parallel_workers, len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_regression_worker, jobs))


def _run_unsupervised_analysis(feature_frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    fraud_rows = feature_frame[feature_frame["class"] == 1]
    normal_rows = feature_frame[feature_frame["class"] == 0]
    rng = np.random.default_rng(config.random_state)
    fraud_sample_size = min(len(fraud_rows), max(config.cluster_sample_size // 3, 1))
    normal_sample_size = max(config.cluster_sample_size - fraud_sample_size, 1)
    if len(fraud_rows) > fraud_sample_size:
        fraud_indices = rng.choice(
            fraud_rows.index.to_numpy(),
            size=fraud_sample_size,
            replace=False,
        )
        fraud_rows = fraud_rows.loc[fraud_indices]
    if len(normal_rows) > normal_sample_size:
        normal_indices = rng.choice(
            normal_rows.index.to_numpy(),
            size=normal_sample_size,
            replace=False,
        )
        normal_rows = normal_rows.loc[normal_indices]
    sample_frame = (
        pd.concat([fraud_rows, normal_rows], ignore_index=False)
        .sort_values("time_seconds")
        .head(config.cluster_sample_size)
        .reset_index(drop=True)
    )

    unsupervised_columns = [
        "log_amount",
        "amount_zscore",
        "hour_of_day",
        "amount_rolling_mean_50",
        "amount_rolling_std_50",
        "v_magnitude",
        "pca_abs_mean",
        "pca_positive_share",
        *PCA_COLUMNS[:8],
    ]
    features = sample_frame.loc[:, unsupervised_columns].to_numpy(dtype=float)
    scaler = StandardScalerScratch()
    scaled_features = scaler.fit_transform(features)

    pca = PCAScratch(n_components=2)
    embedding = pca.fit_transform(scaled_features)
    kmeans = KMeansScratch(n_clusters=3, random_state=config.random_state)
    kmeans_labels = kmeans.fit_predict(embedding)
    hierarchical = HierarchicalClusteringScratch(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(embedding[: min(len(embedding), config.cluster_sample_size)])

    cluster_frame = sample_frame.loc[:, ["source_row_id", "class", "amount", "time_seconds"]].copy()
    cluster_frame["pca_1"] = embedding[:, 0]
    cluster_frame["pca_2"] = embedding[:, 1]
    cluster_frame["kmeans_cluster"] = kmeans_labels
    cluster_frame["hierarchical_cluster"] = -1
    cluster_frame.loc[: len(hierarchical_labels) - 1, "hierarchical_cluster"] = hierarchical_labels
    explained_variance = pca.explained_variance_ratio_ if pca.explained_variance_ratio_ is not None else np.zeros(2)
    cluster_frame["pca_explained_variance_1"] = explained_variance[0] if len(explained_variance) else 0.0
    cluster_frame["pca_explained_variance_2"] = explained_variance[1] if len(explained_variance) > 1 else 0.0
    return cluster_frame


def _time_split(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("time_seconds").reset_index(drop=True)
    split_index = int(len(ordered) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(ordered) - 1))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def _undersample_majority_class(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    fraud_rows = frame[frame["class"] == 1]
    normal_rows = frame[frame["class"] == 0]
    if len(normal_rows) <= config.training_majority_limit:
        return frame
    rng = np.random.default_rng(config.random_state)
    sampled_indices = rng.choice(
        normal_rows.index.to_numpy(),
        size=config.training_majority_limit,
        replace=False,
    )
    sampled_normals = normal_rows.loc[sampled_indices]
    return (
        pd.concat([fraud_rows, sampled_normals], ignore_index=False)
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )


def _extract_feature_importance(best_classifier: dict[str, Any]) -> pd.DataFrame:
    importances = best_classifier.get("feature_importances")
    feature_names = best_classifier.get("feature_names", [])
    if importances is None:
        return pd.DataFrame(columns=["feature_name", "importance"])
    importance_frame = pd.DataFrame(
        {"feature_name": feature_names, "importance": np.asarray(importances, dtype=float)}
    )
    return importance_frame.sort_values("importance", ascending=False).head(15).reset_index(drop=True)


def _classification_model_names() -> list[str]:
    return [
        "LogisticRegressionScratch",
        "DecisionTreeClassifierScratch",
        "RandomForestClassifierScratch",
        "LinearSVMClassifierScratch",
        "KNNClassifierScratch",
        "AdaBoostClassifierScratch",
    ]


def _regression_model_names() -> list[str]:
    return [
        "LinearRegressionScratch",
        "DecisionTreeRegressorScratch",
        "RandomForestRegressorScratch",
        "GradientBoostingRegressorScratch",
    ]


def _classification_worker(job: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PipelineConfig]) -> dict[str, Any]:
    model_name, x_train_scaled, y_train, x_test_scaled, y_test, config = job
    model = _build_classification_model(model_name, config)
    cv_scores = []
    for fold_train, fold_valid in expanding_window_split(len(x_train_scaled), config.cross_validation_splits):
        cv_model = _build_classification_model(model_name, config)
        cv_model.fit(x_train_scaled[fold_train], y_train[fold_train])
        predictions = cv_model.predict(x_train_scaled[fold_valid])
        fold_metrics = binary_classification_metrics(y_train[fold_valid], predictions)
        cv_scores.append(fold_metrics["f1"])

    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test_scaled)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x_test_scaled)
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    else:
        scores = predictions.astype(float)
    test_metrics = binary_classification_metrics(y_test, predictions, scores)
    return {
        "model_name": model_name,
        "task_type": "classification",
        "cv_f1": float(np.mean(cv_scores)) if cv_scores else 0.0,
        "test": test_metrics,
        "feature_names": CLASSIFICATION_FEATURE_COLUMNS,
        "feature_importances": getattr(model, "feature_importances_", None),
    }


def _regression_worker(job: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PipelineConfig]) -> dict[str, Any]:
    model_name, x_train_scaled, y_train, x_test_scaled, y_test, config = job
    model = _build_regression_model(model_name, config)
    cv_scores = []
    for fold_train, fold_valid in expanding_window_split(len(x_train_scaled), config.cross_validation_splits):
        cv_model = _build_regression_model(model_name, config)
        cv_model.fit(x_train_scaled[fold_train], y_train[fold_train])
        predictions = cv_model.predict(x_train_scaled[fold_valid])
        fold_metrics = regression_metrics(y_train[fold_valid], predictions)
        cv_scores.append(fold_metrics["r2"])

    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)
    test_metrics = regression_metrics(y_test, predictions)
    return {
        "model_name": model_name,
        "task_type": "regression",
        "cv_r2": float(np.mean(cv_scores)) if cv_scores else 0.0,
        "test": test_metrics,
    }


def _build_classification_model(model_name: str, config: PipelineConfig) -> Any:
    registry: dict[str, Callable[[], Any]] = {
        "LogisticRegressionScratch": lambda: LogisticRegressionScratch(
            learning_rate=0.03,
            n_iterations=650,
            class_weight=config.fraud_weight,
            l2_penalty=0.0005,
        ),
        "DecisionTreeClassifierScratch": lambda: DecisionTreeClassifierScratch(
            max_depth=5,
            min_samples_split=20,
            random_state=config.random_state,
        ),
        "RandomForestClassifierScratch": lambda: RandomForestClassifierScratch(
            n_estimators=15,
            max_depth=6,
            min_samples_split=20,
            random_state=config.random_state,
        ),
        "LinearSVMClassifierScratch": lambda: LinearSVMClassifierScratch(
            learning_rate=0.001,
            n_iterations=500,
            regularization_strength=0.01,
            class_weight=config.fraud_weight,
        ),
        "KNNClassifierScratch": lambda: KNNClassifierScratch(k=7),
        "AdaBoostClassifierScratch": lambda: AdaBoostClassifierScratch(n_estimators=20),
    }
    return registry[model_name]()


def _build_regression_model(model_name: str, config: PipelineConfig) -> Any:
    registry: dict[str, Callable[[], Any]] = {
        "LinearRegressionScratch": lambda: LinearRegressionScratch(
            learning_rate=0.01,
            n_iterations=600,
            l2_penalty=0.0001,
        ),
        "DecisionTreeRegressorScratch": lambda: DecisionTreeRegressorScratch(
            max_depth=5,
            min_samples_split=20,
            random_state=config.random_state,
        ),
        "RandomForestRegressorScratch": lambda: RandomForestRegressorScratch(
            n_estimators=12,
            max_depth=5,
            min_samples_split=25,
            random_state=config.random_state,
        ),
        "GradientBoostingRegressorScratch": lambda: GradientBoostingRegressorScratch(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=20,
            random_state=config.random_state,
        ),
    }
    return registry[model_name]()


def _resolve_worker_count(requested_workers: int, job_count: int) -> int:
    available_cpus = os.cpu_count() or 1
    return max(1, min(requested_workers, job_count, available_cpus))
