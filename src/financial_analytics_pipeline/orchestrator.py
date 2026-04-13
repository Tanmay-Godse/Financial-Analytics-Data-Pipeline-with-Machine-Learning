from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_pipeline import PipelineArtifacts, build_data_pipeline
from .database import DatabaseManager
from .experiments import ExperimentResults, run_experiments
from .visualization import generate_visualizations


def run_data_pipeline(config: PipelineConfig | None = None) -> PipelineArtifacts:
    config = config or PipelineConfig()
    config.ensure_directories()
    database = DatabaseManager(config.database_path, config.schema_path)
    database.initialize_schema()

    artifacts = build_data_pipeline(config)
    _persist_pipeline_outputs(artifacts, database, config)
    return artifacts


def run_training_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    config = config or PipelineConfig()
    artifacts = run_data_pipeline(config)
    database = DatabaseManager(config.database_path, config.schema_path)
    experiment_results = run_experiments(artifacts.feature_frame, config)
    experiment_results.cluster_frame.to_csv(config.cluster_snapshot_path, index=False)
    visualization_paths = generate_visualizations(
        artifacts.raw_frame,
        artifacts.cleaned_frame,
        artifacts.feature_frame,
        experiment_results,
        config.outputs_dir,
    )
    summary = _build_training_summary(experiment_results, visualization_paths)
    config.model_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _persist_model_results(database, experiment_results, config.model_summary_path)
    return summary


def refresh_dashboard_loop(
    config: PipelineConfig | None = None,
    *,
    once: bool = False,
    interval_minutes: int | None = None,
) -> None:
    config = config or PipelineConfig()
    interval = interval_minutes or config.refresh_interval_minutes
    while True:
        run_training_pipeline(config)
        if once:
            return
        time.sleep(interval * 60)


def run_data_pipeline_cli() -> None:
    run_data_pipeline(PipelineConfig())


def run_training_cli() -> None:
    run_training_pipeline(PipelineConfig())


def run_refresh_cli() -> None:
    refresh_dashboard_loop(PipelineConfig(), once=True)


def _persist_pipeline_outputs(
    artifacts: PipelineArtifacts,
    database: DatabaseManager,
    config: PipelineConfig,
) -> None:
    raw_table = artifacts.raw_frame.copy()
    raw_table["source_name"] = artifacts.source_name
    database.replace_table("raw_credit_transactions", raw_table)
    database.replace_table("cleaned_credit_transactions", artifacts.cleaned_frame)
    database.replace_table("feature_store", artifacts.feature_frame)

    artifacts.raw_frame.to_csv(config.raw_snapshot_path, index=False)
    artifacts.cleaned_frame.to_csv(config.cleaned_snapshot_path, index=False)
    artifacts.feature_frame.to_csv(config.feature_snapshot_path, index=False)

    quality = artifacts.validation_report["quality"]
    database.append_record(
        "pipeline_runs",
        {
            "source_name": artifacts.source_name,
            "raw_rows": quality["raw_rows"],
            "cleaned_rows": quality["cleaned_rows"],
            "feature_rows": int(len(artifacts.feature_frame)),
            "duplicate_rows_removed": quality["duplicate_rows_removed"],
            "missing_value_count": quality["missing_value_count"],
            "fraud_rate": quality["fraud_rate"],
            "validation_report_path": str(config.validation_report_path),
        },
    )


def _persist_model_results(
    database: DatabaseManager,
    experiment_results: ExperimentResults,
    summary_path: Path,
) -> None:
    for result in experiment_results.classification_results:
        database.append_record(
            "model_runs",
            {
                "experiment_name": "credit_card_fraud_detection",
                "model_name": result["model_name"],
                "task_type": "classification",
                "cv_metric": result["cv_f1"],
                "test_metric": result["test"]["accuracy"],
                "precision_value": result["test"]["precision"],
                "recall_value": result["test"]["recall"],
                "f1_value": result["test"]["f1"],
                "roc_auc_value": result["test"]["roc_auc"],
                "rmse_value": None,
                "mae_value": None,
                "r2_value": None,
                "artifacts_path": str(summary_path),
            },
        )
    for result in experiment_results.regression_results:
        database.append_record(
            "model_runs",
            {
                "experiment_name": "credit_card_amount_regression",
                "model_name": result["model_name"],
                "task_type": "regression",
                "cv_metric": result["cv_r2"],
                "test_metric": result["test"]["r2"],
                "precision_value": None,
                "recall_value": None,
                "f1_value": None,
                "roc_auc_value": None,
                "rmse_value": result["test"]["rmse"],
                "mae_value": result["test"]["mae"],
                "r2_value": result["test"]["r2"],
                "artifacts_path": str(summary_path),
            },
        )


def _build_training_summary(
    experiment_results: ExperimentResults,
    visualization_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "classification_results": [_serialize_for_json(result) for result in experiment_results.classification_results],
        "regression_results": [_serialize_for_json(result) for result in experiment_results.regression_results],
        "best_classifier": _serialize_for_json(experiment_results.best_classifier),
        "feature_importance": _serialize_for_json(experiment_results.feature_importance_frame),
        "cluster_sample": _serialize_for_json(experiment_results.cluster_frame.head(50)),
        "visualizations": visualization_paths,
    }


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    return value
