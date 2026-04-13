from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from .config import PipelineConfig

PCA_COLUMNS = [f"V{index}" for index in range(1, 29)]


@dataclass(slots=True)
class PipelineArtifacts:
    source_name: str
    raw_frame: pd.DataFrame
    cleaned_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    validation_report: dict[str, Any]


def load_credit_card_data(config: PipelineConfig) -> tuple[pd.DataFrame, str]:
    """Load the local Kaggle CSV when available, otherwise fall back to an OpenML mirror."""

    if config.local_csv_path.exists():
        frame = pd.read_csv(config.local_csv_path)
        return _normalize_columns(frame), "local_kaggle_csv"

    kaggle_frame = _try_kaggle_download(config)
    if kaggle_frame is not None:
        return kaggle_frame, "kaggle_api"

    openml_frame = _load_openml_mirror(config)
    openml_frame.to_csv(config.raw_snapshot_path, index=False)
    return openml_frame, "openml_mirror_of_kaggle_creditcardfraud"


def _try_kaggle_download(config: PipelineConfig) -> pd.DataFrame | None:
    has_env_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file_credentials = any(
        candidate.exists()
        for candidate in (
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".config" / "kaggle" / "kaggle.json",
        )
    )
    if not has_env_credentials and not has_file_credentials:
        return None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(
            config.kaggle_dataset,
            config.kaggle_filename,
            path=str(config.raw_dir),
            force=False,
            quiet=True,
        )
        zipped_path = config.raw_dir / f"{config.kaggle_filename}.zip"
        if zipped_path.exists():
            import zipfile

            with zipfile.ZipFile(zipped_path) as archive:
                archive.extractall(config.raw_dir)
        if config.local_csv_path.exists():
            return _normalize_columns(pd.read_csv(config.local_csv_path))
    except Exception:
        return None

    return None


def _load_openml_mirror(config: PipelineConfig) -> pd.DataFrame:
    dataset = fetch_openml(data_id=config.openml_data_id, as_frame=True)
    frame = dataset.frame.copy()
    return _normalize_columns(frame)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "Time": "time_seconds",
            "time": "time_seconds",
            "Amount": "amount",
            "amount": "amount",
            "Class": "class",
            "class": "class",
        }
    )
    if "time_seconds" not in renamed.columns:
        renamed["time_seconds"] = np.arange(len(renamed), dtype=float)
    numeric_columns = ["time_seconds", "amount", "class", *PCA_COLUMNS]
    for column in numeric_columns:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["class"] = renamed["class"].fillna(0).astype(int)
    renamed.insert(0, "source_row_id", np.arange(len(renamed), dtype=int))
    ordered_columns = ["source_row_id", "time_seconds", "amount", "class", *PCA_COLUMNS]
    return renamed.loc[:, ordered_columns]


def clean_credit_card_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.copy()
    duplicate_subset = ["time_seconds", "amount", "class", *PCA_COLUMNS]
    duplicate_rows = int(working.duplicated(subset=duplicate_subset).sum())
    missing_values = int(working.isna().sum().sum())

    working = (
        working.drop_duplicates(subset=duplicate_subset)
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )
    working["source_row_id"] = np.arange(len(working), dtype=int)

    numeric_columns = ["time_seconds", "amount", *PCA_COLUMNS]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working[column] = working[column].fillna(working[column].median())

    working["time_seconds"] = working["time_seconds"].clip(lower=0)
    working["amount"] = working["amount"].clip(lower=0)
    working["class"] = working["class"].fillna(0).astype(int)

    quality = {
        "raw_rows": int(len(frame)),
        "cleaned_rows": int(len(working)),
        "duplicate_rows_removed": duplicate_rows,
        "missing_value_count": missing_values,
        "fraud_rate": float(working["class"].mean()),
    }
    return working, quality


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy().sort_values("time_seconds").reset_index(drop=True)
    amount_mean = working["amount"].mean()
    amount_std = working["amount"].std(ddof=0) or 1.0
    clipped_upper = working["amount"].quantile(0.995)

    working["log_amount"] = np.log1p(working["amount"])
    working["amount_zscore"] = (working["amount"] - amount_mean) / amount_std
    working["amount_clipped"] = working["amount"].clip(upper=clipped_upper)
    working["hour_of_day"] = (working["time_seconds"] / 3600.0) % 24.0
    working["hour_bucket"] = np.floor(working["hour_of_day"]).astype(int)
    working["hour_sin"] = np.sin(2 * np.pi * working["hour_of_day"] / 24.0)
    working["hour_cos"] = np.cos(2 * np.pi * working["hour_of_day"] / 24.0)

    hourly_mean = working.groupby("hour_bucket")["amount"].transform("mean")
    working["amount_hour_mean_ratio"] = working["amount"] / (hourly_mean + 1e-6)
    working["amount_rolling_mean_50"] = working["amount"].rolling(50, min_periods=5).mean().bfill()
    working["amount_rolling_std_50"] = (
        working["amount"].rolling(50, min_periods=5).std(ddof=0).fillna(0.0)
    )
    working["amount_velocity"] = working["amount"].diff().fillna(0.0)
    working["time_delta"] = working["time_seconds"].diff().clip(lower=0).fillna(0.0)
    working["v_magnitude"] = np.sqrt((working[PCA_COLUMNS] ** 2).sum(axis=1))
    working["pca_abs_mean"] = working[PCA_COLUMNS].abs().mean(axis=1)
    working["pca_positive_share"] = (working[PCA_COLUMNS] > 0).mean(axis=1)
    return working


def validate_pipeline_frames(
    raw_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    report = {
        "raw": {
            "rows": int(len(raw_frame)),
            "columns": raw_frame.columns.tolist(),
            "fraud_count": int(raw_frame["class"].sum()),
            "null_values": int(raw_frame.isna().sum().sum()),
        },
        "cleaned": {
            "rows": int(len(cleaned_frame)),
            "duplicate_rows": int(cleaned_frame.duplicated().sum()),
            "null_values": int(cleaned_frame.isna().sum().sum()),
            "amount_min": float(cleaned_frame["amount"].min()),
            "amount_max": float(cleaned_frame["amount"].max()),
        },
        "features": {
            "rows": int(len(feature_frame)),
            "null_values": int(feature_frame.isna().sum().sum()),
            "fraud_rate": float(feature_frame["class"].mean()),
            "feature_columns": [
                column
                for column in feature_frame.columns
                if column not in {"source_row_id", "class"}
            ],
        },
    }
    report["checks"] = {
        "non_negative_amounts": bool((cleaned_frame["amount"] >= 0).all()),
        "binary_target": bool(set(feature_frame["class"].unique()).issubset({0, 1})),
        "no_feature_nulls": bool(feature_frame.isna().sum().sum() == 0),
    }
    return report


def write_validation_report(report: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_data_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    raw_frame, source_name = load_credit_card_data(config)
    cleaned_frame, quality = clean_credit_card_data(raw_frame)
    feature_frame = engineer_features(cleaned_frame)
    validation_report = validate_pipeline_frames(raw_frame, cleaned_frame, feature_frame)
    validation_report["quality"] = quality
    write_validation_report(validation_report, config.validation_report_path)
    return PipelineArtifacts(
        source_name=source_name,
        raw_frame=raw_frame,
        cleaned_frame=cleaned_frame,
        feature_frame=feature_frame,
        validation_report=validation_report,
    )
