from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Central configuration for the credit-card fraud analytics project."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    kaggle_dataset: str = "mlg-ulb/creditcardfraud"
    kaggle_filename: str = "creditcard.csv"
    openml_data_id: int = 1597
    random_state: int = 42
    test_size: float = 0.2
    cross_validation_splits: int = 3
    training_majority_limit: int = 12_000
    fraud_weight: float = 35.0
    cluster_sample_size: int = 300
    max_parallel_workers: int = 3
    refresh_interval_minutes: int = 30

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.outputs_dir = self.project_root / "outputs"
        self.artifacts_dir = self.project_root / "artifacts"
        self.sql_dir = self.project_root / "sql"
        self.database_path = self.processed_dir / "financial_pipeline.sqlite"
        self.schema_path = self.sql_dir / "schema.sql"
        self.local_csv_path = self.raw_dir / self.kaggle_filename
        self.raw_snapshot_path = self.raw_dir / "creditcard_snapshot.csv"
        self.cleaned_snapshot_path = self.processed_dir / "creditcard_cleaned.csv"
        self.feature_snapshot_path = self.processed_dir / "creditcard_features.csv"
        self.validation_report_path = self.artifacts_dir / "validation_report.json"
        self.model_summary_path = self.artifacts_dir / "model_results.json"
        self.cluster_snapshot_path = self.artifacts_dir / "cluster_assignments.csv"

    def ensure_directories(self) -> None:
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.outputs_dir,
            self.artifacts_dir,
            self.sql_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
