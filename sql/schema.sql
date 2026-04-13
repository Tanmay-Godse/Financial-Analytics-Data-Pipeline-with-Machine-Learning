PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS raw_credit_transactions (
    source_row_id INTEGER PRIMARY KEY,
    time_seconds REAL NOT NULL,
    amount REAL NOT NULL,
    class INTEGER NOT NULL,
    V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL,
    V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL,
    V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL,
    V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL,
    source_name TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cleaned_credit_transactions (
    source_row_id INTEGER PRIMARY KEY,
    time_seconds REAL NOT NULL,
    amount REAL NOT NULL,
    class INTEGER NOT NULL,
    V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL,
    V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL,
    V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL,
    V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL,
    cleaned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_store (
    source_row_id INTEGER PRIMARY KEY,
    time_seconds REAL NOT NULL,
    amount REAL NOT NULL,
    class INTEGER NOT NULL,
    log_amount REAL NOT NULL,
    amount_zscore REAL NOT NULL,
    amount_clipped REAL NOT NULL,
    hour_of_day REAL NOT NULL,
    hour_bucket INTEGER NOT NULL,
    hour_sin REAL NOT NULL,
    hour_cos REAL NOT NULL,
    amount_hour_mean_ratio REAL NOT NULL,
    amount_rolling_mean_50 REAL NOT NULL,
    amount_rolling_std_50 REAL NOT NULL,
    amount_velocity REAL NOT NULL,
    time_delta REAL NOT NULL,
    v_magnitude REAL NOT NULL,
    pca_abs_mean REAL NOT NULL,
    pca_positive_share REAL NOT NULL,
    V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL, V6 REAL, V7 REAL,
    V8 REAL, V9 REAL, V10 REAL, V11 REAL, V12 REAL, V13 REAL, V14 REAL,
    V15 REAL, V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL, V21 REAL,
    V22 REAL, V23 REAL, V24 REAL, V25 REAL, V26 REAL, V27 REAL, V28 REAL,
    feature_created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    raw_rows INTEGER NOT NULL,
    cleaned_rows INTEGER NOT NULL,
    feature_rows INTEGER NOT NULL,
    duplicate_rows_removed INTEGER NOT NULL,
    missing_value_count INTEGER NOT NULL,
    fraud_rate REAL NOT NULL,
    validation_report_path TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    cv_metric REAL,
    test_metric REAL,
    precision_value REAL,
    recall_value REAL,
    f1_value REAL,
    roc_auc_value REAL,
    rmse_value REAL,
    mae_value REAL,
    r2_value REAL,
    artifacts_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP VIEW IF EXISTS class_distribution_summary;
CREATE VIEW class_distribution_summary AS
SELECT
    class,
    COUNT(*) AS row_count,
    ROUND(COUNT(*) * 1.0 / (SELECT COUNT(*) FROM feature_store), 6) AS class_share
FROM feature_store
GROUP BY class;

DROP VIEW IF EXISTS hourly_fraud_summary;
CREATE VIEW hourly_fraud_summary AS
SELECT
    hour_bucket,
    COUNT(*) AS transactions,
    SUM(class) AS fraud_transactions,
    ROUND(AVG(amount), 2) AS avg_amount
FROM feature_store
GROUP BY hour_bucket
ORDER BY hour_bucket;
