import numpy as np

from financial_analytics_pipeline.models.metrics import (
    binary_classification_metrics,
    confusion_matrix_binary,
    regression_metrics,
)


def test_confusion_matrix_binary_counts() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    matrix = confusion_matrix_binary(y_true, y_pred)
    assert matrix.tolist() == [[1, 1], [1, 1]]


def test_binary_classification_metrics_include_f1() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = binary_classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert round(metrics["precision"], 3) == 1.0
    assert round(metrics["recall"], 3) == round(2 / 3, 3)
    assert metrics["roc_auc"] >= 0.0


def test_regression_metrics_reasonable_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.5])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["rmse"] > 0.0
    assert metrics["mae"] > 0.0
    assert metrics["r2"] <= 1.0
