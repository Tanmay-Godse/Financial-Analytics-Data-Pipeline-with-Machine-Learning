from __future__ import annotations

import numpy as np


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return float(np.mean(true == pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    matrix = confusion_matrix_binary(y_true, y_pred)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = np.asarray(y_true).astype(int)
    score = np.asarray(y_score, dtype=float)
    thresholds = np.unique(score)[::-1]
    thresholds = np.concatenate(([np.inf], thresholds, [-np.inf]))
    tpr_values = []
    fpr_values = []
    for threshold in thresholds:
        predictions = (score >= threshold).astype(int)
        matrix = confusion_matrix_binary(true, predictions)
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    return np.asarray(fpr_values), np.asarray(tpr_values), thresholds


def auc_score(x_values: np.ndarray, y_values: np.ndarray) -> float:
    order = np.argsort(x_values)
    return float(np.trapezoid(y_values[order], x_values[order]))


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, object]:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    results: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix_binary(y_true, y_pred),
    }
    if y_score is not None:
        fpr, tpr, thresholds = roc_curve_binary(y_true, y_score)
        results["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        results["roc_auc"] = auc_score(fpr, tpr)
    else:
        results["roc_auc"] = 0.0
    return results


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((true - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(true - pred)))
    total_variance = float(np.sum((true - np.mean(true)) ** 2))
    explained_variance = float(np.sum((true - pred) ** 2))
    r2 = 1.0 - explained_variance / total_variance if total_variance else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
