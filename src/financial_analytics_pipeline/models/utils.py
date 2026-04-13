from __future__ import annotations

import numpy as np


class StandardScalerScratch:
    """Simple z-score scaler implemented without sklearn transformers."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "StandardScalerScratch":
        self.mean_ = np.mean(features, axis=0)
        self.scale_ = np.std(features, axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before calling transform.")
        return (features - self.mean_) / self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def sample_feature_indices(
    random_state: np.random.Generator,
    n_features: int,
    max_features: str | int | None,
) -> np.ndarray:
    all_indices = np.arange(n_features)
    if max_features is None:
        return all_indices
    if isinstance(max_features, str) and max_features == "sqrt":
        sample_size = max(1, int(np.sqrt(n_features)))
        return np.sort(random_state.choice(all_indices, size=sample_size, replace=False))
    if isinstance(max_features, int):
        sample_size = min(n_features, max_features)
        return np.sort(random_state.choice(all_indices, size=sample_size, replace=False))
    return all_indices


def candidate_thresholds(values: np.ndarray, max_candidates: int = 10) -> np.ndarray:
    unique_values = np.unique(values)
    if unique_values.size <= max_candidates:
        return unique_values
    quantiles = np.linspace(0.1, 0.9, max_candidates)
    return np.unique(np.quantile(values, quantiles))


def expanding_window_split(
    n_samples: int,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    indices = np.arange(n_samples)
    fold_size = n_samples // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split_index in range(1, n_splits + 1):
        train_end = fold_size * split_index
        test_end = fold_size * (split_index + 1)
        train_indices = indices[:train_end]
        test_indices = indices[train_end:test_end]
        if len(test_indices) == 0:
            continue
        splits.append((train_indices, test_indices))
    return splits
