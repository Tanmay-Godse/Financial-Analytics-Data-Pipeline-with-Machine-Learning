from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .supervised import DecisionTreeClassifierScratch, DecisionTreeRegressorScratch
from .utils import candidate_thresholds


class RandomForestClassifierScratch:
    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 6,
        min_samples_split: int = 20,
        max_features: str | int | None = "sqrt",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeClassifierScratch] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RandomForestClassifierScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=int)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        importances = []
        for estimator_index in range(self.n_estimators):
            sample_indices = rng.choice(len(x), size=len(x), replace=True)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + estimator_index,
            )
            tree.fit(x[sample_indices], y[sample_indices])
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances.append(tree.feature_importances_)
        if importances:
            self.feature_importances_ = np.mean(np.vstack(importances), axis=0)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("Model must be fitted before prediction.")
        probabilities = np.mean([tree.predict_proba(features) for tree in self.trees_], axis=0)
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


class RandomForestRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 6,
        min_samples_split: int = 20,
        max_features: str | int | None = "sqrt",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[DecisionTreeRegressorScratch] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RandomForestRegressorScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        importances = []
        for estimator_index in range(self.n_estimators):
            sample_indices = rng.choice(len(x), size=len(x), replace=True)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + estimator_index,
            )
            tree.fit(x[sample_indices], y[sample_indices])
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances.append(tree.feature_importances_)
        if importances:
            self.feature_importances_ = np.mean(np.vstack(importances), axis=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("Model must be fitted before prediction.")
        return np.mean([tree.predict(features) for tree in self.trees_], axis=0)


@dataclass
class _DecisionStump:
    feature_index: int = 0
    threshold: float = 0.0
    polarity: int = 1
    alpha: float = 0.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = np.ones(features.shape[0], dtype=int)
        if self.polarity == 1:
            predictions[features[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[features[:, self.feature_index] >= self.threshold] = -1
        return predictions


class AdaBoostClassifierScratch:
    def __init__(self, n_estimators: int = 25) -> None:
        self.n_estimators = n_estimators
        self.stumps_: list[_DecisionStump] = []

    def fit(self, features: np.ndarray, target: np.ndarray) -> "AdaBoostClassifierScratch":
        x = np.asarray(features, dtype=float)
        y = np.where(np.asarray(target) > 0, 1, -1)
        n_samples, n_features = x.shape
        weights = np.full(n_samples, 1 / n_samples, dtype=float)
        self.stumps_ = []

        for _ in range(self.n_estimators):
            stump = _DecisionStump()
            min_error = np.inf
            for feature_index in range(n_features):
                thresholds = candidate_thresholds(x[:, feature_index], max_candidates=8)
                for threshold in thresholds:
                    for polarity in (1, -1):
                        candidate = _DecisionStump(
                            feature_index=feature_index,
                            threshold=float(threshold),
                            polarity=polarity,
                        )
                        predictions = candidate.predict(x)
                        weighted_error = np.sum(weights[predictions != y])
                        if weighted_error < min_error:
                            min_error = weighted_error
                            stump = candidate
            min_error = float(np.clip(min_error, 1e-10, 1 - 1e-10))
            stump.alpha = 0.5 * np.log((1.0 - min_error) / min_error)
            predictions = stump.predict(x)
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= weights.sum()
            self.stumps_.append(stump)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if not self.stumps_:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        scores = np.zeros(x.shape[0], dtype=float)
        for stump in self.stumps_:
            scores += stump.alpha * stump.predict(x)
        return scores

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        probabilities = 1.0 / (1.0 + np.exp(-2.0 * scores))
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) >= 0.0).astype(int)


class GradientBoostingRegressorScratch:
    def __init__(
        self,
        n_estimators: int = 30,
        learning_rate: float = 0.05,
        max_depth: int = 2,
        min_samples_split: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.base_prediction_: float = 0.0
        self.trees_: list[DecisionTreeRegressorScratch] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "GradientBoostingRegressorScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        self.base_prediction_ = float(np.mean(y))
        current_predictions = np.full(len(y), self.base_prediction_, dtype=float)
        self.trees_ = []
        importances = []

        for estimator_index in range(self.n_estimators):
            residuals = y - current_predictions
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + estimator_index,
            )
            tree.fit(x, residuals)
            update = tree.predict(x)
            current_predictions += self.learning_rate * update
            self.trees_.append(tree)
            if tree.feature_importances_ is not None:
                importances.append(tree.feature_importances_)
        if importances:
            self.feature_importances_ = np.mean(np.vstack(importances), axis=0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        predictions = np.full(np.asarray(features).shape[0], self.base_prediction_, dtype=float)
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(features)
        return predictions
