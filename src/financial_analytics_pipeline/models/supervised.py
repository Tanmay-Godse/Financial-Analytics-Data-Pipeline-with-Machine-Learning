from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import candidate_thresholds, sample_feature_indices, sigmoid


class LinearRegressionScratch:
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 500,
        l2_penalty: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LinearRegressionScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        for _ in range(self.n_iterations):
            predictions = x @ self.weights_ + self.bias_
            errors = predictions - y
            weight_gradient = (x.T @ errors) / n_samples + self.l2_penalty * self.weights_
            bias_gradient = float(np.mean(errors))
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        return x @ self.weights_ + self.bias_


class LogisticRegressionScratch:
    def __init__(
        self,
        learning_rate: float = 0.02,
        n_iterations: int = 700,
        l2_penalty: float = 0.0,
        class_weight: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.class_weight = class_weight
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LogisticRegressionScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        sample_weights = np.ones(n_samples, dtype=float)
        sample_weights[y == 1] = self.class_weight
        for _ in range(self.n_iterations):
            logits = x @ self.weights_ + self.bias_
            probabilities = sigmoid(logits)
            errors = (probabilities - y) * sample_weights
            weight_gradient = (x.T @ errors) / n_samples + self.l2_penalty * self.weights_
            bias_gradient = float(np.mean(errors))
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        probabilities = sigmoid(x @ self.weights_ + self.bias_)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= threshold).astype(int)


class LinearSVMClassifierScratch:
    def __init__(
        self,
        learning_rate: float = 0.001,
        n_iterations: int = 500,
        regularization_strength: float = 0.01,
        class_weight: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_strength = regularization_strength
        self.class_weight = class_weight
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, features: np.ndarray, target: np.ndarray) -> "LinearSVMClassifierScratch":
        x = np.asarray(features, dtype=float)
        y = np.where(np.asarray(target) > 0, 1.0, -1.0)
        n_samples, n_features = x.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        sample_weights = np.ones(n_samples, dtype=float)
        sample_weights[y == 1.0] = self.class_weight

        for _ in range(self.n_iterations):
            margins = y * (x @ self.weights_ + self.bias_)
            active = margins < 1.0
            if np.any(active):
                weighted_y = y[active] * sample_weights[active]
                grad_w = self.regularization_strength * self.weights_ - (x[active].T @ weighted_y) / n_samples
                grad_b = -float(np.sum(weighted_y) / n_samples)
            else:
                grad_w = self.regularization_strength * self.weights_
                grad_b = 0.0
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        return x @ self.weights_ + self.bias_

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) >= 0.0).astype(int)


class KNNClassifierScratch:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.features_: np.ndarray | None = None
        self.target_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "KNNClassifierScratch":
        self.features_ = np.asarray(features, dtype=float)
        self.target_ = np.asarray(target, dtype=int)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.features_ is None or self.target_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        probabilities = np.zeros((len(x), 2), dtype=float)
        for index, row in enumerate(x):
            distances = np.linalg.norm(self.features_ - row, axis=1)
            neighbors = np.argsort(distances)[: self.k]
            neighbor_labels = self.target_[neighbors]
            positive_rate = float(np.mean(neighbor_labels))
            probabilities[index] = [1.0 - positive_rate, positive_rate]
        return probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


@dataclass
class _TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    value: float | None = None
    positive_probability: float | None = None


class DecisionTreeClassifierScratch:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 20,
        max_features: str | int | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "DecisionTreeClassifierScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=int)
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._grow_tree(x, y, depth=0)
        importance_sum = float(self.feature_importances_.sum())
        if importance_sum:
            self.feature_importances_ /= importance_sum
        return self

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        positive_rate = float(np.mean(y))
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return _TreeNode(value=float(positive_rate >= 0.5), positive_probability=positive_rate)

        feature_index, threshold, gain = self._best_split(x, y)
        if feature_index is None or threshold is None or gain <= 0.0:
            return _TreeNode(value=float(positive_rate >= 0.5), positive_probability=positive_rate)

        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask
        self.feature_importances_[feature_index] += gain
        return _TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=self._grow_tree(x[left_mask], y[left_mask], depth + 1),
            right=self._grow_tree(x[right_mask], y[right_mask], depth + 1),
            positive_probability=positive_rate,
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n_samples, n_features = x.shape
        parent_impurity = self._gini(y)
        best_gain = -np.inf
        best_feature: int | None = None
        best_threshold: float | None = None
        feature_indices = sample_feature_indices(self._rng, n_features, self.max_features)

        for feature_index in feature_indices:
            thresholds = candidate_thresholds(x[:, feature_index])
            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                child_impurity = (
                    left_mask.sum() / n_samples * left_impurity
                    + right_mask.sum() / n_samples * right_impurity
                )
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    @staticmethod
    def _gini(target: np.ndarray) -> float:
        positive_rate = np.mean(target)
        return float(1.0 - positive_rate**2 - (1.0 - positive_rate) ** 2)

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> tuple[int, float]:
        if node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
            probability = float(node.positive_probability or 0.0)
            return int((node.value or 0.0) >= 0.5), probability
        branch = node.left if row[node.feature_index] <= node.threshold else node.right
        return self._predict_row(row, branch)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        return np.array([self._predict_row(row, self.root_)[0] for row in x], dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        positive_probabilities = np.array([self._predict_row(row, self.root_)[1] for row in x], dtype=float)
        return np.column_stack([1.0 - positive_probabilities, positive_probabilities])


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 20,
        max_features: str | int | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "DecisionTreeRegressorScratch":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root_ = self._grow_tree(x, y, depth=0)
        importance_sum = float(self.feature_importances_.sum())
        if importance_sum:
            self.feature_importances_ /= importance_sum
        return self

    def _grow_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.allclose(y, y[0]):
            return _TreeNode(value=float(np.mean(y)))

        feature_index, threshold, gain = self._best_split(x, y)
        if feature_index is None or threshold is None or gain <= 0.0:
            return _TreeNode(value=float(np.mean(y)))

        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask
        self.feature_importances_[feature_index] += gain
        return _TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=self._grow_tree(x[left_mask], y[left_mask], depth + 1),
            right=self._grow_tree(x[right_mask], y[right_mask], depth + 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        n_samples, n_features = x.shape
        parent_error = np.var(y) * n_samples
        best_gain = -np.inf
        best_feature: int | None = None
        best_threshold: float | None = None
        feature_indices = sample_feature_indices(self._rng, n_features, self.max_features)

        for feature_index in feature_indices:
            thresholds = candidate_thresholds(x[:, feature_index])
            for threshold in thresholds:
                left_mask = x[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_error = np.var(y[left_mask]) * left_mask.sum()
                right_error = np.var(y[right_mask]) * right_mask.sum()
                gain = parent_error - (left_error + right_error)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        if node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
            return float(node.value or 0.0)
        branch = node.left if row[node.feature_index] <= node.threshold else node.right
        return self._predict_row(row, branch)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(features, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in x], dtype=float)
