from __future__ import annotations

import numpy as np


class PCAScratch:
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "PCAScratch":
        x = np.asarray(features, dtype=float)
        self.mean_ = np.mean(x, axis=0)
        centered = x - self.mean_
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, : self.n_components]
        total_variance = np.sum(eigenvalues) or 1.0
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total_variance
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before calling transform.")
        x = np.asarray(features, dtype=float)
        return (x - self.mean_) @ self.components_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


class KMeansScratch:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit(self, features: np.ndarray) -> "KMeansScratch":
        x = np.asarray(features, dtype=float)
        rng = np.random.default_rng(self.random_state)
        centroid_indices = rng.choice(len(x), size=self.n_clusters, replace=False)
        centroids = x[centroid_indices].copy()

        for _ in range(self.max_iterations):
            distances = self._pairwise_distances(x, centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = centroids.copy()
            for cluster_id in range(self.n_clusters):
                cluster_points = x[labels == cluster_id]
                if len(cluster_points):
                    new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if centroid_shift <= self.tolerance:
                break

        self.centroids_ = centroids
        final_distances = self._pairwise_distances(x, centroids)
        final_labels = np.argmin(final_distances, axis=1)
        self.inertia_ = float(np.sum((x - centroids[final_labels]) ** 2))
        return self

    @staticmethod
    def _pairwise_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        return np.sqrt(((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise ValueError("KMeans must be fitted before calling predict.")
        distances = self._pairwise_distances(np.asarray(features, dtype=float), self.centroids_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).predict(features)


class HierarchicalClusteringScratch:
    def __init__(self, n_clusters: int = 3) -> None:
        self.n_clusters = n_clusters
        self.labels_: np.ndarray | None = None

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        n_samples = len(x)
        clusters: dict[int, list[int]] = {index: [index] for index in range(n_samples)}
        active_ids = list(clusters.keys())
        next_cluster_id = n_samples

        while len(active_ids) > self.n_clusters:
            best_distance = np.inf
            best_pair: tuple[int, int] | None = None
            for left_index in range(len(active_ids)):
                for right_index in range(left_index + 1, len(active_ids)):
                    left_id = active_ids[left_index]
                    right_id = active_ids[right_index]
                    distance = self._average_linkage_distance(x, clusters[left_id], clusters[right_id])
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (left_id, right_id)
            if best_pair is None:
                break
            left_id, right_id = best_pair
            clusters[next_cluster_id] = clusters[left_id] + clusters[right_id]
            active_ids = [cluster_id for cluster_id in active_ids if cluster_id not in best_pair]
            active_ids.append(next_cluster_id)
            next_cluster_id += 1

        labels = np.zeros(n_samples, dtype=int)
        for label, cluster_id in enumerate(active_ids):
            labels[clusters[cluster_id]] = label
        self.labels_ = labels
        return labels

    @staticmethod
    def _average_linkage_distance(
        features: np.ndarray,
        left_indices: list[int],
        right_indices: list[int],
    ) -> float:
        left_points = features[left_indices]
        right_points = features[right_indices]
        distances = np.sqrt(((left_points[:, None, :] - right_points[None, :, :]) ** 2).sum(axis=2))
        return float(np.mean(distances))
