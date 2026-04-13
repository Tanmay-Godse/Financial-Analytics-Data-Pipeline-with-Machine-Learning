import numpy as np

from financial_analytics_pipeline.models import (
    KMeansScratch,
    LinearRegressionScratch,
    LogisticRegressionScratch,
    PCAScratch,
)


def test_linear_regression_scratch_fits_simple_relationship() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    model = LinearRegressionScratch(learning_rate=0.05, n_iterations=1_500)
    model.fit(x, y)
    predictions = model.predict(x)
    assert np.allclose(predictions, y, atol=0.3)


def test_logistic_regression_scratch_learns_binary_boundary() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionScratch(learning_rate=0.2, n_iterations=800)
    model.fit(x, y)
    predictions = model.predict(x)
    assert predictions.tolist() == y.tolist()


def test_pca_and_kmeans_shapes_are_stable() -> None:
    x = np.array(
        [
            [1.0, 1.0, 0.9],
            [0.9, 1.1, 1.0],
            [5.0, 5.1, 4.9],
            [5.2, 4.9, 5.1],
        ]
    )
    pca = PCAScratch(n_components=2)
    embedding = pca.fit_transform(x)
    model = KMeansScratch(n_clusters=2, random_state=42)
    labels = model.fit_predict(embedding)
    assert embedding.shape == (4, 2)
    assert sorted(np.unique(labels).tolist()) == [0, 1]
