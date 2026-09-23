from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    """
    KNN Classifier
    The k-Nearest Neighbors classifier estimates the class of a sample based on
    the k most similar examples of the training dataset.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: KNNClassifier
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """
        Returns the most common label among the k nearest neighbors of the sample
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest = np.argsort(distances)[:self.k]
        k_labels = self.dataset.y[k_nearest]
        labels, counts = np.unique(k_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The test dataset

        Returns
        -------
        predictions: np.ndarray
            The predicted classes
        """
        return np.array([self._get_closest_label(sample) for sample in dataset.X])

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The test dataset
        predictions: np.ndarray
            The predicted classes

        Returns
        -------
        accuracy: float
        """
        return accuracy(dataset.y, predictions)
