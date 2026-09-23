from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor estimates the value of a sample as the
    average of the values of the k most similar examples of the training dataset.
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

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: KNNRegressor
        """
        self.dataset = dataset
        return self

    def _get_mean_value(self, sample: np.ndarray) -> float:
        """
        Returns the average value of the k nearest neighbors of the sample
        """
        distances = self.distance(sample, self.dataset.X)
        k_nearest = np.argsort(distances)[:self.k]
        return np.mean(self.dataset.y[k_nearest])

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the values of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The test dataset

        Returns
        -------
        predictions: np.ndarray
            The predicted values
        """
        return np.array([self._get_mean_value(sample) for sample in dataset.X])

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the RMSE of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The test dataset
        predictions: np.ndarray
            The predicted values

        Returns
        -------
        rmse: float
        """
        return rmse(dataset.y, predictions)
