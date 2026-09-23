from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select features according to the k highest scores computed by score_func.
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: Callable
            Variance analysis function (f_classification by default)
        k: int
            Number of features to select
        """
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estimates the F and p values for each feature using the score_func

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        self: SelectKBest
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the top k features with the highest F value

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
            A dataset with the k selected features
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=features.tolist(), label=dataset.label)
