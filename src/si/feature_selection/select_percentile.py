from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Select a given percentage of features according to the highest scores computed by score_func.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: Callable
            Variance analysis function (f_classification by default)
        percentile: float
            Percentage (0-100) of features to select
        """
        super().__init__(**kwargs)
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values for each feature using the score_func

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        self: SelectPercentile
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the given percentage of features with the highest F values.
        Ties at the threshold are resolved by order of appearance, so that exactly
        int(n_features * percentile / 100) features are selected.

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
            A dataset with the selected features
        """
        n_features = len(self.F)
        n_select = int(n_features * self.percentile / 100)
        if n_select == 0:
            mask = np.zeros(n_features, dtype=bool)
        else:
            threshold = np.percentile(self.F, 100 - self.percentile)
            mask = self.F > threshold
            ties = np.where(self.F == threshold)[0]
            mask[ties[:n_select - mask.sum()]] = True
        features = np.array(dataset.features)[mask]
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=features.tolist(), label=dataset.label)
