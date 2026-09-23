import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Variance Threshold feature selection.
    Features with a variance lower or equal to the threshold are removed.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        threshold: float
            The cut-off value. Features with variance <= threshold are removed.
        """
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Estimates the variance of each feature

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        self: VarianceThreshold
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects all features with variance greater than the threshold

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
            The transformed dataset
        """
        mask = self.variance > self.threshold
        features = np.array(dataset.features)[mask]
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=features.tolist(), label=dataset.label)
