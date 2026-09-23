from typing import Tuple

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scoring function for classification problems. Computes one-way ANOVA F-value
    for each feature, grouping the samples by class.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.ndarray (n_features,)
        F scores
    p: np.ndarray (n_features,)
        p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c].astype(float) for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p
