import copy
from typing import Callable, List

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset


def k_fold_cross_validation(model: Model, dataset: Dataset, scoring: Callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Performs k-fold cross validation on the given model and dataset

    Parameters
    ----------
    model: Model
        The model to validate. It is copied for each fold, so that no fold trains on what
        was already fitted in a previous one
    dataset: Dataset
        The dataset to validate the model on
    scoring: Callable
        The scoring function, called with (y_true, y_pred). If None, model.score is used
    cv: int
        The number of folds
    seed: int
        The seed used to shuffle the samples

    Returns
    -------
    scores: List[float]
        The score of the model on each fold
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset.X.shape[0])

    scores = []
    # array_split garante que todas as amostras entram num fold de teste
    for test_idx in np.array_split(indices, cv):
        train_idx = np.setdiff1d(indices, test_idx)
        train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features, label=dataset.label)
        test = Dataset(dataset.X[test_idx], dataset.y[test_idx], features=dataset.features, label=dataset.label)

        fold_model = copy.deepcopy(model)
        fold_model.fit(train)
        if scoring is None:
            scores.append(fold_model.score(test))
        else:
            scores.append(scoring(test.y, fold_model.predict(test)))
    return scores
