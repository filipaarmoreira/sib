from typing import Callable, Dict, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.grid_search import _make_folds, _cross_validate, _results


def log_uniform(low: float, high: float) -> Callable:
    """
    Returns a sampler of values spread uniformly over orders of magnitude:
    10 ** uniform(log10(low), log10(high))

    Parameters
    ----------
    low: float
        The lowest value (> 0)
    high: float
        The highest value

    Returns
    -------
    sampler: Callable
        A function that receives a numpy random generator and returns one sample
    """
    if low <= 0 or high <= low:
        raise ValueError("log_uniform needs 0 < low < high")

    def sampler(rng: np.random.Generator) -> float:
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    return sampler


def randomized_search_cv(model: Union[Model, Callable], dataset: Dataset, hyperparameter_distributions: Dict,
                         scoring: Callable = None, cv: int = 5, n_iter: int = 10, seed: int = None) -> Dict:
    """
    Evaluates n_iter random hyperparameter configurations with k-fold cross validation

    Parameters
    ----------
    model: Model or Callable
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_distributions: Dict
        Dictionary {name: list of values or sampler function}. A list is sampled uniformly;
        a function is called with the random generator
    scoring: Callable
        The scoring function, called with (y_true, y_pred). If None, model.score is used
    cv: int
        The number of folds
    n_iter: int
        The number of random configurations to evaluate
    seed: int
        The seed used to create the folds and the random generator

    Returns
    -------
    results: Dict
        The same dictionary as grid_search_cv
    """
    folds = _make_folds(dataset, cv, seed)
    rng = np.random.default_rng(seed)

    hyperparameters = []
    scores = []
    for _ in range(n_iter):
        configuration = {}
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                configuration[name] = distribution(rng)
            else:
                # escolher um índice (e não rng.choice(lista)) mantém os tipos Python (int, str...)
                configuration[name] = distribution[rng.integers(len(distribution))]
        hyperparameters.append(configuration)
        scores.append(_cross_validate(model, configuration, folds, scoring))

    return _results(hyperparameters, scores)
