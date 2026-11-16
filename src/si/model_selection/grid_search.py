import copy
import itertools
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset


def _make_folds(dataset: Dataset, cv: int, seed: int = None) -> List[Tuple[Dataset, Dataset]]:
    """
    Creates the k (train, validation) folds once, so that every configuration is compared
    on exactly the same splits. Uses the same splitting as k_fold_cross_validation

    Parameters
    ----------
    dataset: Dataset
        The training dataset
    cv: int
        The number of folds
    seed: int
        The seed used to shuffle the samples

    Returns
    -------
    folds: List[Tuple[Dataset, Dataset]]
        The (train, validation) datasets of each fold
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset.X.shape[0])

    folds = []
    for val_idx in np.array_split(indices, cv):
        train_idx = np.setdiff1d(indices, val_idx)
        train = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features, label=dataset.label)
        val = Dataset(dataset.X[val_idx], dataset.y[val_idx], features=dataset.features, label=dataset.label)
        folds.append((train, val))
    return folds


def _build_model(model: Union[Model, Callable], hyperparameters: Dict) -> Model:
    """
    Builds a fresh, untrained model with the given hyperparameters

    Parameters
    ----------
    model: Model or Callable
        A model, which is copied and has its attributes changed (copy.deepcopy + setattr),
        or a builder function, which is called with the hyperparameters as keyword arguments
    hyperparameters: Dict
        The configuration to use

    Returns
    -------
    model: Model
        A new model
    """
    if isinstance(model, Model):
        new_model = copy.deepcopy(model)
        # atenção: na NeuralNetwork, mudar learning_rate assim não tem efeito (o otimizador
        # já foi criado no __init__); para redes usa-se uma função builder
        for name, value in hyperparameters.items():
            setattr(new_model, name, value)
        return new_model
    return model(**hyperparameters)


def _cross_validate(model: Union[Model, Callable], hyperparameters: Dict,
                    folds: List[Tuple[Dataset, Dataset]], scoring: Callable = None) -> float:
    """
    Trains a fresh model on each fold and returns the mean validation score

    Parameters
    ----------
    model: Model or Callable
        The model or builder function
    hyperparameters: Dict
        The configuration to evaluate
    folds: List[Tuple[Dataset, Dataset]]
        The (train, validation) folds
    scoring: Callable
        The scoring function, called with (y_true, y_pred). If None, model.score is used

    Returns
    -------
    score: float
        The mean of the k validation scores
    """
    scores = []
    for train, val in folds:
        fold_model = _build_model(model, hyperparameters)
        fold_model.fit(train)
        if scoring is None:
            scores.append(fold_model.score(val))
        else:
            scores.append(scoring(val.y, fold_model.predict(val)))
    return float(np.mean(scores))


def _results(hyperparameters: List[Dict], scores: List[float]) -> Dict:
    """
    Builds the results dictionary returned by the searches
    """
    best = int(np.argmax(scores))
    return {
        'hyperparameters': hyperparameters,
        'scores': scores,
        'best_hyperparameters': hyperparameters[best],
        'best_score': scores[best],
    }


def grid_search_cv(model: Union[Model, Callable], dataset: Dataset, hyperparameter_grid: Dict[str, list],
                   scoring: Callable = None, cv: int = 5, seed: int = None) -> Dict:
    """
    Evaluates every combination of hyperparameter values with k-fold cross validation

    Parameters
    ----------
    model: Model or Callable
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_grid: Dict[str, list]
        Dictionary {name: list of values to try}
    scoring: Callable
        The scoring function, called with (y_true, y_pred). If None, model.score is used
    cv: int
        The number of folds
    seed: int
        The seed used to create the folds

    Returns
    -------
    results: Dict
        hyperparameters (every configuration tested), scores (mean validation score of
        each configuration), best_hyperparameters and best_score
    """
    folds = _make_folds(dataset, cv, seed)

    hyperparameters = []
    scores = []
    for values in itertools.product(*hyperparameter_grid.values()):
        configuration = dict(zip(hyperparameter_grid.keys(), values))
        hyperparameters.append(configuration)
        scores.append(_cross_validate(model, configuration, folds, scoring))

    return _results(hyperparameters, scores)
