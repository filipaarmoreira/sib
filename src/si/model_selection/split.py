from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    np.random.seed(random_state)
    n_samples = dataset.shape()[0]
    n_test = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2,
                                random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets, preserving the class proportions

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The stratified training dataset
    test: Dataset
        The stratified testing dataset
    """
    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(dataset.y, return_counts=True)
    train_idxs = []
    test_idxs = []

    for label, count in zip(labels, counts):
        n_test = int(count * test_size)
        class_idxs = rng.permutation(np.where(dataset.y == label)[0])
        test_idxs.extend(class_idxs[:n_test])
        train_idxs.extend(class_idxs[n_test:])

    train_idxs = np.array(train_idxs, dtype=int)
    test_idxs = np.array(test_idxs, dtype=int)
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test
