import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between a sample x and a set of samples y

    Parameters
    ----------
    x: np.ndarray (n_features,)
        A single sample
    y: np.ndarray (n_samples, n_features)
        Multiple samples

    Returns
    -------
    np.ndarray (n_samples,)
        Euclidean distance between x and each sample in y
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))
