import numpy as np


def _to_labels(y: np.ndarray) -> np.ndarray:
    """
    Converts neural network style outputs to a 1D array of labels

    Parameters
    ----------
    y: np.ndarray
        The labels. A (n_samples, 1) array holds probabilities, which are rounded, and a
        (n_samples, n_classes) array holds one-hot labels or probabilities, for which the
        index of the highest value is used. 1D arrays are returned unchanged

    Returns
    -------
    np.ndarray (n_samples,)
        The labels
    """
    y = np.asarray(y)
    if y.ndim == 2:
        return np.round(y[:, 0]) if y.shape[1] == 1 else np.argmax(y, axis=1)
    return y


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the accuracy of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        The proportion of correctly classified samples
    """
    return np.sum(_to_labels(y_true) == _to_labels(y_pred)) / len(y_true)
