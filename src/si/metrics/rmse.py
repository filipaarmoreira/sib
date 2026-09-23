import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the root mean squared error between the true and predicted values

    Parameters
    ----------
    y_true: np.ndarray
        The true values of y
    y_pred: np.ndarray
        The predicted values of y

    Returns
    -------
    rmse: float
        The root mean squared error
    """
    return float(np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true)))
