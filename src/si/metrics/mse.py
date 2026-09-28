import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error entre os valores reais e os previstos.

    MSE = (1/n) * sum((Y_i - Y_hat_i)^2)

    Parameters
    ----------
    y_true: np.ndarray
        Valores reais de y
    y_pred: np.ndarray
        Valores previstos de y

    Returns
    -------
    float
        O erro quadrático médio
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))
