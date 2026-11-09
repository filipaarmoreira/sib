from abc import ABCMeta, abstractmethod

import numpy as np

# evita log(0) e divisões por 0
EPSILON = 1e-15


class LossFunction(metaclass=ABCMeta):
    """
    Base class for the loss functions of a neural network.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the loss between the true and predicted labels.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss with respect to the predicted labels.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error, for regression: L = 1/n * sum((y - y*)^2)
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Binary cross-entropy, for binary classification: L = -sum(y*log(y*) + (1-y)*log(1-y*))
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return float(-np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -y_true / p + (1 - y_true) / (1 - p)


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy, for multi-class classification with one-hot labels: L = -sum(y*log(y*))
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return float(-np.sum(y_true * np.log(p)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, EPSILON, 1 - EPSILON)
        return -y_true / p
