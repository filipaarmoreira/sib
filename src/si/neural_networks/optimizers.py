from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Base class for the optimizers of a neural network.

    Parameters
    ----------
    learning_rate: float
        The learning rate of the optimizer
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Returns the updated parameters (weights or biases).

        Parameters
        ----------
        w: np.ndarray
            The current parameters
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    Parameters
    ----------
    learning_rate: float
        The learning rate
    momentum: float
        The momentum (0 means plain gradient descent)

    Attributes
    ----------
    retained_gradient: np.ndarray
        The gradient accumulated from the previous updates
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)
        # V_t = beta * V_t-1 + (1 - beta) * dL/dW
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        # W_t = W_t-1 - alpha * V_t
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """
    Adam: combines momentum (moving average of the gradient) with RMSprop
    (learning rate scaled by the moving average of the squared gradient).

    Parameters
    ----------
    learning_rate: float
        The learning rate
    beta_1: float
        The exponential decay rate for the 1st moment estimates
    beta_2: float
        The exponential decay rate for the 2nd moment estimates
    epsilon: float
        A small constant for numerical stability

    Attributes
    ----------
    m: np.ndarray
        The moving average of the gradient
    v: np.ndarray
        The moving average of the squared gradient
    t: int
        The time stamp (number of updates)
    """

    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad_loss_w ** 2
        # correção do viés: no início as médias ainda estão puxadas para zero
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
