import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
    patience: int
        The maximum number of iterations without improvement
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    mean: np.ndarray
        Mean of the dataset (for every feature)
    std: np.ndarray
        Standard deviation of the dataset (for every feature)
    cost_history: dict
        The cost function value at each iteration (iteration: cost)
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 2000,
                 patience: int = 5, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Estimate theta, theta_zero, mean, std and cost_history using gradient descent.
        """
        X = dataset.X
        y = dataset.y

        # 1. scale the data if required
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std = np.where(self.std == 0, 1, self.std)  # evita divisão por zero
            X = (X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        m, n = X.shape

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        early_stopping = 0

        for i in range(self.max_iter):
            # 2. predict y
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - y

            # 3. gradiente (com learning rate)
            gradient = (self.alpha / m) * np.dot(error, X)

            # 4. termo de regularização L2 (com learning rate)
            penalization_term = self.theta * (1 - self.alpha * self.l2_penalty / m)

            # 5. update theta
            self.theta = penalization_term - gradient

            # 6. update theta_zero (o intercept não é penalizado)
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            # 7. custo
            self.cost_history[i] = self.cost(dataset)

            # 8. early stopping (patience)
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            if early_stopping >= self.patience:
                break

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict y using the estimated theta and theta_zero.
        """
        X = dataset.X
        if self.scale:
            X = (X - self.mean) / self.std

        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the MSE between the real and predicted y values.
        """
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the regularized cost function J between predicted and real y values.

        J(theta) = 1/(2m) * [ sum((h(x) - y)^2) + lambda * sum(theta_j^2) ]
        """
        y_pred = self._predict(dataset)
        m = dataset.y.shape[0]
        squared_errors = np.sum((y_pred - dataset.y) ** 2)
        regularization = self.l2_penalty * np.sum(self.theta ** 2)
        return float((squared_errors + regularization) / (2 * m))
