import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression resolvida com a solução analítica de mínimos quadrados.

    theta = (X^T X + lambda * I')^-1 X^T y
    (I' é a matriz identidade com a posição [0, 0] a zero, para não penalizar o intercept)

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The coefficients of the model for every feature
    theta_zero: float
        The intercept of the model
    mean: np.ndarray
        Mean of the dataset (for every feature)
    std: np.ndarray
        Standard deviation of the dataset (for every feature)
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimate theta, theta_zero, mean and std.
        """
        X = dataset.X
        y = dataset.y

        # 1. scale the data if required
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std = np.where(self.std == 0, 1, self.std)
            X = (X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        m, n = X.shape

        # 2. add intercept term (coluna de 1s na primeira posição)
        X = np.c_[np.ones(m), X]

        # 3. penalty term = l2_penalty * identity matrix
        penalty_matrix = self.l2_penalty * np.eye(n + 1)

        # 4. o intercept (theta_zero) não é penalizado
        penalty_matrix[0, 0] = 0

        # 5. model parameters
        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(y)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict y using the estimated thetas.
        """
        X = dataset.X

        # 1. scale the data if required (usando mean e std estimados no fit)
        if self.scale:
            X = (X - self.mean) / self.std

        # 2. add intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. predicted y = X * thetas
        thetas = np.r_[self.theta_zero, self.theta]
        return X.dot(thetas)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the MSE between the real and predicted y values.
        """
        return mse(dataset.y, predictions)
