from abc import abstractmethod

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for the activation layers (apply a non-linear function element-wise).
    """

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Applies the activation function to the input.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Multiplies (element-wise) the derivative of the activation function on the
        layer input by the output error.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the activation function on the given input.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function on the given input.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the input shape (activation layers do not change the shape of the input).
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns 0 (activation layers do not have learnable parameters).
        """
        return 0


class ReLUActivation(ActivationLayer):
    """
    Rectified Linear Unit: f(x) = max(0, x)
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        return np.where(input >= 0, 1, 0)


class SigmoidActivation(ActivationLayer):
    """
    Sigmoid: f(x) = 1 / (1 + e^-x)
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray) -> np.ndarray:
        f = self.activation_function(input)
        return f * (1 - f)


class TanhActivation(ActivationLayer):
    """
    Hyperbolic tangent: f(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        return 1 - self.activation_function(input) ** 2


class SoftmaxActivation(ActivationLayer):
    """
    Softmax: transforms the scores into a probability distribution (each row sums to 1).
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        # versão estável: subtrair o máximo de cada linha evita exponenciais enormes
        exp = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        f = self.activation_function(input)
        return f * (1 - f)
