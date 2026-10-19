from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Base class for the layers of a neural network.
    """

    @abstractmethod
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Computes the output of the layer given the input.
        """
        raise NotImplementedError

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Back propagates the output error (implemented in the next class).
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, input_shape: tuple):
        """
        Sets the input shape of the layer.
        """
        self._input_shape = input_shape

    def input_shape(self) -> tuple:
        """
        Returns the input shape of the layer.
        """
        return self._input_shape

    def layer_name(self) -> str:
        """
        Returns the name of the layer.
        """
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Dense (fully connected) layer: each neuron is connected to every input.

    Parameters
    ----------
    n_units: int
        The number of neurons of the layer
    input_shape: tuple
        The input shape of the layer (for tabular data: (n_features,))

    Attributes
    ----------
    input: np.ndarray
        The layer input
    output: np.ndarray
        The layer output
    weights: np.ndarray (n_features, n_units)
        The layer weights
    biases: np.ndarray (1, n_units)
        The layer biases
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Initializes the weights with random values in [-0.5, 0.5[ and the biases with zeros.

        Parameters
        ----------
        optimizer:
            The optimizer used to update the weights and biases (used from the next class on)

        Returns
        -------
        self: DenseLayer
        """
        n_features = self.input_shape()[0]
        # 1. pesos (n_features, n_units) uniformes em [-0.5, 0.5[
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        # 2. biases (1, n_units) a zero
        self.biases = np.zeros((1, self.n_units))
        # 3. optimizer para os pesos e para os biases
        self.optimizer = optimizer
        return self

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer (weights + biases).
        """
        return int(np.prod(self.weights.shape) + np.prod(self.biases.shape))

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Computes the layer output: Y = XW + B

        Parameters
        ----------
        input: np.ndarray (n_samples, n_features)
            The layer input
        training: bool
            Whether the layer is in training mode or not

        Returns
        -------
        np.ndarray (n_samples, n_units)
            The layer output
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.
        """
        return (self.n_units,)
