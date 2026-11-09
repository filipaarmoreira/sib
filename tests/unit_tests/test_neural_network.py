from unittest import TestCase

import numpy as np
import torch

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.model_selection.cross_validation import k_fold_cross_validation
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD, Adam


class TestLosses(TestCase):

    def _check_derivative(self, loss, y, p):
        eps = 1e-6
        numeric = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            plus, minus = p.copy(), p.copy()
            plus[idx] += eps
            minus[idx] -= eps
            numeric[idx] = (loss.loss(y, plus) - loss.loss(y, minus)) / (2 * eps)
        np.testing.assert_allclose(loss.derivative(y, p), numeric, atol=1e-6)

    def test_derivatives(self):
        y = np.array([[1.0], [0.0], [1.0]])
        p = np.array([[0.7], [0.2], [0.4]])
        self._check_derivative(MeanSquaredError(), y, p)
        self._check_derivative(BinaryCrossEntropy(), y, p)
        y = np.array([[1.0, 0.0], [0.0, 1.0]])
        p = np.array([[0.7, 0.3], [0.4, 0.6]])
        self._check_derivative(CategoricalCrossEntropy(), y, p)

    def test_extreme_predictions_are_finite(self):
        y = np.array([[1.0], [0.0]])
        p = np.array([[0.0], [1.0]])
        for loss in (BinaryCrossEntropy(), CategoricalCrossEntropy()):
            self.assertTrue(np.isfinite(loss.loss(y, p)))
            self.assertTrue(np.all(np.isfinite(loss.derivative(y, p))))


class TestOptimizers(TestCase):

    def test_sgd_momentum(self):
        opt = SGD(learning_rate=0.1, momentum=0.9)
        w = opt.update(np.array([1.0]), np.array([2.0]))     # V = 0.2
        np.testing.assert_allclose(w, [0.98])
        w = opt.update(w, np.array([2.0]))                   # V = 0.9*0.2 + 0.2 = 0.38
        np.testing.assert_allclose(w, [0.942])

    def test_adam_matches_pytorch(self):
        w = np.array([1.0, -2.0, 3.0])
        param = torch.tensor(w.copy(), requires_grad=True)
        reference = torch.optim.Adam([param], lr=0.01)
        adam = Adam(learning_rate=0.01)
        for _ in range(50):
            reference.zero_grad()
            (param ** 2).sum().backward()
            reference.step()
            w = adam.update(w, 2 * w)
        np.testing.assert_allclose(w, param.detach().numpy(), rtol=1e-6)


class TestNeuralNetwork(TestCase):

    def _xor(self):
        return Dataset(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), np.array([0, 1, 1, 0], dtype=float))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ValueError):
            NeuralNetwork().predict(self._xor())

    def test_xor(self):
        np.random.seed(1)
        net = NeuralNetwork(epochs=5000, batch_size=4, optimizer=SGD, learning_rate=0.5,
                            loss=MeanSquaredError, metric=accuracy)
        net.add(DenseLayer(4, (2,)))
        net.add(SigmoidActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        net.fit(self._xor())
        self.assertEqual(1.0, net.score(self._xor()))

    def test_score_with_mse_does_not_broadcast(self):
        data = self._xor()
        net = NeuralNetwork(epochs=1, batch_size=4, learning_rate=0.0, metric=mse)
        net.add(DenseLayer(1, (2,)))
        net.fit(data)
        expected = np.mean((data.y - net.predict(data).ravel()) ** 2)
        self.assertAlmostEqual(expected, net.score(data))

    def test_k_fold_does_not_change_the_original_network(self):
        np.random.seed(0)
        X = np.random.randn(60, 4)
        data = Dataset(X, (X[:, 0] > 0).astype(float))
        net = NeuralNetwork(epochs=2, batch_size=8, learning_rate=0.01,
                            loss=BinaryCrossEntropy, metric=accuracy)
        net.add(DenseLayer(4, (4,)))
        net.add(ReLUActivation())
        net.add(DenseLayer(1))
        net.add(SigmoidActivation())
        before = net.layers[0].weights.copy()
        self.assertEqual(3, len(k_fold_cross_validation(net, data, cv=3, seed=0)))
        np.testing.assert_array_equal(before, net.layers[0].weights)
