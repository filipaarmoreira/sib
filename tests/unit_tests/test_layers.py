from unittest import TestCase

import numpy as np

from si.neural_networks.activation import ReLUActivation, SigmoidActivation, TanhActivation, SoftmaxActivation
from si.neural_networks.layers import DenseLayer, Dropout


class TestDenseLayer(TestCase):

    def setUp(self):
        self.layer = DenseLayer(4, (3,)).initialize()

    def test_forward_propagation(self):
        out = self.layer.forward_propagation(np.random.rand(10, 3))
        self.assertEqual((10, 4), out.shape)

    def test_parameters(self):
        self.assertEqual(3 * 4 + 4, self.layer.parameters())

    def test_output_shape(self):
        self.assertEqual((4,), self.layer.output_shape())

    def test_initialize(self):
        self.assertEqual((3, 4), self.layer.weights.shape)
        self.assertTrue(np.all(self.layer.weights >= -0.5) and np.all(self.layer.weights < 0.5))
        self.assertTrue(np.all(self.layer.biases == 0))

    def test_layer_name(self):
        self.assertEqual('DenseLayer', self.layer.layer_name())


class NoUpdate:
    def update(self, w, grad):
        return w


class SGDStep:
    def update(self, w, grad):
        return w - 0.1 * grad


class TestDenseBackward(TestCase):

    def test_gradient_check(self):
        mse = lambda Y, P: np.mean((Y - P) ** 2)
        mse_derivative = lambda Y, P: 2 * (P - Y) / Y.size
        np.random.seed(0)
        X, Y = np.random.rand(4, 3), np.random.rand(4, 2)
        layer = DenseLayer(2, (3,)).initialize(NoUpdate())
        out = layer.forward_propagation(X, training=True)
        analytic = X.T @ mse_derivative(Y, out)
        numeric, eps = np.zeros_like(layer.weights), 1e-6
        for i in range(3):
            for j in range(2):
                layer.weights[i, j] += eps
                lp = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] -= 2 * eps
                lm = mse(Y, layer.forward_propagation(X, True))
                layer.weights[i, j] += eps
                numeric[i, j] = (lp - lm) / (2 * eps)
        self.assertLess(np.abs(numeric - analytic).max(), 1e-8)

    def test_hand_step(self):
        # exercício dos slides 10-11: 2 -> 3 -> 1, um passo de treino
        x = np.array([[1, 2]])
        l1 = DenseLayer(3, (2,)).initialize(SGDStep())
        l1.weights = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.4]])
        l1.biases = np.array([[0.1, 0, -0.1]])
        l2 = DenseLayer(1, (3,)).initialize(SGDStep())
        l2.weights = np.array([[0.2], [-0.3], [0.1]])
        y_hat = l2.forward_propagation(l1.forward_propagation(x))
        self.assertAlmostEqual(0.28, y_hat[0, 0])
        dh = l2.backward_propagation(2 * (y_hat - 1))
        np.testing.assert_allclose(dh, [[-0.288, 0.432, -0.144]])
        np.testing.assert_allclose(l2.weights.ravel(), [0.3152, -0.3288, 0.1864])
        self.assertAlmostEqual(0.144, l2.biases[0, 0])
        l1.backward_propagation(dh)
        np.testing.assert_allclose(l1.weights, [[0.1288, 0.1568, -0.0856], [0.3576, -0.2864, 0.4288]])


class TestDropout(TestCase):

    def test_training(self):
        np.random.seed(0)
        layer = Dropout(0.5)
        X = np.random.rand(100, 10) + 1
        out = layer.forward_propagation(X, training=True)
        self.assertEqual(X.shape, out.shape)
        zeros = np.mean(out == 0)
        self.assertTrue(0.4 < zeros < 0.6)
        np.testing.assert_allclose(out[out != 0], X[out != 0] * 2)
        error = np.ones_like(X)
        np.testing.assert_array_equal(layer.backward_propagation(error), layer.mask)

    def test_inference(self):
        layer = Dropout(0.5)
        X = np.random.rand(5, 3)
        np.testing.assert_array_equal(X, layer.forward_propagation(X, training=False))

    def test_shape_and_parameters(self):
        layer = Dropout(0.3)
        layer.set_input_shape((4,))
        self.assertEqual((4,), layer.output_shape())
        self.assertEqual(0, layer.parameters())


class TestActivations(TestCase):

    def test_derivatives(self):
        x, eps = np.array([-2.5, -1.2, -0.3, 0.4, 1.1, 2.7]), 1e-6
        for act in [SigmoidActivation(), ReLUActivation(), TanhActivation()]:
            num = (act.activation_function(x + eps) - act.activation_function(x - eps)) / (2 * eps)
            self.assertLess(np.abs(num - act.derivative(x)).max(), 1e-6)

    def test_ranges(self):
        x = np.random.randn(10, 4) * 5
        self.assertTrue(np.all(ReLUActivation().activation_function(x) >= 0))
        s = SigmoidActivation().activation_function(x)
        self.assertTrue(np.all((s > 0) & (s < 1)))
        t = TanhActivation().activation_function(x)
        self.assertTrue(np.all((t >= -1) & (t <= 1)))

    def test_softmax(self):
        x = np.array([[1, 2, 3], [1000, 1001, 1002]])
        p = SoftmaxActivation().activation_function(x)
        np.testing.assert_allclose(p.sum(axis=1), [1, 1])
        np.testing.assert_allclose(p[0], p[1])  # estável: sem overflow

    def test_backward_shape(self):
        act = TanhActivation()
        X = np.random.rand(5, 3)
        act.forward_propagation(X)
        self.assertEqual(X.shape, act.backward_propagation(np.ones((5, 3))).shape)
        self.assertEqual(0, act.parameters())
