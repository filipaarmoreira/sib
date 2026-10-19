from unittest import TestCase

import numpy as np

from si.neural_networks.layers import DenseLayer


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
