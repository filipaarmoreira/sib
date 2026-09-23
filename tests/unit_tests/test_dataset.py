import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        dataset = Dataset(X, y).dropna()
        self.assertEqual((2, 3), dataset.shape())
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual([0, 1], dataset.y.tolist())

    def test_fillna_value(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan]])
        dataset = Dataset(X, np.array([0, 1, 0])).fillna(0.0)
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(0.0, dataset.X[1, 0])
        self.assertEqual(0.0, dataset.X[2, 2])

    def test_fillna_mean(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan]])
        dataset = Dataset(X, np.array([0, 1, 0])).fillna("mean")
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(4.0, dataset.X[1, 0])
        self.assertEqual(4.5, dataset.X[2, 2])

    def test_fillna_median(self):
        X = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, np.nan], [10, 11, 12]])
        dataset = Dataset(X, np.array([0, 1, 0, 1])).fillna("median")
        self.assertFalse(np.isnan(dataset.X).any())
        self.assertEqual(7.0, dataset.X[1, 0])
        self.assertEqual(6.0, dataset.X[2, 2])

    def test_fillna_invalid(self):
        dataset = Dataset(np.array([[1.0, np.nan]]))
        with self.assertRaises(ValueError):
            dataset.fillna("mode")

    def test_remove_by_index(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([0, 1, 2])
        dataset = Dataset(X, y).remove_by_index(1)
        self.assertEqual((2, 3), dataset.shape())
        self.assertEqual([[1, 2, 3], [7, 8, 9]], dataset.X.tolist())
        self.assertEqual([0, 2], dataset.y.tolist())
        with self.assertRaises(IndexError):
            dataset.remove_by_index(10)
