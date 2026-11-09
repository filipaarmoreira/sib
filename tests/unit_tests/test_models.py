import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.metrics.accuracy import accuracy
from si.metrics.rmse import rmse
from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation
from si.model_selection.split import train_test_split, stratified_train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.knn_regressor import KNNRegressor


class TestSplit(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(120, train.shape()[0])

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(120, train.shape()[0])
        _, test_counts = np.unique(test.y, return_counts=True)
        _, train_counts = np.unique(train.y, return_counts=True)
        self.assertEqual([10, 10, 10], test_counts.tolist())
        self.assertEqual([40, 40, 40], train_counts.tolist())


class TestMetrics(TestCase):

    def test_accuracy(self):
        self.assertEqual(0.75, accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])))

    def test_rmse(self):
        self.assertAlmostEqual(np.sqrt(1.5), rmse(np.array([1, 2, 3, 4]), np.array([2, 2, 1, 5])))


class TestKNN(TestCase):

    def test_knn_classifier(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)
        train, test = stratified_train_test_split(dataset, test_size=0.2, random_state=42)
        knn = KNNClassifier(k=3).fit(train)
        predictions = knn.predict(test)
        self.assertEqual(test.shape()[0], len(predictions))
        self.assertGreater(knn.score(test), 0.9)

    def test_knn_regressor(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)
        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        knn = KNNRegressor(k=3).fit(train)
        predictions = knn.predict(test)
        self.assertEqual(test.shape()[0], len(predictions))
        self.assertAlmostEqual(rmse(test.y, predictions), knn.score(test))

    def test_predict_not_fitted(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)
        with self.assertRaises(ValueError):
            KNNClassifier(k=3).predict(dataset)


class TestAccuracyFormats(TestCase):

    def test_labels(self):
        self.assertEqual(0.5, accuracy(np.array(['a', 'b']), np.array(['a', 'a'])))

    def test_binary_probabilities(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([[0.1], [0.9], [0.2], [0.4]])
        self.assertEqual(0.75, accuracy(y_true, y_pred))

    def test_one_hot_probabilities(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
        self.assertEqual(0.5, accuracy(y_true, y_pred))


class TestKFold(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_scores(self):
        scores = k_fold_cross_validation(KNNClassifier(k=3), self.dataset, cv=5, seed=1)
        self.assertEqual(5, len(scores))
        self.assertTrue(all(s > 0.8 for s in scores))

    def test_scoring_and_reproducible(self):
        a = k_fold_cross_validation(KNNClassifier(k=3), self.dataset, scoring=accuracy, cv=3, seed=7)
        b = k_fold_cross_validation(KNNClassifier(k=3), self.dataset, cv=3, seed=7)
        self.assertEqual(a, b)

    def test_uneven_folds_use_all_samples(self):
        X, y = np.arange(14).reshape(7, 2).astype(float), np.array([0, 1, 0, 1, 0, 1, 0])
        scores = k_fold_cross_validation(KNNClassifier(k=1), Dataset(X, y), cv=3, seed=0)
        self.assertEqual(3, len(scores))
