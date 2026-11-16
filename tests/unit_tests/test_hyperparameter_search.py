import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.metrics.accuracy import accuracy
from si.model_selection.grid_search import grid_search_cv
from si.model_selection.randomized_search import randomized_search_cv, log_uniform
from si.models.knn_classifier import KNNClassifier


class TestHyperparameterSearch(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_grid_search_cv(self):
        grid = {'k': [1, 3, 5], 'distance': [KNNClassifier().distance]}
        results = grid_search_cv(KNNClassifier(), self.dataset, grid, scoring=accuracy, cv=3, seed=1)
        self.assertEqual(3, len(results['hyperparameters']))
        self.assertEqual(3, len(results['scores']))
        self.assertEqual(max(results['scores']), results['best_score'])
        self.assertIn(results['best_hyperparameters']['k'], [1, 3, 5])

    def test_grid_search_builder_function(self):
        results = grid_search_cv(lambda k: KNNClassifier(k=k), self.dataset, {'k': [1, 3]}, cv=3, seed=1)
        self.assertEqual([{'k': 1}, {'k': 3}], results['hyperparameters'])

    def test_randomized_search_cv(self):
        dist = {'k': [1, 3, 5, 7]}
        results = randomized_search_cv(KNNClassifier(), self.dataset, dist, cv=3, n_iter=4, seed=1)
        self.assertEqual(4, len(results['scores']))
        self.assertTrue(all(h['k'] in [1, 3, 5, 7] for h in results['hyperparameters']))
        again = randomized_search_cv(KNNClassifier(), self.dataset, dist, cv=3, n_iter=4, seed=1)
        self.assertEqual(results['hyperparameters'], again['hyperparameters'])

    def test_log_uniform(self):
        sampler = log_uniform(1e-4, 1e-1)
        rng = np.random.default_rng(0)
        samples = np.array([sampler(rng) for _ in range(1000)])
        self.assertTrue(np.all((samples >= 1e-4) & (samples <= 1e-1)))
        # cerca de 1/3 das amostras em cada ordem de grandeza
        self.assertGreater(np.mean(samples < 1e-3), 0.25)
