import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.feature_selection.variance_threshold import VarianceThreshold
from si.feature_selection.select_k_best import SelectKBest
from si.feature_selection.select_percentile import SelectPercentile
from si.statistics.f_classification import f_classification


class TestFeatureSelection(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_variance_threshold(self):
        selector = VarianceThreshold(threshold=0.5).fit(self.dataset)
        self.assertEqual(4, len(selector.variance))
        new = selector.transform(self.dataset)
        self.assertEqual(['sepal_length', 'petal_length', 'petal_width'], new.features)
        self.assertEqual((150, 3), new.shape())

    def test_select_k_best(self):
        selector = SelectKBest(k=2).fit(self.dataset)
        self.assertEqual(4, len(selector.F))
        new = selector.transform(self.dataset)
        self.assertEqual((150, 2), new.shape())
        self.assertEqual({'petal_length', 'petal_width'}, set(new.features))

    def test_select_percentile(self):
        selector = SelectPercentile(percentile=50).fit(self.dataset)
        self.assertEqual(4, len(selector.F))
        self.assertEqual(4, len(selector.p))
        new = selector.transform(self.dataset)
        self.assertEqual((150, 2), new.shape())
        self.assertEqual({'petal_length', 'petal_width'}, set(new.features))

    def test_select_percentile_ties(self):
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        from si.data.dataset import Dataset
        dataset = Dataset(np.tile(F, (3, 1)), np.array([0, 1, 0]))
        selector = SelectPercentile(score_func=lambda d: (F, F), percentile=40)
        new = selector.fit_transform(dataset)
        self.assertEqual([5.6, 5.6, 7.8, 6.5], new.X[0].tolist())

    def test_f_classification(self):
        F, p = f_classification(self.dataset)
        self.assertEqual(4, len(F))
        self.assertTrue((p < 0.05).all())
