import numpy as np
from si.data.dataset import Dataset
import os
from unittest import TestCase

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv, write_csv


class TestCSVFile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.csv_file_no_label = os.path.join(DATASETS_PATH, 'iris', 'iris_no_label.csv')
        self.csv_file_no_features = os.path.join(DATASETS_PATH, 'iris', 'iris_no_features.csv')

    def test_read_csv_file(self):

        dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)

        self.assertEqual((150, 4), dataset.shape())
        self.assertEqual(True, dataset.has_label())
        self.assertEqual(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dataset.get_classes().tolist())

    def test_read_csv_file_no_label(self):
        dataset = read_csv(filename=self.csv_file_no_label, sep=",", features=True, label=False)

        self.assertEqual((150, 4), dataset.shape())
        self.assertEqual(False, dataset.has_label())

    def test_read_csv_file_no_features(self):

        dataset = read_csv(filename=self.csv_file_no_features, sep=",", features=False, label=True)

        self.assertEqual((150, 0), dataset.shape())
        self.assertEqual(True, dataset.has_label())
        self.assertEqual(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dataset.get_classes().tolist())

    def test_write_csv_file(self):

        dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)
        write_csv(filename='test.csv', dataset=dataset, sep=",", features=True, label=True)
        dataset2 = read_csv(filename='test.csv', sep=",", features=True, label=True)

        self.assertEqual(dataset.shape(), dataset2.shape())
        self.assertEqual(dataset.has_label(), dataset2.has_label())
        self.assertEqual(dataset.get_classes().tolist(), dataset2.get_classes().tolist())
        os.remove('test.csv')

    def test_write_csv_file_no_label(self):

        dataset = read_csv(filename=self.csv_file_no_label, sep=",", features=True, label=False)
        write_csv(filename='test.csv', dataset=dataset, sep=",", features=True, label=False)
        dataset2 = read_csv(filename='test.csv', sep=",", features=True, label=False)

        self.assertEqual(dataset.shape(), dataset2.shape())
        self.assertEqual(dataset.has_label(), dataset2.has_label())
        os.remove('test.csv')

    def test_write_csv_file_no_features(self):

        dataset = read_csv(filename=self.csv_file_no_features, sep=",", features=False, label=True)
        write_csv(filename='test.csv', dataset=dataset, sep=",", features=False, label=True)
        dataset2 = read_csv(filename='test.csv', sep=",", features=False, label=True)

        self.assertEqual(dataset.shape(), dataset2.shape())
        self.assertEqual(dataset.has_label(), dataset2.has_label())
        self.assertEqual(dataset.get_classes().tolist(), dataset2.get_classes().tolist())
        os.remove('test.csv')

class TestCSVHeaderlessFile(TestCase):

    def test_numeric_first_row_is_a_sample(self):
        dataset = read_csv(os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv'), sep=",", features=False, label=True)
        self.assertEqual((699, 9), dataset.shape())
        self.assertEqual(8.0, dataset.X[0, 0])

    def test_write_read_without_features_or_label(self):
        original = Dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
        write_csv(filename='test.csv', dataset=original, sep=",", features=False, label=False)
        dataset = read_csv(filename='test.csv', sep=",", features=False, label=False)
        os.remove('test.csv')
        self.assertEqual((2, 2), dataset.shape())
