import numpy as np
import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file (data file) into a Dataset object

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has feature names, by default False
    label : bool, optional
        Whether the file has a label (last column), by default False

    Returns
    -------
    Dataset
        The dataset object
    """
    data = pd.read_csv(filename, sep=sep)

    if label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        label_name = data.columns[-1] if features else None
        feature_names = data.columns[:-1].tolist() if features else None
    else:
        X = data.to_numpy()
        y = None
        label_name = None
        feature_names = data.columns.tolist() if features else None

    return Dataset(X, y, features=feature_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether to write the feature names, by default False
    label : bool, optional
        Whether to write the label, by default False
    """
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label if dataset.label is not None else 'y'] = dataset.y

    data.to_csv(filename, sep=sep, index=False)
