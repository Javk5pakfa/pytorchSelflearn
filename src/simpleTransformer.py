"""
This script contains my introductory attempt at creating a Transformer algorithm.

Things to-do:
1. DONE: Determine a set of example data that the algorithm should learn from.
2. TODO: Design algorithm by making a flow-chart and a class diagram.
3. TODO: Write class and method signatures in this file.
4. TODO: Implement methods in code.
5. TODO: Implement integrated testing for methods.

*. TODO: Determine additional steps needed.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import abc


def read_csv(filename: str,
             abs_path: str = f"/Users/jackhu/PycharmProjects/pytorchSelflearn"):
    """
    Helper function to read a CSV file into a pandas dataframe. Currently
    only supports CSV file.
    :param filename: Name of the csv file to read.
    :param abs_path: Absolute path to the csv file to read.
    :return: A pandas dataframe.
    """

    if abs_path[-1] != "/":
        df = pd.read_csv(f"{abs_path}/{filename}")
    else:
        df = pd.read_csv(f"{abs_path}{filename}")
    return df


class Observation:
    """
    This object contains the raw data from one given dataset (observation).
    """

    def __init__(self, axis_titles: list,
                 data: pd.DataFrame,
                 metadata: dict = None):
        """
        The constructor method for the Observation class.
        :param axis_titles: Names of the axis titles.
        :param data: Actual data to be read.
        :param metadata: Additional information about the dataset.
        """

        self.axis_titles = axis_titles if axis_titles is not None else []
        self.data = data if data is not None else pd.DataFrame()
        self.metadata = metadata if metadata is not None else {}

        assert len(self.axis_titles) == len(self.data.columns)

    def get_axis_titles(self):
        return self.axis_titles

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata

    def add_axis_titles(self, axis_titles: list):
        self.axis_titles = axis_titles

    def add_data(self, data: pd.DataFrame):
        self.data = data

    def add_metadata(self, metadata: dict):
        self.metadata = metadata

    def has_axis_titles(self):
        return len(self.axis_titles) > 0

    def has_data(self):
        return len(self.data) > 0

    def has_metadata(self):
        return len(self.metadata) > 0

    def print(self):
        print("This observation contains axes: {}".format(self.axis_titles))
        print("with data: ")
        print(self.data)
        print("and metadata: {}".format(self.metadata))


class TPE(nn.Module, metaclass=abc.ABCMeta):
    """
    Very simple time-position encoding interface for my
    synthetic lightcurve data.
    """

    @abc.abstractmethod
    def forward(self, obs: Observation):
        pass


class SimpleTimePositionEncoding(TPE):
    """
    Test.
    """

    def __init__(self):
        super(SimpleTimePositionEncoding, self).__init__()

    def forward(self, obs: Observation):
        pass


if __name__ == "__main__":
    test_data = read_csv("agn_0_synthetic.csv",
                         "/Users/jackhu/PycharmProjects/pytorchSelflearn"
                         + "/data/agn_synthetic")

    test_data_obs = Observation(axis_titles=["x", "y"],
                                data=test_data,
                                metadata={'type': 'agn'})
    test_data_obs.print()
