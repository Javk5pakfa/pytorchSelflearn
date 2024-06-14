import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import abc


class CustomEmbedder(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for custom embedders.
    """

    @abc.abstractmethod
    def parse_labels(self, labels: torch.LongTensor) -> torch.LongTensor:
        """
        Method to process labels associated with input data.
        :param labels:
        :return: torch.Tensor of input labels.
        """

        pass

    @abc.abstractmethod
    def parse_pos_sensitive_data(self, pos_sensitive_data: torch.LongTensor) -> torch.LongTensor:
        """
        Method that handles position-sensitive data, such as temporal data
        or data that has spatial correlations.

        :param pos_sensitive_data: torch.Tensor of input data.
        :return: torch.Tensor of embedded data.
        """

        pass

    @abc.abstractmethod
    def parse_reg_data(self, reg_data: torch.FloatTensor) -> torch.FloatTensor:
        """
        Method that handles regular, non-position-sensitive data and creates
        an embedding.
        :param reg_data:
        :return:
        """

        pass


class SynDataEmbedder(CustomEmbedder):

    def __init__(self):
        super(SynDataEmbedder, self).__init__()
        self.embedding = nn.Embedding(2, 8)

    def parse_labels(self, labels: torch.LongTensor) -> torch.LongTensor:
        pass

    def parse_pos_sensitive_data(self, pos_sensitive_data: torch.LongTensor) -> torch.LongTensor:
        pass

    def parse_reg_data(self, reg_data: torch.FloatTensor) -> torch.FloatTensor:
        pass
