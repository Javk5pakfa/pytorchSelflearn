"""
From https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
"""
import numpy as np
from simpleTransformer import TPE
import torch.nn as nn
import torch


class TrialTPE(TPE):
    """
    Implements positional encoding for the Transformer.

    Preconditions:
        - config must include 'n_input_dimension' and 'max_len'.
    Postconditions:
        - Positional encoding is initialized.
    """

    def __init__(self, d_emb: int, max_d_emb: int):
        """

        :param d_emb: Dimensionality of the embedding, as in for how many output
        features should the input be projected to.
        :param max_d_emb: Maximum dimensionality of the embedding. Essentially
        the sequence length.

        """

        super().__init__()
        self.__pe = None
        self.__tp_emb_tensor = None

        assert d_emb > 0, "Number of dimensionality must be greater than zero."
        self.__d_emb = d_emb
        self.__max_d_emb = max_d_emb

        # nn related variables.
        # Since actual number of features is equal or below __d_emb,
        # initialize linear layer to max number of features possible.
        self.__data_lin = nn.Linear(1, self.__d_emb)

    def forward(self, data: torch.Tensor) -> None:
        """
        :param data: incoming singular Tensor containing a number of sets of
        features. It assumes the first index is the number of sets of features
        contained within.

        :return: None. The class object now holds a Tensor with the combined
        embedding.
        """

        # For each set of features and their positions, find ones that have
        # padded nan values, obtain their indices, and truncate both Tensors
        # until only non-nan values are present.
        self.__tp_emb_tensor = []
        for each_set in data:
            feat_pos = each_set[0]
            feat     = each_set[1]

            # get mask for non-nan values in this set of features, then apply
            # to both feature and feature_position Tensors.
            feat_nans_mask = ~torch.isnan(feat)
            feat     = feat[feat_nans_mask]

            # Do encodings on positions and features.
            feat_pos_emb = self.__positional_encoding(self.__max_d_emb)[feat_nans_mask]
            feat_emb     = self.__data_lin(torch.unsqueeze(feat, -1))

            self.__tp_emb_tensor.append(feat_pos_emb + feat_emb)

        self.__tp_emb_tensor = torch.cat(self.__tp_emb_tensor, 0)

    def __get_angles(self, pos, i):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)).float() / self.__d_emb)
        return pos * angle_rates

    def __positional_encoding(self, n_pos):
        angle_rads = self.__get_angles(torch.arange(n_pos).unsqueeze(1),
                                       torch.arange(self.__d_emb).unsqueeze(0))

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return angle_rads

    def get_representation(self) -> torch.Tensor:
        """
        Getter method for embedded representation.

        :return: torch.Tensor of representation of the tire dataset.
        """

        return self.__tp_emb_tensor
