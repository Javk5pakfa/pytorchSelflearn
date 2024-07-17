from simpleTransformer import FeedForward
import torch
import torch.nn as nn


class TrialFFN(FeedForward):
    """
    Implements a feed-forward neural network within the Transformer.

    Preconditions:
        - config must include 'n_input_dimension' and 'n_ffn_dimension'.
    Postconditions:
        - FFN is initialized.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    def __init__(self, **config):
        """

        :param config: {"n_input_dimension": int, ...}
        """

        super().__init__()
        self.__input_linear = nn.Linear(config["n_input_dimension"], 8 * config["n_input_dimension"])
        self.__activation = nn.ReLU()
        self.__output_linear = nn.Linear(8 * config["n_input_dimension"], config["n_input_dimension"])

    def forward(self, in_vec) -> torch.Tensor:
        """

        :param in_vec: Input torch.Tensor.
        :return: learned torch.Tensor vector.
        """

        in_vec = self.__input_linear(in_vec)
        in_vec = self.__activation(in_vec)
        in_vec = self.__output_linear(in_vec)
        return in_vec
