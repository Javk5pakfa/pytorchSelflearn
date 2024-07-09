from simpleTransformer import FeedForward
import torch.nn as nn


class TrialFFN(FeedForward):
    """
    Experimental implementation of FeedForward NN.
    """

    def __init__(self, **config):
        """

        :param config: {"input_dimension": int, ...}
        """

        super().__init__()
        self.__input_linear = nn.Linear(config["n_input_dimension"], 8 * config["n_input_dimension"])
        self.__activation = nn.ReLU()
        self.__output_linear = nn.Linear(8 * config["n_input_dimension"], config["n_input_dimension"])

    def forward(self, in_vec):
        """

        :param in_vec: Input torch.Tensor.
        :return:
        """

        in_vec = self.__input_linear(in_vec)
        in_vec = self.__activation(in_vec)
        in_vec = self.__output_linear(in_vec)
        return in_vec
