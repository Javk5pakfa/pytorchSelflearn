from simpleTransformer import FeedForward
import torch
import torch.nn as nn


class TrialFFN(FeedForward):
    """
    Implements a feed-forward neural network within the Transformer.

    Preconditions:
        - config must include 'embed_dim' and 'n_ffn_dimension'.
    Postconditions:
        - FFN is initialized.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    def __init__(self, **config):
        """

        :param config: {"embed_dim": int, ...}
        """

        super().__init__()
        d_emb = config['embed_dim']
        self.__input_linear = nn.Linear(d_emb, 4 * d_emb)
        self.__activation = nn.ReLU()
        self.__output_linear = nn.Linear(4 * d_emb, d_emb)

    def forward(self, in_vec) -> torch.Tensor:
        """

        :param in_vec: Input torch.Tensor.
        :return: learned torch.Tensor vector.
        """

        in_vec = self.__input_linear(in_vec)
        in_vec = self.__activation(in_vec)
        in_vec = self.__output_linear(in_vec)
        return in_vec
