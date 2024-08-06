from simpleTransformer import LayerNormalization
import torch
import torch.nn as nn
import torch.nn.functional as f


class TrialLN(LayerNormalization):
    """
    Implements layer normalization.

    Preconditions:
        - config must include 'embed_dim'.
    Postconditions:
        - Layer normalization is initialized.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    def __init__(self, **config):
        super().__init__()
        self.__weight = nn.Parameter(torch.ones(config['embed_dim']))
        if config['bias'] is not None:
            self.__bias = nn.Parameter(torch.zeros(config['embed_dim']))

    def forward(self, in_vec) -> torch.Tensor:
        return f.layer_norm(in_vec, self.__weight.shape, self.__weight, self.__bias)
