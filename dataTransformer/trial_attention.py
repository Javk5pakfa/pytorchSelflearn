import torch
import torch.nn as nn
from simpleTransformer import SelfAttention


class TrialAttention(SelfAttention):
    """
    Basic attention layer.
    """

    def __init__(self, **config):
        super().__init__()

        # Get initial variables from config dictionary.
        self.__d_embd = config['n_input_dimension']
        self.__c_attn = nn.Linear(self.__d_embd, 3 * self.__d_embd, bias=config['bias'])  # PyTorch doesn't support bias=False. TODO.
        self.__softmax = nn.Softmax(dim=-1)

    def forward(self, in_embedded_vecs: torch.Tensor, masks=None):
        # masks.shape = (# of elements in this batch of sequences,)

        # For multi-head,
        # b_size, s_length, d_embd = in_embedded_vecs.size()

        # Obtain q, k, v matrices.
        q, k, v = self.__c_attn(in_embedded_vecs).split(self.__d_embd, dim=1)

        # Attention procedures following the original paper.
        att = q @ k.transpose(-1, 0)
        att = att / k.size(-1) ** 0.5

        if masks is not None:
            att = att.masked_fill_(masks, float('-inf'))

        att = self.__softmax(att)
        att_final = att @ v

        # For multi-head,
        # att_final = att_final.transpose(0, 1)

        return att_final
