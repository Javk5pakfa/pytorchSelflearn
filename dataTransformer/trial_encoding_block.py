import torch
from trial_LN import TrialLN
from trial_ffn import TrialFFN
from trial_attention import TrialAttention
from simpleTransformer import SimpleTransformerBlock


class TrialEncodingBlock(SimpleTransformerBlock):
    """
    Experimental implementation of transformer encoding block
    """

    def __init__(self, **config):
        super().__init__()
        self.__attention = TrialAttention(**config)
        self.__ln1 = TrialLN(**config)
        self.__ffn = TrialFFN(**config)
        self.__ln2 = TrialLN(**config)

    def forward(self, data_embs: torch.Tensor) -> torch.Tensor:
        output_sub_layer_1 = self.__attention(data_embs)
        output_an1 = self.__ln1(data_embs + output_sub_layer_1)
        output_sub_layer_2 = self.__ffn(output_an1)
        output_final = self.__ln2(output_an1 + output_sub_layer_2)
        return output_final
