import torch
from trial_LN import TrialLN
from trial_ffn import TrialFFN
from trial_attention import TrialAttention
from simpleTransformer import SimpleTransformerBlock


class TrialEncodingBlock(SimpleTransformerBlock):
    """
    Represents a single encoding block within a Transformer architecture, integrating attention, layer normalization, and feedforward neural network (FFN) layers.

    Preconditions:
        - config must include keys such as 'embed_dim' to properly configure sub-modules.
    Postconditions:
        - Encoding block is fully initialized and can process input embeddings to produce encoded outputs.

    Parameters:
        config (dict): Configuration parameters for attention, layer normalization, and FFN sub-modules.

    Returns:
        torch.Tensor: The output of the encoding block after processing input embeddings.
    """

    def __init__(self, **config):
        super().__init__()
        self.__attention = TrialAttention(**config)
        self.__ln1 = TrialLN(**config)
        self.__ffn = TrialFFN(**config)
        self.__ln2 = TrialLN(**config)

    def forward(self, data_embs: torch.Tensor, masks=None) -> torch.Tensor:
        output_sub_layer_1 = self.__attention(data_embs, masks=masks if masks is not None else None)
        output_an1 = self.__ln1(data_embs + output_sub_layer_1)
        output_sub_layer_2 = self.__ffn(output_an1)
        output_final = self.__ln2(output_an1 + output_sub_layer_2)
        return output_final
