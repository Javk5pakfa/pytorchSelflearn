from trial_encoding_block import TrialEncodingBlock
from trial_tpe import TrialTPE
from trial_ffn import TrialFFN
from simpleTransformer import SimpleTransformer
import torch
import torch.nn as nn
from tqdm import tqdm


class TrialTransformer(SimpleTransformer):
    """
    Initializes a Transformer model configuration based on passed parameters and composes sub-components including positional encoding, encoding blocks, and a final linear layer.

    Preconditions:
        - config must be a non-empty dictionary containing necessary keys like 'n_input_dimension' and 'n_encoding_blocks'.
    Postconditions:
        - Transformer model fully initialized with specified sub-components ready for training or inference.

    Parameters:
        config (dict): Dictionary containing configuration parameters.

    Returns:
        None
    """

    def __init__(self, **config):
        """
        preconditions: all entries to config are not-none.
        :param config: Configurations for different modules of the transformer.
        """

        super().__init__()
        self.__config = config
        self.__ffn = TrialFFN(**self.__config)

        self.__transformer = nn.ModuleDict(
            dict(
                init_embeds=TrialTPE(self.__config["n_input_dimension"]),
                blocks=nn.ModuleList([TrialEncodingBlock(**self.__config) for _ in range(self.__config["n_encoding_blocks"])]),
                linear_final=TrialFFN(**self.__config),
            )
        )

    def forward(self, initial_dataset: []) -> torch.Tensor:
        """

        :param initial_dataset: List of Observations containing raw tabular data.
        :return: torch.Tensor with learned representation of this dataset.
        """

        pre_transform_embs = []
        for obs in tqdm(initial_dataset, desc="Begin pretrain encoding..."):
            self.__transformer.init_embeds.forward(obs)
            pre_transform_embs.append(self.__transformer.init_embeds.get_representation())
        pre_transform_embs = torch.cat(pre_transform_embs, dim=0)

        x = None
        for block in tqdm(self.__transformer.blocks, desc="Begin transformer block ops..."):
            x = block(pre_transform_embs)
        x = self.__transformer.linear_final.forward(x)

        return x
