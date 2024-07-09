from dataTransformer.basic_FFN import get_batch_datasets
from trial_encoding_block import TrialEncodingBlock
from trial_tpe import TrialTPE
from trial_ffn import TrialFFN
from simpleTransformer import SimpleTransformer
import torch
import torch.nn as nn

from tqdm import tqdm
import json as js


class TrialTransformer(SimpleTransformer):

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

        :param initial_dataset: List of
        :return:
        """

        pre_transform_embs = []
        print("Begin pretrain encoding...")
        for obs in tqdm(initial_dataset):
            self.__transformer.init_embeds.forward(obs)
            pre_transform_embs.append(self.__transformer.init_embeds.get_representation())
        pre_transform_embs = torch.cat(pre_transform_embs, dim=0)

        print("Begin transformer block ops...")
        x = None
        for block in tqdm(self.__transformer.blocks):
            x = block(pre_transform_embs)
        x = self.__transformer.linear_final.forward(x)

        return x


if __name__ == '__main__':

    train_ds = get_batch_datasets(0,
                                  800,
                                  "agn_{}_synthetic.csv",
                                  "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/agn_synthetic",
                                  ["x", "y"],
                                  {'type': 'agn'})

    with open("trial_configs.json", "r") as f:
        config_dic = js.load(f)

    test_transformer = TrialTransformer(**config_dic)

    result = test_transformer.forward(train_ds)
    print(result.shape)
