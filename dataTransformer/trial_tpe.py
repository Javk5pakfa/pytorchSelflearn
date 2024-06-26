"""
From https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
"""
import numpy as np
from simpleTransformer import Observation, TPE
from basic_FFN import get_batch_datasets
import torch.nn as nn
import torch
from tqdm import tqdm


class TrialTPE(TPE):
    """
    Trial one dimensional time-position embedder.
    """

    def __init__(self, d_emb: int, n_bands=1):
        """

        :param d_emb: Dimensionality of the embedding.
        :param n_bands: Number of bands in this Observation.
        """

        super().__init__()
        self.__pe = None
        self.__xy_rep_tensor = None

        assert d_emb > 0, "Number of dimensionality must be greater than zero."
        assert n_bands > 0, "Number of bands must be greater than zero."
        self.__d_emb = d_emb
        self.__n_bands = n_bands

        # nn related variables.
        self.__data_lin = nn.Linear(1, self.__d_emb)

    def forward(self, obs: Observation):

        # Load dataset.
        obs_data = obs.get_data()
        n_pos = len(obs_data.x)
        x_data = obs_data.x.to_numpy()
        y_data = obs_data.y.to_numpy()

        # Positional encoding on the x-axis.
        x_tensor_temp = torch.Tensor(self.__positional_encoding(n_pos))
        x_tensor_final = torch.empty(0, self.__d_emb)

        # Put y values through a linear layer. Since we only want to pass
        # non-nan values to the next layer, the positions of those in x-axis
        # will also be included, with the rest discarded.
        y_emb_vals = []
        for index, val in enumerate(y_data):
            if not np.isnan(val):
                x_tensor_final = torch.cat((x_tensor_final, torch.reshape(x_tensor_temp[index], (1, self.__d_emb))), dim=0)
                y_emb_vals.append(self.__data_lin(torch.Tensor([val])))
        y_tensor_final = torch.stack(y_emb_vals)

        # Finally, linearly combine encodings.
        assert x_tensor_final.shape == y_tensor_final.shape
        self.__xy_rep_tensor = x_tensor_final + y_tensor_final

    def __get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000,
                                   (2 * (i // 2)) / np.float32(self.__d_emb))
        return pos * angle_rates

    def __positional_encoding(self, num_pos: int):
        angle_rads = self.__get_angles(np.arange(num_pos)[:, np.newaxis],
                                       np.arange(self.__d_emb)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return angle_rads

    def __split_list_into_bands(self, original_list):
        # Calculate the length of each band
        total_length = len(original_list)
        band_size = total_length // self.__n_bands  # integer division to find minimum size of each band
        remainder = total_length % self.__n_bands  # elements that won't fit evenly into bands

        # Create the sub-lists
        bands = []
        start_index = 0
        for i in range(self.__n_bands):
            if i < remainder:
                # If there's a remainder, add one more element to the first
                # 'remainder' bands
                end_index = start_index + band_size + 1
            else:
                end_index = start_index + band_size
            bands.append(original_list[start_index:end_index])
            start_index = end_index

        return bands

    def get_representation(self):
        """
        Getter for embedded representation.
        :return: torch.Tensor of representation of one dataset.
        """

        return self.__xy_rep_tensor


if __name__ == '__main__':
    train_ds = get_batch_datasets(0,
                                  800,
                                  "agn_{}_synthetic.csv",
                                  "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/agn_synthetic",
                                  ["x", "y"],
                                  {'type': 'agn'})

    encoder = TrialTPE(32)
    encoded_data = []
    for observe in tqdm(train_ds):
        encoder.forward(observe)
        encoded_data.append(encoder.get_representation())
