import torch
import torch.nn as nn
from simpleTransformer import Observation, read_csv
import numpy as np
from tqdm import tqdm


class RegressionMLP(nn.Module):
    """
    Simple multi-layer perceptron network for regressing tabular data.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def ground_truth(num=100):
    """
    The ground truth of this particular exercise.
    :param num:
    :return:
    """

    a, b, c, d, f, g = -6.5, -2, 5, 1.5, 1, 0.3
    x_data = np.linspace(0, 2, num)

    y_func_agn = g + a * (x_data - f) ** 4 + b * (x_data - f) ** 3 + c * (
                x_data - f) ** 2 + d * (x_data - f)
    return torch.Tensor(y_func_agn)


def get_batch_datasets(start_num, end_num, file_name: str, file_path: str) -> []:
    """
    Getting a batch of datasets from csv files.

    :param start_num: The starting number of the files.
    :param end_num: The ending number of the files.
    :param file_name: The file name of each of the files. Should have some formatting to allow the numbering to work.
    :param file_path: Absolute path of the csv files.
    :return: A list of Observation objects.
    """

    ds = []
    for i in tqdm(np.arange(start_num, end_num), desc='Getting Datasets'):
        df = read_csv(file_name.format(i), file_path).to_numpy()
        df_x, df_y = df[:, 0], df[:, 1]
        df_reformat = np.stack((df_x, df_y), axis=0)
        ds.append(df_reformat)

    return ds


def get_batch_tensors(ds: []) -> [torch.Tensor]:
    """
    Getting a batch of tensors from a list of datasets in Observation format.
    :param ds: List of datasets in pd.Dataframe format.
    :return: A list of data tensors.
    """

    tensors = []
    for df in ds:
        tensors.append(torch.Tensor(df.get_data().y.to_numpy()).requires_grad_(True))
    return tensors
