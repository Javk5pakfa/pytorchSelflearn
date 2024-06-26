import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from simpleTransformer import Observation, read_csv
import numpy as np
import copy
import matplotlib.pyplot as plt


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


def pad_nan(in_array: np.array) -> np.array:

    # Step 1: Identify missing values
    missing_indices = np.isnan(in_array)

    # Step 2: Calculate the minimum and maximum values of the existing data
    existing_values = in_array[~missing_indices]
    min_val = np.min(existing_values)
    max_val = np.max(existing_values)

    # Step 3: Generate random values within the range [min_val, max_val]
    random_values = np.random.uniform(low=min_val, high=max_val,
                                      size=np.sum(missing_indices))

    # Step 4: Fill in the missing values with the generated random values
    in_array[missing_indices] = random_values

    return in_array


def get_batch_datasets(start_num, end_num, file_name: str, file_path: str, axis_titles: [], metadata: {}):

    ds = []
    for i in np.arange(start_num, end_num):
        df = read_csv(file_name.format(i), file_path)
        df_obs = Observation(axis_titles=axis_titles, data=df, metadata=metadata)
        ds.append(df_obs)

    return ds


def get_batch_tensors(ds: []):

    tensors = []
    for df in ds:
        tensors.append(torch.Tensor(pad_nan(df.get_data().y.to_numpy())).requires_grad_(True))
    return tensors


if __name__ == '__main__':

    # Import a bunch of data.
    train_ds = get_batch_datasets(0,
                                  800,
                                  "agn_{}_synthetic.csv",
                                  "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/agn_synthetic",
                                  ["x", "y"],
                                  {'type': 'agn'})
    test_ds = get_batch_datasets(900,
                                 1000,
                                 "agn_{}_synthetic.csv",
                                 "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/test",
                                 ["x", "y"],
                                 {'type': 'agn'})

    train_tensors = torch.stack(get_batch_tensors(train_ds))
    test_tensors = torch.stack(get_batch_tensors(test_ds))

    # Tensor datasets.
    train_dataset = TensorDataset(train_tensors)
    test_dataset = TensorDataset(test_tensors)

    # Tensor dataloaders.
    train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=50, shuffle=False)

    # Initialize ground truth.
    grd_truth = ground_truth()

    # Define RegressionMLP.
    em_mlp = RegressionMLP(100, 200, 100)
    loss_function = nn.MSELoss()
    optimize = optim.SGD(em_mlp.parameters(), lr=0.01)

    best_mse = np.inf
    best_weights = None
    history = []

    num_epoch = 20

    for epoch in range(num_epoch):
        print("_" * 10)
        print("Epoch {}/{}".format(epoch + 1, num_epoch))

        # Training steps
        em_mlp.train()
        running_loss = 0.0

        for batch in train_loader:
            tensor = batch[0]
            # Forward pass
            pred = em_mlp(tensor)
            loss = loss_function(pred, tensor)

            # Backward pass
            optimize.zero_grad()
            loss.backward()
            optimize.step()

            running_loss += loss.item()

        print("Training loss after epoch {}: {:.4f}".format(epoch + 1, running_loss / len(train_loader)))

        # Now, evaluate accuracy
        em_mlp.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                tensor = batch[0]
                test_mlp_pred = em_mlp(tensor)
                loss = loss_function(test_mlp_pred, tensor)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        history.append(test_loss)
        print("Evaluated loss after epoch {}: {:.4f}".format(epoch + 1, test_loss))
        print("_" * 10)

        if test_loss < best_mse:
            best_mse = test_loss
            best_weights = copy.deepcopy(em_mlp.state_dict())

    plt.scatter(range(num_epoch), history[:num_epoch])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    em_mlp.load_state_dict(best_weights)
