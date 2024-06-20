import torch
import torch.nn as nn
import torch.optim as optim
from simpleTransformer import Observation, read_csv
import numpy as np
import copy


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
    return torch.Tensor(np.stack((x_data, y_func_agn), axis=1))


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


# Import data and print out.
train_data = read_csv("agn_0_synthetic.csv",
                     "/Users/jackhu/PycharmProjects/pytorchSelflearn/data"
                     "/agn_synthetic")
test_data = read_csv("agn_900_synthetic.csv",
                     "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/test")

train_data_obs = Observation(axis_titles=["x", "y"],
                             data=train_data,
                             metadata={'type': 'agn'})
test_data_obs = Observation(axis_titles=["x", "y"],
                            data=test_data,
                            metadata={'type': 'agn'})

# Generate 1 band for analyze.
train_tensor = torch.Tensor(pad_nan(train_data_obs.get_data().to_numpy())).requires_grad_(True)
test_tensor = torch.Tensor(pad_nan(test_data_obs.get_data().to_numpy())).requires_grad_(True)

# Initialize ground truth.
grd_truth = ground_truth()


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


# Define RegressionMLP.
em_mlp = RegressionMLP(2, 32, 1)
loss_function = nn.MSELoss()
optimize = optim.SGD(em_mlp.parameters(), lr=0.01)

best_mse = np.inf
best_weights = None
history = []

num_epoch = 100

for epoch in range(num_epoch):
    # Training steps.
    em_mlp.train()

    # Forward pass.
    pred = em_mlp(train_tensor)
    loss = loss_function(pred[:, 0], train_tensor[:, 1])
    print("Forward pass loss: {}".format(loss))

    # Backward pass
    optimize.zero_grad()
    loss.backward()

    # Update weights.
    optimize.step()

    # Now, evaluate accuracy.
    em_mlp.eval()
    test_mlp = em_mlp(test_tensor)
    mse = loss_function(test_mlp[:, 0], test_tensor[:, 1])
    mse = float(mse)
    history.append(mse)
    print("Evaluated loss after 1 epoch: {}".format(mse))

    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(em_mlp.state_dict())
