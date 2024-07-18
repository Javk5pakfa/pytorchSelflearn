import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from trial_transformer import TrialTransformer
from trial_tpe import TrialTPE
from basic_FFN import get_batch_datasets
import torch
import torch.nn as nn
from json import load


if __name__ == '__main__':

    # Get raw datasets.
    train_ds = get_batch_datasets(0,
                                  800,
                            "agn_{}_synthetic.csv",
                            "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/agn_synthetic")
    validate_ds = get_batch_datasets(800,
                                     899,
                                 "agn_{}_synthetic.csv",
                                 "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/validate")

    with open("/Users/jackhu/PycharmProjects/pytorchSelflearn/dataTransformer/trial_configs.json", 'r') as f:
        options = load(f)

    # Initialize transformer object, make dataset list a Tensor.
    train_ds_tensor = torch.Tensor(np.array(train_ds)).requires_grad_(True)
    valid_ds_tensor = torch.Tensor(np.array(validate_ds)).requires_grad_(True)
    transform = TrialTransformer(**options)

    # To training!
    # Get the datasets.
    train_dataset = TensorDataset(train_ds_tensor)
    validate_dataset = TensorDataset(valid_ds_tensor)

    # Get the dataloaders.
    train_loader = DataLoader(dataset=train_dataset, batch_size=options['batch_size'], shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=options['batch_size'], shuffle=True)

    # Define training functions.
    loss_function = nn.MSELoss()
    optimize = optim.SGD(transform.parameters(), lr=0.001)
    best_mse = np.inf
    best_weights = None
    num_epochs = 10

    data_tpe = TrialTPE(options['n_input_dimension'], options['n_max_dimension'])

    # Epoch training loop.
    for epoch in range(num_epochs):
        print("_" * 10, flush=True)
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        transform.train()
        running_loss = 0.0

        for batch in tqdm(train_loader):
            data_tensor = batch[0]

            data_tpe.forward(data_tensor)
            seq2_tensor = data_tpe.get_representation()

            # Currently, mask incoming data's missing values away. Might not
            # be the best approach? Update 07/18: passing the entire sequences
            # through now.
            pred = transform.forward(data_tensor)  # (n_sequence, 32)
            loss = loss_function(pred, seq2_tensor)

            optimize.zero_grad()
            loss.backward()
            optimize.step()

            running_loss += loss.item()

        print("Training loss: {}".format(running_loss))

        # Evaluate accuracy.
        transform.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in validate_loader:
                test_data_tensor = batch[0]
                data_tpe.forward(test_data_tensor)
                test_seq_tensor = data_tpe.get_representation()

                pred = transform.forward(test_data_tensor)
                loss = loss_function(pred, test_seq_tensor)
                test_loss += loss.item()

        test_loss /= len(validate_loader)
        print("Test loss: {}".format(test_loss))
        print("_" * 10)

    # exit(0)  # EXPERIMENTAL: Break.
    # ---------- Prediction section ----------

    # Load a test dataset.
    test_ds = get_batch_datasets(900,
                                 999,
                                 "agn_{}_synthetic.csv",
                                 "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/test")
    test_ds_tensor = torch.Tensor(np.array(test_ds)).requires_grad_(True)
    test_dataset = TensorDataset(test_ds_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=options['batch_size'], shuffle=True)

    test_batch = next(iter(test_loader))
    test_sequence = torch.unsqueeze(test_batch[0][0], 0)  # shape = (1, 2, 100)
