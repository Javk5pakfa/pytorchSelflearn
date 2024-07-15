import numpy as np

from trial_transformer import TrialTransformer
from basic_FFN import get_batch_datasets
import torch
from json import load

if __name__ == '__main__':

    # Get raw datasets.
    ds = get_batch_datasets(0,
                            800,
                            "agn_{}_synthetic.csv",
                            "/Users/jackhu/PycharmProjects/pytorchSelflearn/data/agn_synthetic")

    with open("/Users/jackhu/PycharmProjects/pytorchSelflearn/dataTransformer/trial_configs.json", 'r') as f:
        options = load(f)

    # Initialize transformer object, make dataset list a Tensor.
    ds_tensor = torch.Tensor(np.array(ds)).requires_grad_(True)
    transform = TrialTransformer(**options)

    result = transform.forward(ds_tensor)
    print(result)
