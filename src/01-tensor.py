import torch
from torch import tensor
import numpy as np


def tensor_basic():
    data = np.random.rand(3, 4)
    tensor1 = tensor(data)
    tensor2 = torch.rand((3, 4,))

    tensor_stack = torch.stack(tensors=(tensor1, tensor2))
    tensor_cat = torch.cat(tensors=(tensor1, tensor2))

    print(tensor_stack)
    print(tensor_stack.size())

    print(tensor_cat)
    print(tensor_cat.size())


def tensor_multiplication():
    tensor1 = tensor([[1, 2], [3, 4]])
    tensor2 = tensor([[5, 6], [7, 8]])

    # Element multiplication
    print(tensor1 * tensor2)

    # Matrix ~
    print(tensor1 @ tensor2)


if __name__ == "__main__":
    # tensor_basic()
    tensor_multiplication()
