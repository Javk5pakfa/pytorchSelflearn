# First following the tutorial on fundamentals of pytorch.autograd.

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

def example1():
    # Variable "a" is a leaf tensor because it is created by the user.
    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)

    # These are non-leaf tensors, which were made as a result of a tensor operation
    # done to them.
    b = torch.exp(a)
    c = 2 * b
    d = c + 1
    out = d.sum()

    # Call the backward() fn on the leaf tensor at the end of the sequence allows
    # access to a.grad gradient values.
    out.backward()
    print(a.grad)


def example2():
    # Define a small model for example.

    BATCH_SIZE = 16
    DIM_IN = 1000
    HIDDEN_SIZE = 100
    DIM_OUT = 10

    class TinyModel(torch.nn.Module):

        def __init__(self):
            super(TinyModel, self).__init__()

            self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
            self.relu = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

    model = TinyModel()
    

if __name__ == "__main__":
    example2()
