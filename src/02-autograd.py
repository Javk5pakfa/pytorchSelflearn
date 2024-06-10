# First following the tutorial on fundamentals of pytorch.autograd.

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10


def example1():
    # Variable "a" is a leaf tensor because it is created by the user.
    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)

    # These are non-leaf tensors, which were made as a result of a tensor
    # operation done to them.
    b = torch.exp(a)
    c = 2 * b
    d = c + 1
    out = d.sum()

    # Call the backward() fn on the leaf tensor at the end of the sequence
    # allows access to a.grad gradient values.
    out.backward()
    print(a.grad)


def example2():
    # Define a small model for example.

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

    # Initial weights.
    print("Initial weights and gradients:")
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad)
    print("\n")

    # Perform a one training batch run. Use square of the Euclidean distance
    # between prediction and the ideal_output. Also will use a basic gradient
    # stochastic gradient descent optimizer. SGD.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    prediction = model(some_input)

    loss = (ideal_output - prediction).pow(2).sum()
    print(loss)

    # Run backward() on loss. Obtain gradient values.
    loss.backward()
    print("Weights and gradients before optimizer: ")
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])
    print("\n")

    # Apply optimizer.
    optimizer.step()
    print("Weights and gradients after optimizer: ")
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])
    print("\n")

    print("Loss after optimizer: {}".format(loss))

    # Zero the gradients.
    optimizer.zero_grad()


def example3():
    """
    Autograd needs these intermediate values to perform gradient
    computations. For this reason, you must be careful about using in-place
    operations when using autograd. Doing so can destroy information you need
    to compute derivatives in the backward() call. PyTorch will even stop you
    if you attempt an in-place operation on leaf variable that requires
    autograd, as shown below.
    :return: Nothing
    """

    a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
    b = torch.sin_(a)  # <-- Error!!
    print(b)


if __name__ == "__main__":
    example3()
