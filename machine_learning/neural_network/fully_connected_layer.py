"""The fully connected neural network model."""

import torch


class Net(torch.nn.Module):
    """The fully connected neural network model.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super(Net, self).__init__()
        # Define the fully connected layer
        self.fully_connected = torch.nn.Linear(
            input_size,
            output_size,
        )
        # Initialize the weights and bias
        torch.nn.init.kaiming_normal_(
            self.fully_connected.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        torch.nn.init.zeros_(self.fully_connected.bias)

    def forward(self, x: torch.Tensor):
        """The forward pass of the model."""
        # Apply the fully connected layer
        return self.fully_connected(x)


class NonLinearActivation(torch.nn.Module):
    """The non-linear activation layer.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
    ):
        super(NonLinearActivation, self).__init__()
        self.net = torch.nn.Sequential(
            Net(input_size, output_size),
            torch.nn.BatchNorm1d(output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """The forward pass of the model."""
        return self.net(x)
