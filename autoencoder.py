"""The autoencoder model and related functions."""

import numpy
import torch

from hoisaai.machine_learning.neural_network.fully_connected_layer import (
    Net,
    NonLinearActivation,
)


class Autoencoder(torch.nn.Module):
    """The autoencoder model.

    Args:
        input_size (int): The number of input features.
        k (int): The number of output features.
    """

    def __init__(
        self,
        input_size: int,
        k: int,
    ):
        super(Autoencoder, self).__init__()
        self.input_size: int = input_size
        # Define the encoder in child classes
        self.encoder: torch.nn.Module = torch.nn.Module()
        # Define the decoder, portfolio formed from the characteristics and equal-weighted portfolio
        self.decoder = torch.nn.Linear(self.input_size + 1, k)

    def forward(
        self,
        characteristic_and_portfolio_tensor: torch.Tensor,
    ):
        """The forward pass of the model."""
        return (
            self.encoder(
                characteristic_and_portfolio_tensor[
                    :,
                    : self.input_size,
                ]
            )[
                :,
                numpy.newaxis,
                :,
            ]
            @ self.decoder(
                characteristic_and_portfolio_tensor[
                    :,
                    self.input_size :,
                ]
            )[
                :,
                :,
                numpy.newaxis,
            ]
        ).squeeze()


class CA1(Autoencoder):
    """CA1 model from Gu, Kelly, and Xiu (2019) with a single hidden layer of 32 units.

    Args:
        input_size (int): The number of input features.
        k (int): The number of output features.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        k: int,
        dropout: float,
    ):
        Autoencoder.__init__(self, input_size, k)
        self.encoder = torch.nn.Sequential(
            NonLinearActivation(input_size, 32, dropout),
            Net(32, k),
        )
