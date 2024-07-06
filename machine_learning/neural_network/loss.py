"""The loss functions for the neural network."""

import math
import torch


class MseWithL1(torch.nn.Module):
    """The Mean Squared Error with L1 penalty.

    Args:
        model (torch.nn.Module): The model.
        l1_lambda (float): The L1 regularization lambda.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        l1_lambda: float,
    ):
        super(MseWithL1, self).__init__()
        self.model = model
        self.l1_lambda = l1_lambda

    def forward(
        self,
        outputs: torch.Tensor,
        return_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass of the model."""
        # Calculate the mean squared error loss
        mse_loss = torch.nn.MSELoss()(outputs, return_tensor)
        # Ensure that the loss is not NaN
        assert math.isnan(mse_loss.item()) is False
        # Calculate the L1 regularization term
        l1_term = torch.tensor(0.0, requires_grad=True)
        # Iterate over the parameters
        for name, weights in self.model.named_parameters():
            # Exclude the bias terms
            if "bias" not in name:
                # Calculate the L1 norm
                weights_sum = torch.sum(torch.abs(weights))
                # Add the L1 norm to the L1 term
                l1_term = l1_term + weights_sum
        # Ensure that the L1 term is not NaN
        assert math.isnan(l1_term.item()) is False
        # Calculate the L1 penalty
        l1_penalty = self.l1_lambda * l1_term
        # Aggregate the loss
        return mse_loss + l1_penalty
