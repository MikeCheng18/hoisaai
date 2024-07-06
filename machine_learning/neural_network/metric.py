"""The metric module."""

import typing
import numpy
import torch
from hoisaai.machine_learning.neural_network.dataset import CustomDataset


def r2(
    models: typing.List[torch.nn.Module],
    sample_dataset: CustomDataset,
    batch_size: int,
) -> float:
    """Calculate the R2 score.

    Args:
        models (typing.List[torch.nn.Module]): The models.
        sample_dataset (CustomDataset): The sample dataset.
        batch_size (int): The batch size.
    """
    # Get the number of batches
    num_batches: int = len(sample_dataset) // batch_size
    # Set the models to evaluation mode
    for model in models:
        model.eval()
    # Initialize the sum of squares of residuals and returns
    sum_of_squares_of_residuals: float = 0.0
    sum_of_squares_of_returns: float = 0.0
    # Iterate over the batches
    for i in range(num_batches):
        # Initialize the expected returns
        expected_returns: typing.List[numpy.ndarray] = []
        # Iterate over the models
        for model in models:
            # Get the batch start and end
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            # Get the characteristic and portfolio tensor and return tensor
            characteristic_and_portfolio_tensor, return_tensor = sample_dataset[
                batch_start:batch_end
            ]
            # Calculate the expected returns
            expected_returns.append(
                model(characteristic_and_portfolio_tensor).detach().cpu().numpy()
            )
        # Calculate the mean of the expected returns
        expected_returns: numpy.ndarray = numpy.stack(
            expected_returns,
            axis=0,
        ).mean(
            axis=0,
            keepdims=False,
        )
        # Get the actual returns
        actual_returns: numpy.ndarray = return_tensor.detach().cpu().numpy()
        # Calculate the sum of squares of residuals and returns
        sum_of_squares_of_residuals += numpy.sum(
            (expected_returns - actual_returns) ** 2,
        )
        sum_of_squares_of_returns += numpy.sum(
            actual_returns**2,
        )
    # Return the R2 score
    return 1.0 - sum_of_squares_of_residuals / sum_of_squares_of_returns
