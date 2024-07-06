"""Early stopping."""

import torch
from hoisaai.machine_learning.neural_network.dataset import CustomDataset
from hoisaai.machine_learning.neural_network.metric import r2


class EarlyStopping:
    """Early stopping.

    Args:
        patience (int): The patience.
        validation_sample (CustomDataset): The validation sample.
        batch_size (int): The batch size.
    """

    def __init__(
        self,
        patience: int,
        validation_sample: CustomDataset,
        batch_size: int,
    ):
        self.patience: int = patience
        self.validation_sample: CustomDataset = validation_sample
        self.batch_size: int = batch_size
        self.counter: int = 0
        self.best_score: float = None
        self.early_stop: bool = False
        self.checkpoint: torch.nn.Module = None

    def __call__(self, model: torch.nn.Module):
        # Calculate the R2 score
        score = r2(
            models=[model],
            sample_dataset=self.validation_sample,
            batch_size=self.batch_size,
        )
        # Check if the score is the best score
        if self.best_score is None:
            # Set the best score
            self.best_score = score
            # Set the checkpoint
            self.checkpoint = model
        elif score < self.best_score:
            # Increment the counter
            self.counter += 1
            # Check if the counter is greater than or equal to the patience
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Reset the counter
            # Set the best score
            self.best_score = score
            # Set the checkpoint
            self.checkpoint = model
            # Reset the counter
            self.counter = 0
