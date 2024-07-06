"""The dataset for the characteristics and excess returns."""

import typing
import pandas
import torch


class CustomDataset(torch.utils.data.Dataset):
    """The custom dataset for the characteristics and excess returns.

    Args:
        data (pandas.DataFrame): The data.
        device (torch.device): The device to use.
    """

    def __init__(
        self,
        data: pandas.DataFrame,
        characteristics: typing.List[str],
        device: torch.device,
    ):
        # Get the characteristics
        self.characteristics: typing.List[str] = characteristics
        # Get the data
        self.data = (
            torch.from_numpy(
                data[
                    characteristics
                    + [f"x_{i}" for i in self.characteristics]
                    + ["mktrf"]
                    + ["excess_ret"]
                ].values
            )
            .float()
            .to(device)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx, :-1],
            self.data[idx, -1],
        )
