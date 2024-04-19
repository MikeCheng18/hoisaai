"""
This module contains classes that represent sliding window transformations on tensors.
"""

import typing

import jax
import numpy

from hoisaai.layer_1.model import StatefulTensor, Tensor, get_tensor


class SlidingWindowInSample(StatefulTensor):
    """
    A class that represents a sliding window transformation on a tensor.

    Args:
        tensor (Tensor): The input tensor.
        in_sample_window_size (int): The size of the sliding window.

    Attributes:
        in_sample_window_size (int): The size of the sliding window.
        stateful (bool): Indicates whether the model is stateful or not.
        tensor (Tensor): The input tensor.
        x (jaxlib.xla_extension.ArrayImpl): The underlying tensor data.

    Methods:
        transform(): Applies the sliding window transformation on the tensor.

    """

    def __init__(
        self,
        tensor: Tensor,
        in_sample_window_size: int,
    ) -> None:
        super().__init__(x=None)
        self.in_sample_window_size: int = in_sample_window_size
        self.tensor: Tensor = tensor

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Applies the sliding window transformation on the tensor.
         (..., Observation, Dependent variable and independent variable)
            -> (..., Window, Observation, Dependent variable and independent variable

        Yields:
            typing.Any: The transformed tensor.

        """
        if self.x is None or self.stateful is False:
            self.x = jax.numpy.swapaxes(
                numpy.lib.stride_tricks.sliding_window_view(
                    x=get_tensor(
                        tensor=self.tensor,
                    )[
                        ...,
                        # Observation
                        :-1,
                        # Dependent variable and independent variable
                        :,
                    ],
                    window_shape=self.in_sample_window_size,
                    # Observation
                    axis=-2,
                ),
                -2,
                -1,
            )
        yield self.x


class SlidingWindowOutOfSample(StatefulTensor):
    """
    A class representing a sliding window transformation applied to out-of-sample data.

    Args:
        tensor (Tensor): The input tensor.
        in_sample_window_size (int): The size of the in-sample window.

    Attributes:
        in_sample_window_size (int): The size of the in-sample window.
        stateful (bool): Indicates whether the model is stateful or not.
        tensor (Tensor): The input tensor.
        x (jaxlib.xla_extension.ArrayImpl): The underlying tensor data.

    Methods:
        transform(): Applies the sliding window transformation and yields the result.

    """

    def __init__(
        self,
        tensor: Tensor,
        in_sample_window_size: int,
    ) -> None:
        super().__init__(x=None)
        self.in_sample_window_size: int = in_sample_window_size
        self.tensor: Tensor = tensor

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Applies the sliding window transformation to the input tensor.
        (..., Observation, Dependent variable and independent variable)
            -> (..., Window, 1, Dependent variable and independent variable

        Yields:
            typing.Any: The transformed tensor.

        """
        if self.x is None or self.stateful is False:
            self.x = jax.numpy.swapaxes(
                numpy.lib.stride_tricks.sliding_window_view(
                    x=get_tensor(
                        tensor=self.tensor,
                    )[
                        ...,
                        # Observation
                        self.in_sample_window_size :,
                        # Dependent variable and independent variable
                        :,
                    ],
                    window_shape=1,
                    # Observation
                    axis=-2,
                ),
                -2,
                -1,
            )
        yield self.x
