"""This module contains the implementation of the Module class and its subclasses."""

from __future__ import annotations
import abc
from collections import OrderedDict
import typing

from hoisaai.layer_0.tensor import Parameter, Tensor


class Module(object):
    """Module class for neural networks."""

    def __init__(self) -> None:
        self.module: OrderedDict[str, Module] = OrderedDict()
        self.parameter: OrderedDict[str, Parameter] = OrderedDict()

    def __setattr__(self, key, value):
        # First initialize the attribute we want to add
        self.__dict__[key] = value
        # Then update the inner dictionary '_modules', '_params'
        if issubclass(value.__class__, Module):
            self.module[key] = value
        elif issubclass(value.__class__, Parameter):
            self.parameter[key] = value

    def get_parameter(self) -> typing.Iterator[Parameter]:
        """Get all the parameters in the module."""
        for parameter in self.parameter.values():
            yield parameter
        for module in self.module.values():
            yield from module.get_parameter()

    def evaluate(self) -> None:
        """Set all the parameters to evaluate mode."""
        for parameter in self.get_parameter():
            parameter.require_gradient = False

    @abc.abstractmethod
    def forward(self, *input_tensor: Tensor) -> typing.Tuple[Tensor]:
        """Forward propagation of the module."""
        raise NotImplementedError

    def initialize_gradient(self) -> None:
        """Initialize the gradient of the parameters."""
        for parameter in self.get_parameter():
            parameter.detach()

    def train(self) -> None:
        """Set all the parameters to train mode."""
        for parameter in self.get_parameter():
            parameter.require_gradient = True


class Linear(Module):
    """Linear layer for neural networks.

    :param input_shape: Shape of the input tensor.
    :type input_shape: typing.Tuple[int]
    :param output_shape: Shape of the output tensor.
    :type output_shape: typing.Tuple[int]
    :param bias: Whether to use bias.
    :type bias: bool
    :param data_type: Data type of the tensor.
    :type data_type: Tensor.DataType
    :param seed: Random seed.
    :type seed: int
    """

    def __init__(
        self,
        # (..., Batch, Input dimension)
        input_shape: typing.Tuple[int],
        # (..., Batch, Output dimension)
        output_shape: typing.Tuple[int],
        bias: bool,
        data_type: Tensor.DataType,
        seed: int,
    ) -> None:
        Module.__init__(self)
        assert input_shape[:-1] == output_shape[:-1]
        # (..., Input dimension, Output dimension)
        self.weight: Parameter = Parameter(
            x=Tensor.random_normal(
                shape=(
                    # ,,,
                    *input_shape[:-2],
                    # Input dimension
                    input_shape[-1],
                    # Output dimension
                    output_shape[-1],
                ),
                datatype=data_type,
                seed=seed,
            ),
        )
        # (..., Output dimension)
        self.bias: Parameter = (
            None
            if bias is False
            else Parameter(
                x=Tensor.full(
                    shape=(
                        # ,,,
                        *input_shape[:-2],
                        # Output dimension
                        output_shape[-1],
                    ),
                    value=0.0,
                    datatype=data_type,
                ),
            )
        )
        # (..., Batch, Input dimension)
        self.x: Tensor = None

    def forward(self, *input_tensor: Tensor) -> typing.Tuple[Tensor]:
        assert isinstance(input_tensor, tuple)
        assert len(input_tensor) == 1
        # (..., Batch, Input dimension)
        self.x: Tensor = input_tensor[0]

        return (
            # (..., Batch, Output dimension)
            (
                # (..., Batch, Output dimension)
                (
                    # (..., Batch, Input dimension)
                    self.x
                )
                @ (
                    # (..., Input dimension, Output dimension)
                    self.weight
                )
            )
            + (
                # (..., Batch, Output dimension)
                (
                    # (..., Output dimension)
                    self.bias
                ).expand_dimension(
                    # Batch
                    -2
                )
            ),
        )


class Sequential(Module):
    """Sequential model for neural networks.

    :param module: Modules in the sequential model.
    :type module: typing.Tuple[Module]
    """

    def __init__(self, *module: Module) -> None:
        super().__init__()
        self.model: typing.Tuple[Module] = module

    def forward(self, *input_tensor: Tensor) -> typing.Tuple[Tensor]:
        for module in self.model:
            input_tensor = module.forward(*input_tensor)
        return input_tensor
