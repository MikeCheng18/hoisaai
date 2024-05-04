from __future__ import annotations
from collections import OrderedDict
import typing

import numpy
from hoisaai.layer_0.tensor import Hook, Tensor


class Parameter(Tensor):
    def __init__(
        self,
        x: Tensor | numpy.ndarray | typing.Any,
        require_gradient: bool = True,
    ) -> None:
        Tensor.__init__(
            self,
            x=x,
        )
        self.require_gradient: bool = require_gradient

    def detach(self) -> None:
        self.gradient: Tensor = None
        self.hook: typing.List[Hook] = []


class Module(object):
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
        for parameter in self.parameter.values():
            yield parameter
        for module in self.module.values():
            yield from module.get_parameter()

    def evaluate(self) -> None:
        for parameter in self.get_parameter():
            parameter.require_gradient = False

    def forward(self, *input_tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    def initialize_gradient(self) -> None:
        for parameter in self.get_parameter():
            parameter.detach()

    def train(self) -> None:
        for parameter in self.get_parameter():
            parameter.require_gradient = True


class Linear(Module):
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
                require_gradient=True,
                seed=seed,
            )
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
                    require_gradient=True,
                ),
            )
        )
        # (..., Batch, Input dimension)
        self.x: Tensor = None

    def forward(self, *input_tensor: Tensor) -> Tensor:
        assert len(input_tensor) == 1
        # (..., Batch, Input dimension)
        self.x: Tensor = input_tensor[0]
        # (..., Batch, Output dimension)
        return (
            # (..., Batch, Output dimension)
            (
                # (..., Batch, Input dimension)
                self.x
            )
            @ (
                # (..., Input dimension, Output dimension)
                self.weight
            )
        ) + (
            # (..., Batch, Output dimension)
            (
                # (..., Output dimension)
                self.bias
            ).expand_dimension(
                # Batch
                -2
            )
        )
