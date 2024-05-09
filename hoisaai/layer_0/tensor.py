"""This module implements a tensor."""

# pylint: disable=too-many-lines
from __future__ import annotations

import abc
import dataclasses
import enum
import os
import typing

import jax
import jaxlib.xla_extension
import numpy


class Function(object):
    """Function."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        """Backward function for backpropagation."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        tensor: Tensor,
    ):
        """Forward function."""
        raise NotImplementedError

    def __call__(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """Add a hook if the tensor requires gradient.

        :param tensor: The tensor to be processed.
        :type tensor: Tensor

        :return: The processed tensor.
        :rtype: Tensor
        """
        out: Tensor = self.forward(
            tensor=tensor,
        )
        if tensor.require_gradient:
            out.hook.append(
                Hook(
                    tensor,
                    self.backward,
                )
            )
        return out


class ExpandDimension(Function):
    """Expand the dimension of a tensor.

    :param axis: The axis to be expanded.
    :type axis: Tuple[int]
    """

    def __init__(
        self,
        *axis: int,
    ) -> None:
        super().__init__()
        self.axis: typing.Tuple[int] = axis

    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.reshape(
                a=gradient.x,
                newshape=tensor.shape,
            ),
            require_gradient=False,
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.expand_dims(
                a=tensor.x,
                axis=self.axis,
            ),
            require_gradient=tensor.require_gradient,
        )


class Exponential(Function):
    """Exponential function.

    .. math:: f(x) = e^{x}
    """

    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=gradient.x * tensor.x,
            require_gradient=False,
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.exp(tensor.x),
            require_gradient=tensor.require_gradient,
        )


class Negative(Function):
    """Negative function."""

    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=-(gradient.x),
            require_gradient=False,
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        return Tensor(
            x=-(tensor.x),
            require_gradient=tensor.require_gradient,
        )


class Power(Function):
    """Power function.

    :param exponent: The exponent.
    :type exponent: float
    """

    def __init__(
        self,
        exponent: float,
    ) -> None:
        super().__init__()
        self.exponent: float = float(exponent)

    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=gradient.x * self.exponent * (tensor.x ** (self.exponent - 1.0)),
            require_gradient=False,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return Tensor(
            x=tensor.x**self.exponent,
            require_gradient=tensor.require_gradient,
        )


class Inverse(Power):
    """Inverse function."""

    def __init__(
        self,
    ) -> None:
        Power.__init__(self, -1.0)


class Transpose(Function):
    """Transpose function.

    :param index: The index to be transposed.
    :type index: Tuple[int]
    """

    def __init__(
        self,
        *index: int,
    ) -> None:
        super().__init__()
        self.index: typing.Tuple[int] = index

    def backward(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        inverse = [0] * len(self.index)
        for original_axis, current_axis in enumerate(self.index):
            inverse[current_axis] = original_axis
        return Tensor(
            x=jax.numpy.transpose(
                a=gradient.x,
                axes=inverse,
            ),
            require_gradient=False,
        )

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.transpose(
                a=tensor.x,
                axes=self.index,
            ),
            require_gradient=tensor.require_gradient,
        )


@dataclasses.dataclass
class Hook(object):
    """Hook for backpropagation."""

    tensor: Tensor
    gradient_function: typing.Callable[
        [
            # gradient
            Tensor,
        ],
        Tensor,
    ]


class Operation(object):
    """Operation."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def backward1(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        """Backward function for backpropagation."""
        raise NotImplementedError

    @abc.abstractmethod
    def backward2(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        """Backward function for backpropagation."""
        raise NotImplementedError

    def forward(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ):
        """Forward function."""
        raise NotImplementedError

    def __call__(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ) -> Tensor:
        """Add a hook if the tensor requires gradient."""
        out: Tensor = self.forward(
            tensor1=tensor1,
            tensor2=tensor2,
        )
        if isinstance(tensor1, Tensor):
            if tensor1.require_gradient:
                out.hook.append(
                    Hook(
                        tensor1,
                        self.backward1,
                    )
                )
        if isinstance(tensor2, Tensor):
            if tensor2.require_gradient:
                out.hook.append(
                    Hook(
                        tensor2,
                        self.backward2,
                    )
                )
        return out


class Add(Operation):
    """Add operation"""

    @staticmethod
    def backward(
        # pylint: disable=unused-argument
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        """Calculate the gradient of the addition operation."""
        return Tensor(
            x=gradient.x,
            require_gradient=False,
        )

    def backward1(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Add.backward(gradient=gradient, tensor=tensor)

    def backward2(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Add.backward(gradient=gradient, tensor=tensor)

    def forward(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ) -> Tensor:
        operand2: typing.Any = tensor2.x if isinstance(tensor2, Tensor) else tensor2
        return Tensor(
            x=tensor1.x + operand2,
            require_gradient=tensor1.require_gradient
            or (tensor2.require_gradient if isinstance(tensor2, Tensor) else False),
        )


class MatrixMultiplication(Operation):
    r"""Matrix multiplication operation.
    
    .. math::
        t_{out} = t_{1} \cdot t_{2} \\
        t_{out}[i, j] = \sum_{k=1}^{n} t_{1}[i, k] \times t_{2}[k, j]
        
    """

    def __init__(self) -> None:
        super().__init__()
        self.tensor1: Tensor = None
        self.tensor2: Tensor = None

    def backward1(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=gradient.x
            @ jax.numpy.swapaxes(
                a=self.tensor2.x,
                axis1=-1,
                axis2=-2,
            ),
            require_gradient=False,
        )

    def backward2(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.swapaxes(
                a=self.tensor1.x,
                axis1=-1,
                axis2=-2,
            )
            @ gradient.x,
            require_gradient=False,
        )

    def forward(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ) -> Tensor:
        self.tensor1: Tensor = tensor1
        self.tensor2: Tensor = tensor2
        return Tensor(
            x=tensor1.x @ tensor2.x,
            require_gradient=tensor1.require_gradient or tensor2.require_gradient,
        )


class Multiply(Operation):
    """Multiply operation."""

    def __init__(self) -> None:
        super().__init__()
        self.tensor1: Tensor = None
        self.tensor2: Tensor = None

    @staticmethod
    def backward(
        gradient: Tensor,
        other: float | int | Tensor,
    ) -> Tensor:
        """Calculate the gradient of the multiplication operation."""
        return Tensor(
            x=gradient.x * (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def backward1(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return self.backward(
            gradient=gradient,
            other=self.tensor2,
        )

    def backward2(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return self.backward(
            gradient=gradient,
            other=self.tensor1,
        )

    def forward(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ) -> Tensor:
        self.tensor1: Tensor = tensor1
        self.tensor2: Tensor = tensor2
        operand2: typing.Any = tensor2.x if isinstance(tensor2, Tensor) else tensor2
        return Tensor(
            x=tensor1.x * operand2,
            require_gradient=tensor1.require_gradient
            or (tensor2.require_gradient if isinstance(tensor2, Tensor) else False),
        )


class Where(Operation):
    """Where operation.

    :param condition: The condition.
    :type condition: Tensor
    """

    def __init__(
        self,
        condition: Tensor,
    ) -> None:
        super().__init__()
        self.condition: Tensor = condition

    def backward1(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=gradient.x * jax.numpy.where(self.condition.x, 1.0, 0.0),
            require_gradient=False,
        )

    def backward2(
        self,
        tensor: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        return Tensor(
            x=gradient.x * jax.numpy.where(self.condition.x, 0.0, 1.0),
            require_gradient=False,
        )

    def forward(
        self,
        tensor1: Tensor,
        tensor2: float | int | Tensor,
    ) -> Tensor:
        operand1: typing.Any = tensor1.x if isinstance(tensor1, Tensor) else tensor1
        operand2: typing.Any = tensor2.x if isinstance(tensor2, Tensor) else tensor2
        return Tensor(
            x=jax.numpy.where(self.condition.x, operand1, operand2),
            require_gradient=(
                tensor1.require_gradient if isinstance(tensor1, Tensor) else False
            )
            or (tensor2.require_gradient if isinstance(tensor2, Tensor) else False),
        )


class Tensor(object):
    """This is a class representing a tensor.

    :param x: The tensor.
    :type x: jaxlib.xla_extension.ArrayImpl | numpy.ndarray | Tensor
    :param require_gradient: Whether the tensor requires gradient.
    :type require_gradient: bool, default=False
    :param hook: The hook for backpropagation.
    :type hook: List[Hook], optional
    """

    class DataType(enum.Enum):
        """Data type of a tensor."""

        BOOL: jax.numpy.dtype = jax.numpy.bool
        """Boolean."""
        INT8: jax.numpy.dtype = jax.numpy.int8
        """8-bit integer."""
        INT16: jax.numpy.dtype = jax.numpy.int16
        """16-bit integer."""
        INT32: jax.numpy.dtype = jax.numpy.int32
        """32-bit integer."""
        INT64: jax.numpy.dtype = jax.numpy.int64
        """64-bit integer."""
        FLOAT16: jax.numpy.dtype = jax.numpy.float16
        """16-bit floating point."""
        FLOAT32: jax.numpy.dtype = jax.numpy.float32
        """32-bit floating point."""
        FLOAT64: jax.numpy.dtype = jax.numpy.float64
        """64-bit floating point."""

    class Constant(enum.Enum):
        """Constant value."""

        NAN: float | int = jax.numpy.nan
        """Not a number."""
        INF: float | int = jax.numpy.inf
        """Infinity."""

    def __init__(
        self,
        x: Tensor | numpy.ndarray | jaxlib.xla_extension.ArrayImpl,
        require_gradient: bool = False,
        hook: typing.List[Hook] = None,
    ) -> None:
        self.x: jaxlib.xla_extension.ArrayImpl = None
        self.hook: typing.List[Hook] = hook if hook is not None else []
        self.require_gradient: bool = require_gradient
        self.gradient: Tensor = None
        if isinstance(x, Tensor):
            self.x: jaxlib.xla_extension.ArrayImpl = x.x
        elif isinstance(x, jaxlib.xla_extension.ArrayImpl):
            self.x = jax.numpy.copy(x)
        elif isinstance(x, numpy.ndarray):
            self.x = jax.numpy.array(x)
        assert isinstance(self.x, jaxlib.xla_extension.ArrayImpl), type(self.x)

    def _repr_html_(self) -> str:
        return f"shape: {str(self.shape)}<br>{jax.numpy.array_str(a=self.x, max_line_width=128,precision=4,suppress_small=True,)}"

    def __add__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Add()(self, other)

    def __bool__(self) -> bool:
        assert self.x.size == 1, self.x
        return bool(self.x)

    def __eq__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Tensor(
            x=self.x == (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def __floordiv__(
        self,
        other: int | Tensor,
    ) -> Tensor:
        return Tensor(
            x=self.x // (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def __ge__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Tensor(
            x=self.x >= (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def __gt__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Tensor(
            x=self.x > (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def __getitem__(
        self,
        *args,
    ) -> Tensor:
        return Tensor(
            x=self.x.__getitem__(*args),
            require_gradient=False,
        )

    def __float__(self) -> float:
        assert self.x.size == 1, self.x
        return float(self.x)

    def __iadd__(
        self,
        other: float | int | Tensor,
    ) -> None:
        return Add()(self, other)

    def __int__(self) -> int:
        assert self.x.size == 1, self.x
        return int(self.x)

    def __imul__(self, other: float | int | Tensor) -> Tensor:
        self: Tensor = Multiply()(self, other)
        return self

    def __ipow__(self, other: float | int | Tensor) -> Tensor:
        return Tensor(
            x=self.x ** (other.x if isinstance(other, Tensor) else other),
            require_gradient=False,
        )

    def __isub__(self, other: float | int | Tensor) -> Tensor:
        return Add()(self, Negative()(other) if isinstance(other, Tensor) else -(other))

    def __itruediv__(self, other: float | int | Tensor) -> Tensor:
        return Multiply()(
            self, Inverse()(other) if isinstance(other, Tensor) else (1.0 / other)
        )

    def __invert__(self) -> Tensor:
        if self.datatype == Tensor.DataType.BOOL:
            return Tensor(x=jax.numpy.invert(self.x), require_gradient=False)
        else:
            return -1.0 * self

    def __le__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Tensor(x=self.x <= other, require_gradient=False)

    def __lt__(
        self,
        other: Tensor,
    ) -> Tensor:
        return Tensor(x=self.x < other, require_gradient=False)

    def __matmul__(
        self,
        y: Tensor,
    ) -> Tensor:
        return MatrixMultiplication()(self, y)

    def __mod__(
        self,
        other: int | Tensor,
    ) -> Tensor:
        return Tensor(x=self.x % other, require_gradient=False)

    def __mul__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Multiply()(self, other)

    def __ne__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Tensor(x=self.x != other, require_gradient=False)

    def __pow__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(
                x=self.x**other.x,
                require_gradient=False,
            )
        else:
            return Power(exponent=other)(self)

    def __radd__(
        self,
        other: float | int,
    ) -> Tensor:
        return Add()(self, other)

    def __req__(
        self,
        other: float | int,
    ) -> Tensor:
        return Tensor(x=other == self.x, require_gradient=False)

    def __rfloordiv__(
        self,
        other: int,
    ) -> Tensor:
        return Tensor(x=other // self.x, require_gradient=False)

    def __rge__(
        self,
        other: float | int,
    ):
        return Tensor(x=other >= self.x, require_gradient=False)

    def __rgt__(
        self,
        other: float | int,
    ):
        return Tensor(x=other > self.x, require_gradient=False)

    def __rle__(
        self,
        other: float | int,
    ):
        return Tensor(x=other <= self.x, require_gradient=False)

    def __rlt__(
        self,
        other: float | int,
    ):
        return Tensor(x=other < self.x, require_gradient=False)

    def __rmod__(
        self,
        other: int,
    ) -> Tensor:
        return Tensor(x=other % self.x, require_gradient=False)

    def __rmul__(
        self,
        other: float | int,
    ) -> Tensor:
        return Multiply()(self, other)

    def __rsub__(
        self,
        other: float | int,
    ) -> Tensor:
        return Add()(Negative()(self), other)

    def __rtruediv__(
        self,
        other: float | int,
    ) -> Tensor:
        return Multiply()(Inverse()(self), other)

    def __setitem__(self, key, value) -> None:
        self.x = self.x.at[key].set(value.x if isinstance(value, Tensor) else value)

    def __str__(self) -> str:
        return str(self.x)

    def __sub__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Add()(self, Negative()(other) if isinstance(other, Tensor) else -(other))

    def __truediv__(
        self,
        other: float | int | Tensor,
    ) -> Tensor:
        return Multiply()(
            self, Inverse()(other) if isinstance(other, Tensor) else (1.0 / other)
        )

    @typing.overload
    def all(self) -> bool: ...
    @typing.overload
    def all(self, axis: int) -> Tensor: ...
    def all(self, axis: int = None) -> bool | Tensor:
        """Check if all elements are true.

        :param axis: The axis to be checked.
        :type axis: int, optional

        :return: Whether all elements are true.
        :rtype: bool | Tensor
        """
        if axis is None:
            return bool(jax.numpy.all(self.x))
        return Tensor(
            x=jax.numpy.all(
                self.x,
                axis=axis,
            ),
            require_gradient=False,
        )

    @typing.overload
    def any(self) -> bool: ...
    @typing.overload
    def any(self, axis: int) -> Tensor: ...
    def any(self, axis: int = None) -> bool | Tensor:
        """Check if any element is true.

        :param axis: The axis to be checked.
        :type axis: int, optional

        :return: Whether any element is true.
        :rtype: bool | Tensor
        """
        if axis is None:
            return bool(jax.numpy.any(self.x))
        return Tensor(
            x=jax.numpy.any(
                self.x,
                axis=axis,
            ),
            require_gradient=False,
        )

    @staticmethod
    def arange(
        start: int,
        stop: int,
        step: int,
        datatype: Tensor.DataType,
    ) -> Tensor:
        """Create a tensor with a range of values.

        :param start: The start of the range.
        :type start: int
        :param stop: The stop of the range.
        :type stop: int
        :param step: The step of the range.
        :type step: int
        :param datatype: The data type.
        :type datatype: Tensor.DataType

        :return: The tensor with a range of values.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.arange(start=start, stop=stop, step=step, dtype=datatype.value),
            require_gradient=False,
        )

    def argmax(self, axis: int) -> Tensor:
        """Return the indices of the maximum values along an axis.

        :param axis: The axis to be checked.
        :type axis: int

        :return: The indices of the maximum values along an axis.
        :rtype: Tensor
        """
        return Tensor(x=jax.numpy.nanargmax(self.x, axis=axis))

    def argsort(self, axis: int) -> Tensor:
        """Return the indices that would sort an array.

        :param axis: The axis to be checked.
        :type axis: int

        :return: The indices that would sort an array.
        :rtype: Tensor
        """
        return Tensor(x=jax.numpy.argsort(self.x, axis=axis))

    @staticmethod
    def array(
        x: typing.List[any],
        datatype: DataType,
        require_gradient: bool = False,
    ) -> Tensor:
        """Create a tensor from a list.

        :param x: The list.
        :type x: List[any]
        :param datatype: The data type.
        :type datatype: Tensor.DataType

        :return: The tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.array(
                x,
                dtype=datatype.value,
            ),
            require_gradient=require_gradient,
        )

    def astype(self, datatype: Tensor.DataType) -> Tensor:
        """Return a tensor with a specified data type.

        :param datatype: The data type.
        :type datatype: Tensor.DataType

        :return: The tensor with a specified data type.
        :rtype: Tensor
        """
        return Tensor(
            x=self.x.astype(datatype.value),
            require_gradient=self.require_gradient,
        )

    def backward(self, gradient: Tensor) -> Tensor:
        """Backward function for backpropagation.

        :param gradient: The gradient.
        :type gradient: Tensor, optional

        :return: The tensor.
        :rtype: Tensor
        """
        self.gradient = gradient if self.gradient is None else self.gradient + gradient
        for hook in self.hook:
            hook.tensor.backward(
                gradient=hook.gradient_function(
                    tensor=hook.tensor,
                    gradient=gradient,
                ),
            )
        return self

    @staticmethod
    def concatenate(
        tensors: typing.Sequence[Tensor],
        axis: int,
        require_gradient: bool = False,
    ) -> Tensor:
        """Concatenate tensors along an axis.

        :param tensors: The tensors to be concatenated.
        :type tensors: Sequence[Tensor]
        :param axis: The axis to be concatenated.
        :type axis: int
        :param require_gradient: Whether the tensor requires gradient.
        :type require_gradient: bool, default=False

        :return: The concatenated tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.concatenate(
                arrays=[array.x for array in tensors],
                axis=axis,
            ),
            require_gradient=require_gradient,
        )

    def count_nonzero(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Count the number of non-zero elements.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The number of non-zero elements.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.count_nonzero(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
            require_gradient=False,
        )

    def cumulative_product(
        self,
        axis: int,
    ) -> Tensor:
        """Return the cumulative product of elements along an axis.

        :param axis: The axis to be checked.
        :type axis: int

        :return: The cumulative product of elements along an axis.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.cumprod(self.x, axis=axis),
            require_gradient=False,
        )

    @property
    def datatype(self) -> Tensor.DataType:
        """Return the data type of the tensor."""
        return {
            jax.numpy.dtype("bool"): Tensor.DataType.BOOL,
            jax.numpy.dtype("int8"): Tensor.DataType.INT8,
            jax.numpy.dtype("float16"): Tensor.DataType.FLOAT16,
            jax.numpy.dtype("float32"): Tensor.DataType.FLOAT32,
            jax.numpy.dtype("float64"): Tensor.DataType.FLOAT64,
            jax.numpy.dtype("int16"): Tensor.DataType.INT16,
            jax.numpy.dtype("int32"): Tensor.DataType.INT32,
            jax.numpy.dtype("int64"): Tensor.DataType.INT64,
        }[self.x.dtype]

    @staticmethod
    def diagonal(
        value: float | int,
        size: int,
        datatype: Tensor.DataType,
        require_gradient: bool = False,
    ) -> Tensor:
        """Create a tensor with diagonal values.

        :param value: The diagonal value.
        :type value: float | int
        :param size: The size of the tensor.
        :type size: int
        :param datatype: The data type.
        :type datatype: Tensor.DataType
        :param require_gradient: Whether the tensor requires gradient.
        :type require_gradient: bool, default=False

        :return: The tensor with diagonal values.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.fill_diagonal(
                a=jax.numpy.zeros(
                    shape=(size, size),
                    dtype=datatype.value,
                ),
                val=value,
                inplace=False,
            ),
            require_gradient=require_gradient,
        )

    def exp(self) -> Tensor:
        """Exponential function."""
        return Exponential()(self)

    def expand_dimension(
        self,
        *axis: int,
    ) -> Tensor:
        """Expand the dimension of a tensor.

        :param axis: The axis to be expanded.
        :type axis: Tuple[int]

        :return: The tensor with expanded dimension.
        :rtype: Tensor
        """
        return ExpandDimension(*axis)(self)

    @staticmethod
    def full(
        shape: typing.Tuple[int],
        value: float | int,
        datatype: Tensor.DataType,
    ) -> Tensor:
        """Create a tensor with a constant value.

        :param shape: The shape of the tensor.
        :type shape: Tuple[int]
        :param value: The constant value.
        :type value: float | int
        :param datatype: The data type.
        :type datatype: Tensor.DataType

        :return: The tensor with a constant value.
        :rtype: Tensor
        """
        if isinstance(value, Tensor.Constant):
            value = value.value
        return Tensor(
            x=jax.numpy.full(
                shape=shape,
                fill_value=value,
                dtype=datatype.value,
            ),
            require_gradient=False,
        )

    def get_sample_x_and_y(
        self,
        number_of_target: int,
    ) -> typing.Tuple[Tensor, Tensor]:
        """Get the sample of x and y.

        :param number_of_target: The number of target.
        :type number_of_target: int

        :return: The sample of x and y.
        :rtype: Tuple[Tensor, Tensor]
        """
        return (
            Tensor(
                x=self.x[..., number_of_target:],
                require_gradient=False,
            ),
            Tensor(
                x=self.x[..., :number_of_target],
                require_gradient=False,
            ),
        )

    def get_by_index(
        # (A..., index)
        self,
        # (B...,)
        indexes: Tensor,
    ) -> Tensor:
        """Get the values by index.

        :param indexes: The indexes.
        :type indexes: Tensor

        :return: The values by index.
        :rtype: Tensor
        """
        # (max{A..., B...}, index)
        selection: jaxlib.xla_extension.ArrayImpl = 1 * (
            # (A..., index)
            (
                # (index,)
                jax.numpy.arange(
                    start=0,
                    stop=self.x.shape[-1],
                    dtype=jax.numpy.int16,
                )
                * (
                    # (A..., index)
                    jax.numpy.ones(
                        # (A..., index)
                        shape=self.x.shape,
                        dtype=jax.numpy.int16,
                    )
                )
            )
            # Checking if the list of indexes generated above is equal to the indexes given
            == (
                # Add a new axis to the indexes for broadcasting
                # (B..., 1)
                jax.numpy.expand_dims(
                    # (B...,)
                    a=indexes.x,
                    axis=-1,
                )
            )
        )
        # (max{A..., B...})
        return Tensor(
            x=jax.numpy.nanmax(
                # (max{A..., B...}, index)
                (
                    # Turn 0 to nan
                    # (max{A..., B...}, index)
                    (selection / selection)
                )
                * (
                    # (A..., index)
                    self.x
                ),
                axis=-1,
            ),
            require_gradient=False,
        ).astype(self.datatype)

    def in_sample(
        self,
        in_sample_size: int,
    ) -> Tensor:
        """Get the in-sample data.

        :param in_sample_size: The size of the in-sample data.
        :type in_sample_size: int

        :return: The in-sample data.
        :rtype: Tensor
        """
        return Tensor(
            x=self.x[
                # ...
                ...,
                # Observation
                :in_sample_size,
                # Target and factor
                :,
            ],
            require_gradient=False,
        )

    def insert(
        self,
        index: int,
        value: float | int,
        axis: int,
    ) -> Tensor:
        """Insert a value into a tensor.

        :param index: The index.
        :type index: int
        :param value: The value.
        :type value: float | int
        :param axis: The axis.
        :type axis: int

        :return: The tensor with the value inserted.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.insert(
                arr=self.x,
                obj=index,
                values=value,
                axis=axis,
            ),
            require_gradient=False,
        )

    def inverse(
        self,
    ) -> Tensor:
        """Return the inverse of the tensor.

        :return: The inverse of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.linalg.inv(self.x),
            require_gradient=False,
        )

    def isnan(self) -> Tensor:
        """Check if the tensor is nan.

        :return: The tensor with nan values.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.isnan(self.x),
            require_gradient=False,
        )

    @staticmethod
    def load(
        source: str,
    ) -> Tensor:
        """Load a tensor from a file.

        :param source: The source file.
        :type source: str

        :return: The tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.load(
                file=source,
            ),
            require_gradient=False,
        )

    def log2(self) -> Tensor:
        """Logarithm base 2 function.

        :return: The logarithm base 2 of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.log2(self.x),
            require_gradient=False,
        )

    def mean(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Return the mean of the tensor.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The mean of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.mean(self.x, axis=axis, keepdims=keep_dimension),
            require_gradient=False,
        )

    def nanmean(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Return the mean of the tensor with NAN.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The mean of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.nanmean(self.x, axis=axis, keepdims=keep_dimension),
            require_gradient=False,
        )

    def nansum(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Return the sum of the tensor with NAN.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The sum of the tensor.
        :rtype: Tensor"""
        return Tensor(
            x=jax.numpy.nansum(
                a=self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
            require_gradient=False,
        )

    def nan_to_num(
        self,
        nan: float | int,
        posinf: float | int,
        neginf: float | int,
    ) -> Tensor:
        """Replace NaN with zero and infinity with large finite numbers.

        :param nan: The value to replace NaN.
        :type nan: float | int
        :param posinf: The value to replace positive infinity.
        :type posinf: float | int
        :param neginf: The value to replace negative infinity.
        :type neginf: float | int

        :return: The tensor with NaN replaced.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.nan_to_num(
                x=self.x,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            ),
            require_gradient=False,
        )

    @property
    def number_of_dimension(self) -> int:
        """Return the number of dimensions of the tensor.

        :return: The number of dimensions of the tensor.
        :rtype: int
        """
        return jax.numpy.ndim(self.x)

    def out_of_sample(
        self,
        in_sample_size: int,
    ) -> Tensor:
        """Get the out-of-sample data.

        :param in_sample_size: The size of the in-sample data.
        :type in_sample_size: int

        :return: The out-of-sample data.
        :rtype: Tensor
        """
        return Tensor(
            x=self.x[
                # ...
                ...,
                # Observation
                in_sample_size:,
                # Target and factor
                :,
            ],
            require_gradient=False,
        )

    @staticmethod
    def random_integer(
        shape: typing.Tuple[int],
        minimum_value: int,
        maximum_value: int,
        datatype: Tensor.DataType,
        seed: int,
    ) -> Tensor:
        """Create a tensor with uniformlly distributed random integer values.

        :param shape: The shape of the tensor.
        :type shape: Tuple[int]
        :param minimum_value: The minimum value (inclusive).
        :type minimum_value: int
        :param maximum_value: The maximum value (exclusive).
        :type maximum_value: int
        :param datatype: The data type.
        :type datatype: Tensor.DataType
        :param seed: The seed for random number generation.
        :type seed: int
        """
        return Tensor(
            x=jax.random.randint(
                jax.random.PRNGKey(
                    seed=seed,
                ),
                shape=shape,
                minval=minimum_value,
                maxval=maximum_value,
                dtype=datatype.value,
            ),
            require_gradient=False,
        )

    @staticmethod
    def random_normal(
        shape: typing.Tuple[int],
        datatype: Tensor.DataType,
        seed: int,
    ) -> Tensor:
        """Create a tensor with normally distributed random values.

        :param shape: The shape of the tensor.
        :type shape: Tuple[int]
        :param datatype: The data type.
        :type datatype: Tensor.DataType
        :param seed: The seed for random number generation.
        :type seed: int

        :return: The tensor with normally distributed random values.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.random.normal(
                key=jax.random.PRNGKey(seed),
                shape=shape,
                dtype=datatype.value,
            ),
            require_gradient=False,
        )

    @staticmethod
    def random_uniform(
        shape: typing.Tuple[int],
        minimum_value: int,
        maximum_value: int,
        datatype: Tensor.DataType,
        seed: int,
    ) -> Tensor:
        """Create a tensor with uniformly distributed random values.

        :param shape: The shape of the tensor.
        :type shape: Tuple[int]
        :param minimum_value: The minimum value (inclusive).
        :type minimum_value: int
        :param maximum_value: The maximum value (exclusive).
        :type maximum_value: int
        :param datatype: The data type.
        :type datatype: Tensor.DataType
        :param seed: The seed for random number generation.
        :type seed: int

        :return: The tensor with uniformly distributed random values.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.random.uniform(
                key=jax.random.PRNGKey(seed),
                shape=shape,
                minval=minimum_value,
                maxval=maximum_value,
                dtype=datatype.value,
            ),
            require_gradient=False,
        )

    def relu(self) -> Tensor:
        """Rectified linear unit function.

        :return: The tensor with rectified linear unit function.
        :rtype: Tensor
        """
        return Where(condition=self > 0.0)(self, 0.0)

    def reshape(
        self,
        *shape: int,
    ) -> Tensor:
        """Reshape the tensor.

        :param shape: The shape.
        :type shape: Tuple[int]

        :return: The tensor with reshaped shape.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.reshape(
                a=self.x,
                newshape=shape,
            ),
            require_gradient=False,
        )

    def save(
        self,
        destination: str,
    ) -> typing.Self:
        """Save the tensor to a file.

        :param destination: The destination file.
        :type destination: str

        :return: The tensor itself.
        :rtype: Tensor
        """
        # Save the tensor to a file
        jax.numpy.save(
            file=destination,
            arr=self.x,
        )
        try:
            # Remove the file if it exists
            os.remove(destination)
        except OSError:
            pass
        # Rename the file
        os.rename(
            f"{destination}.npy",
            destination,
        )
        return self

    @property
    def shape(self) -> typing.Tuple[int]:
        """Return the shape of the tensor.

        :return: The shape of the tensor.
        :rtype: Tuple[int]
        """
        return self.x.shape

    def sign(self) -> Tensor:
        """Sign function.

        :return: The sign of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.sign(self.x),
            require_gradient=False,
        )

    def sigmoid(self) -> Tensor:
        """Sigmoid function.

        :return: The sigmoid of the tensor.
        :rtype: Tensor
        """
        return 1.0 / (1.0 + self.exp())

    def sliding_window(
        self,
        window_size: int,
    ) -> Tensor:
        """Create a sliding window.

        :param window_size: The window size.
        :type window_size: int

        :return: The sliding window.
        :rtype: Tensor
        """
        # (..., Batch, Window, Target and factor)
        return Tensor(
            x=(
                # (..., Batch, Target and factor, Window)
                numpy.lib.stride_tricks.sliding_window_view(
                    # (..., Observation, Target and factor)
                    x=self.x,
                    window_shape=window_size,
                    # Observation
                    axis=-2,
                )
            ).swapaxes(
                # Target and factor
                -1,
                # Batch
                -2,
            ),
            require_gradient=False,
        )

    def sort(
        self,
        axis: int,
    ) -> Tensor:
        """Sort the tensor.

        :param axis: The axis to be sorted.
        :type axis: int
        """
        return Tensor(
            x=jax.numpy.sort(
                self.x,
                axis=axis,
            ),
            require_gradient=False,
        )

    def sqrt(self) -> Tensor:
        """Square root function.

        :return: The square root of the tensor.
        :rtype: Tensor
        """
        return Power(exponent=0.5)(self)

    def square(self) -> Tensor:
        """Square function.

        :return: The square of the tensor.
        :rtype: Tensor
        """
        return Power(exponent=2)(self)

    @staticmethod
    def stack(
        tensors: typing.List[Tensor],
        axis: int,
    ) -> Tensor:
        """Stack tensors along an axis.

        :param tensors: The tensors to be stacked.
        :type tensors: List[Tensor]
        :param axis: The axis to be stacked.
        :type axis: int

        :return: The stacked tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.stack(
                arrays=[tensor.x for tensor in tensors],
                axis=axis,
            ),
            require_gradient=False,
        )

    def std(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Return the standard deviation of the tensor.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The standard deviation of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.std(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
            require_gradient=False,
        )

    def sum(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        """Return the sum of the tensor.

        :param axis: The axis to be checked.
        :type axis: int
        :param keep_dimension: Whether to keep the dimension.
        :type keep_dimension: bool

        :return: The sum of the tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=jax.numpy.sum(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
            require_gradient=False,
        )

    def swapaxes(
        self,
        axis1: int,
        axis2: int,
    ) -> Tensor:
        """Swap the axes of the tensor.

        :param axis1: The first axis.
        :type axis1: int
        :param axis2: The second axis.
        :type axis2: int

        :return: The tensor with swapped axes.
        :rtype: Tensor
        """
        axis = list(range(self.number_of_dimension))
        axis[axis1], axis[axis2] = axis[axis2], axis[axis1]
        return self.transpose(*axis)

    def to_list(self) -> typing.List[typing.Any]:
        """Convert the tensor to a list.

        :return: The list.
        :rtype: List[any]
        """
        return jax.numpy.asarray(self.x).tolist()

    def to_numpy(self) -> numpy.ndarray:
        """Convert the tensor to a numpy array.

        :return: The numpy array.
        :rtype: numpy.ndarray
        """
        return jax.numpy.asarray(self.x)

    def transpose(self, *index: int) -> Tensor:
        """Transpose the tensor.

        :param index: The index.
        :type index: Tuple[int]

        :return
        :rtype: Tensor"""
        return Transpose(*index)(self)

    def unique(self) -> typing.Tuple[Tensor, Tensor]:
        """Return the unique values and the count of the unique values.

        :return: The unique values and the count of the unique values.
        :rtype: Tuple[Tensor, Tensor]
        """
        unique, count = jax.numpy.unique(self.x, return_counts=True)
        return (
            Tensor(
                x=unique,
                require_gradient=False,
            ),
            Tensor(
                x=count,
                require_gradient=False,
            ),
        )

    @staticmethod
    def where(
        condition: Tensor,
        if_true: Tensor,
        if_false: Tensor,
    ) -> Tensor:
        """Return elements, either from x or y, depending on condition.

        :param condition: The condition.
        :type condition: Tensor
        :param if_true: The tensor if_true.
        :type if_true: Tensor
        :param if_false: The tensor if_false.
        :type if_false: Tensor
        """
        return Where(condition)(if_true, if_false)


class Parameter(Tensor):
    def __init__(
        self,
        x: Tensor | numpy.ndarray | typing.Any,
    ) -> None:
        Tensor.__init__(
            self,
            x=x,
        )
        self.require_gradient: bool = True

    def detach(self) -> None:
        self.gradient: Tensor = None
        self.hook: typing.List[Hook] = []
