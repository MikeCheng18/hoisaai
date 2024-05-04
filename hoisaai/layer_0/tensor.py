from __future__ import annotations

import dataclasses
import enum
import os
import typing

import jax
import jaxlib.xla_extension
import numpy


def jax_casting(
    x: Tensor | jaxlib.xla_extension.ArrayImpl | int | float,
) -> jaxlib.xla_extension.ArrayImpl | int | float:
    if isinstance(x, Tensor):
        return x.x
    elif isinstance(x, numpy.ndarray):
        return jax.numpy.array(x)
    return x


def tensor_casting(x: typing.Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor.array([x], datatype=Tensor.DataType.FLOAT32)


@dataclasses.dataclass
class Hook(object):
    tensor: Tensor
    gradient_function: typing.Callable[
        [
            # gradient
            Tensor,
        ],
        Tensor,
    ]


class Function(object):
    def __init__(self) -> None:
        pass

    def backward(
        self,
        gradient: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    def forward(
        self,
        tensor: Tensor,
    ):
        raise NotImplementedError()

    def __call__(
        self,
        tensor: Tensor,
    ) -> Tensor:
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
    def __init__(
        self,
        *axis: int,
    ) -> None:
        super().__init__()
        self.axis: typing.Tuple[int] = axis
        self.tensor: Tensor = None

    def backward(
        self,
        gradient: Tensor,
    ) -> Tensor:
        print("C" * 100)
        print(gradient)
        gradient = Tensor(
            x=jax.numpy.reshape(a=gradient.x, newshape=self.tensor.shape),
            require_gradient=False,
        )
        print(gradient)
        return gradient

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        self.tensor: Tensor = tensor
        return Tensor(
            x=jax.numpy.expand_dims(
                a=tensor.x,
                axis=self.axis,
            ),
            require_gradient=tensor.require_gradient,
        )


class Inverse(Function):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.tensor: Tensor = None

    def backward(
        self,
        gradient: Tensor,
    ) -> Tensor:
        return -1.0 / (self.tensor.square()) * gradient

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        self.tensor: Tensor = tensor
        return Tensor(
            x=1.0 / tensor.x,
            require_gradient=tensor.require_gradient,
        )


class Negative(Function):
    def backward(
        self,
        gradient: Tensor,
    ) -> Tensor:
        return -1.0 * gradient

    def forward(
        self,
        tensor: Tensor,
    ) -> Tensor:
        return Tensor(
            x=-tensor.x,
            require_gradient=tensor.require_gradient,
        )


class Transpose(Function):
    def __init__(
        self,
        *index: int,
    ) -> None:
        super().__init__()
        self.index: typing.Tuple[int] = index

    def backward(
        self,
        gradient: Tensor,
    ) -> Tensor:
        inverse = [0] * len(self.index)
        for i, p in enumerate(self.index):
            inverse[p] = i
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


class Operation(object):
    def __init__(self) -> None:
        self.tensor1: Tensor = None
        self.tensor2: Tensor = None

    def backward1(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError()

    def backward2(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ):
        self.tensor1: Tensor = tensor1
        self.tensor2: Tensor = tensor2

    def __call__(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> Tensor:
        out: Tensor = self.forward(
            tensor1=tensor1,
            tensor2=tensor2,
        )
        if tensor1.require_gradient:
            out.hook.append(
                Hook(
                    tensor1,
                    self.backward1,
                )
            )
        if tensor2.require_gradient:
            out.hook.append(
                Hook(
                    tensor2,
                    self.backward2,
                )
            )
        return out


class Add(Operation):
    @staticmethod
    def backward(gradient: Tensor, tensor: Tensor) -> Tensor:
        print("A" * 100)
        print(gradient)
        for _ in range(gradient.number_of_dimension - tensor.number_of_dimension):
            gradient = gradient.mean(
                axis=0,
                keep_dimension=False,
            )
        # Sum across broadcasted (but non-added dims)
        for i, dimension in enumerate(tensor.shape):
            if dimension == 1:
                gradient = gradient.mean(
                    axis=i,
                    keep_dimension=True,
                )
        print(gradient)
        return gradient

    def backward1(self, gradient: Tensor) -> Tensor:
        return Add.backward(gradient=gradient, tensor=self.tensor1)

    def backward2(self, gradient: Tensor) -> Tensor:
        return Add.backward(gradient=gradient, tensor=self.tensor2)

    def forward(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> Tensor:
        Operation.forward(self, tensor1, tensor2)
        return Tensor(
            x=tensor1.x + tensor2.x,
            require_gradient=tensor1.require_gradient or tensor2.require_gradient,
        )


class MatrixMultiplication(Operation):
    def backward1(self, gradient: Tensor) -> Tensor:
        gradient = Tensor(x=gradient @ self.tensor2.swapaxes(-1, -2))
        return gradient

    def backward2(self, gradient: Tensor) -> Tensor:
        print("B" * 100)
        print(gradient)
        gradient = Tensor(x=self.tensor1.swapaxes(-1, -2) @ gradient)
        print(gradient)
        return gradient

    def forward(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> Tensor:
        Operation.forward(self, tensor1, tensor2)
        return Tensor(
            x=tensor1.x @ tensor2.x,
            require_gradient=tensor1.require_gradient or tensor2.require_gradient,
        )


class Multiply(Operation):
    @staticmethod
    def backward(gradient: Tensor, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        gradient = gradient * tensor2
        for _ in range(gradient.number_of_dimension - tensor1.number_of_dimension):
            gradient = gradient.sum(
                axis=0,
                keep_dimension=False,
            )
        # Sum across broadcasted (but non-added dims)
        for i, dimension in enumerate(tensor1.shape):
            if dimension == 1:
                gradient = gradient.sum(
                    axis=i,
                    keep_dimension=True,
                )
        return gradient

    def backward1(self, gradient: Tensor) -> Tensor:
        return self.backward(
            gradient=gradient,
            tensor1=self.tensor1,
            tensor2=self.tensor2,
        )

    def backward2(self, gradient: Tensor) -> Tensor:
        return self.backward(
            gradient=gradient,
            tensor1=self.tensor2,
            tensor2=self.tensor1,
        )

    def forward(
        self,
        tensor1: Tensor,
        tensor2: Tensor,
    ) -> Tensor:
        Operation.forward(self, tensor1, tensor2)
        return Tensor(
            x=tensor1.x * tensor2.x,
            require_gradient=tensor1.require_gradient or tensor2.require_gradient,
        )


class Tensor(object):
    class DataType(enum.Enum):
        BOOL: jax.numpy.dtype = jax.numpy.bool
        INT16: jax.numpy.dtype = jax.numpy.int16
        INT32: jax.numpy.dtype = jax.numpy.int32
        INT64: jax.numpy.dtype = jax.numpy.int64
        FLOAT16: jax.numpy.dtype = jax.numpy.float16
        FLOAT32: jax.numpy.dtype = jax.numpy.float32
        FLOAT64: jax.numpy.dtype = jax.numpy.float64

    class Value(enum.Enum):
        NAN: int | float = jax.numpy.nan
        INF: int | float = jax.numpy.inf

    def __init__(
        self,
        x: Tensor | numpy.ndarray | jaxlib.xla_extension.ArrayImpl,
        require_gradient: bool = False,
        hook: typing.List[Hook] = None,
        pair: Tensor = None,
    ) -> None:
        self.x: jaxlib.xla_extension.ArrayImpl = None
        self.hook: typing.List[Hook] = hook if hook is not None else []
        self.require_gradient: bool = require_gradient
        self.gradient: Tensor = None
        if isinstance(x, Tensor):
            self.x: jaxlib.xla_extension.ArrayImpl = x.x
            self.hook: typing.List[Hook] = x.hook
            self.gradient: Tensor = x.gradient
            self.require_gradient: bool = x.require_gradient
        elif isinstance(x, jaxlib.xla_extension.ArrayImpl):
            self.x = jax.numpy.copy(x)
        elif isinstance(x, numpy.ndarray):
            self.x = jax.numpy.array(x)
        elif isinstance(x, int) or isinstance(x, float) or isinstance(self.x, bool):
            self.x = jax.numpy.full(
                shape=pair.shape,
                fill_value=x,
                dtype=pair.datatype.value,
            )
        assert isinstance(self.x, jaxlib.xla_extension.ArrayImpl), type(self.x)

    def _repr_html_(self) -> str:
        return str(self.x)

    def __add__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Add()(self, tensor_casting(x=other))

    def __bool__(self) -> bool:
        assert self.x.size == 1, self.x
        return bool(self.x)

    def __eq__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x == jax_casting(x=other))

    def __floordiv__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=self.x // jax_casting(x=other))

    def __ge__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x >= jax_casting(x=other))

    def __gt__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x > jax_casting(x=other))

    def __getitem__(
        self,
        *args,
    ) -> Tensor:
        return Tensor(x=self.x.__getitem__(*args))

    def __float__(self) -> float:
        assert self.x.size == 1, self.x
        return float(self.x)

    def __iadd__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        self: Tensor = Add()(self, tensor_casting(x=other))
        return self

    def __int__(self) -> int:
        assert self.x.size == 1, self.x
        return int(self.x)

    def __imul__(self, other: Tensor | int | float) -> Tensor:
        self: Tensor = Multiply()(self, tensor_casting(x=other))
        return self

    def __ipow__(self, other: Tensor | int | float) -> Tensor:
        self.x **= jax_casting(x=other)
        return self

    def __isub__(self, other: Tensor | int | float) -> Tensor:
        self: Tensor = Add()(self, Negative()(tensor_casting(x=other)))
        return self

    def __itruediv__(self, other: Tensor | int | float) -> Tensor:
        self: Tensor = Multiply()(self, Inverse()(tensor_casting(x=other)))
        return self

    def __invert__(self) -> Tensor:
        return Tensor(x=jax.numpy.invert(self.x))

    def __le__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x <= jax_casting(x=other))

    def __lt__(
        self,
        other: Tensor,
    ) -> Tensor:
        return Tensor(x=self.x < jax_casting(x=other))

    def __matmul__(
        self,
        y: Tensor,
    ) -> Tensor:
        return MatrixMultiplication()(self, y)

    def __mod__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=jax.numpy.remainder(self.x, jax_casting(x=other)))

    def __mul__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Multiply()(self, tensor_casting(x=other))

    def __ne__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x != jax_casting(x=other))

    def __pow__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x ** jax_casting(x=other))

    def __radd__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Add()(tensor_casting(x=other), self)

    def __req__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=jax_casting(x=other) == self.x)

    def __rfloordiv__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=jax_casting(x=other) // self.x)

    def __rge__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=jax_casting(x=other) >= self.x)

    def __rgt__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=jax_casting(x=other) > self.x)

    def __rle__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=jax_casting(x=other) <= self.x)

    def __rlt__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=jax_casting(x=other) < self.x)

    def __rmod__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=jax_casting(x=other) % self.x)

    def __rmul__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Multiply()(tensor_casting(x=other), self)

    def __rsub__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Add()(tensor_casting(x=other), Negative()(self))

    def __rtruediv__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Multiply()(tensor_casting(x=other), Inverse()(self))

    def __setitem__(self, key, value) -> None:
        self.x = self.x.at[key].set(jax_casting(x=value))

    def __str__(self) -> str:
        return self.x.__repr__()

    def __sub__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Add()(self, Negative()(tensor_casting(x=other)))

    def __truediv__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Multiply()(self, Inverse()(tensor_casting(x=other)))

    def all(self, axis: int) -> Tensor:
        # if axis is None, return a scalar
        return Tensor(x=jax.numpy.all(self.x, axis=axis))

    def any(self, axis: int) -> Tensor:
        return Tensor(x=jax.numpy.any(self.x, axis=axis))

    @staticmethod
    def arange(
        start: int,
        stop: int,
        step: int,
        datatype: Tensor.DataType,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.arange(start=start, stop=stop, step=step, dtype=datatype.value)
        )

    def argmax(self, axis: int) -> Tensor:
        return Tensor(x=jax.numpy.argmax(self.x, axis=axis))

    def argsort(self, axis: int) -> Tensor:
        return Tensor(x=jax.numpy.argsort(self.x, axis=axis))

    @staticmethod
    def array(
        x: typing.List[any],
        datatype: DataType,
        require_gradient: bool = False,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.array(x, dtype=datatype.value),
            require_gradient=require_gradient,
        )

    def astype(self, datatype: Tensor.DataType) -> Tensor:
        return Tensor(x=self.x.astype(datatype.value))

    def backward(self, gradient: Tensor = None) -> Tensor:
        if gradient is None:
            gradient = Tensor.full(shape=self.shape, value=1.0, datatype=self.datatype)
        self.gradient = gradient if self.gradient is None else self.gradient + gradient
        for hook in self.hook:
            hook.tensor.backward(
                gradient=hook.gradient_function(gradient=gradient),
            )
        return self

    @staticmethod
    def concatenate(
        tensors: typing.Sequence[Tensor],
        axis: int,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.concatenate(
                arrays=[array.x for array in tensors],
                axis=axis,
            ),
        )

    def count_nonzero(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.count_nonzero(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
        )

    @property
    def datatype(self) -> Tensor.DataType:
        return {
            jax.numpy.dtype("bool"): Tensor.DataType.BOOL,
            jax.numpy.dtype("int16"): Tensor.DataType.INT16,
            jax.numpy.dtype("int32"): Tensor.DataType.INT32,
            jax.numpy.dtype("int64"): Tensor.DataType.INT64,
            jax.numpy.dtype("float16"): Tensor.DataType.FLOAT16,
            jax.numpy.dtype("float32"): Tensor.DataType.FLOAT32,
            jax.numpy.dtype("float64"): Tensor.DataType.FLOAT64,
        }[self.x.dtype]

    @staticmethod
    def diagonal(
        value: int | float,
        size: int,
        datatype: Tensor.DataType,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.fill_diagonal(
                a=jax.numpy.zeros(
                    shape=(size, size),
                    dtype=datatype.value,
                ),
                val=value,
                inplace=False,
            ),
        )

    def exp(self) -> Tensor:
        return Tensor(x=jax.numpy.exp(self.x))

    def expand_dimension(
        self,
        *axis: int,
    ) -> Tensor:
        return ExpandDimension(*axis)(self)

    @staticmethod
    def full(
        shape: typing.Tuple[int],
        value: int | float,
        datatype: Tensor.DataType,
        require_gradient: bool = False,
    ) -> Tensor:
        if isinstance(value, Tensor.Value):
            value = value.value
        return Tensor(
            x=jax.numpy.full(
                shape=shape,
                fill_value=value,
                dtype=datatype.value,
            ),
            require_gradient=require_gradient,
        )

    def get_sample_x_and_y(
        self,
        number_of_target: int,
    ) -> typing.Tuple[Tensor, Tensor, Tensor]:
        return (
            Tensor(
                x=self.x[
                    ...,
                    number_of_target:,
                ],
            ),
            Tensor(
                x=self.x[
                    ...,
                    :number_of_target,
                ],
            ),
        )

    def get_by_index(
        # (A..., index)
        self,
        # (B...,)
        indexes: Tensor,
    ) -> Tensor:
        # (max{A..., B...}, index)
        answer: jaxlib.xla_extension.ArrayImpl = 1 * (
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
                    (answer / answer)
                )
                * (
                    # (A..., index)
                    self.x
                ),
                axis=-1,
            )
        ).astype(self.datatype)

    def in_sample(
        self,
        in_sample_size: int,
    ) -> Tensor:
        return Tensor(
            x=self.x[
                # ...
                ...,
                # Observation
                :in_sample_size,
                # Target and factor
                :,
            ]
        )

    def insert(
        self,
        index: int,
        value: int | float,
        axis: int,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.insert(
                arr=self.x,
                obj=index,
                values=value,
                axis=axis,
            ),
        )

    def inverse(
        self,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.linalg.inv(
                self.x,
            ),
        )

    def isnan(self) -> Tensor:
        return Tensor(
            x=jax.numpy.isnan(self.x),
        )

    @staticmethod
    def load(
        source: str,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.load(
                file=source,
            ),
        )

    def log2(self) -> Tensor:
        return Tensor(
            x=jax.numpy.log2(self.x),
        )

    def mean(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        return Tensor(x=jax.numpy.mean(self.x, axis=axis, keepdims=keep_dimension))

    def nanmean(self, axis: int, keep_dimension: bool) -> Tensor:
        return Tensor(x=jax.numpy.nanmean(self.x, axis=axis, keepdims=keep_dimension))

    def nansum(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.nansum(
                a=self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
        )

    def nan_to_num(
        self,
        nan: int | float,
        posinf: int | float,
        neginf: int | float,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.nan_to_num(
                x=self.x,
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            ),
        )

    @property
    def number_of_dimension(self) -> int:
        return jax.numpy.ndim(self.x)

    def out_of_sample(
        self,
        in_sample_size: int,
    ) -> Tensor:
        return Tensor(
            x=self.x[
                # ...
                ...,
                # Observation
                in_sample_size:,
                # Target and factor
                :,
            ]
        )

    @staticmethod
    def random_integer(
        shape: typing.Tuple[int],
        # inclusive
        minimum_value: int,
        # exclusive
        maximum_value: int,
        datatype: Tensor.DataType,
        require_gradient: bool,
        seed: int,
    ) -> Tensor:
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
            require_gradient=require_gradient,
        )

    @staticmethod
    def random_normal(
        shape: typing.Tuple[int],
        datatype: Tensor.DataType,
        require_gradient: bool,
        seed: int,
    ) -> Tensor:
        return Tensor(
            x=jax.random.normal(
                key=jax.random.PRNGKey(seed),
                shape=shape,
                dtype=datatype.value,
            ),
            require_gradient=require_gradient,
        )

    @staticmethod
    def random_uniform(
        shape: typing.Tuple[int],
        # inclusive
        minimum_value: int,
        # exclusive
        maximum_value: int,
        datatype: Tensor.DataType,
        require_gradient: bool,
        seed: int,
    ) -> Tensor:
        return Tensor(
            x=jax.random.uniform(
                key=jax.random.PRNGKey(seed),
                shape=shape,
                minval=minimum_value,
                maxval=maximum_value,
                dtype=datatype.value,
            ),
            require_gradient=require_gradient,
        )

    def reshape(
        self,
        *shape: int,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.reshape(
                a=self.x,
                newshape=shape,
            ),
        )

    def save(
        self,
        destination: str,
    ) -> None:
        jax.numpy.save(
            file=destination,
            arr=self.x,
        )
        try:
            os.remove(destination)
        except OSError:
            pass
        os.rename(
            f"{destination}.npy",
            destination,
        )

    @property
    def shape(self) -> typing.Tuple[int]:
        return self.x.shape

    def sign(self) -> Tensor:
        return Tensor(
            x=jax.numpy.sign(self.x),
        )

    def sigmoid(self) -> Tensor:
        return 1.0 / (1.0 + self.exp())

    def sliding_window(
        self,
        window_size: int,
    ) -> Tensor:
        # (..., Batch, Window, Target and factor)
        return (
            # (..., Batch, Target and factor, Window)
            Tensor(
                x=numpy.lib.stride_tricks.sliding_window_view(
                    # (..., Observation, Target and factor)
                    x=self.x,
                    window_shape=window_size,
                    # Observation
                    axis=-2,
                ),
            )
        ).swapaxes(
            # Batch
            -1,
            # Target and factor
            -2,
        )

    def sort(
        self,
        axis: int,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.sort(
                self.x,
                axis=axis,
            ),
        )

    def sqrt(self) -> Tensor:
        return Tensor(
            x=jax.numpy.sqrt(self.x),
        )

    def square(self) -> Tensor:
        return Tensor(
            x=jax.numpy.square(self.x),
        )

    def std(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.std(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
        )

    def sum(
        self,
        axis: int,
        keep_dimension: bool,
    ) -> Tensor:
        return Tensor(
            x=jax.numpy.sum(
                self.x,
                axis=axis,
                keepdims=keep_dimension,
            ),
        )

    def swapaxes(
        self,
        axis1: int,
        axis2: int,
    ) -> Tensor:
        axis = list(range(self.number_of_dimension))
        axis[axis1], axis[axis2] = axis[axis2], axis[axis1]
        return self.transpose(*axis)

    def tolist(self) -> typing.List[typing.Any]:
        return jax.numpy.asarray(self.x).tolist()

    def transpose(self, *index: int) -> Tensor:
        return Transpose(*index)(self)

    def unique(self) -> Tensor:
        return Tensor(
            x=jax.numpy.unique(self.x),
        )
