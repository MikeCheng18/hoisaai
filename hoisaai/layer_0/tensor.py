from __future__ import annotations

import enum
import os
import typing

import jax
import jaxlib.xla_extension
import numpy

UNICODE_OF_I: str = ord("i")


def tensor_casting(
    x: Tensor | jaxlib.xla_extension.ArrayImpl | int | float,
) -> jaxlib.xla_extension.ArrayImpl | int | float:
    if isinstance(x, Tensor):
        return x.x
    elif isinstance(x, numpy.ndarray):
        return jax.numpy.array(x)
    return x


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
        x: numpy.ndarray | jaxlib.xla_extension.ArrayImpl,
    ) -> None:
        self.x: jaxlib.xla_extension.ArrayImpl = x
        if isinstance(self.x, numpy.ndarray):
            self.x = jax.numpy.array(self.x)
        assert isinstance(self.x, jaxlib.xla_extension.ArrayImpl), type(self.x)

    def _repr_html_(self) -> str:
        return self.x.__repr__()

    def __add__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x + tensor_casting(x=other))

    def __bool__(self) -> bool:
        assert self.x.size == 1, self.x
        return bool(self.x)

    def __eq__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x == tensor_casting(x=other))

    def __floordiv__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=self.x // tensor_casting(x=other))

    def __ge__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x >= tensor_casting(x=other))

    def __gt__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x > tensor_casting(x=other))

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
        self.x += tensor_casting(x=other)
        return self

    def __int__(self) -> int:
        assert self.x.size == 1, self.x
        return int(self.x)

    def __imul__(self, other: Tensor | int | float) -> Tensor:
        self.x *= tensor_casting(x=other)
        return self

    def __ipow__(self, other: Tensor | int | float) -> Tensor:
        self.x **= tensor_casting(x=other)
        return self

    def __isub__(self, other: Tensor | int | float) -> Tensor:
        self.x -= tensor_casting(x=other)
        return self

    def __itruediv__(self, other: Tensor | int | float) -> Tensor:
        self.x /= tensor_casting(x=other)
        return self

    def __invert__(self) -> Tensor:
        return Tensor(x=~self.x)

    def __le__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x <= tensor_casting(x=other))

    def __lt__(
        self,
        other: Tensor,
    ) -> Tensor:
        return Tensor(x=self.x < tensor_casting(x=other))

    def __matmul__(
        self,
        y: Tensor,
    ) -> Tensor:
        # (..., i, j) * (..., j, k) -> (..., i, k)
        number_of_dimension: int = len(self.x.shape)
        subscripts: str = "".join(
            [chr(UNICODE_OF_I + i) for i in range(number_of_dimension - 2)]
        )
        i: str = chr(UNICODE_OF_I + number_of_dimension - 2)
        j: str = chr(UNICODE_OF_I + number_of_dimension - 1)
        k: str = chr(UNICODE_OF_I + number_of_dimension)
        subscripts: str = f"{subscripts}{i}{j},{subscripts}{j}{k}->{subscripts}{i}{k}"
        return Tensor(x=jax.numpy.einsum(subscripts, self.x, y.x))

    def __mod__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=jax.numpy.remainder(self.x, tensor_casting(x=other)))

    def __mul__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x * tensor_casting(x=other))

    def __ne__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x != tensor_casting(x=other))

    def __pow__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x ** tensor_casting(x=other))

    def __radd__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) + self.x)

    def __req__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) == self.x)

    def __rfloordiv__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) // self.x)

    def __rge__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=tensor_casting(x=other) >= self.x)

    def __rgt__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=tensor_casting(x=other) > self.x)

    def __rle__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=tensor_casting(x=other) <= self.x)

    def __rlt__(
        self,
        other: Tensor | int | float,
    ):
        return Tensor(x=tensor_casting(x=other) < self.x)

    def __rmod__(
        self,
        other: Tensor | int,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) % self.x)

    def __rmul__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) * self.x)

    def __rsub__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) - self.x)

    def __rtruediv__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=tensor_casting(x=other) / self.x)

    def __setitem__(self, key, value) -> None:
        self.x = self.x.at[key].set(tensor_casting(x=value))

    def __str__(self) -> str:
        return self.x.__repr__()

    def __sub__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x - tensor_casting(x=other))

    def __truediv__(
        self,
        other: Tensor | int | float,
    ) -> Tensor:
        return Tensor(x=self.x / tensor_casting(x=other))

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
    def array(x: typing.List[any], datatype: DataType) -> Tensor:
        return Tensor(x=jax.numpy.array(x, dtype=datatype.value))

    def astype(self, datatype: Tensor.DataType) -> Tensor:
        return Tensor(x=self.x.astype(datatype.value))

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
        return Tensor(
            x=jax.numpy.expand_dims(
                a=self.x,
                axis=axis,
            ),
        )

    @staticmethod
    def full(
        shape: typing.Tuple[int],
        value: int | float,
        datatype: Tensor.DataType,
    ) -> Tensor:
        if isinstance(value, Tensor.Value):
            value = value.value
        return Tensor(
            x=jax.numpy.full(
                shape=shape,
                fill_value=value,
                dtype=datatype.value,
            ),
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
        )

    @staticmethod
    def random_uniform(
        shape: typing.Tuple[int],
        # inclusive
        minimum_value: int,
        # exclusive
        maximum_value: int,
        datatype: Tensor.DataType,
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
        return Tensor(
            x=jax.numpy.swapaxes(
                a=self.x,
                axis1=axis1,
                axis2=axis2,
            ),
        )

    def tolist(self) -> typing.List[typing.Any]:
        return jax.numpy.asarray(self.x).tolist()

    def unique(self) -> Tensor:
        return Tensor(
            x=jax.numpy.unique(self.x),
        )
