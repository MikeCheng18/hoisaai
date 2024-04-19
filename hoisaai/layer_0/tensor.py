"""
This module contains the functions that are used to perform tensor operations.
"""

import typing
import jax
import jaxlib.xla_extension


def dot_multiplication(
    x: jaxlib.xla_extension.ArrayImpl,
    y: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Perform dot multiplication between two arrays.

    Args:
        x (jaxlib.xla_extension.ArrayImpl): The first input array.
        y (jaxlib.xla_extension.ArrayImpl): The second input array.

    Returns:
        jaxlib.xla_extension.ArrayImpl: The result of dot multiplication.

    Raises:
        None

    Examples:
        >>> x = jax.numpy.array([[1, 2], [3, 4]])
        >>> y = jax.numpy.array([[5, 6], [7, 8]])
        >>> dot_multiplication(x, y)
        Array([[19, 22],[43, 50]], dtype=int32)
    """
    # (..., i, j) * (..., j, k) -> (..., i, k)
    number_of_dimension: int = len(x.shape)
    subscripts: str = "".join(
        [chr(ord("i") + i) for i in range(number_of_dimension - 2)]
    )
    i: str = chr(ord("i") + number_of_dimension - 2)
    j: str = chr(ord("i") + number_of_dimension - 1)
    k: str = chr(ord("i") + number_of_dimension)
    subscripts: str = f"{subscripts}{i}{j},{subscripts}{j}{k}->{subscripts}{i}{k}"
    return jax.numpy.einsum(
        subscripts,
        x,
        y,
    )


def split_x_y(
    tensor: jaxlib.xla_extension.ArrayImpl,
    number_of_dependent_variables: int,
) -> typing.Tuple[jaxlib.xla_extension.ArrayImpl, jaxlib.xla_extension.ArrayImpl]:
    """
    Splits the input tensor into two parts: the dependent variables and the independent variables.

    Args:
        tensor (jaxlib.xla_extension.ArrayImpl): The input tensor to be split.
        number_of_dependent_variables (int): The number of dependent variables.

    Returns:
        Tuple[jaxlib.xla_extension.ArrayImpl, jaxlib.xla_extension.ArrayImpl]:
            A tuple containing two tensors, where the first tensor contains
            the independent variables and the second tensor contains the dependent variables.
    """
    return (
        tensor[..., number_of_dependent_variables:],
        tensor[..., :number_of_dependent_variables],
    )
