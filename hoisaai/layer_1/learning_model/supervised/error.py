"""
This module contains functions for calculating the error
between the predicted values and the actual values.
"""

import jax
import jaxlib.xla_extension


def explained_error(
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the explained error between the prediction and the actual values.

    Args:
        prediction (jaxlib.xla_extension.ArrayImpl): The predicted values.
        actual (jaxlib.xla_extension.ArrayImpl): The actual values.

    Returns:
        jaxlib.xla_extension.ArrayImpl:
            The explained error between the prediction and the actual values.
    """
    # (..., Observation, Independent Variables)
    return (
        # (..., Observation, Independent Variables)
        prediction
    ) - (
        # (..., 1, Independent Variables)
        jax.numpy.average(
            actual,
            axis=-2,
            keepdims=True,
        )
    )


def residual_error(
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the residual error between the prediction and the actual values.

    Args:
        prediction: The predicted values.
        actual: The actual values.

    Returns:
        The residual error, which is the difference between the prediction and the actual values.
    """
    return prediction - actual


def total_error(
    # pylint: disable=unused-argument
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the total error between the predicted values and the actual values.

    Args:
        prediction: An array containing the predicted values.
        actual: An array containing the actual values.

    Returns:
        An array containing the total error between the predicted values and the actual values.
    """
    # (..., Observation, Independent Variables)
    return (
        # (..., Observation, Independent Variables)
        actual
    ) - (
        # (..., 1, Independent Variables)
        jax.numpy.average(
            actual,
            axis=-2,
            keepdims=True,
        )
    )


def squared_error(
    error_function,
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the squared error between the predicted values and the actual values.

    Args:
        error_function: The error function to be applied to the prediction and actual values.
        prediction: The predicted values.
        actual: The actual values.

    Returns:
        The squared error between the prediction and actual values.
    """
    return jax.numpy.square(
        error_function(
            prediction=prediction,
            actual=actual,
        )
    )


def sum_of_square(
    error_function,
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the sum of squared errors between the predicted values and the actual values.

    Parameters:
    - error_function: The error function to be used for calculating the squared error.
    - prediction: The predicted values.
    - actual: The actual values.

    Returns:
    - The sum of squared errors.

    """
    return (
        jax.numpy.sum(
            squared_error(
                error_function=error_function,
                prediction=prediction,
                actual=actual,
            ),
            # Observation
            axis=-2,
        ),
    )


def mean_squared_error(
    error_function,
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the mean squared error between the predicted values and the actual values.

    Args:
        error_function: The error function to be used for calculating the squared error.
        prediction: The predicted values.
        actual: The actual values.

    Returns:
        The mean squared error.

    """
    return (
        jax.numpy.mean(
            squared_error(
                error_function=error_function,
                prediction=prediction,
                actual=actual,
            ),
            # Observation
            axis=-2,
        ),
    )


def root_mean_squared_error(
    error_function,
    prediction: jaxlib.xla_extension.ArrayImpl,
    actual: jaxlib.xla_extension.ArrayImpl,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Calculates the root mean squared error (RMSE)
    between the predicted values and the actual values.

    Args:
        error_function: The error function to be used for calculating the squared error.
        prediction: The predicted values.
        actual: The actual values.

    Returns:
        The root mean squared error.

    """
    return jax.numpy.sqrt(
        jax.numpy.mean(
            squared_error(
                error_function=error_function,
                prediction=prediction,
                actual=actual,
            ),
            # Observation
            axis=-2,
        ),
    )
