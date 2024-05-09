"""Metric module."""

import typing
from hoisaai.layer_0.tensor import Tensor


def explained_error(
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the explained error.

    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    # (..., Observation, Feature)
    return (
        # (..., Observation, Feature)
        prediction
    ) - (
        # (..., 1, Feature)
        actual.mean(
            # Observation
            axis=-2,
            keep_dimension=True,
        )
    )


def residual_error(
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the residual error.

    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    return prediction - actual


def total_error(
    # pylint: disable=unused-argument
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the total error.

    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    # (..., Observation, Feature)
    return (
        # (..., Observation, Feature)
        actual
    ) - (
        # (..., 1, Feature)
        actual.mean(
            # Observation
            axis=-2,
            keep_dimension=True,
        )
    )


def squared_error(
    error_function: typing.Callable[[Tensor, Tensor], Tensor],
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the squared error.

    :param error_function: The error function.
    :type error_function: Callable
    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    return error_function(
        prediction=prediction,
        actual=actual,
    ).square()


def sum_of_square(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the sum of square error.

    :param error_function: The error function.
    :type error_function: Callable
    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    return squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    ).sum(
        # Observation
        axis=-2,
        keep_dimension=False,
    )


def mean_squared_error(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the mean squared error.

    :param error_function: The error function.
    :type error_function: Callable
    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    return squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    ).mean(
        # Observation
        axis=-2,
        keep_dimension=False,
    )


def root_mean_squared_error(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    """Calculate the root mean squared error.

    :param error_function: The error function.
    :type error_function: Callable
    :param prediction: The prediction tensor.
    :type prediction: Tensor
    :param actual: The actual tensor.
    :type actual: Tensor
    """
    return mean_squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    ).sqrt()


class Metric(object):
    """Metric for evaluating the model."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        prediction_y: Tensor,
        sample_y: Tensor,
    ) -> Tensor:
        """Evaluate the model.

        :param prediction_y: The prediction tensor.
        :type prediction_y: Tensor
        :param sample_y: The sample tensor.
        :type sample_y: Tensor
        """
        raise NotImplementedError()


class GoyalAndWelch(Metric):
    """Goyal and Welch metric.
    
    :param in_sample_y: The in-sample tensor.
    :type in_sample_y: Tensor
    """
    def __init__(
        self,
        in_sample_y: Tensor,
    ) -> None:
        Metric.__init__(self)
        self.in_sample_y: Tensor = in_sample_y

    def evaluate(
        self,
        prediction_y: Tensor,
        sample_y: Tensor,
    ) -> Tensor:
        return root_mean_squared_error(
            # prediction - mean(actual) => sample_y - mean(in_sample_y)
            error_function=explained_error,
            prediction=sample_y,
            actual=self.in_sample_y,
        ) - root_mean_squared_error(
            error_function=residual_error,
            prediction=prediction_y,
            actual=sample_y,
        )
