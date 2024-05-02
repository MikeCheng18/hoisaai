from hoisaai.layer_0.tensor import Tensor


def explained_error(
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
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
    return prediction - actual


def total_error(
    # pylint: disable=unused-argument
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
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
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    error: Tensor = error_function(
        prediction=prediction,
        actual=actual,
    )
    return error.square()


def sum_of_square(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    error: Tensor = squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    )
    return error.sum(
        # Observation
        axis=-2,
        keep_dimension=False,
    )


def mean_squared_error(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    error: Tensor = squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    )
    return error.mean(
        # Observation
        axis=-2,
        keep_dimension=False,
    )


def root_mean_squared_error(
    error_function,
    prediction: Tensor,
    actual: Tensor,
) -> Tensor:
    return mean_squared_error(
        error_function=error_function,
        prediction=prediction,
        actual=actual,
    ).sqrt()


class Metric(object):
    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        prediction: Tensor,
        sample: Tensor,
    ) -> Tensor:
        raise NotImplementedError()


class GoyalWelch(Metric):
    def __init__(self) -> None:
        Metric.__init__(self)

    def evaluate(
        self,
        prediction: Tensor,
        sample: Tensor,
    ) -> Tensor:
        return root_mean_squared_error(
            error_function=total_error,
            prediction=prediction,
            actual=sample,
        ) - root_mean_squared_error(
            error_function=residual_error,
            prediction=prediction,
            actual=sample,
        )
