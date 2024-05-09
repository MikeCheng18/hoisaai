"""Loss functions for neural networks."""

import abc
from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_2.metric import mean_squared_error, residual_error


class Loss(object):
    """Loss function for neural networks."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def backward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        """Backward propagation of the loss function."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        """Forward propagation of the loss function."""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean squared error loss function for neural networks."""

    def backward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        return prediction.backward(
            gradient=(
                2.0
                * residual_error(
                    prediction=prediction,
                    actual=actual,
                )
            )
        )

    def forward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        return mean_squared_error(
            error_function=residual_error,
            prediction=prediction,
            actual=actual,
        )
