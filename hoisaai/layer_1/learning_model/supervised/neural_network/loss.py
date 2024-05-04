from hoisaai.layer_0.tensor import Tensor
from hoisaai.lyaer_2.metric import mean_squared_error, residual_error


class Loss(object):
    def __init__(self) -> None:
        pass

    def backward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, prediction: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError()


class MeanSquaredError(Loss):

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
