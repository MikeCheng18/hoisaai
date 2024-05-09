"""Linear regression model."""

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.regression.regression import Regression


class LinearRegression(Regression):
    """Linear regression model."""

    def __init__(self) -> None:
        Regression.__init__(self)
        # (..., Target + 1, Feature)
        self.beta: Tensor = None

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        self.number_of_target: int = number_of_target

    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:  # (..., Sample observation, Feature)
        assert self.beta is not None
        # (..., Sample observation, Feature)
        return (
            # (..., Sample observation, Target + 1)
            (
                # (..., Sample observation, Target)
                sample_x
            ).insert(
                index=0,
                value=1.0,
                # Target
                axis=-1,
            )
            # (..., Target + 1, Feature)
            @ self.beta
        )


class GradientDescentLinearRegression(LinearRegression):
    """Linear regression model with gradient descent optimization.

    :param maximum_iteration: Maximum iteration.
    :type maximum_iteration: int
    :param learning_rate: Learning rate.
    :type learning_rate: float
    :param seed: Random seed.
    :type seed: int
    """

    def __init__(
        self,
        maximum_iteration: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        LinearRegression.__init__(self)
        self.maximum_iteration: int = maximum_iteration
        self.learning_rate: float = learning_rate
        self.seed: int = seed

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        Regression.fit(
            self,
            in_sample=in_sample,
            number_of_target=number_of_target,
        )
