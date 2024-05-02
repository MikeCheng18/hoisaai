import math
from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.classification import (
    SupervisedClassificationModel,
)
from hoisaai.layer_1.learning_model.supervised.core.linear import (
    GradientDescentLinearRegression,
)


class LogisticRegression(
    GradientDescentLinearRegression,
    SupervisedClassificationModel,
):
    def __init__(
        self,
        maximum_iteration: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        GradientDescentLinearRegression.__init__(
            self,
            maximum_iteration=maximum_iteration,
            learning_rate=learning_rate,
            seed=seed,
        )
        SupervisedClassificationModel.__init__(self)
        self.maximum_iteration: int = maximum_iteration

    def __str__(self) -> str:
        return (
            "Logestic Regression: "
            + f"maximum_iteration={self.maximum_iteration}, "
            + f"learning_rate={self.learning_rate}"
        )

    def fit(
        self,
        # (..., In-sample observations, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        (
            # (..., In-sample observation, Feature)
            in_sample_x,
            # (..., In-sample observation, Target)
            in_sample_y,
        ) = in_sample.get_sample_x_and_y(
            number_of_target=number_of_target,
        )
        # (..., Feature + 1, In-sample observation)
        x_with_y_intercept: Tensor = (
            # (..., In-sample observation, Feature + 1)
            (
                # (..., In-sample observation, Feature)
                in_sample_x
            ).insert(
                index=0,
                value=1.0,
                # Feature
                axis=-1,
            )
        ).swapaxes(
            # Feature + 1
            axis1=-1,
            # In-sample observation
            axis2=-2,
        )
        # Feature + 1
        number_of_features: int = x_with_y_intercept.shape[-2]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(number_of_features)
        # (..., Feature + 1, Target)
        self.beta = Tensor.random_uniform(
            shape=(
                *x_with_y_intercept.shape[:-2],
                number_of_features,
                number_of_target,
            ),
            minimum_value=-limit,
            maximum_value=limit,
            datatype=Tensor.DataType.FLOAT32,
            seed=self.seed,
        )
        for _ in range(self.maximum_iteration):
            self.beta -= self.learning_rate * (
                # (..., Feature + 1, In-sample observation)
                x_with_y_intercept
                # (..., In-sample observation, Feature)
                @ (
                    # (..., In-sample observation, Target)
                    in_sample_y
                    # (..., In-sample observation, Target)
                    - GradientDescentLinearRegression.predict(
                        self,
                        # (..., In-sample observation, Feature)
                        sample_x=in_sample_x,
                    ).sigmoid()
                )
            )
            assert bool((self.beta == Tensor.Value.INF.value).any(axis=None)) is False
            assert (
                bool((self.beta == -1 * Tensor.Value.INF.value).any(axis=None)) is False
            )

    def predict(
        self,
        # (..., Sample observation, Feature)
        sample_x: Tensor,
    ) -> Tensor:
        return SupervisedClassificationModel.predict(
            self,
            sample_x=sample_x,
        )

    def predict_with_probability(
        self,
        sample_x: Tensor,
    ) -> SupervisedClassificationModel.Probability:
        # (..., Sample observation, Target)
        probability: Tensor = GradientDescentLinearRegression.predict(
            self, sample_x=sample_x
        ).sigmoid()
        return SupervisedClassificationModel.Probability(
            # (..., Sample observation, Target, 2)
            probability=(
                # (..., Sample observation, Target * 2)
                Tensor.concatenate(
                    tensors=[
                        # (..., Sample observation, Target)
                        1.0 - probability,
                        # (..., Sample observation, Target)
                        probability,
                    ],
                    axis=-1,
                )
            ).reshape(
                *probability.shape,
                2,
            ),
            # (unique_y,)
            unique_y=Tensor.array(x=[0.0, 1.0], datatype=Tensor.DataType.FLOAT32),
        )
