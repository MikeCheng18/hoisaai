from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.core.linear import LinearRegression


class ElasticNetRegression(LinearRegression):
    def __init__(
        self,
        elastic_lambda: float,
        l1_ratio: float,
        learning_rate: float,
        maximum_iteration: int,
        tolerance: float,
    ) -> None:
        LinearRegression.__init__(
            self,
        )
        self.elastic_lambda: float = elastic_lambda
        self.l1_ratio: float = l1_ratio
        self.learning_rate: float = learning_rate
        self.maximum_iteration: int = maximum_iteration
        self.tolerance: float = tolerance

    def __str__(self) -> str:
        return (
            "Elastic Net: "
            + f"Lambda={self.elastic_lambda:.6f}, "
            + f"L1 Ratio={self.l1_ratio:.6f}, "
            + f"Learning Rate={self.learning_rate:.6f}, "
            + f"Maximum Iteration={self.maximum_iteration}, "
            + f"Tolerance={self.tolerance}"
        )

    def fit(
        self,
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
        del in_sample
        # (..., Feature + 1, Target)
        self.beta: Tensor = Tensor.full(
            shape=(
                *in_sample_y.shape[:-2],
                in_sample_x.shape[-1] + 1,
                in_sample_y.shape[-1],
            ),
            value=1 / in_sample_x.shape[-1],
            datatype=Tensor.DataType.FLOAT32,
        )
        for _ in range(self.maximum_iteration):
            # (..., Feature + 1, Target)
            gradient: Tensor = -1.0 * (
                # (..., Feature + 1, Target)
                (
                    # (..., Feature + 1, In-sample observation)
                    (
                        # (..., In-sample observation, Feature + 1)
                        (
                            # (..., In-sample observation, Feature)
                            in_sample_x
                        ).insert(
                            index=0,
                            value=1.0,
                            axis=-1,
                        )
                    ).swapaxes(
                        axis1=-1,
                        axis2=-2,
                    )
                    @ (
                        # (..., In-sample observation, Target)
                        in_sample_y
                        - self.predict(
                            # (..., In-sample observation, Feature)
                            sample_x=in_sample_x,
                        )
                    )
                )
            ) + self.elastic_lambda * (
                # (..., Feature + 1, Target)
                self.l1_ratio * self.beta.sign()
                # (..., Feature + 1, Target)
                + (1 - self.l1_ratio) * self.beta
            )
            self.beta -= self.learning_rate * gradient
            assert not bool(self.beta.isnan().any(axis=None))
            if bool((gradient <= self.tolerance).all(axis=None)):
                break


class LassoRegression(LinearRegression):
    def __init__(
        self,
        lasso_lambda: float,
        learning_rate: float,
        maximum_iteration: int,
        tolerance: float,
    ) -> None:
        LinearRegression.__init__(
            self,
        )
        self.lasso_lambda: float = lasso_lambda
        self.learning_rate: float = learning_rate
        self.maximum_iteration: int = maximum_iteration
        self.tolerance: float = tolerance

    def __str__(self) -> str:
        return (
            "Lasso Regression: "
            + f"Lambda={self.lasso_lambda:.6f}, "
            + f"Learning Rate={self.learning_rate:.6f}, "
            + f"Maximum Iteration={self.maximum_iteration}, "
            + f"Tolerance={self.tolerance}"
        )

    def fit(
        self,
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
        del in_sample
        # (..., Feature + 1, Target)
        self.beta: Tensor = Tensor.full(
            shape=(
                *in_sample_y.shape[:-2],
                in_sample_x.shape[-1] + 1,
                in_sample_y.shape[-1],
            ),
            value=1 / in_sample_x.shape[-1],
            datatype=Tensor.DataType.FLOAT32,
        )
        for _ in range(self.maximum_iteration):
            # (..., Feature + 1, Target)
            gradient: Tensor = -1.0 * (
                # (..., Feature + 1, Target)
                (
                    # (..., Feature + 1, In-sample observation)
                    (
                        # (..., In-sample observation, Feature + 1)
                        (
                            # (..., In-sample observation, Feature)
                            in_sample_x
                        ).insert(
                            index=0,
                            value=1.0,
                            axis=-1,
                        )
                    ).swapaxes(
                        axis1=-1,
                        axis2=-2,
                    )
                    @ (
                        # (..., In-sample observation, Target)
                        in_sample_y
                        - self.predict(
                            # (..., In-sample observation, Feature)
                            sample_x=in_sample_x,
                        )
                    )
                )
            ) + (
                self.lasso_lambda
                # (..., Feature + 1, Target)
                * self.beta.sign()
            )
            self.beta -= self.learning_rate * gradient
            assert not bool(self.beta.isnan().any(axis=None))
            if bool((gradient <= self.tolerance).all(axis=None)):
                break


class OrdinaryLeastSquares(LinearRegression):
    def __str__(self) -> str:
        return "Ordinary least squares"

    def fit(
        self,
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
        del in_sample
        # (..., In-sample observation, Feature + 1)
        x_with_y_intercept: Tensor = (
            # (..., In-sample observation, Feature)
            in_sample_x
        ).insert(
            index=0,
            value=1.0,
            # Feature
            axis=-1,
        )
        # (..., Feature + 1, In-sample observation)
        transpose_x_with_y_intercept: Tensor = x_with_y_intercept.swapaxes(
            axis1=-1,
            axis2=-2,
        )
        # (..., Feature + 1, Target)
        self.beta: Tensor = (
            # (..., Feature + 1, In-sample observation)
            transpose_x_with_y_intercept
            # (..., In-sample observation, Feature + 1)
            @ x_with_y_intercept
        ).inverse() @ (
            # (..., Feature + 1, In-sample observation)
            transpose_x_with_y_intercept
            # (..., In-sample observation, Target)
            @ in_sample_y
        )
        assert not bool(self.beta.isnan().any(axis=None))


class RidgeRegression(LinearRegression):
    def __init__(
        self,
        ridge_lambda: float,
    ) -> None:
        super().__init__()
        self.ridge_lambda: float = ridge_lambda

    def __str__(self) -> str:
        return f"Ridge Regression: Lambda={self.ridge_lambda:.6f}"

    def fit(
        self,
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
        del in_sample
        # (..., In-sample observation, Feature + 1)
        x_with_y_intercept: Tensor = (
            # (..., In-sample observation, Feature)
            in_sample_x
        ).insert(
            index=0,
            value=1.0,
            # Feature
            axis=-1,
        )
        # (..., Feature + 1, In-sample observation)
        transpose_x_with_y_intercept: Tensor = x_with_y_intercept.swapaxes(
            axis1=-1,
            axis2=-2,
        )
        # (..., Feature + 1, Target)
        self.beta: Tensor = (
            (  # (..., Feature + 1, In-sample observation)
                transpose_x_with_y_intercept
                # (..., In-sample observation, Feature + 1)
                @ x_with_y_intercept
            )
            + Tensor.diagonal(
                value=self.ridge_lambda,
                # Feature + 1
                size=x_with_y_intercept.shape[-1],
                datatype=Tensor.DataType.FLOAT32,
            )
        ).inverse() @ (
            # (..., Feature + 1, In-sample observation)
            transpose_x_with_y_intercept
            # (..., In-sample observation, Target)
            @ in_sample_y
        )
        assert not bool(self.beta.isnan().any(axis=None))
