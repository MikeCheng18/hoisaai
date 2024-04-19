"""
This module contains the implementation of the linear regression algorithms.
"""

import typing

import jax
import jaxlib.xla_extension

from hoisaai.layer_0.tensor import dot_multiplication, split_x_y
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel
from hoisaai.layer_1.model import Tensor, get_tensor


class OrdinaryLeastSquares(SupervisedLearningModel):
    """
    Implementation of the Ordinary Least Squares (OLS) algorithm for linear regression.

    Args:
        tensor (Tensor): The input tensor containing the independent and dependent variables.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.

    Attributes:
        betas (jaxlib.xla_extension.ArrayImpl): The coefficients of the linear regression model.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.
        tensor (Tensor): The input tensor containing the independent and dependent variables.

    Methods:
        predict(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Predicts the dependent variable values for the given input tensor.

        shapley_value(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Computes the Shapley value for the given input tensor.

        transform() -> typing.Iterator[typing.Any]:
            Raises a NotImplementedError.

    """

    def __init__(
        self,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__(
            tensor=tensor,
            number_of_dependent_variables=number_of_dependent_variables,
        )
        self.betas: jaxlib.xla_extension.ArrayImpl = None

    def predict(
        self,
        tensor: Tensor,
    ) -> jaxlib.xla_extension.ArrayImpl:
        """
        Predicts the dependent variable values for the given input tensor.

        Args:
            tensor (Tensor): The input tensor containing the independent variables.

        Returns:
            jaxlib.xla_extension.ArrayImpl: The predicted dependent variable values.

        """
        sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=tensor,
        )
        assert self.betas is not None
        return dot_multiplication(
            x=jax.numpy.insert(
                # Only independent variables
                arr=sample[
                    ...,
                    self.number_of_dependent_variables :,
                ],
                # add y-intercept
                obj=0,
                values=1.0,
                axis=-1,
            ),
            y=self.betas,
        )

    def shapley_value(
        self,
        tensor: Tensor,
    ) -> jaxlib.xla_extension.ArrayImpl:
        """
        Computes the Shapley value for the given input tensor.

        Args:
            tensor (Tensor): The input tensor containing the independent variables.

        Returns:
            jaxlib.xla_extension.ArrayImpl: The Shapley value.

        """
        in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=self.tensor,
        )
        out_of_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=tensor,
        )
        assert self.betas is not None
        return (
            # (..., Observation, Independent variable, 1)
            jax.numpy.expand_dims(
                a=(
                    # (..., Observation, Independent variable)
                    out_of_sample[
                        ...,
                        self.number_of_dependent_variables :,
                    ]
                )
                - (
                    # Expected value of independent variables
                    # (..., 1, Independent variable)
                    jax.numpy.average(
                        # (..., Observation, Independent variable)
                        in_sample[
                            ...,
                            self.number_of_dependent_variables :,
                        ],
                        # Observation
                        axis=-2,
                        keepdims=True,
                    )
                ),
                axis=-1,
            )
        ) * (
            # (..., Y-intercept and independent variable, Dependent variables)
            self.betas[
                ...,
                # Removing the y-intercept
                1:,
                :,
            ]
        )

    def transform(self) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


class LinearRegression(OrdinaryLeastSquares):
    """
    Implementation of the linear regression.

    Args:
        tensor (Tensor): The input tensor containing the independent and dependent variables.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.

    Attributes:
        betas (jaxlib.xla_extension.ArrayImpl): The coefficients of the linear regression model.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.
        tensor (Tensor): The input tensor containing the independent and dependent variables.

    Methods:
        predict(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Predicts the dependent variable values for the given input tensor.

        shapley_value(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Computes the Shapley value for the given input tensor.

        transform() -> typing.Iterator[typing.Any]:
            Raises a NotImplementedError.

    """

    def __str__(self) -> str:
        return "Linear Regression"

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the input tensor by performing linear regression.

        Returns:
            An iterator that yields the calculated beta values.
        """
        if self.betas is None or self.stateful is False:
            in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
                tensor=self.tensor,
            )
            assert in_sample is not None
            (
                # (..., Observation, Independent variable)
                x,
                # (..., Observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., Observation, Independent variable + 1)
            x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = jax.numpy.insert(
                arr=x,
                obj=0,
                values=1.0,
                axis=-1,
            )
            # (..., Independent variable + 1, Observation)
            transpose_x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = (
                jax.numpy.swapaxes(
                    x_with_y_intercept,
                    -1,
                    -2,
                )
            )
            # (..., Independent variable + 1, Dependent variable)
            self.betas: jaxlib.xla_extension.ArrayImpl = dot_multiplication(
                # (..., Independent variable + 1, Independent variable + 1)
                jax.numpy.linalg.inv(
                    # (..., Independent variable + 1, Independent variable + 1)
                    a=dot_multiplication(
                        # (..., Independent variable + 1, Observation)
                        transpose_x_with_y_intercept,
                        # (..., Observation, Independent variable + 1)
                        x_with_y_intercept,
                    ),
                ),
                # (..., Independent variable + 1, Dependent variable)
                dot_multiplication(
                    # (..., Independent variable + 1, Observation)
                    transpose_x_with_y_intercept,
                    # (..., Observation, Dependent variable)
                    y,
                ),
            )
        yield self.betas


class RidgeRegression(OrdinaryLeastSquares):
    """
    Implementation of the linear regression.

    Args:
        ridge_lambda (int): The regularization parameter.
        tensor (Tensor): The input tensor containing the independent and dependent variables.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.

    Attributes:
        betas (jaxlib.xla_extension.ArrayImpl): The coefficients of the linear regression model.
        number_of_dependent_variables (int): The number of dependent variables in the tensor.
        ridge_lambda (int): The regularization parameter.
        tensor (Tensor): The input tensor containing the independent and dependent variables.

    Methods:
        predict(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Predicts the dependent variable values for the given input tensor.

        shapley_value(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Computes the Shapley value for the given input tensor.

        transform() -> typing.Iterator[typing.Any]:
            Raises a NotImplementedError.

    """

    def __init__(
        self,
        ridge_lambda: float,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__(
            tensor=tensor,
            number_of_dependent_variables=number_of_dependent_variables,
        )
        self.betas: jaxlib.xla_extension.ArrayImpl = None
        self.ridge_lambda: float = ridge_lambda

    def __str__(self) -> str:
        return f"Ridge Regression: Lambda={self.ridge_lambda:.3f}"

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the input tensor by performing linear regression.

        Returns:
            An iterator containing the calculated beta values.
        """
        if self.betas is None or self.stateful is False:
            in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
                tensor=self.tensor,
            )
            assert in_sample is not None
            (
                # (..., Observation, Independent variable)
                x,
                # (..., Observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., Observation, Independent variable + 1)
            x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = jax.numpy.insert(
                arr=x,
                obj=0,
                values=1.0,
                axis=-1,
            )
            # (..., Independent variable + 1, Observation)
            transpose_x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = (
                jax.numpy.swapaxes(
                    x_with_y_intercept,
                    -1,
                    -2,
                )
            )
            # (..., Independent variable + 1, Dependent variable)
            self.betas: jaxlib.xla_extension.ArrayImpl = dot_multiplication(
                # (..., Independent variable + 1, Independent variable + 1)
                jax.numpy.linalg.inv(
                    # (..., Independent variable + 1, Independent variable + 1)
                    a=dot_multiplication(
                        # (..., Independent variable + 1, Observation)
                        transpose_x_with_y_intercept,
                        # (..., Observation, Independent variable + 1)
                        x_with_y_intercept,
                    )  # (..., Independent variable + 1, Independent variable + 1)
                    + jax.numpy.diag(
                        jax.numpy.full(
                            x_with_y_intercept.shape[-1],
                            fill_value=self.ridge_lambda,
                        )
                    ),
                ),
                # (..., Independent variable + 1, Dependent variable)
                dot_multiplication(
                    # (..., Independent variable + 1, Observation)
                    transpose_x_with_y_intercept,
                    # (..., Observation, Dependent variable)
                    y,
                ),
            )
        yield self.betas
