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
        # (..., Independent variable + 1, Dependent variable)
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
        # (..., Out-of-sample observation, Dependent variable and independent variable)
        out_of_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=tensor,
        )
        assert self.betas is not None
        return dot_multiplication(
            x=jax.numpy.insert(
                # Only independent variables
                arr=out_of_sample[
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
        # (..., In-sample observation, Dependent variable and independent variable)
        in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=self.tensor,
        )
        # (..., Out-of-sample observation, Dependent variable and independent variable)
        out_of_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
            tensor=tensor,
        )
        assert self.betas is not None

        return (
            # (..., Out-of-sample observation, Dependent variables, Independent variable)
            jax.numpy.swapaxes(
                # (..., Out-of-sample observation, Independent variable, Dependent variables)
                a=(
                    # (..., Out-of-sample observation, Independent variable, 1)
                    jax.numpy.expand_dims(
                        a=(
                            # (..., Out-of-sample observation, Independent variable)
                            out_of_sample[
                                ...,
                                self.number_of_dependent_variables :,
                            ]
                        )
                        - (
                            # Expected value of independent variables
                            # (..., 1, Independent variable)
                            jax.numpy.average(
                                # (..., In-sample observation, Independent variable)
                                in_sample[
                                    ...,
                                    self.number_of_dependent_variables :,
                                ],
                                # In-sample observation
                                axis=-2,
                                keepdims=True,
                            )
                        ),
                        axis=-1,
                    )
                )
                * (
                    # (..., Independent variable, Dependent variables)
                    (
                        # (..., Y-intercept and independent variable, Dependent variables)
                        self.betas[
                            ...,
                            # Removing the y-intercept
                            1:,
                            :,
                        ]
                    )
                ),
                # Independent variable
                axis1=-2,
                # Dependent variables
                axis2=-1,
            )
        )

    def transform(self) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


class ElasticNet(OrdinaryLeastSquares):
    """
    ElasticNet class for performing linear regression with elastic net regularization.

    Args:
        elastic_lambda (float): The regularization parameter lambda.
        l1_ratio (float): The mixing parameter for L1 and L2 regularization.
        learning_rate (float): The learning rate for gradient descent.
        maximum_iteration (int): The maximum number of iterations for gradient descent.
        tolerance (float): The tolerance value for convergence.
        tensor (Tensor, optional): The input tensor. Defaults to None.
        number_of_dependent_variables (int, optional):
            The number of dependent variables. Defaults to None.
    """

    def __init__(
        self,
        elastic_lambda: float,
        l1_ratio: float,
        learning_rate: float,
        maximum_iteration: int,
        tolerance: float,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__(
            tensor=tensor,
            number_of_dependent_variables=number_of_dependent_variables,
        )
        self.betas: jaxlib.xla_extension.ArrayImpl = None
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

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the input data using linear regression.

        Returns:
            An iterator that yields the updated beta values at each iteration.
        """
        if self.betas is None or self.stateful is False:
            in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
                tensor=self.tensor,
            )
            assert in_sample is not None
            (
                # (..., In-sample observation, Independent variable)
                x,
                # (..., In-sample observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., Independent variable + 1, Dependent variable)
            self.betas = jax.numpy.full(
                shape=(
                    *y.shape[:-2],
                    x.shape[-1] + 1,
                    y.shape[-1],
                ),
                fill_value=1.0,
            )
            for _ in range(self.maximum_iteration):
                # (..., Independent variable + 1, Dependent variable)
                gradient = (
                    -1.0
                    # (..., Independent variable + 1, Dependent variable)
                    * dot_multiplication(
                        # (..., Independent variable + 1, In-sample observation)
                        x=jax.numpy.swapaxes(
                            # (..., In-sample observation, Independent variable + 1)
                            jax.numpy.insert(
                                # Only independent variables
                                arr=x,
                                # add y-intercept
                                obj=0,
                                values=1.0,
                                axis=-1,
                            ),
                            -2,
                            -1,
                        ),
                        y=(
                            # (..., In-sample observation, Dependent variable)
                            y
                            # (..., In-sample observation, Dependent variable)
                            - self.predict(
                                tensor=self.tensor,
                            )
                        ),
                    )
                ) + (
                    self.elastic_lambda
                    # (..., Independent variable + 1, Dependent variable)
                    * (
                        # L1 regularization
                        # (..., Independent variable + 1, Dependent variable)
                        (
                            self.l1_ratio
                            * jax.numpy.sign(
                                # (..., Independent variable + 1, Dependent variable)
                                self.betas
                            )
                        )
                        # L2 regularization
                        # (..., Independent variable + 1, Dependent variable)
                        + (
                            (1 - self.l1_ratio)
                            # (..., Independent variable + 1, Dependent variable)
                            * self.betas
                        )
                    )
                )
                self.betas -= self.learning_rate * gradient
                yield self.betas
                assert not jax.numpy.any(jax.numpy.isnan(self.betas))
                if jax.numpy.all(
                    gradient.ravel() <= self.tolerance,
                ):
                    break
        yield self.betas


class LassoRegression(OrdinaryLeastSquares):
    """
    Lasso Regression model.

    This class represents a Lasso Regression model, which is a linear regression model
    with L1 regularization. It inherits from the ElasticNet class.

    Parameters:
    - lasso_lambda (float): The regularization parameter for L1 regularization.
    - learning_rate (float): The learning rate for the optimization algorithm.
    - maximum_iteration (int): The maximum number of iterations for the optimization algorithm.
    - tolerance (float): The convergence tolerance for the optimization algorithm.
    - tensor (Tensor, optional): The input tensor for the model. Defaults to None.
    - number_of_dependent_variables (int, optional):
        The number of dependent variables. Defaults to None.
    """

    def __init__(
        self,
        lasso_lambda: float,
        learning_rate: float,
        maximum_iteration: int,
        tolerance: float,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__(
            tensor=tensor,
            number_of_dependent_variables=number_of_dependent_variables,
        )
        self.betas: jaxlib.xla_extension.ArrayImpl = None
        self.lasso_lambda: float = lasso_lambda
        self.learning_rate: float = learning_rate
        self.maximum_iteration: int = maximum_iteration
        self.tolerance: float = tolerance

    def __str__(self) -> str:
        return (
            "Lasso Regression:"
            + f"Lambda={self.lasso_lambda:.6f}, "
            + f"Learning Rate={self.learning_rate:.6f}, "
            + f"Maximum Iteration={self.maximum_iteration}, "
            + f"Tolerance={self.tolerance}"
        )

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the input data using linear regression.

        Returns:
            An iterator that yields the updated beta values at each iteration.
        """
        if self.betas is None or self.stateful is False:
            in_sample: jaxlib.xla_extension.ArrayImpl = get_tensor(
                tensor=self.tensor,
            )
            assert in_sample is not None
            (
                # (..., In-sample observation, Independent variable)
                x,
                # (..., In-sample observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., Independent variable + 1, Dependent variable)
            self.betas = jax.numpy.full(
                shape=(
                    *y.shape[:-2],
                    x.shape[-1] + 1,
                    y.shape[-1],
                ),
                fill_value=1.0,
            )
            for _ in range(self.maximum_iteration):
                # (..., Independent variable + 1, Dependent variable)
                gradient = (
                    -1.0
                    # (..., Independent variable + 1, Dependent variable)
                    * dot_multiplication(
                        # (..., Independent variable + 1, In-sample observation)
                        x=jax.numpy.swapaxes(
                            # (..., In-sample observation, Independent variable + 1)
                            jax.numpy.insert(
                                # Only independent variables
                                arr=x,
                                # add y-intercept
                                obj=0,
                                values=1.0,
                                axis=-1,
                            ),
                            -2,
                            -1,
                        ),
                        y=(
                            # (..., In-sample observation, Dependent variable)
                            y
                            # (..., In-sample observation, Dependent variable)
                            - self.predict(
                                tensor=self.tensor,
                            )
                        ),
                    )
                ) + (
                    self.lasso_lambda
                    # (..., Independent variable + 1, Dependent variable)
                    * jax.numpy.sign(
                        # (..., Independent variable + 1, Dependent variable)
                        self.betas
                    )
                )
                self.betas -= self.learning_rate * gradient
                yield self.betas
                assert not jax.numpy.any(jax.numpy.isnan(self.betas))
                if jax.numpy.all(
                    gradient.ravel() <= self.tolerance,
                ):
                    break
        yield self.betas


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
                # (..., In-sample observation, Independent variable)
                x,
                # (..., In-sample observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., In-sample observation, Independent variable + 1)
            x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = jax.numpy.insert(
                arr=x,
                obj=0,
                values=1.0,
                axis=-1,
            )
            # (..., Independent variable + 1, In-sample observation)
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
                        # (..., Independent variable + 1, In-sample observation)
                        transpose_x_with_y_intercept,
                        # (..., In-sample observation, Independent variable + 1)
                        x_with_y_intercept,
                    ),
                ),
                # (..., Independent variable + 1, Dependent variable)
                dot_multiplication(
                    # (..., Independent variable + 1, In-sample observation)
                    transpose_x_with_y_intercept,
                    # (..., In-sample observation, Dependent variable)
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
        return f"Ridge Regression: Lambda={self.ridge_lambda:.6f}"

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
                # (..., In-sample observation, Independent variable)
                x,
                # (..., In-sample observation, Dependent variable)
                y,
            ) = split_x_y(
                tensor=in_sample,
                number_of_dependent_variables=self.number_of_dependent_variables,
            )
            # (..., In-sample observation, Independent variable + 1)
            x_with_y_intercept: jaxlib.xla_extension.ArrayImpl = jax.numpy.insert(
                arr=x,
                obj=0,
                values=1.0,
                axis=-1,
            )
            # (..., Independent variable + 1, In-sample observation)
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
                        # (..., Independent variable + 1, In-sample observation)
                        transpose_x_with_y_intercept,
                        # (..., In-sample observation, Independent variable + 1)
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
                    # (..., Independent variable + 1, In-sample observation)
                    transpose_x_with_y_intercept,
                    # (..., In-sample observation, Dependent variable)
                    y,
                ),
            )
        yield self.betas
