"""
This module contains the `SupervisedLearningModel` class,
which is a base class for supervised learning models.
"""

import jaxlib.xla_extension

from hoisaai.layer_1.model import Stateful, Tensor


class SupervisedLearningModel(Stateful):
    """
    A class representing a supervised learning model.

    Attributes:
        tensor (Tensor): The input tensor for the model.
        number_of_dependent_variables (int): The number of dependent variables in the model.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        predict(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Predicts the output for the given input tensor.

        shapley_value(tensor: Tensor) -> jaxlib.xla_extension.ArrayImpl:
            Calculates the Shapley value for the given input tensor.
    """

    def __init__(
        self,
        tensor: Tensor = None,
        number_of_dependent_variables: int = None,
    ) -> None:
        super().__init__()
        self.tensor: Tensor = tensor
        self.number_of_dependent_variables: int = number_of_dependent_variables

    def predict(
        self,
        tensor: Tensor,
    ) -> jaxlib.xla_extension.ArrayImpl:
        """
        Predicts the output for the given input tensor.

        Args:
            tensor (Tensor): The input tensor for prediction.

        Returns:
            jaxlib.xla_extension.ArrayImpl: The predicted output tensor.
        """
        raise NotImplementedError()

    def shapley_value(
        self,
        tensor: Tensor,
    ) -> jaxlib.xla_extension.ArrayImpl:
        """
        Calculates the Shapley value for a given tensor.

        Args:
            tensor (Tensor): The input tensor for which the Shapley value needs to be calculated.

        Returns:
            jaxlib.xla_extension.ArrayImpl: The calculated Shapley value.

        Raises:
            NotImplementedError:
                This method is not implemented and needs to be overridden in a subclass.
        """
        raise NotImplementedError()
