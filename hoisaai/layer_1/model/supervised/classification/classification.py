"""This module contains the `SupervisedClassificationModel` class."""

import abc
import dataclasses

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.supervised import SupervisedLearningModel


class SupervisedClassificationModel(SupervisedLearningModel):
    """Supervised classification model."""

    @dataclasses.dataclass
    class Probability:
        """Probability.

        :param probability: Probability (..., Sample observation, Target, Unique y).
        :type probability: Tensor
        :param unique_y: Unique y (Unique y, ).
        :type unique_y: Tensor
        """

        probability: Tensor
        unique_y: Tensor

    def __init__(self) -> None:
        SupervisedLearningModel.__init__(self)
        # (Unique y)
        self.unique_y: Tensor = None

    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:
        probability: SupervisedClassificationModel.Probability = (
            self.predict_with_probability(
                sample_x=sample_x,
            )
        )
        return (
            # (Unique y)
            probability.unique_y
        ).get_by_index(
            # (..., Sample observation, Target)
            indexes=(
                # (..., Sample observation, Target, unique_y)
                probability.probability
            ).argmax(
                # unique_y
                axis=-1,
            ),
        )

    @abc.abstractmethod
    def predict_with_probability(
        self,
        sample_x: Tensor,
    ) -> Probability:
        """Predict with probability.

        :param sample_x: Sample x (..., Sample observation, Feature).
        :type sample_x: Tensor

        :return: Probability.
        :rtype: Probability
        """
        raise NotImplementedError
