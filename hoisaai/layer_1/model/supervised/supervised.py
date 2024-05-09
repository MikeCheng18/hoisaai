"""Supervised learning model."""

import abc
import itertools
import typing
from hoisaai.layer_0.tensor import Tensor


class SupervisedLearningModel(object):
    """Supervised learning model.

    :param number_of_target: Number of target.
    :type number_of_target: int
    """

    def __init__(
        self,
    ) -> None:
        self.number_of_target: int = None

    @abc.abstractmethod
    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        """Fit the model.

        :param in_sample: In-sample observations (..., In-sample observations, Target and Feature).
        :type in_sample: Tensor
        :param number_of_target: Number of target.
        :type number_of_target: int
        """
        raise NotImplementedError

    @staticmethod
    def permuation(
        hyperparameter_space: typing.Dict[str, typing.Any | typing.List[typing.Any]],
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Permute the hyperparameter space.

        :param hyperparameter_space: Hyperparameter space.
        :type hyperparameter_space: Dict[str, Any | List[Any]]

        :return: Hyperparameter space.
        :rtype: List[Dict[str, typing.Any]]
        """
        constants: typing.Dict[str, typing.Any] = {
            key: value
            for key, value in hyperparameter_space.items()
            if not isinstance(value, list)
        }
        if not any(isinstance(value, list) for value in hyperparameter_space.values()):
            return [constants]
        keys, values = zip(
            *{
                key: value
                for key, value in hyperparameter_space.items()
                if isinstance(value, list)
            }.items()
        )
        return [
            {
                **constants,
                **dict(zip(keys, v)),
            }
            for v in itertools.product(*values)
        ]

    @abc.abstractmethod
    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:
        """Predict the target.

        :param sample_x: Sample observations (..., Sample observations, Feature).
        :type sample_x: Tensor

        :return: In-sample observations (..., Sample observations, Target).
        :rtype: Tensor
        """
        raise NotImplementedError
