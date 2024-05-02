import dataclasses

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel


class SupervisedClassificationModel(SupervisedLearningModel):
    @dataclasses.dataclass
    class Probability:
        # # (..., Sample observation, Target, unique_y)
        probability: Tensor
        # (Unique y)
        unique_y: Tensor

    def __init__(self) -> None:
        SupervisedLearningModel.__init__(self)
        # (Unique y)
        self.unique_y: Tensor = None

    def predict(
        self,
        # (..., Sample observation, Feature)
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

    def predict_with_probability(
        self,
        # (..., Sample observation, Feature)
        sample_x: Tensor,
    ) -> Probability:
        raise NotImplementedError()
