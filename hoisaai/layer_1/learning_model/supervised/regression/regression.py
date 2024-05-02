from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel


class Regression(SupervisedLearningModel):
    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        raise NotImplementedError()

    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:
        raise NotImplementedError()
