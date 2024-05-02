from hoisaai.layer_0.tensor import Tensor


class SupervisedLearningModel(object):
    def __init__(
        self,
    ) -> None:
        self.number_of_target: int = None

    def fit(
        self,
        # (..., In-sample observations, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        raise NotImplementedError()

    def predict(
        self,
        # (..., Sample observations, Feature)
        sample_x: Tensor,
    ) -> Tensor:  # (..., In-sample observations, Target)
        raise NotImplementedError()
