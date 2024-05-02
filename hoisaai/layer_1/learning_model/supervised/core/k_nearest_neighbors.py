from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel


class KNearestNeighbors(SupervisedLearningModel):

    def __init__(
        self,
        k: int,
    ) -> None:
        SupervisedLearningModel.__init__(self)
        self.k: int = k
        self.in_sample: Tensor = None

    def fit(
        self,
        # (..., Observations, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        self.in_sample: Tensor = in_sample
        self.number_of_target: int = number_of_target

    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:
        (
            # (..., In-sample observation, Feature)
            in_sample_x,
            # (..., In-sample observation, Target)
            in_sample_y,
        ) = self.in_sample.get_sample_x_and_y(
            number_of_target=self.number_of_target,
        )
        # (..., Sample observation, Target, k)
        return (
            # (..., 1 {Sample observation}, Target, 1 {k}, In-sample observation)
            (
                # (..., Target, In-sample observation)
                (
                    # (..., In-sample observation, Target)
                    in_sample_y
                ).swapaxes(
                    axis1=-1,
                    axis2=-2,
                )
            ).expand_dimension(
                # Sample observation
                -4,
                # k
                -2,
            )
        ).get_by_index(
            # (..., Sample observation, 1 {Target}, k)
            indexes=(
                # (..., Sample observation, k)
                (
                    # (..., Sample observation, In-sample observation)
                    (
                        # (..., Sample observation, In-sample observation, Feature)
                        (
                            (
                                # (..., Sample observation, 1 {In-sample observation}, Feature)
                                (
                                    # (..., Sample observation, Feature)
                                    sample_x
                                ).expand_dimension(
                                    # In-sample observation
                                    -2,
                                )
                            )
                            - (
                                # (..., 1 {Sample observation}, In-sample observation, Feature)
                                (
                                    # (..., In-sample observation, Feature)
                                    in_sample_x
                                ).expand_dimension(
                                    # Sample observation
                                    -3,
                                )
                            )
                        )
                    )
                    .square()
                    # (..., Sample observation, In-sample observation)
                    .sum(
                        # Feature
                        axis=-1,
                        keep_dimension=False,
                    )
                    .sqrt()
                    .argsort(
                        # In-sample observation
                        axis=-1,
                    )
                )[
                    ...,
                    : self.k,
                ]
            ).expand_dimension(
                # Target
                -2,
            )
        )
