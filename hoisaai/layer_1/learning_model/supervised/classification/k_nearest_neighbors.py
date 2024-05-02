from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.classification import (
    SupervisedClassificationModel,
)
from hoisaai.layer_1.learning_model.supervised.core.k_nearest_neighbors import (
    KNearestNeighbors,
)


class KNearestNeighborsClassifier(
    KNearestNeighbors,
    SupervisedClassificationModel,
):
    def __init__(
        self,
        k: int,
    ) -> None:
        KNearestNeighbors.__init__(
            self,
            k=k,
        )
        SupervisedClassificationModel.__init__(self)

    def __str__(self) -> str:
        return "K Nearest Neighbors Classifier: " + f"k={self.k}"

    def fit(
        self,
        # (..., In-sample observations, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        KNearestNeighbors.fit(
            self,
            in_sample=in_sample,
            number_of_target=number_of_target,
        )
        self.unique_y: Tensor = in_sample.get_sample_x_and_y(
            number_of_target=number_of_target,
        )[1].unique()

    def predict(
        self,
        # (..., Sample observation, Feature)
        sample_x: Tensor,
    ) -> Tensor:
        return SupervisedClassificationModel.predict(
            self,
            sample_x=sample_x,
        )

    def predict_with_probability(
        self,
        sample_x: Tensor,
    ) -> SupervisedClassificationModel.Probability:
        return SupervisedClassificationModel.Probability(
            # (..., Sample observation, Target, unique_y)
            probability=(
                (
                    # (..., Sample observation, Target, unique_y)
                    (
                        1
                        # (..., Sample observation, Target, k, Unique y)
                        * (
                            (
                                # (..., Sample observation, Target, k, 1 {Unique y})
                                (
                                    # (..., Sample observation, Target, k)
                                    KNearestNeighbors.predict(
                                        self,
                                        sample_x=sample_x,
                                    )
                                ).expand_dimension(
                                    # Unique y
                                    -1,
                                )
                            )
                            == (
                                # (Unique y,)
                                self.unique_y
                            )
                        )
                    ).count_nonzero(
                        # k
                        axis=-2,
                        keep_dimension=False,
                    )
                )
                / self.k
            ),
            # (unique_y,)
            unique_y=self.unique_y,
        )
