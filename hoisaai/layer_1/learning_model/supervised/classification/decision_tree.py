from enum import Enum
from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.classification import (
    SupervisedClassificationModel,
)
from hoisaai.layer_1.learning_model.supervised.core.decision_tree import DecisionTree


class DecisionTreeClassifier(
    DecisionTree,
    SupervisedClassificationModel,
):
    class Impurity(Enum):
        ENTROPY = "Entropy"

    def __init__(
        self,
        depth: int,
        impurity: Impurity,
    ) -> None:
        DecisionTree.__init__(
            self,
            depth=depth,
            information_gain={
                DecisionTreeClassifier.Impurity.ENTROPY.value: self.information_gain_entropy,
            }[impurity.value],
            pre_fit=self.pre_fit,
            post_fit=self.post_fit,
        )
        SupervisedClassificationModel.__init__(self=self)
        self.impurity: str = impurity.value
        # (..., Target, Unique y, Branch): int32
        self.count: Tensor = None

    def __str__(self) -> str:
        return (
            "Decision Tree Classifier: "
            + f"depth={self.depth}, "
            + f"impurity={self.impurity}"
        )

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        return DecisionTree.fit(
            self,
            in_sample=in_sample,
            number_of_target=number_of_target,
        )

    def information_gain_entropy(
        self,
        # (..., In-sample observation, Target)
        in_sample_y: Tensor,
        # (..., In-sample observation, Target, In-sample observation - 1, InTarget)
        branches: Tensor,
        number_of_branches: int,
    ) -> Tensor:  # (..., Target, In-sample observation - 1, InTarget)
        branch = 1 * (
            # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch)
            (
                # (..., In-sample observation, Target, In-sample observation - 1, InTarget, 1 {Branch})
                (
                    # (..., In-sample observation, Target, In-sample observation - 1, InTarget)
                    branches
                ).expand_dimension(
                    # Branch
                    -1,
                )
            )
            == (
                # (Branch,)
                Tensor.arange(
                    start=0,
                    stop=number_of_branches,
                    step=1,
                    datatype=Tensor.DataType.INT32,
                )
            )
        )
        # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
        category_count: Tensor = (
            # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch, Unique y)
            1
            * (
                # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch, 1 {Unique y})
                (
                    # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch)
                    (
                        # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch)
                        branch
                    )
                    * (
                        # (..., In-sample observation, Target, 1 {In-sample observation - 1}, 1 {InTarget}, 1 {Branch})
                        (
                            # (..., In-sample observation, Target)
                            in_sample_y
                        ).expand_dimension(
                            # number_of_branch
                            -1,
                            # InTarget
                            -2,
                            # In-sample observation - 1
                            -3,
                        )
                    )
                ).expand_dimension(
                    # Unique y
                    -1,
                )
                # (Unique y,)
                == self.unique_y
            )
        ).count_nonzero(
            # In-sample observation
            axis=-6,
            keep_dimension=False,
        )
        # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
        probability: Tensor = (
            # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
            category_count
        ) / (
            # (..., Target, In-sample observation - 1, InTarget, Branch, 1 {Unique y})
            (
                # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch)
                branch
            )
            .count_nonzero(
                # In-sample observation
                axis=-5,
                keep_dimension=False,
            )
            .expand_dimension(
                # Unique y
                -1,
            )
        )
        # (..., Target, In-sample observation - 1, InTarget)
        return 1.0 + (
            # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
            (
                (
                    # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
                    probability.log2()
                    * probability
                ).nan_to_num(
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                * (
                    # (..., Target, In-sample observation - 1, InTarget, Branch, Unique y)
                    category_count
                    # In-sample observation
                    / in_sample_y.shape[-2]
                )
            )
            .sum(
                # Unique y
                axis=-1,
                keep_dimension=False,
            )
            .sum(
                # Branch
                axis=-1,
                keep_dimension=False,
            )
        )

    def predict(
        self,
        # (..., Sample observations, InTarget)
        sample_x: Tensor = None,
    ) -> Tensor:
        return SupervisedClassificationModel.predict(
            self,
            sample_x=sample_x,
        )

    def predict_with_probability(
        self,
        # (..., Sample observation, InTarget)
        sample_x: Tensor,
    ) -> SupervisedClassificationModel.Probability:
        # (..., Sample observation, Target)
        count: Tensor = (
            # (..., 1 {Sample observation}, Target, Unique y, Branch)
            (
                # (..., Target, Unique y, Branch)
                self.count
            ).expand_dimension(
                # Sample observation
                -4,
            )
        ).get_by_index(
            # (..., Sample observation, Target, 1 {Unique y})
            indexes=(
                # (..., Sample observation, Target)
                self.pre_predict(
                    sample_x=sample_x,
                )
            ).expand_dimension(
                # Unique y
                -1,
            ),
        )
        return SupervisedClassificationModel.Probability(
            # (..., Sample observation, Target, Unique y)
            probability=(
                count
                / count.sum(
                    # Unique y
                    axis=-1,
                    keep_dimension=True,
                )
            ).nan_to_num(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            # (Unique y,)
            unique_y=self.unique_y,
        )

    def pre_fit(
        self,
        # (..., In-sample observation, Target)
        in_sample_y: Tensor,
    ) -> None:
        if self.unique_y is None:
            self.unique_y: Tensor = in_sample_y.unique()

    def post_fit(
        self,
        # (..., In-sample observation, Target)
        in_sample_y: Tensor,
        # (..., In-sample observation, Target)
        branch: Tensor,
        number_of_branches: int,
    ) -> None:
        branch = (
            1
            *
            # (..., In-sample observation, Target, Branch)
            (
                # (..., In-sample observation, Target, 1 {Branch})
                (
                    # (..., In-sample observation, Target)
                    branch
                ).expand_dimension(
                    # Branch
                    -1,
                )
                # (Branch,)
                == Tensor.arange(
                    start=0,
                    stop=number_of_branches,
                    step=1,
                    datatype=Tensor.DataType.INT32,
                )
            )
        )
        # (..., Target, Unique y, Branch)
        self.count: Tensor = (
            # (..., In-sample observation, Target, Unique y, Branch)
            (
                # (..., In-sample observation, Target, 1 {Unique y}, Branch)
                (
                    # (..., In-sample observation, Target, Branch)
                    (
                        (
                            # (..., In-sample observation, Target, Branch)
                            branch
                            / branch
                        )
                        * (
                            # (..., In-sample observation, Target, 1 {Branch})
                            (
                                # (..., In-sample observation, Target)
                                in_sample_y
                            ).expand_dimension(
                                # Branch
                                -1,
                            )
                        )
                    ).expand_dimension(
                        # Unique y
                        -2,
                    )
                )
                == (
                    # (Unique y,)
                    self.unique_y.expand_dimension(
                        # Branch
                        -1
                    )
                )
                * 1
            )
        ).count_nonzero(
            # In-sample observation
            axis=-4,
            keep_dimension=False,
        )


class BaggingClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        depth: int,
        impurity: DecisionTreeClassifier.Impurity,
        number_of_subset: int,
        subset_size: int,
        seed: int = 0,
    ) -> None:
        DecisionTreeClassifier.__init__(
            self,
            depth=depth,
            impurity=impurity,
        )
        self.number_of_subset: int = number_of_subset
        self.subset_size: int = subset_size
        self.seed: int = seed

    def __str__(self) -> str:
        return (
            "Bagging Classifier: "
            + f"depth={self.depth}, "
            + f"impurity={self.impurity}, "
            + f"number_of_subset={self.number_of_subset}, "
            + f"subset_size={self.subset_size}, "
            + f"seed={self.seed}"
        )

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        return DecisionTreeClassifier.fit(
            self,
            in_sample=DecisionTree.bagging_preparation(
                in_sample=in_sample,
                number_of_subset=self.number_of_subset,
                subset_size=self.subset_size,
                seed=self.seed,
            ),
            number_of_target=number_of_target,
        )

    def predict_with_probability(
        self,
        # (..., Sample observation, InTarget)
        sample_x: Tensor,
    ) -> SupervisedClassificationModel.Probability:
        probability: SupervisedClassificationModel.Probability = (
            super().predict_with_probability(sample_x=sample_x)
        )
        return SupervisedClassificationModel.Probability(
            # (..., In-sample observation, Target, Unique y)
            probability=(
                # (Aggregation, ..., In-sample observation, Target, Unique y)
                probability.probability
            ).nanmean(
                # Aggregation
                axis=0,
                keep_dimension=False,
            ),
            unique_y=probability.unique_y,
        )


class RandomForestClassifier(DecisionTreeClassifier):
    def __init__(
        self,
        depth: int,
        impurity: DecisionTreeClassifier.Impurity,
        number_of_chosen_feature: int,
        number_of_subset: int,
        subset_size: int,
        seed: int = 0,
    ) -> None:
        DecisionTreeClassifier.__init__(
            self,
            depth=depth,
            impurity=impurity,
        )
        self.number_of_chosen_feature: int = (
            number_of_chosen_feature
        )
        self.number_of_subset: int = number_of_subset
        self.subset_size: int = subset_size
        self.seed: int = seed

    def __str__(self) -> str:
        return (
            "Random Forest Classifier: "
            + f"depth={self.depth}, "
            + f"impurity={self.impurity}, "
            + f"number_of_chosen_feature={self.number_of_chosen_feature}, "
            + f"number_of_subset={self.number_of_subset}, "
            + f"subset_size={self.subset_size}, "
            + f"seed={self.seed}"
        )

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        return DecisionTreeClassifier.fit(
            self,
            in_sample=DecisionTree.random_forest_preparation(
                in_sample=in_sample,
                number_of_target=number_of_target,
                number_of_chosen_feature=self.number_of_chosen_feature,
                number_of_subset=self.number_of_subset,
                subset_size=self.subset_size,
                seed=self.seed,
            ),
            number_of_target=number_of_target,
        )

    def predict_with_probability(
        self,
        sample_x: Tensor,
    ) -> SupervisedClassificationModel.Probability:
        probability: SupervisedClassificationModel.Probability = (
            super().predict_with_probability(sample_x=sample_x)
        )
        return SupervisedClassificationModel.Probability(
            # (..., In-sample observation, Target, Unique y)
            probability=(
                # (Aggregation, ..., In-sample observation, Target, Unique y)
                probability.probability
            ).nanmean(
                # Aggregation
                axis=0,
                keep_dimension=False,
            ),
            unique_y=probability.unique_y,
        )
