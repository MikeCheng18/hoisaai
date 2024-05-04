import typing

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.supervised import SupervisedLearningModel


class DecisionTree(SupervisedLearningModel):
    def __init__(
        self,
        depth: int,
        information_gain: typing.Callable[
            [
                # in_sample_y: (..., In-sample observation, Target)
                Tensor,
                # branches: (..., In-sample observation, Target, In-sample observation - 1, Feature)
                Tensor,
                # number_of_branches
                int,
            ],
            # (..., Target, In-sample observation - 1, Feature)
            Tensor,
        ],
        pre_fit: typing.Callable[
            [
                # in_sample_y: (..., In-sample observation, Target)
                Tensor,
            ],
            None,
        ],
        post_fit: typing.Callable[
            [
                # in_sample_y: (..., In-sample observation, Target)
                Tensor,
                # branch: (..., In-sample observation, Target)
                Tensor,
                # number_of_branches
                int,
            ],
            None,
        ],
    ) -> None:
        SupervisedLearningModel.__init__(self)
        self.depth: int = depth
        self.information_gain: typing.Callable = information_gain
        self.pre_fit_function: typing.Callable = pre_fit
        self.post_fit_function: typing.Callable = post_fit
        # (..., Target, depth): int16
        self.feature_index: Tensor = None
        # (..., Target, depth): float32
        self.threshold: Tensor = None

    @staticmethod
    def bagging_preparation(
        # (..., In-sample observation, Target and Feature)
        in_sample: Tensor,
        number_of_subset: int,
        subset_size: int,
        seed: int,
    ) -> Tensor:
        # (Aggregation, ..., In-sample observation subset, Target and Feature)
        return (
            # (1 {Aggregation}, ..., 1 {In-sample observation subset}, Target and Feature, In-sample observation)
            (
                # (..., Target and Feature, In-sample observation)
                (
                    # (..., In-sample observation, Target and Feature)
                    in_sample.swapaxes(
                        # Target and Feature
                        axis1=-1,
                        # In-sample observation
                        axis2=-2,
                    )
                ).expand_dimension(
                    # Aggregation
                    0,
                    # In-sample observation subset
                    -3,
                )
            )
        ).get_by_index(
            # (Aggregation, ..., In-sample observation subset, 1 {Target and Feature})
            indexes=(
                # (Aggregation, ..., In-sample observation subset)
                Tensor.random_integer(
                    shape=(
                        # Aggregation
                        number_of_subset,
                        # ...
                        *in_sample.shape[:-2],
                        # In-sample observation subset
                        subset_size,
                    ),
                    minimum_value=0,
                    maximum_value=in_sample.shape[-2],  # In-sample observation
                    datatype=Tensor.DataType.INT32,
                    require_gradient=False,
                    seed=seed,
                )
            ).expand_dimension(
                # Target and Feature
                -1
            ),
        )

    def fit(
        self,
        # (..., In-sample observations, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        self.number_of_target = number_of_target
        (
            # (..., In-sample observation, Feature)
            in_sample_x,
            # (..., In-sample observation, Target)
            in_sample_y,
        ) = in_sample.get_sample_x_and_y(
            number_of_target=number_of_target,
        )
        del in_sample
        self.pre_fit_function(
            # (..., In-sample observation, Target)
            in_sample_y=in_sample_y,
        )
        # (..., In-sample observation, Feature)
        sorted_x: Tensor = in_sample_x.sort(
            # In-sample observation
            axis=-2,
        )
        # (..., In-sample observation - 1, Feature)
        threshold: Tensor = (
            sorted_x[
                ...,
                :-1,
                :,
            ]
            + sorted_x[
                ...,
                1:,
                :,
            ]
        ) / 2.0
        # (..., In-sample observation, In-sample observation - 1, Feature)
        condition: Tensor = (
            1
            *
            # (..., In-sample observation, In-sample observation - 1, Feature)
            (
                (
                    # (..., In-sample observation, 1 {In-sample observation - 1}, Feature)
                    (
                        # (..., In-sample observation, Feature)
                        in_sample_x
                    ).expand_dimension(
                        # In-sample observation - 1
                        -2,
                    )
                )
                < (
                    # (..., 1 {In-sample observation}, In-sample observation - 1, Feature)
                    (
                        # (..., In-sample observation - 1, Feature)
                        threshold
                    ).expand_dimension(
                        # In-sample observation
                        -3,
                    )
                )
            )
        )
        # (..., In-sample observation, 1 {Target}, In-sample observation - 1, Feature)
        condition: Tensor = condition.expand_dimension(
            # Target
            -3,
        )
        # (..., 1 {Target}, (In-sample observation - 1) * Feature)
        threshold: Tensor = threshold.reshape(
            # ...
            *threshold.shape[:-2],
            1,
            # (In-sample observation - 1) * Feature
            threshold.shape[-2] * threshold.shape[-1],
        )
        # (..., Target, depth)
        self.feature_index: Tensor = Tensor.full(
            shape=(
                # ...
                *(
                    # (..., In-sample observation, Target)
                    in_sample_y
                ).shape[:-2],
                # Target
                in_sample_y.shape[-1],
                self.depth,
            ),
            value=Tensor.Value.NAN,
            datatype=Tensor.DataType.INT32,
        )
        # (..., Target, depth)
        self.threshold: Tensor = Tensor.full(
            shape=self.feature_index.shape,
            value=Tensor.Value.NAN,
            datatype=Tensor.DataType.FLOAT32,
        )
        # (..., In-sample observation, Target, In-sample observation - 1, Feature): int16
        branches: Tensor = Tensor.full(
            shape=(
                # (..., In-sample observation, 1, In-sample observation - 1, Feature)
                *condition.shape[:-3],
                # Target
                number_of_target,
                # In-sample observation - 1, Feature
                *condition.shape[-2:],
            ),
            value=0,
            datatype=Tensor.DataType.INT32,
        )
        # Sequencial iterative process, cannot be parallelized
        for depth_index in range(self.depth):
            number_of_branches: int = 2 ** (depth_index + 1)
            # (..., In-sample observation, Target, In-sample observation - 1, Feature)
            branches: Tensor = (
                # (..., In-sample observation, Target, In-sample observation - 1, Feature)
                2
                * (
                    # (..., In-sample observation, Target, In-sample observation - 1, Feature)
                    branches
                )
                + (
                    # (..., In-sample observation, 1, In-sample observation - 1, Feature)
                    condition
                )
            )
            # (..., Target, In-sample observation - 1, Feature)
            information_gain: Tensor = self.information_gain(
                # (..., In-sample observation, Target)
                in_sample_y=in_sample_y,
                # (..., In-sample observation, Target, In-sample observation - 1, Feature)
                branches=branches,
                number_of_branches=number_of_branches,
            )
            # (..., Target)
            argmax: Tensor = (
                (
                    # (..., Target, In-sample observation - 1, Feature)
                    information_gain
                )
                .reshape(
                    # ..., Target
                    *information_gain.shape[:-2],
                    # (In-sample observation - 1) * Feature
                    information_gain.shape[-2] * information_gain.shape[-1],
                )
                .argmax(
                    # (In-sample observation - 1) * Feature
                    axis=-1,
                )
            )
            # (..., Target, depth)
            self.feature_index[
                ...,
                depth_index,
            ] = (
                # (..., Target)
                argmax
                # Feature
                % information_gain.shape[-2]
            )
            # (..., Target, depth)
            self.threshold[
                ...,
                depth_index,
            ] = (
                # (..., 1 {Target}, (In-sample observation - 1) * Feature)
                threshold
            ).get_by_index(
                # (..., Target)
                indexes=argmax,
            )
            # (..., In-sample observation, Target, 1 {In-sample observation - 1}, 1 {Feature})
            branches: Tensor = (
                # (..., In-sample observation, Target)
                (
                    # (..., In-sample observation, Target, (In-sample observation - 1) * Feature)
                    (
                        # (..., In-sample observation, Target, In-sample observation - 1, Feature)
                        branches
                    ).reshape(
                        # ..., In-sample observation, Target
                        *branches.shape[:-2],
                        # (In-sample observation - 1) * Feature
                        branches.shape[-2] * branches.shape[-1],
                    )
                ).get_by_index(
                    indexes=(
                        # (..., 1 {In-sample observation}, Target)
                        argmax
                    ).expand_dimension(
                        # In-sample observation
                        -2,
                    ),
                )
            ).expand_dimension(
                # Feature
                -1,
                # In-sample observation - 1
                -2,
            )
            continue
        # (..., In-sample observation, Target)
        branch: Tensor = (
            # (..., In-sample observation, Target, 1 {In-sample observation - 1}, 1 {Feature})
            branches
        ).reshape(
            *branches.shape[:-2],
        )
        self.post_fit_function(
            # (..., In-sample observation, Target)
            in_sample_y=in_sample_y,
            # (..., In-sample observation, Target)
            branch=branch,
            number_of_branches=number_of_branches,
        )

    def predict(
        self,
        sample_x: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    def pre_predict(
        self,
        # (..., Sample observations, Feature)
        sample_x: Tensor = None,
    ) -> Tensor:  # (..., Sample observation, Target)
        # (..., Sample observation, Target)
        branches: Tensor = Tensor.full(
            shape=(
                # ..., Sample observation
                *sample_x.shape[:-1],
                # Target
                self.number_of_target,
            ),
            value=0,
            datatype=sample_x.datatype,
        )
        # Sequencial iterative process, cannot be parallelized
        for depth_index in range(self.depth):
            # (..., Sample observation, Target)
            branches: Tensor = (
                # (..., Sample observation, Target)
                2
                * (
                    # (..., Sample observation, Target)
                    branches
                )
            ) + (
                1
                # (..., Sample observation, Target)
                * (
                    # (..., Sample observation, Target)
                    (
                        # (..., Sample observation, 1 {Target}, Feature)
                        (
                            # (..., Sample observation, Feature)
                            sample_x
                        ).expand_dimension(
                            # Target
                            -2,
                        )
                    ).get_by_index(
                        # (..., 1 {Sample observation}, Target)
                        indexes=(
                            # (..., Target)
                            (
                                # (..., Target, depth)
                                self.feature_index
                            )[
                                ...,
                                depth_index,
                            ]
                        ).expand_dimension(
                            # Sample observation
                            -2,
                        ),
                    )
                    < (
                        # (..., 1 {Sample observation}, Target)
                        (
                            # (..., Target)
                            (
                                # (..., Target, depth)
                                self.threshold
                            )[
                                ...,
                                depth_index,
                            ]
                        ).expand_dimension(
                            # Sample observation
                            -2,
                        )
                    )
                )
            )
        # (..., Sample observation, Target)
        return branches

    @staticmethod
    def random_forest_preparation(
        # (..., In-sample observation, Target and Feature)
        in_sample: Tensor,
        number_of_target: int,
        number_of_chosen_feature: int,
        number_of_subset: int,
        subset_size: int,
        seed: int,
    ) -> Tensor:
        # (Aggregation, ..., In-sample observation subset, Target and Feature)
        sample: Tensor = DecisionTree.bagging_preparation(
            in_sample=in_sample,
            number_of_subset=number_of_subset,
            subset_size=subset_size,
            seed=seed,
        )
        # (Aggregation, ..., In-sample observation subset, Target and Feature subset)
        return (
            # (Aggregation, ..., In-sample observation subset, 1 {Target and Feature subset}, Target and Feature)
            (
                # (Aggregation, ..., In-sample observation subset, Target and Feature)
                sample
            ).expand_dimension(
                # Target and Feature subset
                -2
            )
        ).get_by_index(
            indexes=Tensor.concatenate(
                tensors=[
                    (
                        # (Aggregation, ..., In-sample observation subset, Target)
                        Tensor.arange(
                            start=0,
                            stop=number_of_target,
                            step=1,
                            datatype=Tensor.DataType.INT32,
                        )
                        * Tensor.full(
                            shape=(
                                # Aggregation, ..., In-sample observation subset
                                *sample.shape[:-1],
                                number_of_target,
                            ),
                            value=1,
                            datatype=Tensor.DataType.INT32,
                        )
                    ),
                    # (Aggregation, ..., In-sample observation subset, Feature subset)
                    Tensor.random_integer(
                        shape=(
                            # Aggregation, ..., In-sample observation subset
                            *sample.shape[:-1],
                            number_of_chosen_feature,
                        ),
                        minimum_value=number_of_target,
                        # Target and Feature
                        maximum_value=in_sample.shape[-1],
                        datatype=Tensor.DataType.INT16,
                        require_gradient=False,
                        seed=seed,
                    ),
                ],
                # Target and Feature subset
                axis=-1,
            )
        )
