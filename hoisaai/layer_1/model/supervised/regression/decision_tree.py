"""
This module contains the DecisionTreeRegressor,
BaggingRegressor,
and RandomForestRegressor classes.
"""

import enum
import typing

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.core.decision_tree import DecisionTree
from hoisaai.layer_1.model.supervised.regression.regression import Regression


class DecisionTreeRegressor(
    DecisionTree,
    Regression,
):
    """Decision Tree Regressor.

    :param depth: Depth of the decision tree.
    :type depth: int
    :param criterion: Criterion of the decision tree.
    :type criterion: DecisionTreeRegressor.Criterion
    """

    class Criterion(enum.Enum):
        """Criterion of the decision tree regressor."""

        SQUARED_ERROR = "Squared Error"

    def __init__(
        self,
        depth: int,
        criterion: Criterion,
    ) -> None:
        DecisionTree.__init__(
            self,
            depth=depth,
            information_gain=DecisionTreeRegressor.CRITERION[criterion.value],
            pre_fit=self.pre_fit,
            post_fit=self.post_fit,
        )
        Regression.__init__(
            self,
        )
        self.criterion: str = criterion.value
        # (..., Target, Branch): float32
        self.average: Tensor = None

    def __str__(self) -> str:
        return (
            "Decision Tree Regressor: "
            + f"depth={self.depth}, "
            + f"criterion={self.criterion}"
        )

    @staticmethod
    def information_gain_squared_error(
        in_sample_y: Tensor,
        branches: Tensor,
        number_of_branches: int,
    ) -> Tensor:
        """Return the information gain of the branches.

        :param in_sample_y: In-sample y (..., In-sample observation, Target).
        :type in_sample_y: Tensor
        :param branches: Branches
            (..., In-sample observation, Target, In-sample observation - 1, InTarget).
        :type branches: Tensor
        :param number_of_branches: Number of branches.
        :type number_of_branches: int

        :return: Information gain of the branches
            (..., Target, In-sample observation - 1, InTarget).
        :rtype: Tensor"""
        # (..., In-sample observation, Target, In-sample observation - 1, InTarget, Branch)
        criterion: Tensor = 1 * (
            branches.expand_dimension(-1)
            == Tensor.arange(
                start=0,
                stop=number_of_branches,
                step=1,
                datatype=Tensor.DataType.INT32,
            )
        )
        criterion = criterion / criterion
        criterion = criterion * in_sample_y.expand_dimension(-1, -2, -3)
        criterion = criterion - (
            criterion.nansum(axis=-5, keep_dimension=True)
            / (~criterion.isnan()).sum(axis=-5, keep_dimension=True)
        )
        criterion = criterion.square()
        criterion = criterion.nansum(axis=-1, keep_dimension=False)
        criterion = criterion.nansum(axis=-4, keep_dimension=False) / (
            ~criterion.isnan()
        ).sum(axis=-4, keep_dimension=False)
        return 1.0 - criterion

    CRITERION: typing.Dict[str, typing.Callable] = {
        Criterion.SQUARED_ERROR.value: information_gain_squared_error,
    }

    def predict(
        self,
        # (..., Sample observation, InTarget)
        sample_x: Tensor,
    ) -> Tensor:
        # (..., Sample observation, Target)
        return (
            # (..., 1 {Sample observation}, Target, Branch)
            (
                # (..., Target, Branch)
                self.average
            ).expand_dimension(
                # Sample observation
                -3,
            )
        ).get_by_index(
            # (..., Sample observation, Target)
            indexes=self.pre_predict(
                sample_x=sample_x,
            )
        )

    def pre_fit(
        self,
        # (..., In-sample observation, Target)
        # pylint: disable=W0613
        in_sample_y: Tensor,
    ) -> None:
        """Pre-fit before the model.

        :param in_sample_y: In-sample y
            (..., In-sample observation, Target).
        :type in_sample_y: Tensor
        """
        return None

    def post_fit(
        self,
        # (..., In-sample observation, Target)
        in_sample_y: Tensor,
        # (..., In-sample observation, Target)
        branch: Tensor,
        number_of_branches: int,
    ) -> None:
        """Post-fit after the model.

        :param in_sample_y: In-sample y
            (..., In-sample observation, Target).
        :type in_sample_y: Tensor
        :param branch: Branch
            (..., In-sample observation, Target).
        :type branch: Tensor
        :param number_of_branches: Number of branches.
        :type number_of_branches: int
        """
        # (..., In-sample observation, Target, Branch)
        self.average: Tensor = 1 * (
            branch.expand_dimension(-1)
            == Tensor.arange(
                start=0,
                stop=number_of_branches,
                step=1,
                datatype=Tensor.DataType.INT32,
            )
        )
        self.average = self.average / self.average
        self.average = self.average * in_sample_y.expand_dimension(-1)
        # (..., Target, Branch)
        self.average: Tensor = self.average.nansum(axis=-3, keep_dimension=False) / (
            (~self.average.isnan()).sum(axis=-3, keep_dimension=False)
        )


class BaggingRegressor(DecisionTreeRegressor):
    """Bagging Regressor.

    :param depth: Depth of the decision tree.
    :type depth: int
    :param criterion: Criterion of the decision tree.
    :type criterion: DecisionTreeRegressor.Criterion
    :param number_of_subset: Number of subset.
    :type number_of_subset: int
    :param subset_size: Subset size.
    :type subset_size: int
    :param seed: Seed.
    :type seed: int
    """

    def __init__(
        self,
        depth: int,
        criterion: DecisionTreeRegressor.Criterion,
        number_of_subset: int,
        subset_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__(
            depth=depth,
            criterion=criterion,
        )
        self.number_of_subset: int = number_of_subset
        self.subset_size: int = subset_size
        self.seed: int = seed

    def __str__(self) -> str:
        return (
            "Bagging Regressor: "
            + f"depth={self.depth}, "
            + f"criterion={self.criterion}, "
            + f"number_of_subset={self.number_of_subset}, "
            + f"subset_size={self.subset_size}, "
            + f"seed={self.seed}"
        )

    def fit(
        self,
        in_sample: Tensor,
        number_of_target: int,
    ) -> None:
        return DecisionTreeRegressor.fit(
            self,
            in_sample=DecisionTree.bagging_preparation(
                in_sample=in_sample,
                number_of_subset=self.number_of_subset,
                subset_size=self.subset_size,
                seed=self.seed,
            ),
            number_of_target=number_of_target,
        )

    def predict(
        self,
        # (..., Sample observation, InTarget)
        sample_x: Tensor,
    ) -> Tensor:
        # (..., Sample observation, Target)
        return (
            # (Aggregation, ..., Sample observation, Target)
            DecisionTreeRegressor.predict(
                self,
                sample_x=sample_x,
            )
        ).nanmean(
            # Aggregation
            axis=0,
            keep_dimension=False,
        )


class RandomForestRegressor(DecisionTreeRegressor):
    """Random Forest Regressor.

    :param depth: Depth of the decision tree.
    :type depth: int
    :param criterion: Criterion of the decision tree.
    :type criterion: DecisionTreeRegressor.Criterion
    :param number_of_chosen_feature: Number of chosen feature.
    :type number_of_chosen_feature: int
    :param number_of_subset: Number of subset.
    :type number_of_subset: int
    :param subset_size: Subset size.
    :type subset_size: int
    :param seed: Seed.
    :type seed: int
    """

    def __init__(
        self,
        depth: int,
        criterion: DecisionTreeRegressor.Criterion,
        number_of_chosen_feature: int,
        number_of_subset: int,
        subset_size: int,
        seed: int = 0,
    ) -> None:
        DecisionTreeRegressor.__init__(
            self,
            depth=depth,
            criterion=criterion,
        )
        self.number_of_chosen_feature: int = number_of_chosen_feature
        self.number_of_subset: int = number_of_subset
        self.subset_size: int = subset_size
        self.seed: int = seed

    def __str__(self) -> str:
        return (
            "Random Forest Regressor: "
            + f"depth={self.depth}, "
            + f"criterion={self.criterion}, "
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
        return DecisionTreeRegressor.fit(
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

    def predict(self, sample_x: Tensor) -> Tensor:
        # (..., Sample observation, Target)
        return (
            # (Aggregation, ..., Sample observation, Target)
            DecisionTreeRegressor.predict(
                self,
                sample_x=sample_x,
            )
        ).nanmean(
            # Number of subset
            axis=0,
            keep_dimension=False,
        )
