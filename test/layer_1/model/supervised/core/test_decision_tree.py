"""Test DecisionTree."""

import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.core.decision_tree import DecisionTree


def infromation_gain(
    # (..., In-sample observation, Dependent variable)
    # pylint: disable=unused-argument
    in_sample_y: Tensor,
    # (..., In-sample observation, Dependent variable, In-sample observation - 1, Independent variable)
    branches: Tensor,
    number_of_branches: int,
) -> (
    Tensor
):  # (..., Dependent variable, In-sample observation - 1, Independent variable)
    """Return the information gain of the branches."""
    return Tensor.full(
        shape=(
            *branches.shape[:-4],
            *branches.shape[-3:],
        ),
        value=1.0,
        datatype=Tensor.DataType.FLOAT32,
    )


def pre_fit(
    # (..., In-sample observation, Dependent variable)
    # pylint: disable=unused-argument
    in_sample_y: Tensor,
) -> None:
    """Pre-fit before the model."""
    return None


def post_fit(
    # (..., In-sample observation, Dependent variable)
    # pylint: disable=unused-argument
    in_sample_y: Tensor,
    #  (..., In-sample observation, Dependent variable)
    # pylint: disable=unused-argument
    branch,
    number_of_branches: int,
) -> None:
    """Post-fit after the model."""
    return None


class TestDecisionTree(unittest.TestCase):
    """Test DecisionTree."""

    def test_decision_tree_init(self):
        """Test DecisionTree.__init__."""
        dt: DecisionTree = DecisionTree(
            depth=3,
            information_gain=infromation_gain,
            pre_fit=pre_fit,
            post_fit=post_fit,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            dt.pre_predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).to_list(),
            [[7], [0], [0]],
        )

    def test_bagging_preparation(self):
        """Test bagging_preparation."""
        self.assertListEqual(
            DecisionTree.bagging_preparation(
                in_sample=Tensor.array(
                    x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
                ),
                number_of_subset=3,
                subset_size=2,
                seed=0,
            ).to_list(),
            [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [7, 8, 9]], [[1, 2, 3], [7, 8, 9]]],
        )

    def test_random_forest_preparation(self):
        """Test random_forest_preparation."""
        self.assertListEqual(
            DecisionTree.random_forest_preparation(
                in_sample=Tensor.array(
                    x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
                ),
                number_of_target=1,
                number_of_chosen_feature=1,
                number_of_subset=4,
                subset_size=2,
                seed=0,
            ).to_list(),
            [[[7, 8], [1, 2]], [[7, 9], [7, 9]], [[4, 6], [4, 5]], [[7, 8], [7, 8]]],
        )
