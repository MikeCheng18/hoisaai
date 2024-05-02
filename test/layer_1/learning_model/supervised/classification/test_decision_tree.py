import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.decision_tree import (
    BaggingClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
)


class TestDecisionTreeClassifier(unittest.TestCase):
    def test_decision_tree_init(self):
        dt: DecisionTreeClassifier = DecisionTreeClassifier(
            depth=3,
            impurity=DecisionTreeClassifier.Impurity.ENTROPY,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            dt.predict_with_probability(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).probability.tolist(),
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
        )
        self.assertListEqual(
            dt.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).tolist(),
            [[1], [4], [7]],
        )

    def test_bagging_preparation(self):
        dt: BaggingClassifier = BaggingClassifier(
            depth=3,
            impurity=DecisionTreeClassifier.Impurity.ENTROPY,
            number_of_subset=128,
            subset_size=2,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            dt.predict_with_probability(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).probability.tolist(),
            [
                [[0.546875, 0.2421875, 0.0]],
                [[0.125, 0.59375, 0.1640625]],
                [[0.125, 0.3515625, 0.5234375]],
            ],
        )
        self.assertListEqual(
            dt.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).tolist(),
            [[1], [4], [7]],
        )

    def test_random_forest_preparation(self):
        dt: RandomForestClassifier = RandomForestClassifier(
            depth=3,
            impurity=DecisionTreeClassifier.Impurity.ENTROPY,
            number_of_chosen_feature=1,
            number_of_subset=100,
            subset_size=2,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            dt.predict_with_probability(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).probability.tolist(),
            [
                [[0.4099999964237213, 0.3499999940395355, 0.10000000149011612]],
                [[0.20000000298023224, 0.5699999928474426, 0.14000000059604645]],
                [[0.09000000357627869, 0.30000001192092896, 0.6100000143051147]],
            ],
        )
        self.assertListEqual(
            dt.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).tolist(),
            [[1], [4], [7]],
        )
