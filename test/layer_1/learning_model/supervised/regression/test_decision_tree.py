import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.regression.decision_tree import (
    BaggingRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
)


class TestDecisionTreeRegressor(unittest.TestCase):
    def test_decision_tree_init(self):
        dt: DecisionTreeRegressor = DecisionTreeRegressor(
            depth=2,
            criterion=DecisionTreeRegressor.Criterion.SQUARED_ERROR,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        prediction: Tensor = dt.predict(
            sample_x=Tensor.array(
                x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
            )
        )
        self.assertTrue(
            bool(
                (
                    (
                        prediction
                        - Tensor.array(
                            [[[1.0], [4.0], [7.0]]],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.tolist(),
        )

    def test_bagging_preparation(self):
        dt: BaggingRegressor = BaggingRegressor(
            depth=3,
            criterion=DecisionTreeRegressor.Criterion.SQUARED_ERROR,
            number_of_subset=128,
            subset_size=3,
        )
        dt.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            dt.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).tolist(),
            [[1.4455446004867554], [4.5279998779296875], [6.0625]],
        )

    def test_random_forest_preparation(self):
        dt: RandomForestRegressor = RandomForestRegressor(
            depth=3,
            criterion=DecisionTreeRegressor.Criterion.SQUARED_ERROR,
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
            dt.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]], datatype=Tensor.DataType.INT32
                )
            ).tolist(),
            [[2.918604612350464], [3.8021976947784424], [5.559999942779541]],
        )
