"""Test for KNearestNeighborsRegressor."""

import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.regression.k_nearest_neighbors import (
    KNearestNeighborsRegressor,
)


class TestKNearestNeighborsRegressor(unittest.TestCase):
    """Test KNearestNeighborsRegressor."""

    def test_decision_tree_init(self):
        """Test DecisionTreeRegressor.__init__."""
        knn: KNearestNeighborsRegressor = KNearestNeighborsRegressor(
            k=2,
        )
        knn.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            ),
            number_of_target=1,
        )
        prediction: Tensor = knn.predict(
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
                            [[2.5], [2.5], [5.5]],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.to_list(),
        )
