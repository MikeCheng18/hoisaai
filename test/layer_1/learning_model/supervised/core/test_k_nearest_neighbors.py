import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.core.k_nearest_neighbors import (
    KNearestNeighbors,
)


class TestKNearestNeighbors(unittest.TestCase):
    def test_decision_tree_init(self):
        dt: KNearestNeighbors = KNearestNeighbors(
            k=2,
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
            [[[1, 4]], [[4, 1]], [[7, 4]]],
        )
