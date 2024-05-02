import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.k_nearest_neighbors import (
    KNearestNeighborsClassifier,
)


class TestKNearestNeighborsClassifier(unittest.TestCase):
    def test_decision_tree_init(self):
        knn: KNearestNeighborsClassifier = KNearestNeighborsClassifier(
            k=1,
        )
        knn.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            knn.predict_with_probability(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]],
                    datatype=Tensor.DataType.INT32,
                )
            ).probability.tolist(),
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
        )
        self.assertListEqual(
            knn.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]],
                    datatype=Tensor.DataType.INT32,
                )
            ).tolist(),
            [[1], [4], [7]],
        )
        knn: KNearestNeighborsClassifier = KNearestNeighborsClassifier(
            k=2,
        )
        knn.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        self.assertListEqual(
            knn.predict_with_probability(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]],
                    datatype=Tensor.DataType.INT32,
                )
            ).probability.tolist(),
            [[[0.5, 0.5, 0.0]], [[0.5, 0.5, 0.0]], [[0.0, 0.5, 0.5]]],
        )
        self.assertListEqual(
            knn.predict(
                sample_x=Tensor.array(
                    x=[[2, 3], [5, 6], [8, 9]],
                    datatype=Tensor.DataType.INT32,
                )
            ).tolist(),
            [[1], [1], [4]],
        )
