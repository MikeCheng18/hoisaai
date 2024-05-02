import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.classification.logistic_regression import (
    LogisticRegression,
)


class TestLogisticRegression(unittest.TestCase):
    def test_decision_tree_init(self):
        lg: LogisticRegression = LogisticRegression(
            maximum_iteration=int(1e3),
            learning_rate=1e-4,
            seed=0,
        )
        lg.fit(
            in_sample=Tensor.array(
                x=[[1, -3, -2, -1], [1, -2, -1, 0], [0, 4, 5, 6], [0, 7, 8, 9]],
                datatype=Tensor.DataType.FLOAT32,
            ),
            number_of_target=1,
        )
        probability: Tensor = lg.predict_with_probability(
            sample_x=Tensor.array(
                x=[[-3, -2, -1], [-2, -1, 0], [4, 5, 6], [7, 8, 9]],
                datatype=Tensor.DataType.INT32,
            )
        ).probability
        self.assertTrue(
            bool(
                (
                    (
                        probability
                        - Tensor.array(
                            [
                                [[0.28854042291641235, 0.7114595770835876]],
                                [[0.4286777377128601, 0.5713222622871399]],
                                [[0.9678344130516052, 0.032165609300136566]],
                                [[0.9947791695594788, 0.0052208062261343]],
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=probability.tolist(),
        )
        self.assertListEqual(
            lg.predict(
                sample_x=Tensor.array(
                    x=[[-3, -2, -1], [-2, -1, 0], [4, 5, 6], [7, 8, 9]],
                    datatype=Tensor.DataType.INT32,
                )
            ).tolist(),
            [[1.0], [1.0], [0.0], [0.0]],
        )
