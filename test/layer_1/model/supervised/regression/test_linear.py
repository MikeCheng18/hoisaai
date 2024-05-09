"""Test for linear regression models."""

import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.regression.linear import (
    ElasticNetRegression,
    LassoRegression,
    OrdinaryLeastSquares,
    RidgeRegression,
)


class TestLinear(unittest.TestCase):
    """Test for linear regression models."""

    def test_elastic_net(self):
        """Test ElasticNetRegression."""
        lr: ElasticNetRegression = ElasticNetRegression(
            elastic_lambda=0.5,
            l1_ratio=0.5,
            learning_rate=1e-3,
            maximum_iteration=100,
            tolerance=1e-4,
        )
        lr.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        prediction: Tensor = lr.predict(
            sample_x=Tensor.array(
                x=[[2, 3], [5, 6], [8, 9]],
                datatype=Tensor.DataType.INT32,
            )
        )
        self.assertTrue(
            bool(
                (
                    (
                        prediction
                        - Tensor.array(
                            [
                                [2.107433795928955],
                                [4.274905204772949],
                                [6.442376613616943],
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.to_list(),
        )

    def test_lasso(self):
        """Test LassoRegression."""
        lr: LassoRegression = LassoRegression(
            lasso_lambda=0.5,
            learning_rate=1e-3,
            maximum_iteration=100,
            tolerance=1e-4,
        )
        lr.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        prediction: Tensor = lr.predict(
            sample_x=Tensor.array(
                x=[[2, 3], [5, 6], [8, 9]],
                datatype=Tensor.DataType.INT32,
            )
        )
        self.assertTrue(
            bool(
                (
                    (
                        prediction
                        - Tensor.array(
                            [
                                [2.107433795928955],
                                [4.274905204772949],
                                [6.442376613616943],
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.to_list(),
        )

    def test_ordinary_least_squares(self):
        """Test OrdinaryLeastSquares."""
        lr: OrdinaryLeastSquares = OrdinaryLeastSquares()
        lr.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, -7]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        prediction: Tensor = lr.predict(
            sample_x=Tensor.array(
                x=[[2, 3], [5, -7], [8, 9]],
                datatype=Tensor.DataType.INT32,
            )
        )
        self.assertTrue(
            bool(
                (
                    (
                        prediction
                        - Tensor.array(
                            [[0.77734375], [4.80078125], [4.45703125]],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.to_list(),
        )

    def test_ridge(self):
        """Test RidgeRegression."""
        lr: RidgeRegression = RidgeRegression(
            ridge_lambda=0.5,
        )
        lr.fit(
            in_sample=Tensor.array(
                x=[[1, 2, 3], [4, 5, -7]],
                datatype=Tensor.DataType.INT32,
            ),
            number_of_target=1,
        )
        prediction: Tensor = lr.predict(
            sample_x=Tensor.array(
                x=[[2, 3], [5, -7], [8, 9]],
                datatype=Tensor.DataType.INT32,
            )
        )
        self.assertTrue(
            bool(
                (
                    (
                        prediction
                        - Tensor.array(
                            [
                                [0.9419455528259277],
                                [3.965819835662842],
                                [3.6049203872680664],
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
            msg=prediction.to_list(),
        )
