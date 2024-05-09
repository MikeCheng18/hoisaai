"""Test cases for the loss module."""

import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.neural_network.loss import (
    MeanSquaredError,
)


class TestMeanSquaredError(unittest.TestCase):
    """Test cases for the MeanSquaredError loss function."""

    def test_backward(self):
        """Test MeanSquaredError.backward."""
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[4.0, 5.0, 6.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.backward(prediction, actual).to_list(), [[4.0, 5.0, 6.0]]
        )
        self.assertEqual(
            prediction.gradient.to_list(),
            [[6.0, 6.0, 6.0]],
        )
        self.assertIsNone(actual.gradient)

    def test_forward(self):
        """Test MeanSquaredError.forward."""
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.forward(prediction, actual).to_list(), [0.0, 0.0, 0.0]
        )
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[-1.0, -2.0, -3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.forward(prediction, actual).to_list(), [4.0, 16.0, 36.0]
        )
