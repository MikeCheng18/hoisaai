import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.neural_network.loss import (
    MeanSquaredError,
)


class TestMeanSquaredError(unittest.TestCase):

    def test_backward(self):
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[4.0, 5.0, 6.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.backward(prediction, actual).tolist(), [[4.0, 5.0, 6.0]]
        )
        self.assertEqual(
            prediction.gradient.tolist(),
            [[6.0, 6.0, 6.0]],
        )
        self.assertIsNone(actual.gradient)

    def test_forward(self):
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.forward(prediction, actual).tolist(), [0.0, 0.0, 0.0]
        )
        mean_squared_error: MeanSquaredError = MeanSquaredError()
        prediction = Tensor.array([[1.0, 2.0, 3.0]], Tensor.DataType.FLOAT32)
        actual = Tensor.array([[-1.0, -2.0, -3.0]], Tensor.DataType.FLOAT32)
        self.assertEqual(
            mean_squared_error.forward(prediction, actual).tolist(), [4.0, 16.0, 36.0]
        )
