"""Test optimizer module."""

import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.neural_network.module import Parameter
from hoisaai.layer_1.model.supervised.neural_network.optimizer import Adam, SGD


class TestAdam(unittest.TestCase):
    """Test cases for the Adam optimizer."""

    def setUp(self) -> None:
        self.parameter: Parameter = Parameter(
            x=Tensor.array(
                x=[1.0, -1.0, 2.0],
                datatype=Tensor.DataType.FLOAT32,
                require_gradient=True,
            )
        )
        self.assertIsNone(self.parameter.gradient)
        self.parameter.gradient = Tensor.array(
            [1.0, 2.0, 3.0],
            datatype=Tensor.DataType.FLOAT32,
            require_gradient=False,
        )
        self.assertIsNotNone(self.parameter.gradient)
        self.optimizer: Adam = Adam(
            parameter=[self.parameter],
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        self.optimizer.initialize_gradient()
        self.assertIsNone(self.parameter.gradient)

    def test_initialize_gradient(self):
        """Test Adam.initialize_gradient."""

    def test_step(self):
        """Test Adam.step."""
        self.parameter.gradient = Tensor.array(
            [1.0, 2.0, 3.0],
            datatype=Tensor.DataType.FLOAT32,
            require_gradient=False,
        )
        self.optimizer.step()
        self.assertListEqual(
            self.parameter.to_list(),
            [0.8999999761581421, -1.100000023841858, 1.899999976158142],
        )


class TestSGD(unittest.TestCase):
    """Test cases for the SGD optimizer."""

    def setUp(self) -> None:
        self.parameter: Parameter = Parameter(
            x=Tensor.array(
                x=[1.0, -1.0, 2.0],
                datatype=Tensor.DataType.FLOAT32,
                require_gradient=True,
            )
        )
        self.assertIsNone(self.parameter.gradient)
        self.parameter.gradient = Tensor.array(
            [1.0, 2.0, 3.0],
            datatype=Tensor.DataType.FLOAT32,
            require_gradient=False,
        )
        self.assertIsNotNone(self.parameter.gradient)
        self.optimizer: SGD = SGD(
            parameter=[self.parameter],
            learning_rate=0.1,
            momentum=0.9,
        )
        self.optimizer.initialize_gradient()
        self.assertIsNone(self.parameter.gradient)

    def test_initialize_gradient(self):
        """Test SGD.initialize_gradient."""

    def test_step(self):
        """Test SGD.step."""
        self.parameter.gradient = Tensor.array(
            [1.0, 2.0, 3.0],
            datatype=Tensor.DataType.FLOAT32,
            require_gradient=False,
        )
        self.optimizer.step()
        self.assertListEqual(
            self.parameter.to_list(),
            [0.8999999761581421, -1.2000000476837158, 1.7000000476837158],
        )
