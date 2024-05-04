import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.learning_model.supervised.neural_network.module import Parameter
from hoisaai.layer_1.learning_model.supervised.neural_network.optimizer import Adam


class TestAdam(unittest.TestCase):
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
        pass

    def test_step(self):
        self.parameter.gradient = Tensor.array(
            [1.0, 2.0, 3.0],
            datatype=Tensor.DataType.FLOAT32,
            require_gradient=False,
        )
        self.optimizer.step()
        self.assertListEqual(
            self.parameter.tolist(),
            [0.8999999761581421, -1.100000023841858, 1.899999976158142],
        )
