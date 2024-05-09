"""Test cases for the module module."""

import typing
import unittest

from hoisaai.layer_0.tensor import Tensor
from hoisaai.layer_1.model.supervised.neural_network.loss import (
    MeanSquaredError,
)
from hoisaai.layer_1.model.supervised.neural_network.module import (
    Linear,
    Parameter,
)
from hoisaai.layer_1.model.supervised.neural_network.module import Sequential


class TestParameter(unittest.TestCase):
    """Test cases for the Parameter module."""

    def test_init(self):
        """Test Parameter.__init__."""
        self.assertEqual(
            Parameter(
                x=Tensor.array(x=1, datatype=Tensor.DataType.INT32),
            ).require_gradient,
            True,
        )


class TestLinear(unittest.TestCase):
    """Test cases for the Linear module."""

    WEIGHT: typing.List[float] = [
        [0.08482573926448822],
        [1.9097647666931152],
        [0.2956174314022064],
        [1.1209479570388794],
        [0.3343234360218048],
        [-0.8260677456855774],
        [0.6481276750564575],
        [1.043487310409546],
        [-0.7824838757514954],
        [-0.45394620299339294],
        [0.629797101020813],
        [0.8152464628219604],
        [-0.3278767764568329],
        [-1.1234447956085205],
        [-1.6607415676116943],
        [0.27290546894073486],
    ]

    def setUp(self) -> None:
        self.linear: Linear = Linear(
            input_shape=(2, 16),
            output_shape=(2, 1),
            bias=True,
            data_type=Tensor.DataType.FLOAT32,
            seed=0,
        )

    def test_init(self):
        """Test Linear.__init__."""
        self.assertListEqual(
            self.linear.weight.to_list(),
            TestLinear.WEIGHT,
        )
        self.assertListEqual(
            self.linear.bias.to_list(),
            [0.0],
        )

    def test_multiple_output_dimension(self):
        """Test Linear.__init__ with multiple output dimension."""
        self.linear: Linear = Linear(
            input_shape=(2, 2),
            output_shape=(2, 2),
            bias=True,
            data_type=Tensor.DataType.FLOAT32,
            seed=0,
        )
        self.assertListEqual(
            self.linear.weight.to_list(),
            [
                [1.8160862922668457, -0.7548851370811462],
                [0.3398890793323517, -0.5348353385925293],
            ],
        )
        self.assertListEqual(
            self.linear.bias.to_list(),
            [0.0, 0.0],
        )
        self.assertTrue(
            bool(
                (
                    (
                        self.linear.forward(
                            Tensor.array(
                                [[1.0, 1.0], [2.0, 2.0]],
                                datatype=Tensor.DataType.FLOAT32,
                            )
                        )[0]
                        - Tensor.array(
                            [
                                [
                                    (1.8160862922668457 + 0.3398890793323517),
                                    (-0.7548851370811462 - 0.5348353385925293),
                                ],
                                [
                                    (1.8160862922668457 + 0.3398890793323517) * 2.0,
                                    (-0.7548851370811462 - 0.5348353385925293) * 2.0,
                                ],
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e0
                ).all(axis=None)
            ),
            msg=(
                self.linear.forward(
                    Tensor.array(
                        [[1.0, 1.0], [2.0, 2.0]],
                        datatype=Tensor.DataType.FLOAT32,
                    )
                )[0].to_list()
            ),
        )

    def test_parameter(self):
        """Test Linear.get_parameter."""
        parameters: typing.List[Parameter] = list(self.linear.get_parameter())
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0].shape, (16, 1))
        self.assertEqual(parameters[1].shape, (1,))

    def test_forward(self):
        """Test Linear.forward."""
        self.assertListEqual(
            self.linear.forward(
                Tensor.array(
                    [[1.0 for _ in range(16)]],
                    datatype=Tensor.DataType.FLOAT32,
                    require_gradient=False,
                )
            )[0].to_list(),
            [[1.9804831743240356]],
        )

    def test_gradient(self):
        """Test Linear.gradient."""
        self.linear.train()
        loss: MeanSquaredError = MeanSquaredError()
        actual: Tensor = Tensor.array([[1.0]], Tensor.DataType.FLOAT32)
        prediction: Tensor = self.linear.forward(
            Tensor.array(
                [[2.0 for _ in range(16)]],
                datatype=Tensor.DataType.FLOAT32,
                require_gradient=True,
            )
        )[0]
        self.assertListEqual(
            prediction.to_list(),
            [[2.0 * 1.9804831743240356]],
        )
        self.assertAlmostEqual(
            loss.forward(
                prediction=prediction,
                actual=actual,
            ).to_list()[0],
            (2.0 * 1.9804831743240356 - 1.0) ** 2,
            places=5,
        )
        self.assertIsNone(self.linear.weight.gradient)
        self.assertIsNone(self.linear.bias.gradient)
        loss.backward(prediction, actual)
        self.assertIsNotNone(self.linear.weight.gradient)
        self.assertIsNotNone(self.linear.bias.gradient)
        self.assertTrue(
            bool(
                (
                    (
                        self.linear.weight.gradient
                        - Tensor.array(
                            [
                                [
                                    2.0
                                    * (2.0 * 1.9804831743240356 - 1.0)
                                    * 1.9804831743240356
                                ]
                                for _ in range(16)
                            ],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e0
                ).all(axis=None)
            ),
            msg=(
                self.linear.weight.gradient.to_list(),
                2.0 * (2.0 * 1.9804831743240356 - 1.0) * 1.9804831743240356,
            ),
        )
        self.assertListEqual(
            self.linear.bias.gradient.to_list(),
            [2.0 * (2.0 * 1.9804831743240356 - 1.0)],
        )


class TestSequential(unittest.TestCase):
    """Test cases for the Sequential module."""

    def setUp(self):
        self.linear1: Linear = Linear(
            input_shape=(2, 2),
            output_shape=(2, 2),
            bias=True,
            data_type=Tensor.DataType.FLOAT32,
            seed=0,
        )
        self.linear2: Linear = Linear(
            input_shape=(2, 2),
            output_shape=(2, 1),
            bias=True,
            data_type=Tensor.DataType.FLOAT32,
            seed=0,
        )
        self.sequential: Sequential = Sequential(self.linear1, self.linear2)

    def test_init(self):
        """Test Sequential.__init__."""
        self.assertIsNotNone(self.sequential)

    def test_forward(self):
        """Test Sequential.forward."""
        self.assertListEqual(
            self.sequential.forward(
                Tensor.array(
                    [[1.0, 1.0], [2.0, 2.0]],
                    datatype=Tensor.DataType.FLOAT32,
                    require_gradient=False,
                )
            )[0].to_list(),
            (
                Tensor.array(
                    [
                        [
                            (1.8160862922668457 + 0.3398890793323517),
                            (-0.7548851370811462 - 0.5348353385925293),
                        ],
                        [
                            (1.8160862922668457 + 0.3398890793323517) * 2.0,
                            (-0.7548851370811462 - 0.5348353385925293) * 2.0,
                        ],
                    ],
                    datatype=Tensor.DataType.FLOAT32,
                )
                @ self.linear2.weight
            ).to_list(),
        )
