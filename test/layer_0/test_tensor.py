"""Test the tensor module."""

# pylint: disable=too-many-lines
import math
import unittest

import jax

from hoisaai.layer_0.tensor import (
    Add,
    ExpandDimension,
    Exponential,
    Function,
    Hook,
    Inverse,
    MatrixMultiplication,
    Multiply,
    Negative,
    Power,
    Tensor,
    Transpose,
    Where,
)


class TestFunction(unittest.TestCase):
    """Test the Function class."""

    def test_expand_dimension(self):
        """Test the ExpandDimension class"""
        expand_dimension: ExpandDimension = ExpandDimension(-1)
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            expand_dimension(tensor).to_list(),
            [[1], [2], [3]],
        )
        self.assertListEqual(
            expand_dimension.backward(tensor=tensor, gradient=tensor).to_list(),
            [1, 2, 3],
        )

    def test_exponential(self):
        """Test the Exponential class"""
        exponential: Exponential = Exponential()
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertTrue(
            bool(
                #                 (
                (
                    exponential.forward(
                        tensor=Tensor.array(
                            [[1.0, 2.0, 3.0]], datatype=Tensor.DataType.FLOAT32
                        )
                    )
                    - Tensor.array(
                        [math.exp(1), math.exp(2), math.exp(3)],
                        datatype=Tensor.DataType.FLOAT32,
                        require_gradient=False,
                    )
                    < 1e-6
                ).all(axis=None)
            ),
        )
        self.assertListEqual(
            exponential.backward(tensor=tensor, gradient=tensor).to_list(),
            [1, 4, 9],
        )

    def test_negative(self):
        """Test the Negative class"""
        inverse: Negative = Negative()
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            inverse(tensor).to_list(),
            [-1, -2, -3],
        )
        self.assertListEqual(
            inverse.backward(tensor=tensor, gradient=tensor).to_list(),
            [-1, -2, -3],
        )

    def test_power(self):
        """Test the Power class"""
        power: Function = Power(exponent=3.0)
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            power(tensor).to_list(),
            [1, 8, 27],
        )
        self.assertListEqual(
            power.backward(tensor=tensor, gradient=tensor).to_list(),
            [3.0, 24.0, 81.0],
        )

    def test_inverse(self):
        """Test the Inverse class"""
        inverse: Inverse = Inverse()
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            inverse(tensor).to_list(),
            [1.0, 0.5, 0.3333333432674408],
        )
        self.assertListEqual(
            inverse.backward(tensor=tensor, gradient=tensor).to_list(),
            [-1.0, -0.5, -0.3333333432674408],
        )

    def test_transpose(self):
        """Test the Transpose class"""
        inverse: Transpose = Transpose(1, 0)
        tensor: Tensor = Tensor.array(
            [[1], [2], [3]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            inverse(tensor).to_list(),
            [[1, 2, 3]],
        )
        self.assertListEqual(
            inverse.backward(
                tensor=tensor,
                gradient=Tensor.array(
                    [[1, 2, 3]],
                    datatype=Tensor.DataType.INT32,
                    require_gradient=True,
                ),
            ).to_list(),
            [[1], [2], [3]],
        )


class TestHook(unittest.TestCase):
    """Test the Hook class."""

    def dummy_function(self, tensor: Tensor) -> Tensor:
        """Dummy function for testing."""
        return tensor

    def test_init(self):
        """Test the initialization of a Hook."""
        tensor: Tensor = Tensor.array(
            [1, 2, 3], datatype=Tensor.DataType.INT32, require_gradient=True
        )
        hook = Hook(
            tensor=tensor,
            gradient_function=self.dummy_function,
        )
        self.assertIsInstance(hook, Hook)
        self.assertListEqual(
            hook.tensor.to_list(),
            [1, 2, 3],
        )
        self.assertListEqual(
            hook.gradient_function(tensor).to_list(),
            [1, 2, 3],
        )


class TestOperation(unittest.TestCase):
    def test_add(self):
        add: Add = Add()
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [4, 5, 6],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = add(tensor1, tensor2)
        self.assertListEqual(
            output.to_list(),
            [5, 7, 9],
        )
        self.assertEqual(
            len(output.hook),
            2,
        )
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=output.hook[0].tensor,
            )
            .to_list(),
            [1, 2, 3],
        )
        self.assertListEqual(
            output.hook[1]
            .gradient_function(
                tensor=output.hook[1].tensor,
                gradient=output.hook[1].tensor,
            )
            .to_list(),
            [4, 5, 6],
        )
        tensor1: Tensor = Tensor.array(
            [[1], [2], [3]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [[4], [5], [6]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = add(tensor1, tensor2)
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=output.hook[0].tensor,
            )
            .to_list(),
            [[1], [2], [3]],
        )

    def test_matrix_multiplication(self):
        matrix_multiplication: MatrixMultiplication = MatrixMultiplication()
        tensor1: Tensor = Tensor.array(
            [[1, 2], [3, 4]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [[5, 6], [7, 8]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = matrix_multiplication(tensor1, tensor2)
        self.assertListEqual(
            output.to_list(),
            [[19, 22], [43, 50]],
        )
        self.assertEqual(
            len(output.hook),
            2,
        )
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=output.hook[0].tensor,
            )
            .to_list(),
            [[17, 23], [39, 53]],
        )
        self.assertListEqual(
            output.hook[1]
            .gradient_function(
                tensor=output.hook[1].tensor,
                gradient=output.hook[1].tensor,
            )
            .to_list(),
            [[26, 30], [38, 44]],
        )

    def test_multiply(self):
        multiply: Multiply = Multiply()
        tensor1: Tensor = Tensor.array(
            [[1, 2], [3, 4]], datatype=Tensor.DataType.INT32, require_gradient=True
        )
        tensor2: Tensor = Tensor.array(
            [[5, 6], [7, 8]], datatype=Tensor.DataType.INT32, require_gradient=True
        )
        output: Tensor = multiply(tensor1, tensor2)
        self.assertListEqual(
            output.to_list(),
            [[5, 12], [21, 32]],
        )
        self.assertEqual(
            len(output.hook),
            2,
        )
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=output.hook[0].tensor,
            )
            .to_list(),
            [[5, 12], [21, 32]],
        )
        self.assertListEqual(
            output.hook[1]
            .gradient_function(
                tensor=output.hook[1].tensor,
                gradient=output.hook[1].tensor,
            )
            .to_list(),
            [[5, 12], [21, 32]],
        )
        tensor1: Tensor = Tensor.array(
            [[1], [2], [3]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [[4], [5], [6]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = multiply(tensor1, tensor2)
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=Tensor.array(
                    [[1, 1], [2, 2], [3, 3]],
                    datatype=Tensor.DataType.INT32,
                    require_gradient=True,
                ),
            )
            .to_list(),
            [[4, 4], [10, 10], [18, 18]],
        )

    def test_where(self):
        where: Where = Where(
            condition=Tensor.array([True, False, False], datatype=Tensor.DataType.BOOL),
        )
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [4, 5, 6],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = where(tensor1, tensor2)
        self.assertListEqual(
            output.to_list(),
            [1, 5, 6],
        )
        self.assertEqual(
            len(output.hook),
            2,
        )
        self.assertListEqual(
            output.hook[0]
            .gradient_function(
                tensor=output.hook[0].tensor,
                gradient=output.hook[0].tensor,
            )
            .to_list(),
            [1.0, 0.0, 0.0],
        )
        self.assertListEqual(
            output.hook[1]
            .gradient_function(
                tensor=output.hook[1].tensor,
                gradient=output.hook[1].tensor,
            )
            .to_list(),
            [0.0, 5.0, 6.0],
        )


class TestTensor(unittest.TestCase):
    """Test the Tensor class."""

    def test_tensor_init(self):
        """Test the initialization of a Tensor."""
        tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
        )
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))
        # From jax array
        tensor = Tensor(x=jax.numpy.array([1, 2, 3]))
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))
        # From Tensor
        tensor = Tensor(x=tensor)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))

    def test_tensor_add(self):
        """Test the __add__ method of a Tensor."""
        tensor1: Tensor = Tensor.array(
            [[1, 2, 3]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [[4, 5, 6]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 + tensor2
        self.assertListEqual(
            output.to_list(),
            [[5, 7, 9]],
        )
        output.backward(
            gradient=Tensor.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], datatype=Tensor.DataType.INT32
            )
        )
        self.assertListEqual(
            output.gradient.to_list(),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        self.assertListEqual(
            tensor1.gradient.to_list(),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        self.assertListEqual(
            tensor2.gradient.to_list(),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 + 1
        self.assertListEqual(
            output.to_list(),
            [2, 3, 4],
        )

    def test_tensor_bool(self):
        """Test the __bool__ method of a Tensor."""
        self.assertEqual(
            bool(
                Tensor.array(
                    [True, False, True],
                    datatype=Tensor.DataType.BOOL,
                )[0]
            ),
            True,
        )

    def test_tensor_eq(self):
        """Test the __eq__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([2, 2, 3], datatype=Tensor.DataType.INT32)
                == Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [False, True, True],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) == 1).to_list(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32) == 1.0
            ).to_list(),
            [True, False, False],
        )

    def test_tensor_floordiv(self):
        """Test the __floordiv__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([64, 32, 16], datatype=Tensor.DataType.INT32)
                // Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [64, 16, 5],
        )
        self.assertListEqual(
            (Tensor.array([2, 4, 6], datatype=Tensor.DataType.INT32) // 2).to_list(),
            [1, 2, 3],
        )

    def test_tensor_ge(self):
        """Test the __ge__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                >= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [False, True, True],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) >= 2).to_list(),
            [False, True, True],
        )
        self.assertListEqual(
            (
                Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32) >= 2.0
            ).to_list(),
            [False, True, True],
        )

    def test_tensor_gt(self):
        """Test the __gt__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                > Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [False, False, True],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) > 2).to_list(),
            [False, False, True],
        )
        self.assertListEqual(
            (
                Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32) > 2.0
            ).to_list(),
            [False, False, True],
        )

    def test_tensor_getitem(self):
        """Test the __getitem__ method of a Tensor."""
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0:1]).to_list(),
            [1],
        )

    def test_tensor_float(self):
        """Test the __float__ method of a Tensor."""
        self.assertEqual(
            float(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0]),
            1.0,
        )

    def test_tensor_iadd(self):
        """Test the __iadd__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x += Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.to_list(),
            [2, 4, 6],
        )
        x += 1
        self.assertListEqual(
            x.to_list(),
            [3, 5, 7],
        )

    def test_tensor_int(self):
        """Test the __int__ method of a Tensor."""
        self.assertEqual(
            int(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0]),
            1,
        )

    def test_tensor_imul(self):
        """Test the __imul__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x *= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.to_list(),
            [1, 4, 9],
        )
        x *= 2
        self.assertListEqual(
            x.to_list(),
            [2, 8, 18],
        )

    def test_tensor_ipow(self):
        """Test the __ipow__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x **= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.to_list(),
            [1, 4, 27],
        )
        x **= 2
        self.assertListEqual(
            x.to_list(),
            [1, 16, 27 * 27],
        )

    def test_tensor_isub(self):
        """Test the __isub__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x -= Tensor.array([-1, -2, -3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.to_list(),
            [2, 4, 6],
        )
        x -= 2
        self.assertListEqual(
            x.to_list(),
            [0, 2, 4],
        )

    def test_tensor_itruediv(self):
        """Test the __itruediv__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x /= Tensor.array([0.5, 1.0, 2.0], datatype=Tensor.DataType.FLOAT32)
        self.assertListEqual(
            x.to_list(),
            [2.0, 2.0, 1.5],
        )
        x /= 2.0
        self.assertListEqual(
            x.to_list(),
            [1.0, 1.0, 0.75],
        )

    def test_tensor_invert(self):
        """Test the __invert__ method of a Tensor."""
        self.assertListEqual(
            (
                ~Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT16)
            ).to_list(),
            [-1.0, -2.0, -3.0],
        )
        self.assertListEqual(
            (
                ~Tensor.array([True, False, False], datatype=Tensor.DataType.BOOL)
            ).to_list(),
            [False, True, True],
        )

    def test_tensor_le(self):
        """Test the __le__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                <= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [True, True, False],
        )

    def test_tensor_lt(self):
        """Test the __lt__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                < Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [True, False, False],
        )

    def test_tensor_matmul(self):
        self.assertListEqual(
            (
                Tensor.array(
                    [
                        [
                            [1, 2],
                            [3, 4],
                        ],
                        [
                            [5, 6],
                            [7, 8],
                        ],
                    ],
                    datatype=Tensor.DataType.INT32,
                )
                @ Tensor.array(
                    [
                        [
                            [1, 2],
                            [3, 4],
                        ],
                        [
                            [5, 6],
                            [7, 8],
                        ],
                    ],
                    datatype=Tensor.DataType.INT32,
                )
            ).to_list(),
            [[[7, 10], [15, 22]], [[67, 78], [91, 106]]],
        )

    def test_tensor_mod(self):
        self.assertListEqual(
            (
                Tensor.array([17, 13, 11], datatype=Tensor.DataType.INT32)
                % Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [0, 1, 2],
        )
        self.assertListEqual(
            (Tensor.array([17, 13, 11], datatype=Tensor.DataType.INT32) % 4).to_list(),
            [1, 1, 3],
        )

    def test_tensor_mul(self):
        """Test the __mul__ method of a Tensor."""
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 * tensor2
        self.assertListEqual(
            output.to_list(),
            [1, 4, 9],
        )
        output: Tensor = output.backward(
            gradient=Tensor.array([1, 1, 1], datatype=Tensor.DataType.INT32)
        )
        self.assertListEqual(
            tensor1.gradient.to_list(),
            [1.0, 2.0, 3.0],
        )
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 * 2
        self.assertListEqual(
            output.to_list(),
            [2, 4, 6],
        )

    def test_tensor_ne(self):
        """Test the __ne__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([2, 2, 3], datatype=Tensor.DataType.INT32)
                != Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [True, False, False],
        )

    def test_tensor_pow(self):
        """Test the __pow__ method of a Tensor."""
        self.assertListEqual(
            (
                Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
                ** Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).to_list(),
            [1, 4, 27],
        )
        tensor: Tensor = (
            Tensor.array(
                [1, 2, 3],
                datatype=Tensor.DataType.INT32,
                require_gradient=True,
            )
            ** 2
        )
        self.assertListEqual(
            tensor.to_list(),
            [1, 4, 9],
        )
        self.assertTrue(tensor.require_gradient)

    def test_tensor_radd(self):
        """Test the __radd__ method of a Tensor."""
        self.assertListEqual(
            (1 + Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [2, 3, 4],
        )

    def test_tensor_req(self):
        """Test the __req__ method of a Tensor."""
        self.assertListEqual(
            (1 == Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                1.0 == Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).to_list(),
            [True, False, False],
        )

    def test_tensor_rge(self):
        """Test the __rge__ method of a Tensor."""
        self.assertListEqual(
            (2 >= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [True, True, False],
        )
        self.assertListEqual(
            (
                2.0 >= Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).to_list(),
            [True, True, False],
        )

    def test_tensor_rgt(self):
        """Test the __rgt__ method of a Tensor."""
        self.assertListEqual(
            (2 > Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                2.0 > Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).to_list(),
            [True, False, False],
        )

    def test_tensor_rle(self):
        """Test the __rle__ method of a Tensor."""
        self.assertListEqual(
            (2 <= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [False, True, True],
        )
        self.assertListEqual(
            (
                2.0 <= Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).to_list(),
            [False, True, True],
        )

    def test_tensor_rlt(self):
        """Test the __rlt__ method of a Tensor."""
        self.assertListEqual(
            (2 < Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [False, False, True],
        )
        self.assertListEqual(
            (
                2.0 < Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).to_list(),
            [False, False, True],
        )

    def test_tensor_rfloordiv(self):
        """Test the __rfloordiv__ method of a Tensor."""
        self.assertListEqual(
            (128 // Tensor.array([2, 4, 6], datatype=Tensor.DataType.INT32)).to_list(),
            [64, 32, 21],
        )

    def test_tensor_rmod(self):
        """Test the __rmod__ method of a Tensor."""
        self.assertListEqual(
            (17 % Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [0, 1, 2],
        )

    def test_tensor_rmul(self):
        """Test the __rmul__ method of a Tensor."""
        self.assertListEqual(
            (17 * Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [17, 34, 51],
        )

    def test_tensor_rsub(self):
        """Test the __rsub__ method of a Tensor."""
        self.assertListEqual(
            (17 - Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [16, 15, 14],
        )

    def test_tensor_rtruediv(self):
        """Test the __rtruediv__ method of a Tensor."""
        self.assertListEqual(
            (21 / Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).to_list(),
            [21.0, 10.5, 7.0],
        )

    def test_tensor_setitem(self):
        """Test the __setitem__ method of a Tensor."""
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x[0] = 0
        self.assertListEqual(
            x.to_list(),
            [0, 2, 3],
        )
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x[0:2] = Tensor.array([-1, -2], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.to_list(),
            [-1, -2, 3],
        )

    def test_tensor_sub(self):
        """Test the __sub__ method of a Tensor."""
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 - tensor2
        self.assertListEqual(
            output.to_list(),
            [0, 0, 0],
        )
        output: Tensor = output.backward(
            gradient=Tensor.array([1, 1, 1], datatype=Tensor.DataType.INT32)
        )
        self.assertListEqual(
            tensor1.gradient.to_list(),
            [1, 1, 1],
        )
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 - 2
        self.assertListEqual(
            output.to_list(),
            [-1, 0, 1],
        )

    def test_tensor_truediv(self):
        """Test the __truediv__ method of a Tensor."""
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        tensor2: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 / tensor2
        self.assertListEqual(
            output.to_list(),
            [1.0, 1.0, 1.0],
        )
        output: Tensor = output.backward(
            gradient=Tensor.array([1, 1, 1], datatype=Tensor.DataType.INT32)
        )
        self.assertListEqual(
            tensor1.gradient.to_list(),
            [1.0, 0.5, 0.3333333432674408],
        )
        tensor1: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        output: Tensor = tensor1 / 2
        self.assertListEqual(
            output.to_list(),
            [0.5, 1.0, 1.5],
        )

    def test_all(self):
        """Test the all method."""
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .all(axis=0)
            .to_list(),
            [True, False],
        )
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .all(axis=1)
            .to_list(),
            [False, False],
        )
        self.assertFalse(
            Tensor.array(
                [[True, False], [True, False]], datatype=Tensor.DataType.BOOL
            ).all()
        )

    def test_any(self):
        """Test the any method."""
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .any(axis=0)
            .to_list(),
            [True, False],
        )
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .any(axis=1)
            .to_list(),
            [True, True],
        )
        self.assertTrue(
            Tensor.array(
                [[True, False], [True, False]], datatype=Tensor.DataType.BOOL
            ).any()
        )

    def test_arange(self):
        """Test the arange function."""
        self.assertListEqual(
            Tensor.arange(
                start=0,
                stop=3,
                step=1,
                datatype=Tensor.DataType.INT16,
            ).to_list(),
            [0, 1, 2],
        )

    def test_argmax(self):
        """Test the argmax method."""
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argmax(axis=0)
            .to_list(),
            [1, 1],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argmax(axis=1)
            .to_list(),
            [1, 0],
        )

    def test_argsort(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argsort(axis=0)
            .to_list(),
            [[0, 0], [1, 1]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argsort(axis=1)
            .to_list(),
            [[0, 1], [1, 0]],
        )

    def test_array(self):
        """Test the array function."""
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).to_list(),
            [1, 2, 3],
        )

    def test_astype(self):
        """Test the astype method."""
        tensor: Tensor = Tensor.array(
            [[1, 2], [3, 4]],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        )
        self.assertListEqual(
            tensor.astype(Tensor.DataType.FLOAT16).to_list(),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertTrue(tensor.require_gradient)

    def test_concatenate(self):
        """Test the concatenate method."""
        tensor: Tensor = Tensor.concatenate(
            tensors=[
                Tensor.array([1, 2], datatype=Tensor.DataType.INT32),
                Tensor.array([4, 3], datatype=Tensor.DataType.INT32),
            ],
            axis=0,
        )
        self.assertListEqual(
            tensor.to_list(),
            [1, 2, 4, 3],
        )
        self.assertFalse(tensor.require_gradient)
        self.assertTrue(
            Tensor.concatenate(
                tensors=[
                    Tensor.array([1, 2], datatype=Tensor.DataType.INT32),
                    Tensor.array([4, 3], datatype=Tensor.DataType.INT32),
                ],
                axis=0,
                require_gradient=True,
            ).require_gradient
        )

    def test_count_nonzero(self):
        """Test the count_nonzero method."""
        self.assertListEqual(
            Tensor.array([[1, 0], [3, 0]], datatype=Tensor.DataType.INT32)
            .count_nonzero(axis=0, keep_dimension=False)
            .to_list(),
            [2, 0],
        )
        self.assertListEqual(
            Tensor.array([[1, 0], [3, 0]], datatype=Tensor.DataType.INT32)
            .count_nonzero(axis=1, keep_dimension=False)
            .to_list(),
            [1, 1],
        )

    def test_cumulative_product(self):
        """Test the cumulative_product method."""
        self.assertListEqual(
            Tensor.array([[1, 2, 3], [4, 5, 6]], datatype=Tensor.DataType.INT16)
            .cumulative_product(axis=0)
            .to_list(),
            [[1, 2, 3], [4, 10, 18]],
        )

    def test_datatype(self):
        """Test the datatype method."""
        self.assertEqual(
            Tensor.array([True, False, False], datatype=Tensor.DataType.BOOL).datatype,
            Tensor.DataType.BOOL,
        )
        self.assertEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).datatype,
            Tensor.DataType.INT32,
        )
        self.assertEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.FLOAT16).datatype,
            Tensor.DataType.FLOAT16,
        )

    def test_diagonal(self):
        """Test the diagonal method."""
        self.assertListEqual(
            Tensor.diagonal(value=1, size=2, datatype=Tensor.DataType.INT16).to_list(),
            [[1, 0], [0, 1]],
        )
        tensor: Tensor = Tensor.diagonal(
            value=1, size=2, datatype=Tensor.DataType.INT16, require_gradient=True
        )
        self.assertTrue(
            tensor.require_gradient,
        )

    def test_exp(self):
        """Test the exp method."""
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).exp().to_list(),
            [2.7182817459106445, 7.389056205749512, 20.08553695678711],
        )

    def test_expand_dimension(self):
        """Test the expand_dimension method."""
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            .expand_dimension(-1)
            .to_list(),
            [[1], [2], [3]],
        )

    def test_full(self):
        """Test the full function."""
        self.assertListEqual(
            Tensor.full(
                value=1, shape=(2, 2), datatype=Tensor.DataType.INT16
            ).to_list(),
            [[1, 1], [1, 1]],
        )
        self.assertListEqual(
            Tensor.full(
                value=Tensor.Constant.NAN, shape=(2, 2), datatype=Tensor.DataType.INT16
            ).to_list(),
            [[0, 0], [0, 0]],
        )
        self.assertFalse(
            Tensor.full(
                value=1,
                shape=(2, 2),
                datatype=Tensor.DataType.INT16,
            ).require_gradient
        )

    def test_get_sample_x_and_y(self):
        """Test the get_sample_x_and_y method of a Tensor."""
        (
            sample_x,
            sample_y,
        ) = Tensor.array(
            [[1, 2], [3, 4]], datatype=Tensor.DataType.INT32
        ).get_sample_x_and_y(
            number_of_target=1,
        )
        self.assertListEqual(
            sample_x.to_list(),
            [[2], [4]],
        )
        self.assertListEqual(
            sample_y.to_list(),
            [[1], [3]],
        )

    def test_get_by_index(self):
        """Test the get_by_index method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .get_by_index(indexes=Tensor.array([0, 1], datatype=Tensor.DataType.INT32))
            .to_list(),
            [1, 4],
        )
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .get_by_index(indexes=Tensor.array([0, 1], datatype=Tensor.DataType.INT32))
            .to_list(),
            [1.0, 4.0],
        )

    def test_in_sample(self):
        """Test the in_sample method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .in_sample(in_sample_size=1)
            .to_list(),
            [[1.0, 2.0]],
        )

    def test_insert(self):
        """Test the insert method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .insert(index=0, value=0.0, axis=1)
            .to_list(),
            [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]],
        )

    def test_inverse(self):
        """Test the inverse method of a Tensor."""
        self.assertTrue(
            bool(
                (
                    (
                        Tensor.array(
                            [[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32
                        ).inverse()
                        - Tensor.array(
                            [[-2.0, 1.0], [1.5, -0.5]], datatype=Tensor.DataType.FLOAT32
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
        )

    def test_isnan(self):
        """Test the isnan method of a Tensor."""
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .isnan()
            .to_list(),
            [[False, False], [False, True]],
        )

    def test_log2(self):
        """Test the log2 method of a Tensor."""
        self.assertTrue(
            bool(
                (
                    (
                        Tensor.array(
                            [[1.0, 2.0, 3.0, 4.0]], datatype=Tensor.DataType.FLOAT32
                        ).log2()
                        - Tensor.array(
                            [[0, 1.0, math.log2(3.0), 2.0]],
                            datatype=Tensor.DataType.FLOAT32,
                        )
                    )
                    < 1e-6
                ).all(axis=None)
            ),
        )

    def test_mean(self):
        """Test the mean method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=0, keep_dimension=False)
            .to_list(),
            [2.5, 2.5],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=0, keep_dimension=True)
            .to_list(),
            [[2.5, 2.5]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=1, keep_dimension=False)
            .to_list(),
            [1.5, 3.5],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=1, keep_dimension=True)
            .to_list(),
            [[1.5], [3.5]],
        )

    def test_nanmean(self):
        """Test the nanmean method of a Tensor."""
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .nanmean(axis=0, keep_dimension=False)
            .to_list(),
            [2.0, 2.0],
        )

    def test_nansum(self):
        """Test the nansum method of a Tensor."""
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .nansum(axis=0, keep_dimension=False)
            .to_list(),
            [4.0, 2.0],
        )

    def test_nan_to_num(self):
        """Test the nan_to_num method of a Tensor."""
        self.assertListEqual(
            Tensor.array(
                [[0.0, jax.numpy.inf], [-1 * jax.numpy.inf, jax.numpy.nan]],
                datatype=Tensor.DataType.FLOAT32,
            )
            .nan_to_num(
                nan=1.0,
                posinf=2.0,
                neginf=3.0,
            )
            .to_list(),
            [[0.0, 2.0], [3.0, 1.0]],
        )

    def test_number_of_dimension(self):
        """Test the number_of_dimension method of a Tensor."""
        self.assertEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).number_of_dimension,
            1,
        )

    def test_out_of_sample(self):
        """Test the out_of_sample method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .out_of_sample(in_sample_size=1)
            .to_list(),
            [[3.0, 4.0]],
        )

    def test_random_integer(self):
        """Test the random_integer method of a Tensor."""
        self.assertListEqual(
            Tensor.random_integer(
                shape=(2, 2),
                minimum_value=0,
                maximum_value=10,
                datatype=Tensor.DataType.INT16,
                seed=0,
            ).to_list(),
            [[8, 8], [0, 2]],
        )
        self.assertListEqual(
            Tensor.random_integer(
                shape=(2, 2),
                minimum_value=0,
                maximum_value=10,
                datatype=Tensor.DataType.INT16,
                seed=1,
            ).to_list(),
            [[4, 6], [2, 7]],
        )

    def test_random_normal(self):
        """Test the random_normal method of a Tensor."""
        self.assertListEqual(
            Tensor.random_normal(
                shape=(2, 2),
                datatype=Tensor.DataType.FLOAT32,
                seed=0,
            ).to_list(),
            [
                [1.8160862922668457, -0.7548851370811462],
                [0.3398890793323517, -0.5348353385925293],
            ],
        )
        self.assertListEqual(
            Tensor.random_normal(
                shape=(2, 2),
                datatype=Tensor.DataType.FLOAT32,
                seed=1,
            ).to_list(),
            [
                [0.17269018292427063, -0.46084192395210266],
                [1.2229712009429932, -0.07101909071207047],
            ],
        )

    def test_random_uniform(self):
        """Test the random_uniform method of a Tensor."""
        self.assertListEqual(
            Tensor.random_uniform(
                minimum_value=0,
                maximum_value=10,
                shape=(2, 2),
                datatype=Tensor.DataType.FLOAT32,
                seed=0,
            ).to_list(),
            [
                [9.653214454650879, 2.251589298248291],
                [6.330299377441406, 2.963818311691284],
            ],
        )
        self.assertListEqual(
            Tensor.random_uniform(
                minimum_value=0,
                maximum_value=10,
                shape=(2, 2),
                datatype=Tensor.DataType.FLOAT32,
                seed=1,
            ).to_list(),
            [
                [5.685524940490723, 3.224560022354126],
                [8.89329719543457, 4.716912269592285],
            ],
        )

    def test_relu(self):
        """Test the relu method of a Tensor."""
        self.assertListEqual(
            Tensor.array([-1, 2, -3], datatype=Tensor.DataType.INT32).relu().to_list(),
            [0, 2, 0],
        )

    def test_reshape(self):
        """Test the reshape method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .reshape(-1)
            .to_list(),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_shape(self):
        """Test the shape method of a Tensor."""
        self.assertListEqual(
            list(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).shape),
            list((3,)),
        )

    def test_sign(self):
        """Test the sign method of a Tensor."""
        self.assertListEqual(
            Tensor.array([-1, 2, -3], datatype=Tensor.DataType.INT32).sign().to_list(),
            [-1, 1, -1],
        )

    def test_sigmoid(self):
        """Test the sigmoid method of a Tensor."""
        tensor: Tensor = Tensor.array(
            [1, 2, 3],
            datatype=Tensor.DataType.INT32,
            require_gradient=True,
        ).sigmoid()
        self.assertListEqual(
            tensor.to_list(),
            [0.2689414322376251, 0.11920291930437088, 0.04742587357759476],
        )
        self.assertTrue(tensor.require_gradient)
        self.assertEqual(len(tensor.hook), 1)

    def test_sliding_window(self):
        """Test the sliding_window method of a Tensor."""
        self.assertListEqual(
            Tensor.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]], datatype=Tensor.DataType.INT32
            )
            .sliding_window(window_size=2)
            .to_list(),
            [[[1, 2], [3, 4]], [[3, 4], [5, 6]], [[5, 6], [7, 8]]],
        )

    def test_sort(self):
        """Test the sort method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .sort(axis=0)
            .to_list(),
            [[1, 2], [4, 3]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .sort(axis=1)
            .to_list(),
            [[1, 2], [3, 4]],
        )

    def test_sqrt(self):
        """Test the sqrt method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 4], [9, 16]], datatype=Tensor.DataType.INT32)
            .sqrt()
            .to_list(),
            [[1, 2], [3, 4]],
        )

    def test_square(self):
        """Test the square method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .square()
            .to_list(),
            [[1, 4], [9, 16]],
        )

    def test_stack(self):
        """Test the stack method of a Tensor."""
        self.assertListEqual(
            Tensor.stack(
                tensors=[
                    Tensor.array([1, 2], datatype=Tensor.DataType.INT32),
                    Tensor.array([4, 3], datatype=Tensor.DataType.INT32),
                ],
                axis=0,
            ).to_list(),
            [[1, 2], [4, 3]],
        )

    def test_std(self):
        self.assertListEqual(
            Tensor.array([[1, 2, 3], [3, 4, 5]], datatype=Tensor.DataType.INT32)
            .std(axis=1, keep_dimension=False)
            .to_list(),
            [0.8164966106414795, 0.8164966106414795],
        )

    def test_sum(self):
        """Test the sum method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=0, keep_dimension=False)
            .to_list(),
            [4, 6],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=0, keep_dimension=True)
            .to_list(),
            [[4, 6]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=1, keep_dimension=False)
            .to_list(),
            [3, 7],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=1, keep_dimension=True)
            .to_list(),
            [[3], [7]],
        )

    def test_swapaxes(self):
        """Test the swapaxes method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .swapaxes(axis1=0, axis2=1)
            .to_list(),
            [[1, 3], [2, 4]],
        )

    def test_to_list(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).to_list(),
            [1, 2, 3],
        )

    def test_to_numpy(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).to_numpy().tolist(),
            [1, 2, 3],
        )

    def test_transpose(self):
        """Test the transpose method of a Tensor."""
        self.assertListEqual(
            Tensor.array([[1], [2], [3]], datatype=Tensor.DataType.INT32)
            .transpose(1, 0)
            .to_list(),
            [[1, 2, 3]],
        )

    def test_unique(self):
        unique, count = Tensor.array(
            [[1, 2, 3], [1, 2, 3], [4, 5, 6]], datatype=Tensor.DataType.INT32
        ).unique()
        self.assertListEqual(
            unique.to_list(),
            [1, 2, 3, 4, 5, 6],
        )
        self.assertListEqual(
            count.to_list(),
            [2, 2, 2, 1, 1, 1],
        )

    def test_where(self):
        self.assertListEqual(
            Tensor.where(
                condition=Tensor.array(
                    [True, False, True], datatype=Tensor.DataType.BOOL
                ),
                if_true=Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32),
                if_false=Tensor.array([4, 5, 6], datatype=Tensor.DataType.INT32),
            ).to_list(),
            [1, 5, 3],
        )


if __name__ == "__main__":
    unittest.main()
