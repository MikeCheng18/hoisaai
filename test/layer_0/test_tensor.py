import math
import unittest

import jax
import jaxlib.xla_extension
import numpy

from hoisaai.layer_0.tensor import Tensor, tensor_casting


class TestTensor(unittest.TestCase):
    def test_tensor_casting(self):
        # Tensor
        result = tensor_casting(x=Tensor(x=numpy.array([1, 2, 3])))
        self.assertIsInstance(result, jaxlib.xla_extension.ArrayImpl)
        # Numpy
        result = tensor_casting(x=numpy.array([1, 2, 3]))
        self.assertIsInstance(result, jaxlib.xla_extension.ArrayImpl)
        # Int
        result = tensor_casting(x=1)
        self.assertIsInstance(result, int)
        # Float
        result = tensor_casting(x=1.0)
        self.assertIsInstance(result, float)

    def test_tensor_init(self):
        # From numpy array
        tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))
        # From jax array
        tensor = Tensor(x=jax.numpy.array([1, 2, 3]))
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape, (3,))

    def test_tensor_add(self):
        self.assertListEqual(
            (
                Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
                + Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [2, 4, 6],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) + 1).tolist(),
            [2, 3, 4],
        )

    def test_tensor_bool(self):
        self.assertEqual(
            int(Tensor.array([True, False, True], datatype=Tensor.DataType.BOOL)[0]),
            True,
        )

    def test_tensor_eq(self):
        self.assertListEqual(
            (
                Tensor.array([2, 2, 3], datatype=Tensor.DataType.INT32)
                == Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [False, True, True],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) == 1).tolist(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32) == 1.0
            ).tolist(),
            [True, False, False],
        )

    def test_tensor_floordiv(self):
        self.assertListEqual(
            (
                Tensor.array([64, 32, 16], datatype=Tensor.DataType.INT32)
                // Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [64, 16, 5],
        )
        self.assertListEqual(
            (Tensor.array([2, 4, 6], datatype=Tensor.DataType.INT32) // 2).tolist(),
            [1, 2, 3],
        )

    def test_tensor_ge(self):
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                >= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [False, True, True],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) >= 2).tolist(),
            [False, True, True],
        )
        self.assertListEqual(
            (
                Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32) >= 2.0
            ).tolist(),
            [False, True, True],
        )

    def test_tensor_gt(self):
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                > Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [False, False, True],
        )

    def test_tensor_getitem(self):
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0:1]).tolist(),
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0:1].tolist(),
        )

    def test_tensor_float(self):
        self.assertEqual(
            float(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0]), 1.0
        )

    def test_tensor_iadd(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x += Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.tolist(),
            [2, 4, 6],
        )
        x += 1
        self.assertListEqual(
            x.tolist(),
            [3, 5, 7],
        )

    def test_tensor_int(self):
        self.assertEqual(
            int(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)[0]),
            1,
        )

    def test_tensor_imul(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x *= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.tolist(),
            [1, 4, 9],
        )
        x *= 2
        self.assertListEqual(
            x.tolist(),
            [2, 8, 18],
        )

    def test_tensor_ipow(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x **= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.tolist(),
            [1, 4, 27],
        )
        x **= 2
        self.assertListEqual(
            x.tolist(),
            [1, 16, 27 * 27],
        )

    def test_tensor_isub(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x -= Tensor.array([-1, -2, -3], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.tolist(),
            [2, 4, 6],
        )
        x -= 2
        self.assertListEqual(
            x.tolist(),
            [0, 2, 4],
        )

    def test_tensor_itruediv(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x /= Tensor.array([0.5, 1.0, 2.0], datatype=Tensor.DataType.FLOAT32)
        self.assertListEqual(
            x.tolist(),
            [2.0, 2.0, 1.5],
        )
        x /= 2
        self.assertListEqual(
            x.tolist(),
            [1.0, 1.0, 0.75],
        )

    def test_tensor_invert(self):
        self.assertListEqual(
            (
                ~Tensor.array([True, False, True], datatype=Tensor.DataType.BOOL)
            ).tolist(),
            [False, True, False],
        )

    def test_tensor_le(self):
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                <= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [True, True, False],
        )

    def test_tensor_lt(self):
        self.assertListEqual(
            (
                Tensor.array([-1, 2, 4], datatype=Tensor.DataType.INT32)
                < Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
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
            ).tolist(),
            [[[7, 10], [15, 22]], [[67, 78], [91, 106]]],
        )

    def test_tensor_mod(self):
        self.assertListEqual(
            (
                Tensor.array([17, 13, 11], datatype=Tensor.DataType.INT32)
                % Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [0, 1, 2],
        )
        self.assertListEqual(
            (Tensor.array([17, 13, 11], datatype=Tensor.DataType.INT32) % 4).tolist(),
            [1, 1, 3],
        )

    def test_tensor_mul(self):
        self.assertListEqual(
            (
                Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
                * Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [1, 4, 9],
        )

    def test_tensor_ne(self):
        self.assertListEqual(
            (
                Tensor.array([2, 2, 3], datatype=Tensor.DataType.INT32)
                != Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [True, False, False],
        )

    def test_tensor_pow(self):
        self.assertListEqual(
            (
                Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
                ** Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [1, 4, 27],
        )

    def test_tensor_radd(self):
        self.assertListEqual(
            (1 + Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [2, 3, 4],
        )

    def test_tensor_req(self):
        self.assertListEqual(
            (1 == Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                1.0 == Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).tolist(),
            [True, False, False],
        )

    def test_tensor_rge(self):
        self.assertListEqual(
            (2 >= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [True, True, False],
        )
        self.assertListEqual(
            (
                2.0 >= Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).tolist(),
            [True, True, False],
        )

    def test_tensor_rgt(self):
        self.assertListEqual(
            (2 > Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [True, False, False],
        )
        self.assertListEqual(
            (
                2.0 > Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).tolist(),
            [True, False, False],
        )

    def test_tensor_rle(self):
        self.assertListEqual(
            (2 <= Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [False, True, True],
        )
        self.assertListEqual(
            (
                2.0 <= Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).tolist(),
            [False, True, True],
        )

    def test_tensor_rlt(self):
        self.assertListEqual(
            (2 < Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [False, False, True],
        )
        self.assertListEqual(
            (
                2.0 < Tensor.array([1.0, 2.0, 3.0], datatype=Tensor.DataType.FLOAT32)
            ).tolist(),
            [False, False, True],
        )

    def test_tensor_rfloordiv(self):
        self.assertListEqual(
            (128 // Tensor.array([2, 4, 6], datatype=Tensor.DataType.INT32)).tolist(),
            [64, 32, 21],
        )

    def test_tensor_rmod(self):
        self.assertListEqual(
            (17 % Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [0, 1, 2],
        )

    def test_tensor_rmul(self):
        self.assertListEqual(
            (17 * Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [17, 34, 51],
        )

    def test_tensor_rsub(self):
        self.assertListEqual(
            (17 - Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [16, 15, 14],
        )

    def test_tensor_rtruediv(self):
        self.assertListEqual(
            (21 / Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)).tolist(),
            [21.0, 10.5, 7.0],
        )

    def test_tensor_setitem(self):
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x[0] = 0
        self.assertListEqual(
            x.tolist(),
            [0, 2, 3],
        )
        x: Tensor = Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
        x[0:2] = Tensor.array([-1, -2], datatype=Tensor.DataType.INT32)
        self.assertListEqual(
            x.tolist(),
            [-1, -2, 3],
        )

    def test_tensor_str(self):
        self.assertEqual(
            str(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)),
            "Array([1, 2, 3], dtype=int32)",
        )

    def test_tensor_sub(self):
        self.assertListEqual(
            (
                Tensor.array([3, 2, 1], datatype=Tensor.DataType.INT32)
                - Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [2, 0, -2],
        )
        self.assertListEqual(
            (Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32) - 1).tolist(),
            [0, 1, 2],
        )

    def test_tensor_truediv(self):
        self.assertListEqual(
            (
                Tensor.array([3, 2, 3], datatype=Tensor.DataType.INT32)
                / Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            ).tolist(),
            [3.0, 1.0, 1.0],
        )
        self.assertListEqual(
            (Tensor.array([17, 13, 11], datatype=Tensor.DataType.INT32) / 2.0).tolist(),
            [8.5, 6.5, 5.5],
        )

    def test_all(self):
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .all(axis=0)
            .tolist(),
            [True, False],
        )
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .all(axis=1)
            .tolist(),
            [False, False],
        )

    def test_any(self):
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .any(axis=0)
            .tolist(),
            [True, False],
        )
        self.assertListEqual(
            Tensor.array([[True, False], [True, False]], datatype=Tensor.DataType.BOOL)
            .any(axis=1)
            .tolist(),
            [True, True],
        )

    def test_arange(self):
        self.assertListEqual(
            Tensor.arange(
                start=0, stop=3, step=1, datatype=Tensor.DataType.INT16
            ).tolist(),
            [0, 1, 2],
        )

    def test_argmax(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argmax(axis=0)
            .tolist(),
            [1, 1],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argmax(axis=1)
            .tolist(),
            [1, 0],
        )

    def test_argsort(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argsort(axis=0)
            .tolist(),
            [[0, 0], [1, 1]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .argsort(axis=1)
            .tolist(),
            [[0, 1], [1, 0]],
        )

    def test_array(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).tolist(),
            [1, 2, 3],
        )

    def test_astype(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .astype(Tensor.DataType.FLOAT16)
            .tolist(),
            [[1.0, 2.0], [3.0, 4.0]],
        )

    def test_concatenate(self):
        self.assertListEqual(
            Tensor.concatenate(
                tensors=[
                    Tensor.array([1, 2], datatype=Tensor.DataType.INT32),
                    Tensor.array([4, 3], datatype=Tensor.DataType.INT32),
                ],
                axis=0,
            ).tolist(),
            [1, 2, 4, 3],
        )

    def test_count_nonzero(self):
        self.assertListEqual(
            Tensor.array([[1, 0], [3, 0]], datatype=Tensor.DataType.INT32)
            .count_nonzero(axis=0, keep_dimension=False)
            .tolist(),
            [2, 0],
        )
        self.assertListEqual(
            Tensor.array([[1, 0], [3, 0]], datatype=Tensor.DataType.INT32)
            .count_nonzero(axis=1, keep_dimension=False)
            .tolist(),
            [1, 1],
        )

    def test_datatype(self):
        self.assertEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).datatype,
            Tensor.DataType.INT32,
        )

    def test_diagonal(self):
        self.assertListEqual(
            Tensor.diagonal(value=1, size=2, datatype=Tensor.DataType.INT16).tolist(),
            [[1, 0], [0, 1]],
        )

    def test_exp(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).exp().tolist(),
            [2.7182817459106445, 7.389056205749512, 20.08553695678711],
        )

    def test_expand_dimension(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32)
            .expand_dimension(-1)
            .tolist(),
            [[1], [2], [3]],
        )

    def test_full(self):
        self.assertListEqual(
            Tensor.full(value=1, shape=(2, 2), datatype=Tensor.DataType.INT16).tolist(),
            [[1, 1], [1, 1]],
        )
        self.assertListEqual(
            Tensor.full(
                value=Tensor.Value.NAN, shape=(2, 2), datatype=Tensor.DataType.INT16
            ).tolist(),
            [[0, 0], [0, 0]],
        )

    def test_get_sample_x_and_y(self):
        (
            sample_x,
            sample_y,
        ) = Tensor.array(
            [[1, 2], [3, 4]], datatype=Tensor.DataType.INT32
        ).get_sample_x_and_y(
            number_of_target=1,
        )
        self.assertListEqual(
            sample_x.tolist(),
            [[2], [4]],
        )
        self.assertListEqual(
            sample_y.tolist(),
            [[1], [3]],
        )

    def test_get_by_index(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .get_by_index(indexes=Tensor.array([0, 1], datatype=Tensor.DataType.INT32))
            .tolist(),
            [1, 4],
        )
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .get_by_index(indexes=Tensor.array([0, 1], datatype=Tensor.DataType.INT32))
            .tolist(),
            [1.0, 4.0],
        )

    def test_in_sample(self):
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .in_sample(in_sample_size=1)
            .tolist(),
            [[1.0, 2.0]],
        )

    def test_insert(self):
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .insert(index=0, value=0.0, axis=1)
            .tolist(),
            [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]],
        )
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .insert(index=0, value=0.0, axis=0)
            .tolist(),
            [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
        )

    def test_inverse(self):
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
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .isnan()
            .tolist(),
            [[False, False], [False, True]],
        )

    def test_log2(self):
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
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=0, keep_dimension=False)
            .tolist(),
            [2.5, 2.5],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=0, keep_dimension=True)
            .tolist(),
            [[2.5, 2.5]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=1, keep_dimension=False)
            .tolist(),
            [1.5, 3.5],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .mean(axis=1, keep_dimension=True)
            .tolist(),
            [[1.5], [3.5]],
        )

    def test_nanmean(self):
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .nanmean(axis=0, keep_dimension=False)
            .tolist(),
            [2.0, 2.0],
        )

    def test_nansum(self):
        self.assertListEqual(
            Tensor.array(
                [[1.0, 2.0], [3.0, jax.numpy.nan]], datatype=Tensor.DataType.FLOAT32
            )
            .nansum(axis=0, keep_dimension=False)
            .tolist(),
            [4.0, 2.0],
        )

    def test_nan_to_num(self):
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
            .tolist(),
            [[0.0, 2.0], [3.0, 1.0]],
        )

    def test_number_of_dimension(self):
        self.assertEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).number_of_dimension,
            1,
        )

    def test_out_of_sample(self):
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .out_of_sample(in_sample_size=1)
            .tolist(),
            [[3.0, 4.0]],
        )

    def test_random_integer(self):
        self.assertListEqual(
            Tensor.random_integer(
                minimum_value=0,
                maximum_value=10,
                shape=(2, 2),
                datatype=Tensor.DataType.INT16,
                seed=0,
            ).tolist(),
            [[8, 8], [0, 2]],
        )
        self.assertListEqual(
            Tensor.random_integer(
                minimum_value=0,
                maximum_value=10,
                shape=(2, 2),
                datatype=Tensor.DataType.INT16,
                seed=1,
            ).tolist(),
            [[4, 6], [2, 7]],
        )

    def test_random_uniform(self):
        self.assertListEqual(
            Tensor.random_uniform(
                minimum_value=0,
                maximum_value=10,
                shape=(2, 2),
                datatype=Tensor.DataType.FLOAT32,
                seed=0,
            ).tolist(),
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
            ).tolist(),
            [
                [5.685524940490723, 3.224560022354126],
                [8.89329719543457, 4.716912269592285],
            ],
        )

    def test_reshape(self):
        self.assertListEqual(
            Tensor.array([[1.0, 2.0], [3.0, 4.0]], datatype=Tensor.DataType.FLOAT32)
            .reshape(-1)
            .tolist(),
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_shape(self):
        self.assertListEqual(
            list(Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).shape),
            list((3,)),
        )

    def test_sign(self):
        self.assertListEqual(
            Tensor.array([-1, 2, -3], datatype=Tensor.DataType.INT32).sign().tolist(),
            [-1, 1, -1],
        )

    def test_sigmoid(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).sigmoid().tolist(),
            [0.2689414322376251, 0.11920291930437088, 0.04742587357759476],
        )

    def test_sliding_window(self):
        self.assertListEqual(
            Tensor.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]], datatype=Tensor.DataType.INT32
            )
            .sliding_window(window_size=2)
            .tolist(),
            [[[1, 2], [3, 4]], [[3, 4], [5, 6]], [[5, 6], [7, 8]]],
        )

    def test_sort(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .sort(axis=0)
            .tolist(),
            [[1, 2], [4, 3]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [4, 3]], datatype=Tensor.DataType.INT32)
            .sort(axis=1)
            .tolist(),
            [[1, 2], [3, 4]],
        )

    def test_sqrt(self):
        self.assertListEqual(
            Tensor.array([[1, 4], [9, 16]], datatype=Tensor.DataType.INT32)
            .sqrt()
            .tolist(),
            [[1, 2], [3, 4]],
        )

    def test_square(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .square()
            .tolist(),
            [[1, 4], [9, 16]],
        )

    def test_sum(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=0, keep_dimension=False)
            .tolist(),
            [4, 6],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=0, keep_dimension=True)
            .tolist(),
            [[4, 6]],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=1, keep_dimension=False)
            .tolist(),
            [3, 7],
        )
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .sum(axis=1, keep_dimension=True)
            .tolist(),
            [[3], [7]],
        )

    def test_swapaxes(self):
        self.assertListEqual(
            Tensor.array([[1, 2], [3, 4]], datatype=Tensor.DataType.INT32)
            .swapaxes(axis1=0, axis2=1)
            .tolist(),
            [[1, 3], [2, 4]],
        )

    def test_tolist(self):
        self.assertListEqual(
            Tensor.array([1, 2, 3], datatype=Tensor.DataType.INT32).tolist(),
            [1, 2, 3],
        )

    def test_unique(self):
        self.assertListEqual(
            Tensor.array(
                [[1, 2, 3], [1, 2, 3], [4, 5, 6]], datatype=Tensor.DataType.INT32
            )
            .unique()
            .tolist(),
            [1, 2, 3, 4, 5, 6],
        )


if __name__ == "__main__":
    unittest.main()
