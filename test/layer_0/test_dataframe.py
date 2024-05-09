"""Test the DataFrame and Series classes."""

import datetime
import unittest

import polars

from hoisaai.layer_0.dataframe import DataFrame, Series
from hoisaai.layer_0.tensor import Tensor


class TestDataFrame(unittest.TestCase):
    """Test the DataFrame class."""

    def setUp(self):
        """Set up a DataFrame for testing."""
        self.df: DataFrame = DataFrame(
            df=polars.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [4, 5, 6],
                }
            )
        )

    def test_init(self):
        """Test the initialization of a DataFrame."""
        self.assertListEqual(self.df["a"].to_list(), [1, 2, 3])
        self.assertListEqual(self.df["b"].to_list(), [4, 5, 6])

    def test_getitem(self):
        """Test the __getitem__ method of a DataFrame."""
        # pylint: disable=unsubscriptable-object
        # slice -> DataFrame
        self.assertListEqual(self.df[1:]["a"].to_list(), [2, 3])
        self.assertListEqual(self.df[:-1]["b"].to_list(), [4, 5])
        # str -> Series
        self.assertListEqual(self.df["a"].to_list(), [1, 2, 3])
        self.assertListEqual(self.df["b"].to_list(), [4, 5, 6])
        # typing.List[str] -> DataFrame
        self.assertListEqual(self.df[["a", "b"]]["a"].to_list(), [1, 2, 3])
        self.assertListEqual(self.df[["a", "b"]]["b"].to_list(), [4, 5, 6])
        self.assertListEqual(self.df[["a"]]["a"].to_list(), [1, 2, 3])
        self.assertListEqual(self.df[["b"]]["b"].to_list(), [4, 5, 6])
        # typing.Tuple[int, int] -> typing.Any
        self.assertEqual(self.df[0, 0], 1)
        self.assertEqual(self.df[1, 1], 5)
        # typing.Tuple[int, slice] -> DataFrame
        self.assertEqual(self.df[0, :]["a"].to_list(), [1])
        self.assertEqual(self.df[1, :]["b"].to_list(), [5])
        # typing.Tuple[slice, int] -> Series
        self.assertEqual(self.df[:, 0].to_list(), [1, 2, 3])
        self.assertEqual(self.df[:, 1].to_list(), [4, 5, 6])
        # typing.Tuple[slice, slice] -> DataFrame
        self.assertListEqual(self.df[1:, :]["a"].to_list(), [2, 3])
        self.assertListEqual(self.df[:-1, :]["b"].to_list(), [4, 5])

    def test_setitem(self):
        """Test the __setitem__ method of a DataFrame."""
        self.df["c"] = Series(series=polars.Series([7, 8, 9]))
        self.assertListEqual(self.df["c"].to_list(), [7, 8, 9])

    def test_clone(self):
        """Test the clone property of a DataFrame."""
        df: DataFrame = self.df.clone
        self.assertListEqual(df["a"].to_list(), [1, 2, 3])
        self.assertListEqual(df["b"].to_list(), [4, 5, 6])

    def test_drop(self):
        """Test the drop method of a DataFrame."""
        self.df.drop(columns=["a"])
        self.assertListEqual(self.df["a"].to_list(), [1, 2, 3])
        tmp: DataFrame = self.df.drop(columns=["a"], inplace=False)
        self.assertListEqual(self.df["a"].to_list(), [1, 2, 3])
        with self.assertRaises(Exception) as _:
            tmp["a"].to_list()
        self.df.drop(columns=["a"], inplace=True)
        with self.assertRaises(Exception) as _:
            self.df["a"].to_list()
        self.assertListEqual(self.df["b"].to_list(), [4, 5, 6])

    def test_drop_null(self):
        """Test the drop_null method of a DataFrame."""
        df: DataFrame = DataFrame(
            df=polars.DataFrame(
                {
                    "a": [1, None, 3],
                    "b": [4, 5, None],
                }
            )
        )
        self.assertListEqual(df["a"].to_list(), [1, None, 3])
        df.drop_null()
        self.assertListEqual(df["a"].to_list(), [1, None, 3])
        tmp: DataFrame = df.drop_null(inplace=False)
        self.assertListEqual(df["a"].to_list(), [1, None, 3])
        self.assertListEqual(tmp["a"].to_list(), [1])
        df.drop_null(inplace=True)
        self.assertListEqual(df["a"].to_list(), [1])

    def test_from_list(self):
        """Test the from_list method of a DataFrame."""
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        )
        self.assertListEqual(df["a"].to_list(), [1, 2, 3])
        self.assertListEqual(df["b"].to_list(), [4, 5, 6])

    def test_join(self):
        """Test the join method of a DataFrame."""
        df: DataFrame = DataFrame.join(
            dataframes=[
                DataFrame.from_list(
                    data=[[1, 4], [2, 5], [3, 6]],
                    schema={
                        "a": DataFrame.DataType.INT32.value,
                        "b": DataFrame.DataType.INT32.value,
                    },
                ),
                DataFrame.from_list(
                    data=[[7, 1], [8, 2], [9, 3]],
                    schema={
                        "c": DataFrame.DataType.INT32.value,
                        "a": DataFrame.DataType.INT32.value,
                    },
                ),
            ],
            on="a",
        )
        self.assertListEqual(df["a"].to_list(), [1, 2, 3])
        self.assertListEqual(df["b"].to_list(), [4, 5, 6])
        self.assertListEqual(df["c"].to_list(), [7, 8, 9])

    def test_tensor(self):
        """Test the tensor property of a DataFrame."""
        tensor: Tensor = self.df.tensor
        self.assertListEqual(tensor.to_list(), [[1, 4], [2, 5], [3, 6]])
        self.assertFalse(tensor.require_gradient)


class TestSeries(unittest.TestCase):
    """Test the Series class."""

    def test_series_casting(self):
        self.assertEqual(
            Series.series_casting(
                object=Series(series=polars.Series([1, 2, 3])),
            ).__class__,
            polars.Series,
        )
        self.assertEqual(
            Series.series_casting(
                object=1,
            ).__class__,
            int,
        )
        self.assertEqual(
            Series.series_casting(
                object=1.0,
            ).__class__,
            float,
        )

    def test_add(self):
        """Test the __add__ method of a Series."""
        self.assertEqual(
            (
                Series(series=polars.Series([1, 2, 3]))
                + Series(
                    series=polars.Series([4, 5, 6]),
                )
            ).to_list(),
            [5, 7, 9],
        )

    def test_mul(self):
        """Test the __mul__ method of a Series."""
        self.assertEqual(
            (
                Series(series=polars.Series([1, 2, 3]))
                * Series(
                    series=polars.Series([4, 5, 6]),
                )
            ).to_list(),
            [4, 10, 18],
        )

    def test_sub(self):
        """Test the __sub__ method of a Series."""
        self.assertEqual(
            (
                Series(series=polars.Series([1, 2, 3]))
                - Series(
                    series=polars.Series([4, 5, 6]),
                )
            ).to_list(),
            [-3, -3, -3],
        )

    def test_truediv(self):
        """Test the __truediv__ method of a Series."""
        self.assertEqual(
            (
                Series(series=polars.Series([1, 2, 3]))
                / Series(
                    series=polars.Series([4, 5, 6]),
                )
            ).to_list(),
            [0.25, 0.4, 0.5],
        )

    def test_to_datetime(self):
        """Test the to_datetime method of a Series."""
        self.assertListEqual(
            Series(series=polars.Series(["2001-11-18", "2000-10-7"]))
            .to_datetime(format="%Y-%m-%d")
            .to_list(),
            [
                datetime.datetime(2001, 11, 18, 0, 0),
                datetime.datetime(2000, 10, 7, 0, 0),
            ],
        )

    def test_to_float(self):
        """Test the to_float method of a Series."""
        self.assertListEqual(
            Series(series=polars.Series([1, 2, 3]))
            .to_float(datatype=DataFrame.DataType.FLOAT32)
            .to_list(),
            [1.0, 2.0, 3.0],
        )
        self.assertListEqual(
            Series(series=polars.Series(["1", "2", "3"]))
            .to_float(datatype=DataFrame.DataType.FLOAT32)
            .to_list(),
            [1.0, 2.0, 3.0],
        )

    def test_to_str(self):
        """Test the to_str method of a Series."""
        self.assertListEqual(
            Series(series=polars.Series([1, 2, 3])).to_str().to_list(),
            ["1", "2", "3"],
        )

    def test_percentage_change(self):
        """Test the percentage_change method of a Series."""
        self.assertListEqual(
            Series(series=polars.Series([1, 2, 3])).percentage_change().to_list(),
            [None, 1.0, 0.5],
        )


if __name__ == "__main__":
    unittest.main()
