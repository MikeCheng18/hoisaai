import unittest


import polars

from hoisaai.layer_0.dataframe import DataFrame, Series
from hoisaai.layer_0.tensor import Tensor


class TestSeries(unittest.TestCase):
    def test_series_init(self):
        series = Series(series=polars.Series([1, 2, 3]))
        self.assertTrue(series is not None)

    def test_tolist(self):
        series = Series(series=polars.Series([1, 2, 3]))
        self.assertListEqual(series.tolist(), [1, 2, 3])


class TestDataFrame(unittest.TestCase):
    def test_dataframe_init(self):
        df: DataFrame = DataFrame(
            df=polars.DataFrame(
                {
                    "a": [1, 2, 3],
                    "b": [4, 5, 6],
                }
            )
        )
        self.assertListEqual(df["a"].tolist(), [1, 2, 3])

    def test_drop_null(self):
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, None], [3, None]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        ).drop_null()
        self.assertListEqual(df["a"].tolist(), [1])

    def test_from_list(self):
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        )
        self.assertListEqual(df["a"].tolist(), [1, 2, 3])

    def test_join(self):
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
        self.assertListEqual(df["a"].tolist(), [1, 2, 3])
        self.assertListEqual(df["b"].tolist(), [4, 5, 6])
        self.assertListEqual(df["c"].tolist(), [7, 8, 9])

    def test_percent_change(self):
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        ).percent_change(
            columns=["a"],
            inplace=False,
        )
        self.assertListEqual(df["a"].tolist(), [100, 50])
        self.assertListEqual(df["b"].tolist(), [5, 6])
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        )
        df.percent_change(
            columns=["b"],
            inplace=True,
        )
        self.assertListEqual(df["a"].tolist(), [2, 3])
        self.assertListEqual(df["b"].tolist(), [25, 20])

    def test_rename(self):
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        ).rename(
            mapping={"a": "c"},
            inplace=False,
        )
        with self.assertRaises(polars.exceptions.ColumnNotFoundError):
            df["a"].tolist()
        self.assertListEqual(df["b"].tolist(), [4, 5, 6])
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        )
        df.rename(
            mapping={"a": "c"},
            inplace=True,
        )
        with self.assertRaises(polars.exceptions.ColumnNotFoundError):
            df["a"].tolist()
        self.assertListEqual(df["b"].tolist(), [4, 5, 6])

    def test_select(self):
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        ).select("b", inplace=False)
        with self.assertRaises(polars.exceptions.ColumnNotFoundError):
            df["a"].tolist()
        self.assertListEqual(df["b"].tolist(), [4, 5, 6])
        df: DataFrame = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        )
        df.select("b", inplace=True)
        with self.assertRaises(polars.exceptions.ColumnNotFoundError):
            df["a"].tolist()
        self.assertListEqual(df["b"].tolist(), [4, 5, 6])

    def test_tensor(self):
        x: Tensor = DataFrame.from_list(
            data=[[1, 4], [2, 5], [3, 6]],
            schema={
                "a": DataFrame.DataType.INT32.value,
                "b": DataFrame.DataType.INT32.value,
            },
        ).tensor
        self.assertListEqual(x.tolist(), [[1, 4], [2, 5], [3, 6]])


if __name__ == "__main__":
    unittest.main()
