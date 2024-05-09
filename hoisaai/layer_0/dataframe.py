"""
This module contains the DataFrame and Series classes that are used to interact with the panel data.
"""

from __future__ import annotations

import enum
import typing

import polars

from hoisaai.layer_0.tensor import Tensor


class DataFrame(object):
    """This is a class that represents a DataFrame of data.

    :param df: A DataFrame of data.
    :type df: class:`polars.DataFrame`
    """

    class DataType(enum.Enum):
        """This is an enumeration class that represents the data types of the DataFrame."""

        DATETIME: polars.DataType = polars.Datetime(time_unit="us")
        """Represents the datetime data type."""
        FLOAT32: polars.DataType = polars.Float32
        """Represents the 32-bit floating-point data type."""
        FLOAT64: polars.DataType = polars.Float64
        """Represents the 64-bit floating-point data type."""
        INT16: polars.DataType = polars.Int16
        """Represents the 16-bit integer data type."""
        INT32: polars.DataType = polars.Int32
        """Represents the 32-bit integer data type."""
        INT64: polars.DataType = polars.Int64
        """Represents the 64-bit integer data type."""

    def __init__(
        self,
        df: polars.DataFrame,
    ) -> None:
        self.df: polars.DataFrame = df

    @typing.overload
    def __getitem__(self, key: slice) -> DataFrame: ...
    @typing.overload
    def __getitem__(self, key: str) -> Series: ...
    @typing.overload
    def __getitem__(self, key: typing.List[str]) -> DataFrame: ...
    @typing.overload
    def __getitem__(
        self, key: typing.Tuple[int | slice]
    ) -> (
        # typing.Tuple[int, slice]
        # typing.Tuple[slice, slice]
        DataFrame
        # typing.Tuple[slice, int]
        | Series
        # typing.Tuple[int, int]
        | typing.Any
    ): ...
    def __getitem__(
        self, key: slice | str | typing.List[str] | typing.Tuple[int | slice]
    ) -> DataFrame | Series | typing.Any:
        if key.__class__ == list:
            # typing.List[str] -> DataFrame
            return DataFrame(
                df=self.df.select(
                    polars.col(key),
                )
            )
        elif key.__class__ == slice:
            # slice -> DataFrame
            return DataFrame(
                df=self.df[key],
            )
        elif key.__class__ == str:
            # str -> Series
            return Series(
                series=self.df[key],
            )
        elif key.__class__ == tuple:
            assert len(key) == 2, "The key must be a tuple of two integers or slice."
            if key[0].__class__ == int and key[1].__class__ == int:
                # typing.Tuple[int, int] -> typing.Any
                return self.df[key[0], key[1]]
            elif key[0].__class__ == int and key[1].__class__ == slice:
                # typing.Tuple[int, slice] -> DataFrame
                return DataFrame(
                    df=self.df[key[0], key[1]],
                )
            elif key[0].__class__ == slice and key[1].__class__ == int:
                # typing.Tuple[slice, int] -> Series
                return Series(
                    series=self.df[key[0], key[1]],
                )
            elif key[0].__class__ == slice and key[1].__class__ == slice:
                # typing.Tuple[slice, slice] -> DataFrame
                return DataFrame(
                    df=self.df[key[0], key[1]],
                )

    def __setitem__(self, name: str, value: Series) -> None:
        self.df: polars.DataFrame = self.df.with_columns(
            **{
                name: value.series,
            }
        )

    def __str__(self) -> str:
        return self.df.__str__()

    def _repr_html_(self) -> str:
        # pylint: disable=protected-access
        return self.df._repr_html_()

    @property
    def clone(self):
        """Returns a clone of the DataFrame.

        :return: A clone of the DataFrame.
        :rtype: DataFrame
        """
        return DataFrame(
            df=self.df.clone(),
        )

    def drop(
        self, columns: typing.List[str], inplace: bool = False
    ) -> DataFrame | None:
        """Removes the specified columns from the DataFrame.

        :param columns: The columns to be removed.
        :type columns: List[str]
        :param inplace: A boolean value that indicates whether the operation is done in place.
        :type inplace: bool, optional

        :return: Returns the DataFrame with the specified columns removed.
        :rtype: DataFrame, None
        """
        df: polars.DataFrame = self.df.drop(columns)
        if inplace is True:
            self.df = df
            return self
        else:
            return DataFrame(
                df=df,
            )

    def drop_null(
        self,
        inplace: bool = False,
    ) -> DataFrame | None:
        """Removes the rows with null values from the DataFrame.

        :param inplace: A boolean value that indicates whether the operation is done in place.
        :type inplace: bool, optional

        :return: Returns the DataFrame with the rows with null values removed.
        :rtype: DataFrame, None
        """
        df: polars.DataFrame = self.df.drop_nulls()
        if inplace is True:
            self.df = df
            return self
        else:
            return DataFrame(
                df=df,
            )

    @staticmethod
    def from_list(
        data: typing.List[typing.List[typing.Any]],
        schema: typing.Dict[str, DataFrame.DataType],
    ) -> DataFrame:
        """Creates a DataFrame from a list of data.

        :param data: The list of data.
        :type data: List[List[Any]]
        :param schema: The schema of the data.
        :type schema: Dict[str, :class`DataFrame.DataType`]

        :return: Returns the DataFrame created from the list of data.
        :rtype: DataFrame
        """
        return DataFrame(
            df=polars.DataFrame(
                data=data,
                schema=schema,
            ),
        )

    @staticmethod
    def join(
        dataframes: typing.List[DataFrame],
        on: str,
    ) -> DataFrame:
        """Joins the DataFrames on the specified column.

        :param dataframes: The DataFrames to be joined.
        :type dataframes: List[DataFrame]
        :param on: The column to join on.
        :type on: str

        :return: Returns the joined DataFrame.
        :rtype: DataFrame
        """
        df: polars.DataFrame = dataframes[0].df
        for other_df in dataframes[1:]:
            df = df.join(
                other_df.df,
                on=on,
                how="outer_coalesce",
            )
        return DataFrame(
            df=df,
        )

    @staticmethod
    def read_csv(
        source: str,
        skip_rows: int = 0,
    ) -> DataFrame:
        """Reads a CSV file and returns a DataFrame.

        :param source: The source of the CSV file.
        :type source: str
        :param skip_rows: The number of rows to skip.
        :type skip_rows: int, optional

        :return: Returns the DataFrame read from the CSV file.
        :rtype: DataFrame
        """
        return DataFrame(
            df=polars.read_csv(
                source=source,
                skip_rows=skip_rows,
            )
        )

    @property
    def tensor(self) -> Tensor:
        """Returns the DataFrame as a tensor.

        :return: Returns the DataFrame as a tensor.
        :rtype: Tensor
        """
        return Tensor(
            x=self.df.select(
                polars.selectors.by_dtype(
                    polars.Float32,
                    polars.Float64,
                    polars.Int8,
                    polars.Int16,
                    polars.Int32,
                    polars.Int64,
                )
            ).to_numpy(),
            require_gradient=False,
        )

    def to_csv(
        self,
        destination: str,
    ) -> None:
        """Writes the DataFrame to a CSV file.

        :param destination: The destination of the CSV file.
        :type destination: str
        """
        self.df.write_csv(file=destination)


class Series(object):
    """This is a class that represents a series of data for :class:`hoisaai.layer_0.dataframe.DataFrame`.

    :param series: A series of data.
    :type series: class:`polars.Series`
    """

    @staticmethod
    def series_casting(
        # pylint: disable=redefined-builtin
        object: Series | int | float,
    ) -> polars.Series | int | float:
        """Casts the object to a polars.Series or int or float.

        :param object:  The object to be casted.
        :type object: polars.Series | int | float

        :return: Returns the object casted to a polars.Series or int or float.
        :rtype: polars.Series | int | float
        """
        if object.__class__ == Series:
            return object.series
        return object

    def __init__(
        self,
        series: polars.Series,
    ) -> None:
        self.series: polars.Series = series

    def _repr_html_(self) -> str:
        # pylint: disable=protected-access
        return self.series._repr_html_()

    def __add__(self, other: Series | int | float) -> Series:
        return Series(
            series=self.series + Series.series_casting(object=other),
        )

    def __mul__(self, other: Series | int | float) -> Series:
        return Series(
            series=self.series * Series.series_casting(object=other),
        )

    def __sub__(self, other: Series | int | float) -> Series:
        return Series(
            series=self.series - Series.series_casting(object=other),
        )

    def __truediv__(self, other: Series | int | float) -> Series:
        return Series(
            series=self.series / Series.series_casting(object=other),
        )

    def to_datetime(
        self,
        # pylint: disable=redefined-builtin
        format: str,
    ) -> Series:
        """Converts the series to a datetime series.

        :param format: The format of the datetime.
        :type format: str

        :return: Returns the series converted to a datetime series.
        :rtype: Series
        """
        assert self.series.dtype == polars.Utf8, "The series must be of type Utf8."
        return Series(
            series=self.series.str.to_datetime(format=format).cast(
                DataFrame.DataType.DATETIME.value
            ),
        )

    def to_float(
        self,
        datatype: DataFrame.DataType,
    ) -> Series:
        """Converts the series to a float series.

        :param datatype: The data type of the float series.
        :type datatype: DataFrame.DataType

        :return: Returns the series converted to a float series.
        :rtype: Series
        """
        if self.series.dtype == polars.Utf8:
            return Series(
                series=self.series.str.strip().cast(datatype.value),
            )
        return Series(
            series=self.series.cast(datatype.value),
        )

    def to_list(self) -> typing.List[typing.Any]:
        """Converts the series to a list.

        :return: Returns the series converted to a list.
        :rtype: List[Any]"""
        return self.series.to_list()

    def to_str(self) -> Series:
        """Converts the series to a string series.

        :return: Returns the series converted to a string series.
        :rtype: Series
        """
        return Series(
            series=self.series.cast(polars.Utf8),
        )

    def percentage_change(self) -> Series:
        """Calculates the percentage change of the series.

        :return: Returns the percentage change of the series.
        :rtype: Series
        """
        return Series(
            series=self.series.pct_change(),
        )
