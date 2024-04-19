"""
This module contains classes that represent data preprocessing steps on a HistoricalDataFrame.
"""
import typing

import jax
import jaxlib.xla_extension
import polars

from hoisaai.layer_1.model import (
    HistoricalData,
    HistoricalDataFrame,
    StatefulTensor,
    get_dataframe,
)


class DropNull(HistoricalDataFrame):
    """
    A class that represents a data preprocessing step to drop null values from a dataframe.

    Args:
        dataframe (HistoricalDataFrame): The input dataframe to be processed.

    Attributes:
        dataframe (HistoricalDataFrame): The input dataframe to be processed.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform: Applies the drop null transformation to the dataframe.

    """

    def __init__(
        self,
        dataframe: HistoricalDataFrame,
    ) -> None:
        super().__init__(
            datetime=dataframe.datetime,
        )
        self.dataframe: HistoricalDataFrame = dataframe

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Applies the drop null transformation to the dataframe.

        Yields:
            polars.DataFrame: The processed dataframe with null values dropped.

        """
        if self.df is None or self.stateful is False:
            df: polars.DataFrame = get_dataframe(
                dataframe=self.dataframe,
            )
            self.df: polars.DataFrame = df.drop_nulls()
        yield self.df


class Join(HistoricalDataFrame):
    """
    A class representing a join operation on multiple historical dataframes.

    Args:
        dataframes (Dict[str, HistoricalData]): A dictionary of historical dataframes to join.

    Attributes:
        dataframes (Dict[str, HistoricalData]): A dictionary of historical dataframes to join.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform(): Performs the join operation and returns an iterator of the resulting dataframe.

    """

    def __init__(
        self,
        dataframes: typing.Dict[str, HistoricalData],
    ) -> None:
        datetime_column_names: typing.List[str] = list(
            set([df.datetime.datetime_column_name for df in dataframes.values()])
        )
        assert len(datetime_column_names) == 1
        super().__init__(
            datetime=list(dataframes.values())[0].datetime,
        )
        self.dataframes: typing.Dict[str, HistoricalData] = dataframes

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the dataframes by renaming columns and joining them based on a datetime column.

        Returns:
            An iterator that yields the transformed dataframe.

        """
        if self.df is None or self.stateful is False:
            dfs: typing.List[polars.DataFrame] = []
            for dataframe_label, dataframe in self.dataframes.items():
                df: polars.DataFrame = get_dataframe(
                    dataframe=dataframe,
                )
                dfs.append(
                    df.rename(
                        mapping={
                            column_name: f"{dataframe_label} : {column_name}"
                            for column_name in df.columns
                            if column_name != self.datetime.datetime_column_name
                        },
                    )
                )
            self.df: polars.DataFrame = dfs[0]
            for df in dfs[1:]:
                self.df = self.df.join(
                    df,
                    on=self.datetime.datetime_column_name,
                    how="outer_coalesce",
                )
                yield self.df
        yield self.df


class PercentageChange(HistoricalDataFrame):
    """
    A class that calculates the percentage change of specified columns in a HistoricalDataFrame.

    Args:
        dataframe (HistoricalDataFrame): The input HistoricalDataFrame.
        *column_names (str): Variable number of column names to calculate the percentage change for.

    Attributes:
        column_names (List[str]): List of column names to calculate the percentage change for.
        dataframe (HistoricalDataFrame): The input HistoricalDataFrame.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform():
            Calculates the percentage change for the specified columns
            and yields the resulting DataFrame.

    """

    def __init__(
        self,
        dataframe: HistoricalDataFrame,
        *column_names: str,
    ) -> None:
        super().__init__(
            datetime=dataframe.datetime,
        )
        self.column_names: typing.List[str] = column_names
        self.dataframe: HistoricalDataFrame = dataframe

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Calculates the percentage change for the specified columns
        and yields the resulting DataFrame.

        Yields:
            polars.DataFrame:
                The DataFrame with the calculated percentage change for the specified columns.

        """
        if self.df is None or self.stateful is False:
            df: polars.DataFrame = get_dataframe(
                dataframe=self.dataframe,
            )
            self.df: polars.DataFrame = df.with_columns(
                polars.col(self.column_names).pct_change() * 100.0,
            )[
                # Remove the first row
                1:
            ]
        yield self.df


class Select(HistoricalDataFrame):
    """
    A class that represents a selection operation on a HistoricalDataFrame.

    Args:
        dataframe (HistoricalDataFrame): The input HistoricalDataFrame.
        *column_names (str): Variable number of column names to select.

    Attributes:
        column_names (List[str]): List of column names to select.
        dataframe (HistoricalDataFrame): The input HistoricalDataFrame.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform(): Performs the selection operation and yields the resulting DataFrame.

    """

    def __init__(
        self,
        dataframe: HistoricalDataFrame,
        *column_names: str,
    ) -> None:
        super().__init__(
            datetime=dataframe.datetime,
        )
        self.column_names: typing.List[str] = column_names
        self.dataframe: HistoricalDataFrame = dataframe

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Performs the selection operation on the input HistoricalDataFrame.

        Yields:
            typing.Any: The resulting DataFrame after the selection operation.

        """
        if self.df is None or self.stateful is False:
            df: polars.DataFrame = get_dataframe(
                dataframe=self.dataframe,
            )
            self.df: polars.DataFrame = df.select(
                polars.col(
                    [
                        self.datetime.datetime_column_name,
                        *self.column_names,
                    ],
                ),
            )
        yield self.df


class Subtract(HistoricalDataFrame):
    """
    A class that represents a data transformation operation to subtract a column
    from one or more columns in a dataframe.

    Args:
        dataframe (HistoricalDataFrame): The input dataframe.
        minuend_column_name (str | typing.List[str]):
            The name(s) of the column(s) to be subtracted from.
        subtrahend_column_name (str): The name of the column to subtract.

    Attributes:
        dataframe (HistoricalDataFrame): The input dataframe.
        df (polars.DataFrame): The underlying data frame.
        minuend_column_name (str): The name of the column(s) to be subtracted from.
        stateful (bool): Indicates whether the model is stateful or not.
        subtrahend_column_name (str): The name of the column to subtract.

    Methods:
        transform(): Performs the subtraction operation and yields the resulting dataframe.

    """

    def __init__(
        self,
        dataframe: HistoricalDataFrame,
        minuend_column_name: str | typing.List[str],
        subtrahend_column_name: str,
    ) -> None:
        super().__init__(
            datetime=dataframe.datetime,
        )
        self.dataframe: HistoricalDataFrame = dataframe
        self.minuend_column_name: str = minuend_column_name
        self.subtrahend_column_name: str = subtrahend_column_name

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Performs the subtraction operation on the dataframe.

        Yields:
            typing.Any: The resulting dataframe after the subtraction operation.

        """
        if self.df is None or self.stateful is False:
            df: polars.DataFrame = get_dataframe(
                dataframe=self.dataframe,
            )
            self.df: polars.DataFrame = df.with_columns(
                [
                    (
                        polars.col(column_name)
                        - polars.col(self.subtrahend_column_name)
                    ).alias(f"{column_name} - {self.subtrahend_column_name}")
                    for column_name in self.minuend_column_name
                ]
            )
        yield self.df


class ToTensor(StatefulTensor):
    """
    Converts a dataframe to a tensor.

    Args:
        dataframe (HistoricalDataFrame): The input dataframe to be converted.

    Attributes:
        dataframe (HistoricalDataFrame): The input HistoricalDataFrame.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Returns:
        Iterator: An iterator that yields the converted tensor.
    """

    def __init__(
        self,
        dataframe: HistoricalDataFrame,
    ) -> None:
        super().__init__(
            x=None,
        )
        self.dataframe: HistoricalDataFrame = dataframe

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the data and yields the transformed data.

        Returns:
            An iterator that yields the transformed data.
        """
        if self.x is None or self.stateful is False:
            df: polars.DataFrame = get_dataframe(
                dataframe=self.dataframe,
            )
            self.x: jaxlib.xla_extension.ArrayImpl = jax.numpy.array(
                df.drop(
                    # Drop the datetime column
                    self.dataframe.datetime.datetime_column_name,
                ).to_numpy()
            )
        yield self.x
