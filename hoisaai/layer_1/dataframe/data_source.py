"""
This module contains the data source classes for reading data.
"""
import typing

import polars

from hoisaai.layer_1.model import Datetime, HistoricalDataFrame


class DataSource(HistoricalDataFrame):
    """
    A class representing a data source for historical data.

    Args:
        datetime (Datetime): The datetime of the data.

    Attributes:
        datetime (Datetime): The datetime of the data.
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    """

    def __init__(
        self,
        datetime: Datetime,
    ) -> None:
        super().__init__(
            datetime=datetime,
        )

    def transform(self) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


class ReadCSV(DataSource):
    """
    A class representing a CSV data source for reading data.

    Args:
        datetime (Datetime): The datetime object.
        source (str): The path or URL of the CSV file.
        source_datetime_column_name (str): The name of the datetime column in the CSV file.
        source_datetime_format (str): The format of the datetime values in the CSV file.

    Attributes:
        datetime (Datetime): The datetime of the data.
        df (polars.DataFrame): The underlying data frame.
        source (str): The path or URL of the CSV file.
        source_datetime_column_name (str): The name of the datetime column in the CSV file.
        source_datetime_format (str): The format of the datetime values in the CSV file.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform(): Transforms the CSV data and yields the resulting DataFrame.

    """

    def __init__(
        self,
        datetime: Datetime,
        source: str,
        source_datetime_column_name: str,
        source_datetime_format: str,
    ) -> None:
        super().__init__(
            datetime=datetime,
        )
        self.source: str = source
        self.source_datetime_column_name: str = source_datetime_column_name
        self.source_datetime_format: str = source_datetime_format

    def transform(self) -> typing.Iterator[typing.Any]:
        """
        Transforms the CSV data and yields the resulting DataFrame.

        Yields:
            polars.DataFrame: The transformed DataFrame.

        """
        if self.df is None or self.stateful is False:
            self.df: polars.DataFrame = (
                polars.read_csv(
                    source=self.source,
                )
                .rename(
                    mapping={
                        self.source_datetime_column_name: self.datetime.datetime_column_name,
                    },
                )
                .with_columns(
                    polars.col(self.datetime.datetime_column_name)
                    .cast(polars.Utf8)
                    .str.to_datetime(
                        format=self.source_datetime_format,
                    )
                )
                .with_columns(
                    polars.selectors.by_dtype(polars.Utf8)
                    .exclude(
                        self.datetime.datetime_column_name,
                    )
                    .str.strip_chars()
                    .cast(polars.Float32)
                )
            )
        yield self.df
