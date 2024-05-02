from __future__ import annotations

import enum
import typing

import polars

from hoisaai.layer_0.tensor import Tensor


class Series(object):
    def __init__(
        self,
        series: polars.Series,
    ) -> None:
        self.series: polars.Series = series

    def tolist(self) -> typing.List[typing.Any]:
        return self.series.to_list()


class DataFrame(object):
    class DataType(enum.Enum):
        FLOAT32: polars.DataType = polars.Float32
        FLOAT64: polars.DataType = polars.Float64
        INT16: polars.DataType = polars.Int16
        INT32: polars.DataType = polars.Int32
        INT64: polars.DataType = polars.Int64
        DATETIME: polars.DataType = polars.Datetime

    def __init__(
        self,
        df: polars.DataFrame,
    ) -> None:
        self.df: polars.DataFrame = df

    def __getitem__(self, key: str) -> Series:
        return Series(
            series=self.df[key],
        )

    def __str__(self) -> str:
        return str(self.df)

    def _repr_html_(self) -> str:
        # pylint: disable=protected-access
        return self.df._repr_html_()

    @property
    def clone(self):
        return DataFrame(
            df=self.df.clone(),
        )

    def drop_null(
        self,
        inplace: bool = False,
    ) -> DataFrame:
        if inplace is True:
            self.df = self.df.drop_nulls()
            return self
        else:
            return DataFrame(
                df=self.df.drop_nulls(),
            )

    @staticmethod
    def from_list(
        data: typing.List[typing.List[typing.Any]],
        schema: typing.Dict[str, DataFrame.DataType],
    ) -> DataFrame:
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

    def percent_change(
        self,
        columns: typing.List[str],
        inplace: bool = False,
    ) -> DataFrame:
        if inplace is True:
            self.df = self.df.with_columns(
                polars.col(columns).pct_change() * 100.0,
            )[
                # Remove the first row
                1:
            ]
            return self
        else:
            return DataFrame(
                df=self.df.with_columns(
                    polars.col(columns).pct_change() * 100.0,
                )[
                    # Remove the first row
                    1:
                ],
            )

    @staticmethod
    def read_csv(
        source: str,
        date_time_column_name: str,
        source_datetime_column_name: str,
        source_datetime_format: str,
    ) -> DataFrame:
        return DataFrame(
            df=(
                polars.read_csv(
                    source=source,
                )
                .rename(
                    mapping={
                        source_datetime_column_name: date_time_column_name,
                    },
                )
                .with_columns(
                    polars.col(date_time_column_name)
                    .cast(polars.Utf8)
                    .str.to_datetime(
                        format=source_datetime_format,
                    )
                )
                .with_columns(
                    polars.selectors.by_dtype(polars.Utf8)
                    .exclude(
                        date_time_column_name,
                    )
                    .str.strip_chars()
                    .cast(polars.Float32)
                )
            ),
        )

    def rename(
        self,
        mapping: typing.Dict[str, str],
        inplace: bool = False,
    ):
        if inplace is True:
            self.df = self.df.rename(
                mapping=mapping,
            )
            return self
        else:
            return DataFrame(
                df=self.df.rename(
                    mapping=mapping,
                ),
            )

    def save(
        self,
        destination: str,
    ) -> None:
        self.df.write_csv(
            destination,
        )

    def select(
        self,
        columns: typing.List[str],
        inplace: bool = False,
    ) -> DataFrame:
        if inplace is True:
            self.df = self.df.select(
                polars.col(columns),
            )
            return self
        else:
            return DataFrame(
                df=self.df.select(
                    polars.col(columns),
                ),
            )

    @property
    def tensor(self) -> Tensor:
        return Tensor(
            x=self.df.select(
                polars.selectors.by_dtype(
                    polars.Float32,
                    polars.Float64,
                    polars.Int16,
                    polars.Int32,
                    polars.Int64,
                )
            ).to_numpy(),
        )
