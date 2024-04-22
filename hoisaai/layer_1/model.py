"""
This module contains the base classes for defining models and data structures.
"""
import typing

import jaxlib.xla_extension
import polars


class Datetime:
    """
    A class defining datetime column name.

    Args:
        datetime_column_name (str): The name of the datetime column.

    Attributes:
        datetime_column_name (str): The name of the datetime column.

    """

    def __init__(
        self,
        datetime_column_name: str,
    ) -> None:
        self.datetime_column_name: str = datetime_column_name


class Model:
    """
    This class represents a model.

    Methods:
        transform: Perform the transformation.
    """

    def transform(
        self,
    ) -> typing.Iterator[typing.Any]:
        """
        Transforms the data using a specific algorithm.

        Returns:
            An iterator containing the transformed data.
        """
        raise NotImplementedError()


class Stateful(Model):
    """
    A stateful model that maintains internal state.

    Attributes:
        stateful (bool): Indicates whether the model is stateful or not.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stateful: bool = True

    def transform(
        self,
    ) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


class HistoricalData(Model):
    """
    Represents historical data.

    Args:
        datetime (Datetime): The datetime of the historical data.

    Attributes:
        datetime (Datetime): The datetime of the historical data.
    """

    def __init__(
        self,
        datetime: Datetime,
    ) -> None:
        super().__init__()
        self.datetime: Datetime = datetime

    def transform(
        self,
    ) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


def get_dataframe(
    dataframe: HistoricalData,
) -> polars.DataFrame:
    """
    Retrieves a DataFrame from the given HistoricalData object.

    Args:
        dataframe (HistoricalData): The HistoricalData object to retrieve the DataFrame from.

    Returns:
        polars.DataFrame: The DataFrame extracted from the HistoricalData object.
    """
    df: polars.DataFrame = None
    for process in dataframe.transform():
        if isinstance(process, polars.DataFrame):
            df = process
    return df


class HistoricalDataFrame(
    HistoricalData,
    Stateful,
):
    """
    A class representing a historical data frame.

    This class inherits from the HistoricalData and Stateful classes.

    Attributes:
        df (polars.DataFrame): The underlying data frame.
        stateful (bool): Indicates whether the model is stateful or not.

    Methods:
        transform: A method to transform the data frame.

    """

    def __init__(
        self,
        datetime: Datetime,
    ) -> None:
        super().__init__(
            datetime=datetime,
        )
        self.df: polars.DataFrame = None

    def transform(
        self,
    ) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


class Tensor(Model):
    """
    A class representing a tensor.

    Tensors are multi-dimensional arrays that can store and manipulate data efficiently.
    This class provides a base implementation for tensors.

    Methods:
        transform: A method that performs a transformation on the tensor.
    """

    def transform(self) -> typing.Iterator[typing.Any]:
        raise NotImplementedError()


def get_tensor(
    tensor: Tensor,
) -> jaxlib.xla_extension.ArrayImpl:
    """
    Retrieves the first instance of `jaxlib.xla_extension.ArrayImpl` from the given `tensor`.

    Args:
        tensor (Tensor): The input tensor.

    Returns:
        jaxlib.xla_extension.ArrayImpl:
            The last instance of `jaxlib.xla_extension.ArrayImpl` found in the tensor.
    """
    sample: jaxlib.xla_extension.ArrayImpl = None
    for process in tensor.transform():
        if isinstance(process, jaxlib.xla_extension.ArrayImpl):
            sample = process
    return sample


class StatefulTensor(
    Stateful,
    Tensor,
):
    """
    A class representing a stateful tensor.

    Attributes:
        x (jaxlib.xla_extension.ArrayImpl): The underlying tensor data.

    Methods:
        transform(): A generator function that yields the tensor data.
    """

    def __init__(
        self,
        x: jaxlib.xla_extension.ArrayImpl,
    ) -> None:
        super().__init__()
        self.x: jaxlib.xla_extension.ArrayImpl = x

    def transform(
        self,
    ) -> typing.Iterator[typing.Any]:
        yield self.x
