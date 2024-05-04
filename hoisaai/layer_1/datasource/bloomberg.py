import datetime
from enum import Enum
import typing

import blpapi

from hoisaai.layer_0.dataframe import DataFrame


class Bloomberg(object):

    class Equity(Enum):
        AAPL: str = "AAPL US Equity"

    class Forex(Enum):
        # US Dollar per Euro
        EURUSD: str = "EUR BGN Curncy"
        # Japanese Yen per US dollar
        USDJPY: str = "JPY BGN Curncy"
        # British Pound per US dollar
        USDGBP: str = "GBP BGN Curncy"
        # Australian Dollar per US dollar
        USDAUD: str = "AUD BGN Curncy"
        # Swiss Franc per US dollar
        USDCHF: str = "CHF BGN Curncy"
        # Canadian dollar per US dollar
        USDCAD: str = "CAD BGN Curncy"
        # New Zealand per US dollar
        USDNZD: str = "NZD BGN Curncy"
        # Swedish Krona per US dollar
        USDSEK: str = "SEK BGN Curncy"
        # Brazilian Real per US dollar
        USDBRL: str = "BRL BGN Curncy"
        # Russian Ruble per US dollar
        USDRUB: str = "RUB BGN Curncy"
        # Indian Rupee per US dollar
        USDINR: str = "INR BGN Curncy"
        # Chinese Yuan per US dollar
        USDCNH: str = "CNH BGN Curncy"

    def __init__(
        self,
        schema: typing.Dict[str, DataFrame.DataType],
        field: typing.List[str],
    ) -> None:
        self.schema: typing.Dict[str, DataFrame.DataType] = schema
        self.field: typing.List[str] = field
        # Fill SessionOptions
        self.session_options = blpapi.SessionOptions()
        self.session_options.setServerHost(
            serverHost="localhost",
        )
        self.session_options.setServerPort(
            serverPort=8194,
        )

    def get(
        self,
        security: Enum,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
    ) -> DataFrame:
        raise NotImplementedError()


class BloombergDaily(Bloomberg):
    class Field(Enum):
        ASK_PRICE: str = "Ask Price"
        BID_PRICE: str = "Bid Price"
        CLOSE_PRICE: str = "Close Price"
        HIGH_PRICE: str = "High Price"
        LAST_PRICE: str = "Last Price"
        LOW_PRICE: str = "Low Price"
        OPEN_PRICE: str = "Open Price"

    FIELD: typing.List[str] = [
        Field.BID_PRICE,
        Field.ASK_PRICE,
        Field.LAST_PRICE,
        Field.OPEN_PRICE,
        Field.HIGH_PRICE,
        Field.LOW_PRICE,
    ]

    def __init__(
        self,
        date_time_column_name: str,
    ) -> None:
        Bloomberg.__init__(
            self,
            schema={
                date_time_column_name: DataFrame.DataType.DATETIME.value,
                **{
                    column_name: DataFrame.DataType.FLOAT32.value
                    for column_name in BloombergDaily.FIELD
                },
            },
            field=BloombergDaily.FIELD,
        )

    def get(
        self,
        security: Enum,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
    ) -> DataFrame:
        data: typing.List[typing.List[typing.Any]] = []
        # Create a Session
        session: blpapi.Session = blpapi.Session(self.session_options)
        assert session.start()
        try:
            assert session.openService(
                serviceName="//blp/refdata",
            )
            # Obtain previously opened service
            reference_data_service = session.getService(
                serviceName="//blp/refdata",
            )
            # Create a request
            request = reference_data_service.createRequest(
                operation="HistoricalDataRequest",
            )
            # Append security to request
            request.getElement("securities").appendValue(
                security,
            )
            # Append fields to request
            for field in self.field:
                request.getElement("fields").appendValue(
                    field,
                )
            request.set("startDate", start_datetime.strftime("%Y%m%d"))
            request.set("endDate", end_datetime.strftime("%Y%m%d"))
            request.set("periodicitySelection", "DAILY")
            # Send the request
            session.sendRequest(request)
            # Process received events
            while True:
                # We provide timeout to give the chance for Ctrl+C handling:
                environment: blpapi.event.Event = session.nextEvent(500)
                for element in environment:
                    if element.hasElement("securityData"):
                        element: blpapi.message.Element = element.getElement(
                            name="securityData",
                        )
                        if element.hasElement("fieldData"):
                            elements: blpapi.message.Element = element.getElement(
                                nameOrIndex="fieldData",
                            )
                            for element in elements:
                                date_time: datetime.date = element.getElement(
                                    "date"
                                ).getValueAsDatetime()
                                date_time: datetime.datetime = datetime.datetime(
                                    date_time.year,
                                    date_time.month,
                                    date_time.day,
                                )
                                data.append(
                                    [date_time]
                                    + [
                                        (
                                            element.getElement(field).getValueAsFloat()
                                            if element.hasElement(field)
                                            else None
                                        )
                                        for field in BloombergDaily.FIELD
                                    ],
                                )
                environment: blpapi.event.Event = environment
                if environment.eventType() == blpapi.event.Event.RESPONSE:
                    # Response completly received, so we could exit
                    break
        finally:
            # Stop the session
            session.stop()
        return DataFrame.from_list(
            data=data,
            schema=self.schema,
        )


class BloombergEconomicIndicator(BloombergDaily):
    class EconomicIndicator(Enum):
        CPIYOY: str = "CPI YOY Index"


class BloombergMinutely(Bloomberg):
    class Field(Enum):
        CLOSE_PRICE: str = "close"
        HIGH_PRICE: str = "high"
        LOW_PRICE: str = "low"
        OPEN_PRICE: str = "open"
        VOLUME: str = "volume"
        NUMBER_OF_EVENTS: str = "numEvents"
        VALUE: str = "value"

    FIELD: typing.List[str] = [
        Field.OPEN_PRICE,
        Field.HIGH_PRICE,
        Field.LOW_PRICE,
        Field.CLOSE_PRICE,
        Field.VOLUME,
        Field.NUMBER_OF_EVENTS,
        Field,
    ]

    def __init__(
        self,
        date_time_column_name: str,
    ) -> None:
        super().__init__(
            schema={
                date_time_column_name: DataFrame.DataType.DATETIME.value,
                **{
                    column_name: DataFrame.DataType.FLOAT32.value
                    for column_name in BloombergMinutely.FIELD
                },
            },
            field=BloombergMinutely.FIELD,
        )

    def get(
        self,
        security: Enum,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
    ) -> DataFrame:
        data: typing.List[typing.List[typing.Any]] = []
        # Create a Session
        session: blpapi.Session = blpapi.Session(self.session_options)
        assert session.start()
        try:
            assert session.openService(
                serviceName="//blp/refdata",
            )
            # Obtain previously opened service
            reference_data_service = session.getService(
                serviceName="//blp/refdata",
            )
            # Create a request
            request = reference_data_service.createRequest(
                operation="IntradayBarRequest",
            )
            # Append security to request
            request.getElement("security").appendValue(
                security,
            )
            request.set("startDate", start_datetime.strftime("%Y%m%d"))
            request.set("endDate", end_datetime.strftime("%Y%m%d"))
            request.set(name="interval", value=1)
            # Send the request
            session.sendRequest(request)
            # Process received events
            while True:
                # We provide timeout to give the chance for Ctrl+C handling:
                environment: typing.List[blpapi.message.Element] = session.nextEvent(
                    500
                )
                for element in environment:
                    if element.hasElement("barData"):
                        element: blpapi.message.Element = element.getElement(
                            name="barData",
                        )
                        if element.hasElement("barTickData"):
                            elements: blpapi.message.Element = element.getElement(
                                nameOrIndex="barTickData",
                            )
                            for element in elements:
                                data.append(
                                    [element.getElement("time").getValueAsDatetime()]
                                    + [
                                        (
                                            element.getElement(field).getValueAsFloat()
                                            if element.hasElement(field)
                                            else None
                                        )
                                        for field in BloombergMinutely.FIELD
                                    ],
                                )
                environment: blpapi.event.Event = environment
                if environment.eventType() == blpapi.event.Event.RESPONSE:
                    # Response completly received, so we could exit
                    break
        finally:
            # Stop the session
            session.stop()
        return DataFrame.from_list(
            data=data,
            schema=self.schema,
        )
