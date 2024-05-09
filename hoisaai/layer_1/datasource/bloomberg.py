"""This module provides a class to interact with Bloomberg API."""

import dataclasses
import datetime
import enum
import typing

import blpapi

from hoisaai.layer_0.dataframe import DataFrame


@dataclasses.dataclass
class FieldDataStructure:
    """Field data structure.

    :attr field: str: Field name.
    :attr datatype: DataFrame.DataType: Data type.
    """

    field: str
    datatype: DataFrame.DataType


class Bloomberg(object):
    """Bloomberg data provider."""

    class Security(enum.Enum):
        """Security."""

    class Crypto(Security):
        """Cryptocurrency."""

        BTCUSD: str = "XBT BGN Curncy"
        """
        Bitcoin
        """

    class EconomicIndicator(enum.Enum):
        """Economic Indicator."""

        CPIYOY: str = "CPI YOY Index"
        """
        The Consumer Price Index (CPI) year over year
        """

    class Equity(Security):
        """Equity."""

        AAPL: str = "AAPL US Equity"
        """
        Apple Inc
        """

    class Field(enum.Enum):
        """Field."""

        LAST_PRICE: str = FieldDataStructure(
            field="PR005",
            datatype=DataFrame.DataType.FLOAT32,
        )
        """Last Price
        Last price for the security.

        Equities:
            Returns the last price provided by the exchange.
            For securities that trade Monday through Friday,
            this field will be populated only if
            such information has been provided by the exchange in the past 30 trading days.
            For initial public offerings (IPO),
            the day before the first actual trading day may return the IPO price.
            For all other securities,
            this field will be populated only if
            such information was provided by the exchange in the last 30 calendar days.
            This applies to common stocks, receipts, warrants,
            and real estate investment trusts (REITs).

        Equity Derivatives:
            Equity Options, Spot Indices, Index Futures and Commodity Futures:
                Returns the last trade price. No value is returned for expired contracts.

        Synthetic Options:
            Returns N.A.

        Fixed Income:
            Returns the last price received from the current pricing source.
            The last price will always come from the date and time in LAST_UPDATE/LAST_UPDATE_DT.
            If there was no contributed last at that time
            the first valid value from mid/bid/ask will be used.
            The value returned will be a discount
            if Pricing Source Quote Type (DS962, PCS_QUOTE_TYP) is 2 (Discount Quoted).
            For information specific to the last trade see the price (PR088, PX_LAST_ACTUAL),
            time (P2788, LAST_TRADE_TIME),
            and date (P2789, LAST_TRADE_DATE) fields.

            Returns the last price received from the current pricing source.
            If last price is not available,
            then a mid computed from bid and ask will be returned.
            If either bid or ask is not available to compute mid,
            the field returns whichever side that is received.

        Equity Indices:
            Returns either the current quote price of the index
            or the last available close price of the index.

        Custom Indices:
            Returns the value the custom index expression evaluates to.
            Since the expression is user defined, the value has no units.

        Economic Statistics:
            Provides the revision of the prior release. 

        Futures and Options:
            Returns the last traded price until settlement price is received,
            at which time the settlement price is returned.
            If no trade or settlement price is available for the current day,
            then the last settlement price received is provided.
            No value is returned for expired contracts.
            Settlement Price (PR277, PX_SETTLE) and Futures Trade Price (PR083, FUT_PX)
            can be used instead to return settlement price
            and closing price respectively at all times regardless of these parameters.

        Swaps and Credit Default Swaps:
            Not supported for synthetics.

        Mutual Funds:
            Closed-End,
            Exchange Traded
            and Open-End Funds Receiving Intraday Pricing from Exchange Feeds:
                Returns the most recent trade price.

        Open-End and Hedge Funds:
            Returns the net asset value (NAV).
            If no NAV is available, the bid is returned,
            and if no bid is available then the ask is returned.

        Money Market Funds that Display Days to Maturity and Yield:
            Returns a yield.

        Currencies:
            Broken Date Type Currencies (e.g. USD/JPY 3M Curncy):
                Returns the average of the bid and ask.

        For All Other Currency Types:
            Returns the last trade price if it is valid and available.
            If last trade is not available then mid price is returned.
            Mid price is the average of the bid and ask.
            If a valid bid and ask are not available,
            then a bid or ask is returned based on which is non-zero.
            If no data is available for the current day,
            then the previous day's last trade is returned.

        OTC FX Options:
            Returns the premium of the option in nominal amount.
            Returns the price of the option expressed in
            a currency opposite of the notional currency.

        Mortgages:
            Returns the last price received from the current pricing source.
            If this field is empty for any reason,
            then last ask is returned and if no ask is available,
            then last bid is returned.

        Municipals:
            Returns the last price received from the current pricing source.

        Portfolio:
            Net asset value (NAV) as computed in the Portfolio & Risk Analytics function
            and used for Total Return computations.
            It is the cumulated daily total returns applied to
            the user-defined price/value at portfolio's inception date.
        """
        CUMMULATIVE_TOTAL_RETURN_WITH_NET_DIVIDENDS: str = FieldDataStructure(
            field="RT115",
            datatype=DataFrame.DataType.FLOAT32,
        )
        """Cumulative Total Return (Net Dividends)
        One day total return as of today.
        The start date is one day prior to the end date (as of date).
        Historically, this is a series of cumulative total return values. 
        Applicable periodicity values are
        daily, weekly, monthly, quarterly, semi-annually and annually.
        Net dividends are used.
        """

    class Forex(enum.Enum):
        """
        US Dollar per Euro
        """

        EURUSD: str = "EUR BGN Curncy"
        """
        Japanese Yen per US dollar
        """
        USDJPY: str = "JPY BGN Curncy"
        """
        British Pound per US dollar
        """
        USDGBP: str = "GBP BGN Curncy"
        """
        Australian Dollar per US dollar
        """
        USDAUD: str = "AUD BGN Curncy"
        """
        Swiss Franc per US dollar
        """
        USDCHF: str = "CHF BGN Curncy"
        """
        Canadian dollar per US dollar
        """
        USDCAD: str = "CAD BGN Curncy"
        """
        New Zealand per US dollar
        """
        USDNZD: str = "NZD BGN Curncy"
        """
        Swedish Krona per US dollar
        """
        USDSEK: str = "SEK BGN Curncy"
        """
        Brazilian Real per US dollar
        """
        USDBRL: str = "BRL BGN Curncy"
        """
        Russian Ruble per US dollar
        """
        USDRUB: str = "RUB BGN Curncy"
        """
        Indian Rupee per US dollar
        """
        USDINR: str = "INR BGN Curncy"
        """
        Chinese Yuan per US dollar
        """
        USDCNH: str = "CNH BGN Curncy"

    def __init__(
        self,
        date_time_column_name: str,
    ) -> None:
        self.date_time_column_name: str = date_time_column_name

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
        security: Security,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
    ) -> DataFrame:
        """Get data from Bloomberg API.

        :param security: The security to get data from.
        :type security: Bloomberg.Security: Security.
        :param start_datetime: The start date and time.
        :type start_datetime: datetime.datetime.
        :param end_datetime: The end date and time.
        :type end_datetime: datetime.datetime.

        :return: The data.
        :rtype: DataFrame.
        """
        raise NotImplementedError()


class BloombergDaily(Bloomberg):
    """Bloomberg daily data provider.

    :param date_time_column_name: The date time column name.
    :type date_time_column_name: str.
    :param schema: The schema.
    :type schema: Dict[str, Bloomberg.Field].
    """

    def __init__(
        self,
        date_time_column_name: str,
        schema: typing.Dict[str, Bloomberg.Field],
    ) -> None:
        Bloomberg.__init__(
            self,
            date_time_column_name=date_time_column_name,
        )
        self.schema: typing.Dict[str, Bloomberg.Field] = schema

    def get(
        self,
        security: Bloomberg.Security,
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
                security.value,
            )
            # Append fields to request
            for attribute in self.schema.values():
                field: FieldDataStructure = attribute.value
                request.getElement("fields").appendValue(
                    field.field,
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
                                            element.getElement(
                                                field.value.field
                                            ).getValueAsFloat()
                                            if element.hasElement(field.value.field)
                                            else None
                                        )
                                        for field in self.schema.values()
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
            schema={
                self.date_time_column_name: DataFrame.DataType.DATETIME.value,
                **{
                    label: field.value.datatype.value
                    for label, field in self.schema.items()
                },
            },
        )
