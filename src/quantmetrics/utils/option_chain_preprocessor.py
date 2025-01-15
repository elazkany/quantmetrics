# utils/option_chain_preprocessor

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import yfinance as yf


class OptionChainPreprocessor:
    """
    A class for preprocessing historical option chain data downloaded from Option Metrics.

    This class provides methods to clean, calculate, and augment the option chain data, making it ready for analysis.
    Upon initialization, it converts date columns to datetime objects, computes the days to maturity, and calculates the mid price of options.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        A DataFrame containing historical option chain data. The DataFrame is expected to have the following columns:

                - date : object
                        The trading date of the option.
                - exdate : object
                        The expiration date of the option.
                - best_bid : float64
                        The bid price of the option.
                - best_offer : float64
                        The offer price of the option.
                - strike_price : int64
                        The strike price of the option.
                - volume : int64
                        The number of contracts traded within a specific period, providing a snapshot of trading activity.
                - open_interest : int64
                        The number of outstanding contracts that have been opened but not yet closed or settled.
                - impl_volatility : float64
                        The implied volatility of the option.
    Methods:
    --------
    preprocess(
        adjust_strike_price=False,
        strike_price_divisor=1000,
        open_interest_range=None,
        volume_range=None,
        download_underlying_price=False,
        underlying_price_params=None,
        download_vix=False,
        vix_params=None,
        download_interest_rate=False,
        interest_rate_params=None,
        subset_by_moneyness=False,
        moneyness_params=None,
        select_columns=None
        )
        Executes all preprocessing steps with optional adjustments.

    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the OptionChainPreprocessor with the provided dataframe.

        This constructor performs several preprocessing steps on the input dataframe:
        1. Converts the 'date' and 'exdate' columns to datetime objects to ensure proper date operations.
        2. Calculates the number of days to maturity for each option contract by taking the difference between 'exdate' and 'date', and adds this as a new column 'days_to_maturity' in the dataframe.
        3. Computes the mid price of each option contract using the average of 'best_bid' and 'best_offer', and adds this as a new column 'mid_price' in the dataframe.

        Parameters:
        -----------
        dataframe : pd.DataFrame
                A pandas DataFrame containing the option chain data to be processed.
        """
        self.dataframe = dataframe.copy()

        self._convert_dates()
        self._calculate_days_to_maturity()
        self._calculate_mid_price()
        self.underlying_ticker = None

    def _convert_dates(self):
        """
        Convert date columns to datetime.
        """
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"])
        self.dataframe["exdate"] = pd.to_datetime(self.dataframe["exdate"])

    def _calculate_days_to_maturity(self):
        """
        Calculate the days to maturity.
        """
        self.dataframe["days_to_maturity"] = (
            self.dataframe["exdate"] - self.dataframe["date"]
        ).dt.days

    def _calculate_mid_price(self):
        """
        Calculate the mid price of the option.
        """
        self.dataframe["mid_price"] = (
            self.dataframe["best_bid"] + self.dataframe["best_offer"]
        ) / 2

    def _adjust_strike_price(self, divisor: int = 1000):
        """
        Adjust the strike price by dividing by the specified divisor.
        """
        self.dataframe["strike_price"] = (
            self.dataframe["strike_price"].astype(float) / divisor
        )

    def _subset_open_interest(self, open_interest_range: tuple = (0, float("inf"))):
        """
        Subset the dataframe based on the open interest range inclusive.
        """
        min_oi, max_oi = open_interest_range
        self.dataframe = self.dataframe[
            (self.dataframe["open_interest"] >= min_oi)
            & (self.dataframe["open_interest"] <= max_oi)
        ]

    def _subset_volume(self, volume_range: tuple = (0, float("inf"))):
        """
        Subset the dataframe based on the volume range inclusive.
        """
        min_volume, max_volume = volume_range
        self.dataframe = self.dataframe[
            (self.dataframe["volume"] >= min_volume)
            & (self.dataframe["volume"] <= max_volume)
        ]

    def _download_underlying_price(
        self,
        underlying_ticker: str,
        start_date: datetime,
        end_date: datetime,
        auto_adjust: bool = True,
        price_col: str = "Close",
    ):
        """
        Download the underlying asset historical prices from yahoo finance and merge it with the dataframe.

        Parameters:
        -----------
        underlying_ticker : str
                The underlying asset ticker symbol.
        start_date : datetime64
                The start date of the series of historical prices.
        end_date : datetime64
                The end date of the series of historical prices.
        price_col : str
                The price column to be considered. For example, either 'Close' or 'Adj Close'.
        """
        self.underlying_ticker = underlying_ticker

        try:
            asset_data = yf.download(
                underlying_ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=auto_adjust,
            )[[price_col]]
            asset_data.columns = [underlying_ticker]
            self.dataframe = self.dataframe.merge(
                asset_data, left_on="date", right_index=True, how="left"
            )
        except Exception as e:
            print(f"Error fetching asset prices for {underlying_ticker}: {e}")

    def _download_interest_rate(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Download historical intereset rates from FRED and merge them with the dataframe.

        Parameters:
        -----------
        ticker : str
                The interest rate symbol. For example, 'DTB3'.
        start_date : datetime64
                The start date of the series of historical prices.
        end_date : datetime64
                The end date of the series of historical prices.
        """
        try:
            interest_rate_data = web.DataReader(ticker, "fred", start_date, end_date)
            interest_rate_data = interest_rate_data / 100 
            self.dataframe = self.dataframe.merge(
                interest_rate_data, left_on="date", right_index=True, how="left"
            )
        except Exception as e:
            print(f"Error fetching interest rates for {ticker}: {e}")

    def _download_vix(
        self, start_date: datetime, end_date: datetime, price_col: str = "Close"
    ):
        """
        Download VIX data from yahoo finance and merge it with the dataframe.

        Parameters:
        -----------
        start_date : datetime64
                The start date of the series of historical VIX.
        end_date : datetime64
                The end date of the series of historical VIX.
        price_col : str
                The price column to be considered. For example, either 'Close' or 'Adj Close'.
        """
        try:
            asset_data = yf.download(
                "^VIX", start=start_date, end=end_date, progress=False
            )[[price_col]]
            asset_data.columns = ["^VIX"]
            self.dataframe = self.dataframe.merge(
                asset_data, left_on="date", right_index=True, how="left"
            )
        except Exception as e:
            print(f"Error fetching VIX data: {e}")

    def _subset_by_moneyness(self, moneyness_rate_tuple: tuple):
        """
        Subset the dataframe based on moneyness bounds inclusive.

        This method filters the DataFrame based on specified moneyness bounds. Moneyness refers to the relative position of the current asset price to the option's strike price. The moneyness is calculated by dividing the underlying asset price (S0) by the strike price (K). Options with moneyness values within the specified bounds will be included in the resulting DataFrame.

        Parameters:
        -----------
        moneyness_rate_tuple : tuple
                The tuple containing (lower_rate, upper_rate, step_size) for moneyness range. For example, (0.8, 1.21, 0.1).
        """

        # Check if the underlying_ticker has been set by _download_underlying_price
        if self.underlying_ticker is None:
            raise ValueError(
                "Underlying ticker not found in the dataframe. Please ensure the underlying asset price has been downloaded by editing the download_underlying_price input in the preprocess method"
            )

        lower_rate, upper_rate, step_size = moneyness_rate_tuple
        result = pd.DataFrame()

        # Group by date and days_to_maturity
        grouped = self.dataframe.groupby(["date", "days_to_maturity"])

        for (date, days_to_maturity), group in grouped:
            S0 = group[self.underlying_ticker].iloc[0]
            min_strike = np.min(
                np.floor(S0 * np.arange(lower_rate, upper_rate + step_size, step_size))
            )
            max_strike = np.max(
                np.floor(S0 * np.arange(lower_rate, upper_rate + step_size, step_size))
            )

            subset = group[
                (group["strike_price"] >= min_strike)
                & (group["strike_price"] <= max_strike)
            ]
            result = pd.concat([result, subset])
        self.dataframe = result.copy()

    def _select_columns(self, columns: list):
        """
        Select specified columns from the dataframe.

        Parameters:
        -----------
        columns : list
                List of columns to select from the dataframe. For example, ['date', 'exdate', 'days_to_maturity', 'mid_price', 'strike_price', '^SPX', 'DTB3', '^VIX', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility']
        """
        self.dataframe = self.dataframe[columns]

    def preprocess(
        self,
        adjust_strike_price: bool = False,
        strike_price_divisor: int = 1000,
        open_interest_range: tuple = None,
        volume_range: tuple = None,
        download_underlying_price: bool = False,
        underlying_price_params: tuple = None,
        download_vix: bool = False,
        vix_params: tuple = None,
        download_interest_rate: bool = False,
        interest_rate_params: tuple = None,
        subset_by_moneyness: bool = False,
        moneyness_params: tuple = None,
        select_columns: list = None,
    ):
        """
        Execute all preprocessing steps with optional adjustments.

        Parameters:
        adjust_strike_price : bool
                Whether to adjust the strike price.
        strike_price_divisor : int
                The divisor for adjusting the strike price.
        open_interest_range : tuple
                The range (min, max) for subsetting based on open interest.
        volume_range : tuple
                The range (min, max) for subsetting based on volume.
        download_underlying_price : bool
                Whether to download and merge the underlying asset price.
        underlying_price_params : tuple
                The parameters (ticker, start_date, end_date, price_col) for downloading the asset price.
        download_vix : bool
                Whether to download and merge VIX data.
        vix_params : tuple
                The parameters (start_date, end_date, price_col) for downloading VIX data.
        download_interest_rate : bool
                Whether to download and merge interest rates.
        interest_rates_params : tuple
                The parameters (ticker, start_date, end_date) for downloading the interest rates from FRED.
        select_columns : list
                The list of columns to select from the dataframe.
        """

        if adjust_strike_price:
            self._adjust_strike_price(strike_price_divisor)

        if open_interest_range is not None:
            self._subset_open_interest(open_interest_range)

        if volume_range is not None:
            self._subset_volume(volume_range)

        if download_underlying_price and underlying_price_params is not None:
            self._download_underlying_price(*underlying_price_params)

        if download_interest_rate and interest_rate_params is not None:
            self._download_interest_rate(*interest_rate_params)

        if download_vix and vix_params is not None:
            self._download_vix(*vix_params)

        if select_columns is not None:
            self._select_columns(select_columns)

        if subset_by_moneyness and moneyness_params is not None:
            self._subset_by_moneyness(moneyness_params)

        return self.dataframe
