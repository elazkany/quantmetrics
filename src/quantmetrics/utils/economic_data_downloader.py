# utils/economic_data_downloader

import pandas as pd
from pandas_datareader import data as web
import yfinance as yf
from typing import Dict, List, Union
from datetime import datetime


class EconomicDataDownloader:
    """
    A class to download and aggregate economic indicators from multiple sources.

    Parameters:
    ----------
    tickers : Dict[str, List[str]]
        Dictionary where keys are sources (str) and values are lists of ticker symbols (List[str]).
    start_date : Union[str, datetime]
        Start date for data retrieval.
    end_date : Union[str, datetime]
        End date for data retrieval.
    """

    def __init__(
        self,
        tickers: Dict[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        auto_adjust: bool = True,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.auto_adjust = auto_adjust

    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """Internal method to download data for all specified tickers from their respective sources."""
        all_data = {}
        for source, tickers in self.tickers.items():
            for ticker in tickers:
                try:
                    if source == "fred":
                        all_data[ticker] = web.DataReader(
                            ticker, "fred", self.start_date, self.end_date
                        )
                    elif source == "yahoo":
                        all_data[ticker] = yf.download(
                            ticker,
                            start=self.start_date,
                            end=self.end_date,
                            progress=False,
                            auto_adjust=self.auto_adjust,  #
                        )
                    else:
                        print(f"Unsupported data source: {source}")
                        continue
                except Exception as e:
                    print(f"Error fetching data for {ticker} from {source}: {e}")
        return all_data

    def _flatten_yahoo_columns(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Internal method to flatten the multi-index columns from Yahoo Finance."""
        data.columns = ["_".join(col).strip() for col in data.columns]
        return data

    def _align_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Internal method to align all data on the date index of the most frequent ticker.

        Parameters:
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary where keys are ticker symbols and values are DataFrames of the respective data.

        Returns:
        -------
        pd.DataFrame
            DataFrame with all data aligned on the date index of the most frequent ticker.
        """
        # Flatten Yahoo Finance columns and combine data
        for ticker, data in data_dict.items():
            if isinstance(data.columns, pd.MultiIndex):
                data_dict[ticker] = self._flatten_yahoo_columns(data, ticker)

        # Find the ticker with the most frequent dates
        max_freq_ticker = max(data_dict, key=lambda k: data_dict[k].shape[0])

        # Use the dates from the most frequent ticker as the index
        df = pd.DataFrame(index=data_dict[max_freq_ticker].index)

        # Align all other data to this index
        for ticker, data in data_dict.items():
            df = df.join(data, how="outer")

        return df

    def _fill_na(self, data: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """
        Internal method to fill NaN values in the DataFrame.

        Parameters:
        ----------
        data : pd.DataFrame
            DataFrame to fill NaN values in.
        method : str, optional
            Method to fill NaN values. Default is 'ffill' (forward fill).
            Other options can be added as needed.

        Returns:
        -------
        pd.DataFrame
            DataFrame with NaN values filled.
        """
        if method == "ffill":
            return data.ffill()
        else:
            raise ValueError(f"Unsupported fill method: {method}")

    def get_data(self, fill_method: str = None) -> pd.DataFrame:
        """Return the downloaded and aligned data."""
        data_dict = self._download_data()
        aligned_data = self._align_data(data_dict)

        if fill_method:
            aligned_data = self._fill_na(aligned_data, method=fill_method)

        return aligned_data
