"""
Yahoo Finance client for fetching stock data.

This module provides a client for fetching stock data from Yahoo Finance
using the yfinance package. It complements the Polygon.io client and can
be used as a backup or alternative data source.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv
from requests.exceptions import RequestException

from src.ingest.base_client import BaseAPIClient

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Cache directory for storing data
CACHE_DIR = Path(os.getenv('CACHE_DIR', 'data/cache/yahoo'))

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

class YahooFinanceClient(BaseAPIClient):
    """Client for fetching data from Yahoo Finance."""

    def __init__(self, use_cache: bool = True, cache_expiry: int = 24, max_retries: int = 3, backoff_factor: float = 0.5):
        """Initialize the Yahoo Finance client.

        Args:
            use_cache: Whether to use caching for API requests
            cache_expiry: Cache expiry time in hours
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        logger.info("Initializing Yahoo Finance client")

        # Initialize the base client
        super().__init__(
            base_url="https://query2.finance.yahoo.com",  # Not actually used for yfinance
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )

        self.use_cache = use_cache
        self.cache_expiry = cache_expiry

    def _get_cache_path(self, symbol: str, data_type: str, start: datetime, end: datetime, interval: str = "1d") -> Path:
        """Get the cache file path for a specific request.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data (ticks, fundamentals, dividends, etc.)
            start: Start datetime
            end: End datetime
            interval: Time interval for tick data

        Returns:
            Path to the cache file
        """
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        filename = f"{symbol}_{data_type}_{start_str}_{end_str}_{interval}.parquet"
        return CACHE_DIR / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cache file is valid (exists and not expired).

        Args:
            cache_path: Path to the cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False

        # Check if file is older than cache_expiry hours
        file_age = time.time() - os.path.getmtime(cache_path)
        max_age = self.cache_expiry * 3600  # Convert hours to seconds

        return file_age < max_age

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5, exceptions=(Exception,))
    def fetch_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch tick data for a given symbol and time range.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            interval: Time interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {interval} data for {symbol} from {start} to {end}")

        # Validate interval
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}. Must be one of: {', '.join(valid_intervals)}")

        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, "ticks", start, end, interval)
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} data from cache: {cache_path}")
                try:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} records for {symbol} from cache")
                        return df
                    logger.warning(f"Cache file exists but is empty for {symbol}. Will fetch fresh data.")
                except Exception as e:
                    logger.warning(f"Error reading from cache: {e}. Will fetch fresh data.")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval=interval
            )

            # Check if data is empty
            if df.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Reset index to make Date a column
            df = df.reset_index()

            # Rename columns to match Polygon format
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Validate data
            self._validate_tick_data(df, symbol)

            logger.info(f"Fetched {len(df)} records for {symbol}")

            # Save to cache if enabled
            if self.use_cache and not df.empty:
                try:
                    cache_path = self._get_cache_path(symbol, "ticks", start, end, interval)
                    df.to_parquet(cache_path)
                    logger.info(f"Saved {symbol} data to cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Error saving to cache: {e}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def _validate_tick_data(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validate tick data for quality and completeness.

        Args:
            df: DataFrame with tick data
            symbol: Stock ticker symbol

        Raises:
            ValueError: If data validation fails
        """
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = np.nan

        # Check for NaN values in critical columns
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and df[col].isna().any():
                nan_count = df[col].isna().sum()
                logger.warning(f"Found {nan_count} NaN values in {col} column for {symbol}")

        # Check for duplicate timestamps
        if df['timestamp'].duplicated().any():
            dup_count = df['timestamp'].duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate timestamps for {symbol}")
            # Remove duplicates
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)

        # Check for out-of-order timestamps
        if not df['timestamp'].equals(df['timestamp'].sort_values()):
            logger.warning(f"Timestamps are not in order for {symbol}")
            # Sort by timestamp
            df.sort_values('timestamp', inplace=True)

        # Check for price anomalies (e.g., negative prices)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                logger.warning(f"Found {neg_count} negative values in {col} column for {symbol}")

        # Check for high-low inconsistency
        if 'high' in df.columns and 'low' in df.columns and (df['high'] < df['low']).any():
            inconsistent_count = (df['high'] < df['low']).sum()
            logger.warning(f"Found {inconsistent_count} rows where high < low for {symbol}")

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5, exceptions=(Exception,))
    def fetch_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with fundamental data
        """
        logger.info(f"Fetching fundamental data for {symbol}")

        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, "fundamentals",
                                           datetime.now() - timedelta(days=1),
                                           datetime.now(), "")
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} fundamentals from cache: {cache_path}")
                try:
                    return pd.read_parquet(cache_path).to_dict('records')[0]
                except Exception as e:
                    logger.warning(f"Error reading from cache: {e}. Will fetch fresh data.")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)

            # Get various fundamental data
            info = ticker.info

            # Extract key metrics
            fundamentals = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'eps': info.get('trailingEps', None),
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'profit_margins': info.get('profitMargins', None),
                'return_on_equity': info.get('returnOnEquity', None),
                'return_on_assets': info.get('returnOnAssets', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'timestamp': datetime.now()
            }

            # Save to cache if enabled
            if self.use_cache:
                try:
                    cache_path = self._get_cache_path(symbol, "fundamentals",
                                                   datetime.now() - timedelta(days=1),
                                                   datetime.now(), "")
                    pd.DataFrame([fundamentals]).to_parquet(cache_path)
                    logger.info(f"Saved {symbol} fundamentals to cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Error saving to cache: {e}")

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamental data from Yahoo Finance: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

    def fetch_multiple_ticks(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch tick data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start: Start datetime
            end: End datetime
            interval: Time interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_ticks(symbol, start, end, interval)
                results[symbol] = df
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                results[symbol] = pd.DataFrame()

        return results

    def fetch_dividends(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch dividend data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with dividend data
        """
        logger.info(f"Fetching dividend data for {symbol} from {start} to {end}")

        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, "dividends", start, end, "")
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} dividends from cache: {cache_path}")
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(f"Error reading from cache: {e}. Will fetch fresh data.")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.dividends.reset_index()

            # Filter by date range
            if not df.empty:
                df = df[(df['Date'] >= pd.Timestamp(start)) & (df['Date'] <= pd.Timestamp(end))]

            # Rename columns
            if not df.empty:
                df = df.rename(columns={'Date': 'timestamp', 'Dividends': 'amount'})

            # Save to cache if enabled
            if self.use_cache and not df.empty:
                try:
                    cache_path = self._get_cache_path(symbol, "dividends", start, end, "")
                    df.to_parquet(cache_path)
                    logger.info(f"Saved {symbol} dividends to cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Error saving to cache: {e}")

            return df

        except Exception as e:
            logger.error(f"Error fetching dividend data from Yahoo Finance: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'amount'])

    def fetch_earnings(self, symbol: str) -> pd.DataFrame:
        """
        Fetch earnings data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            DataFrame with earnings data
        """
        logger.info(f"Fetching earnings data for {symbol}")

        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, "earnings",
                                           datetime.now() - timedelta(days=30),
                                           datetime.now(), "")
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} earnings from cache: {cache_path}")
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(f"Error reading from cache: {e}. Will fetch fresh data.")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates

            if earnings is not None and not earnings.empty:
                # Reset index to make Date a column
                df = earnings.reset_index()

                # Rename columns
                df = df.rename(columns={
                    'Earnings Date': 'timestamp',
                    'EPS Estimate': 'eps_estimate',
                    'Reported EPS': 'eps_actual',
                    'Surprise(%)': 'surprise_percent'
                })

                # Save to cache if enabled
                if self.use_cache:
                    try:
                        cache_path = self._get_cache_path(symbol, "earnings",
                                                       datetime.now() - timedelta(days=30),
                                                       datetime.now(), "")
                        df.to_parquet(cache_path)
                        logger.info(f"Saved {symbol} earnings to cache: {cache_path}")
                    except Exception as e:
                        logger.warning(f"Error saving to cache: {e}")

                return df
            else:
                return pd.DataFrame(columns=['timestamp', 'eps_estimate', 'eps_actual', 'surprise_percent'])

        except Exception as e:
            logger.error(f"Error fetching earnings data from Yahoo Finance: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'eps_estimate', 'eps_actual', 'surprise_percent'])

    def fetch_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Fetch analyst recommendations for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            DataFrame with recommendation data
        """
        logger.info(f"Fetching recommendations for {symbol}")

        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(symbol, "recommendations",
                                           datetime.now() - timedelta(days=30),
                                           datetime.now(), "")
            if self._is_cache_valid(cache_path):
                logger.info(f"Loading {symbol} recommendations from cache: {cache_path}")
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    logger.warning(f"Error reading from cache: {e}. Will fetch fresh data.")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations

            if recommendations is not None and not recommendations.empty:
                # Reset index to make Date a column
                df = recommendations.reset_index()

                # Save to cache if enabled
                if self.use_cache:
                    try:
                        cache_path = self._get_cache_path(symbol, "recommendations",
                                                       datetime.now() - timedelta(days=30),
                                                       datetime.now(), "")
                        df.to_parquet(cache_path)
                        logger.info(f"Saved {symbol} recommendations to cache: {cache_path}")
                    except Exception as e:
                        logger.warning(f"Error saving to cache: {e}")

                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching recommendations from Yahoo Finance: {str(e)}")
            return pd.DataFrame()


def fetch_ticks(symbol: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    """
    Convenience function to fetch tick data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        start: Start datetime
        end: End datetime
        interval: Time interval

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    client = YahooFinanceClient()
    return client.fetch_ticks(symbol, start, end, interval)


def fetch_fundamentals(symbol: str) -> Dict:
    """
    Convenience function to fetch fundamental data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with fundamental data
    """
    client = YahooFinanceClient()
    return client.fetch_fundamentals(symbol)


def fetch_dividends(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Convenience function to fetch dividend data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with dividend data
    """
    client = YahooFinanceClient()
    return client.fetch_dividends(symbol, start, end)


def fetch_earnings(symbol: str) -> pd.DataFrame:
    """
    Convenience function to fetch earnings data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")

    Returns:
        DataFrame with earnings data
    """
    client = YahooFinanceClient()
    return client.fetch_earnings(symbol)


def fetch_recommendations(symbol: str) -> pd.DataFrame:
    """
    Convenience function to fetch analyst recommendations from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")

    Returns:
        DataFrame with recommendation data
    """
    client = YahooFinanceClient()
    return client.fetch_recommendations(symbol)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch data for Apple
    client = YahooFinanceClient()
    df = client.fetch_ticks("AAPL", start_date, end_date)

    # Print the first few rows
    print(df.head())

    # Fetch fundamental data
    fundamentals = client.fetch_fundamentals("AAPL")
    print(fundamentals)
