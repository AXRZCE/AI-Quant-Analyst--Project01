"""
Polygon.io API client for fetching stock market tick data.
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
from requests.exceptions import RequestException

from src.ingest.base_client import BaseAPIClient, T

# Configure logging
logger = logging.getLogger(__name__)


class PolygonClient(BaseAPIClient):
    """Client for interacting with the Polygon.io API."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, backoff_factor: float = 0.5):
        """
        Initialize the Polygon.io API client.

        Args:
            api_key: Polygon.io API key. If not provided, will look for POLYGON_API_KEY env var.
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key must be provided or set as POLYGON_API_KEY env var")

        # Initialize the base client
        super().__init__(
            base_url="https://api.polygon.io",
            api_key=self.api_key,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )

        # Update session headers for Polygon.io authentication
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5)
    def fetch_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timespan: str = "minute",
        limit: int = 1000,
        max_requests: int = 10
    ) -> pd.DataFrame:
        """
        Fetch tick data for a given symbol and time range.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            timespan: Time granularity, either "second" or "minute"
            limit: Number of results per page (max 50000)
            max_requests: Maximum number of API requests to make for pagination

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {timespan} data for {symbol} from {start} to {end}")

        if timespan not in ["second", "minute", "day", "week", "month", "quarter", "year"]:
            raise ValueError(f"Invalid timespan: {timespan}. Must be one of: second, minute, day, week, month, quarter, year")

        # Format dates for Polygon API
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        # Initialize results list and pagination variables
        all_results = []
        next_url = None
        request_count = 0

        # Build initial URL
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_str}/{end_str}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
        }

        # Paginate through results
        while request_count < max_requests:
            try:
                if next_url:
                    response = self.session.get(next_url, timeout=self.timeout)
                else:
                    response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)

                data = self.handle_response(response)

                # Check if we have results
                if data.get("results") and len(data["results"]) > 0:
                    all_results.extend(data["results"])
                    logger.debug(f"Retrieved {len(data['results'])} results for {symbol}")
                else:
                    logger.warning(f"No results found for {symbol} in this request")

                # Check for next page
                next_url = data.get("next_url")
                if not next_url:
                    break

                # Increment request counter and add small delay to avoid rate limits
                request_count += 1
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching data from Polygon API: {str(e)}")
                raise

        # Convert results to DataFrame
        if not all_results:
            logger.warning(f"No data found for {symbol} from {start} to {end}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        logger.info(f"Successfully retrieved {len(all_results)} data points for {symbol}")

        try:
            df = pd.DataFrame(all_results)

            # Rename columns to standard format
            column_mapping = {
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "transactions",  # Number of transactions
                "vw": "vwap"  # Volume-weighted average price
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Convert timestamp from milliseconds to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Select and order columns that exist in the DataFrame
            columns = [col for col in ["timestamp", "open", "high", "low", "close", "volume", "vwap", "transactions"] if col in df.columns]
            return df[columns]
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            raise ValueError(f"Failed to process data for {symbol}: {str(e)}")


    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5)
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            Dictionary with fundamental data
        """
        logger.info(f"Fetching fundamental data for {symbol}")

        endpoint = f"/v2/reference/financials/{symbol}"
        params = {
            "limit": 1,
            "sort": "reportPeriod",
            "order": "desc",
            "type": "Q"  # Quarterly reports
        }

        try:
            data = self.get(endpoint, params=params)

            if not data.get("results"):
                logger.warning(f"No fundamental data found for {symbol}")
                return {"symbol": symbol}

            # Extract the latest report
            report = data["results"][0]

            # Extract key metrics
            fundamentals = {
                "symbol": symbol,
                "report_date": report.get("reportPeriod"),
                "revenue": report.get("revenue"),
                "net_income": report.get("netIncome"),
                "eps": report.get("eps"),
                "assets": report.get("assets"),
                "liabilities": report.get("liabilities"),
                "equity": report.get("equity")
            }

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {"symbol": symbol, "error": str(e)}

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5)
    def fetch_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            limit: Maximum number of news articles to return

        Returns:
            List of dictionaries containing news articles
        """
        logger.info(f"Fetching news for {symbol}")

        endpoint = f"/v2/reference/news"
        params = {
            "ticker": symbol,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc"
        }

        try:
            data = self.get(endpoint, params=params)

            if not data.get("results"):
                logger.warning(f"No news found for {symbol}")
                return []

            # Extract and normalize news articles
            articles = []
            for article in data["results"]:
                articles.append({
                    "title": article.get("title"),
                    "url": article.get("article_url"),
                    "published_at": article.get("published_utc"),
                    "source": article.get("publisher", {}).get("name")
                })

            return articles

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []


def fetch_ticks(symbol: str, start: datetime, end: datetime, timespan: str = "minute") -> pd.DataFrame:
    """
    Convenience function to fetch tick data from Polygon.io.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        start: Start datetime
        end: End datetime
        timespan: Time granularity, either "second" or "minute"

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    client = PolygonClient()
    return client.fetch_ticks(symbol, start, end, timespan)


def fetch_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Convenience function to fetch fundamental data from Polygon.io.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary with fundamental data
    """
    client = PolygonClient()
    return client.fetch_fundamentals(symbol)


def fetch_news(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch news from Polygon.io.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        limit: Maximum number of news articles to return

    Returns:
        List of dictionaries containing news articles
    """
    client = PolygonClient()
    return client.fetch_news(symbol, limit)
