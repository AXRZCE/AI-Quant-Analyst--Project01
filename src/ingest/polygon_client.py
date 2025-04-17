"""
Polygon.io API client for fetching stock market tick data.
"""
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from requests.exceptions import RequestException


class PolygonClient:
    """Client for interacting with the Polygon.io API."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Polygon.io API client.
        
        Args:
            api_key: Polygon.io API key. If not provided, will look for POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key must be provided or set as POLYGON_API_KEY env var")
        
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
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
        if timespan not in ["second", "minute"]:
            raise ValueError("timespan must be either 'second' or 'minute'")
        
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
                    response = self.session.get(next_url)
                else:
                    response = self.session.get(f"{self.BASE_URL}{endpoint}", params=params)
                
                response.raise_for_status()
                data = response.json()
                
                # Check if we have results
                if data.get("results") and len(data["results"]) > 0:
                    all_results.extend(data["results"])
                
                # Check for next page
                next_url = data.get("next_url")
                if not next_url:
                    break
                
                # Increment request counter and add small delay to avoid rate limits
                request_count += 1
                time.sleep(0.1)
                
            except RequestException as e:
                raise Exception(f"Error fetching data from Polygon API: {str(e)}")
        
        # Convert results to DataFrame
        if not all_results:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        df = pd.DataFrame(all_results)
        
        # Rename columns to standard format
        column_mapping = {
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Select and order columns
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        return df[columns]


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
