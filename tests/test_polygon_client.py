"""
Unit tests for the Polygon.io API client.
"""
import json
import os
from datetime import datetime, timedelta
from unittest import mock

import pandas as pd
import pytest
from requests.exceptions import RequestException

from src.ingest.polygon_client import PolygonClient, fetch_ticks


class MockResponse:
    """Mock response object for requests."""
    
    def __init__(self, json_data, status_code=200, raise_error=False):
        self.json_data = json_data
        self.status_code = status_code
        self.raise_error = raise_error
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.raise_error:
            raise RequestException("Mock API error")


@pytest.fixture
def mock_polygon_response():
    """Sample response data from Polygon API."""
    return {
        "ticker": "AAPL",
        "status": "OK",
        "queryCount": 2,
        "resultsCount": 2,
        "adjusted": True,
        "results": [
            {
                "v": 10000,
                "o": 150.0,
                "c": 152.0,
                "h": 153.0,
                "l": 149.0,
                "t": 1625097600000,  # 2021-07-01 00:00:00 UTC
                "n": 100
            },
            {
                "v": 12000,
                "o": 152.0,
                "c": 154.0,
                "h": 155.0,
                "l": 151.0,
                "t": 1625097660000,  # 2021-07-01 00:01:00 UTC
                "n": 120
            }
        ],
        "next_url": None
    }


@pytest.fixture
def mock_polygon_client():
    """Create a mock Polygon client with a mocked API key."""
    with mock.patch.dict(os.environ, {"POLYGON_API_KEY": "mock_api_key"}):
        return PolygonClient()


def test_polygon_client_init():
    """Test PolygonClient initialization."""
    # Test with explicit API key
    client = PolygonClient(api_key="test_key")
    assert client.api_key == "test_key"
    
    # Test with environment variable
    with mock.patch.dict(os.environ, {"POLYGON_API_KEY": "env_key"}):
        client = PolygonClient()
        assert client.api_key == "env_key"
    
    # Test with missing API key
    with mock.patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError):
            PolygonClient()


@mock.patch("src.ingest.polygon_client.requests.Session.get")
def test_fetch_ticks_success(mock_get, mock_polygon_response, mock_polygon_client):
    """Test successful fetch_ticks method."""
    mock_get.return_value = MockResponse(mock_polygon_response)
    
    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 2)
    
    df = mock_polygon_client.fetch_ticks("AAPL", start, end)
    
    # Check that the API was called with correct parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "AAPL" in args[0]
    assert "2021-07-01" in args[0]
    assert "2021-07-02" in args[0]
    
    # Check DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Check data conversion
    assert df["timestamp"].iloc[0] == pd.Timestamp("2021-07-01 00:00:00", unit="ms")
    assert df["open"].iloc[0] == 150.0
    assert df["close"].iloc[0] == 152.0


@mock.patch("src.ingest.polygon_client.requests.Session.get")
def test_fetch_ticks_pagination(mock_get, mock_polygon_client):
    """Test pagination in fetch_ticks method."""
    # Create responses for pagination
    first_response = {
        "ticker": "AAPL",
        "status": "OK",
        "results": [{"v": 10000, "o": 150.0, "c": 152.0, "h": 153.0, "l": 149.0, "t": 1625097600000}],
        "next_url": "https://api.polygon.io/next_page"
    }
    
    second_response = {
        "ticker": "AAPL",
        "status": "OK",
        "results": [{"v": 12000, "o": 152.0, "c": 154.0, "h": 155.0, "l": 151.0, "t": 1625097660000}],
        "next_url": None
    }
    
    # Set up mock to return different responses on consecutive calls
    mock_get.side_effect = [
        MockResponse(first_response),
        MockResponse(second_response)
    ]
    
    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 2)
    
    df = mock_polygon_client.fetch_ticks("AAPL", start, end)
    
    # Check that both API calls were made
    assert mock_get.call_count == 2
    
    # Check DataFrame has combined results
    assert len(df) == 2


@mock.patch("src.ingest.polygon_client.requests.Session.get")
def test_fetch_ticks_empty_results(mock_get, mock_polygon_client):
    """Test fetch_ticks with empty results."""
    mock_get.return_value = MockResponse({"status": "OK", "results": []})
    
    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 2)
    
    df = mock_polygon_client.fetch_ticks("AAPL", start, end)
    
    # Check that an empty DataFrame with correct columns is returned
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


@mock.patch("src.ingest.polygon_client.requests.Session.get")
def test_fetch_ticks_api_error(mock_get, mock_polygon_client):
    """Test fetch_ticks with API error."""
    mock_get.return_value = MockResponse({}, raise_error=True)
    
    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 2)
    
    with pytest.raises(Exception) as excinfo:
        mock_polygon_client.fetch_ticks("AAPL", start, end)
    
    assert "Error fetching data from Polygon API" in str(excinfo.value)


@mock.patch("src.ingest.polygon_client.PolygonClient")
def test_fetch_ticks_function(mock_client_class):
    """Test the convenience function fetch_ticks."""
    # Create a mock instance
    mock_instance = mock.MagicMock()
    mock_client_class.return_value = mock_instance
    
    # Set up the mock return value
    expected_df = pd.DataFrame({"timestamp": [pd.Timestamp("2021-07-01")]})
    mock_instance.fetch_ticks.return_value = expected_df
    
    # Call the function
    start = datetime(2021, 7, 1)
    end = datetime(2021, 7, 2)
    result = fetch_ticks("AAPL", start, end)
    
    # Verify the client was created and method was called
    mock_client_class.assert_called_once()
    mock_instance.fetch_ticks.assert_called_once_with("AAPL", start, end, "minute")
    
    # Check result
    assert result is expected_df
