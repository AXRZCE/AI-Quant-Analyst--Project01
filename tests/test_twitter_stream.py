"""
Unit tests for the Twitter API v2 streaming client.
"""
import json
import os
from datetime import datetime
from unittest import mock

import pytest
from requests.exceptions import ChunkedEncodingError, RequestException

from src.ingest.twitter_stream import TweetStreamer


class MockResponse:
    """Mock response object for requests."""

    def __init__(self, json_data=None, status_code=200, raise_error=False, stream_data=None):
        self.json_data = json_data or {}
        self.status_code = status_code
        self.raise_error = raise_error
        self.stream_data = stream_data or []
        self._iter_lines_called = False

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.raise_error:
            raise RequestException("Mock API error")

    def iter_lines(self):
        self._iter_lines_called = True
        for line in self.stream_data:
            yield line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_twitter_rules_response():
    """Sample response data for Twitter rules endpoint."""
    return {
        "data": [
            {
                "id": "1234567890",
                "value": "stocks OR finance"
            }
        ],
        "meta": {
            "sent": "2021-07-01T12:00:00Z"
        }
    }


@pytest.fixture
def mock_twitter_stream_data():
    """Sample stream data from Twitter API v2."""
    tweet1 = {
        "data": {
            "id": "1234567890",
            "text": "The stock market is looking bullish today! #stocks #investing",
            "created_at": "2021-07-01T12:00:00Z",
            "author_id": "987654321"
        },
        "includes": {
            "users": [
                {
                    "id": "987654321",
                    "username": "investor123"
                }
            ]
        }
    }

    tweet2 = {
        "data": {
            "id": "1234567891",
            "text": "Just bought some $AAPL shares. Looking good for Q3 earnings.",
            "created_at": "2021-07-01T12:05:00Z",
            "author_id": "987654322"
        },
        "includes": {
            "users": [
                {
                    "id": "987654322",
                    "username": "techtrader"
                }
            ]
        }
    }

    return [
        json.dumps(tweet1).encode(),
        b'',  # Empty line
        json.dumps(tweet2).encode()
    ]


@pytest.fixture
def mock_twitter_streamer():
    """Create a mock Twitter streamer with a mocked bearer token."""
    with mock.patch.dict(os.environ, {"TWITTER_BEARER_TOKEN": "mock_bearer_token"}):
        return TweetStreamer()


def test_twitter_streamer_init():
    """Test TweetStreamer initialization."""
    # Test with explicit bearer token
    streamer = TweetStreamer(bearer_token="test_token")
    assert streamer.bearer_token == "test_token"

    # Test with environment variable
    with mock.patch.dict(os.environ, {"TWITTER_BEARER_TOKEN": "env_token"}):
        streamer = TweetStreamer()
        assert streamer.bearer_token == "env_token"

    # Test with missing bearer token
    with mock.patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError):
            TweetStreamer()


@mock.patch("src.ingest.twitter_stream.requests.Session.get")
def test_get_rules(mock_get, mock_twitter_rules_response, mock_twitter_streamer):
    """Test _get_rules method."""
    mock_get.return_value = MockResponse(mock_twitter_rules_response)

    rules = mock_twitter_streamer._get_rules()

    # Check that the API was called correctly
    mock_get.assert_called_once_with(mock_twitter_streamer.RULES_URL)

    # Check rules content
    assert len(rules) == 1
    assert rules[0]["id"] == "1234567890"
    assert rules[0]["value"] == "stocks OR finance"


@mock.patch("src.ingest.twitter_stream.requests.Session.post")
@mock.patch("src.ingest.twitter_stream.TweetStreamer._get_rules")
def test_delete_all_rules(mock_get_rules, mock_post, mock_twitter_streamer):
    """Test _delete_all_rules method."""
    # Mock existing rules
    mock_get_rules.return_value = [{"id": "1234567890"}, {"id": "0987654321"}]
    mock_post.return_value = MockResponse()

    mock_twitter_streamer._delete_all_rules()

    # Check that the API was called with correct payload
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == mock_twitter_streamer.RULES_URL
    assert kwargs["json"]["delete"]["ids"] == ["1234567890", "0987654321"]


@mock.patch("src.ingest.twitter_stream.requests.Session.post")
@mock.patch("src.ingest.twitter_stream.TweetStreamer._delete_all_rules")
def test_set_rules(mock_delete_rules, mock_post, mock_twitter_streamer):
    """Test _set_rules method."""
    mock_post.return_value = MockResponse()

    keywords = ["stocks", "finance", "investing", "nasdaq", "dow jones", "sp500"]
    mock_twitter_streamer._set_rules(keywords)

    # Check that delete_all_rules was called
    mock_delete_rules.assert_called_once()

    # Check that the API was called with correct payload
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == mock_twitter_streamer.RULES_URL

    # Check that rules were batched correctly (max 5 per rule)
    rules = kwargs["json"]["add"]
    assert len(rules) == 2
    assert "stocks OR finance OR investing OR nasdaq OR dow jones" in rules[0]["value"]
    assert "sp500" in rules[1]["value"]


@mock.patch("src.ingest.twitter_stream.requests.Session.get")
@mock.patch("src.ingest.twitter_stream.TweetStreamer._set_rules")
def test_start_stream(mock_set_rules, mock_get, mock_twitter_stream_data, mock_twitter_streamer):
    """Test start_stream method."""
    # Mock the streaming response
    mock_get.return_value = MockResponse(stream_data=mock_twitter_stream_data)

    # Create a mock callback function
    mock_callback = mock.MagicMock()

    # Call start_stream with a very short timeout to avoid hanging tests
    mock_twitter_streamer.start_stream(
        callback=mock_callback,
        keywords=["stocks", "finance"],
        timeout=0.1,  # Very short timeout
        max_retries=1
    )

    # Check that set_rules was called with the keywords
    mock_set_rules.assert_called_once_with(["stocks", "finance"])

    # Check that the streaming API was called with correct parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == mock_twitter_streamer.STREAM_URL
    assert "tweet.fields" in kwargs["params"]
    assert kwargs["stream"] is True

    # Check that the callback was called for each tweet
    assert mock_callback.call_count == 2

    # Check the content of the first callback
    first_call_args = mock_callback.call_args_list[0][0][0]
    assert first_call_args["id"] == "1234567890"
    assert "stock market" in first_call_args["text"]
    assert first_call_args["username"] == "investor123"


@mock.patch("src.ingest.twitter_stream.requests.Session.get")
@mock.patch("src.ingest.twitter_stream.TweetStreamer._set_rules")
def test_start_stream_connection_error(mock_set_rules, mock_get, mock_twitter_streamer):
    """Test start_stream method with connection error."""
    # Mock a connection error on the first call, then success
    mock_get.side_effect = [
        ChunkedEncodingError("Connection broken"),
        MockResponse(stream_data=[b'{"data": {"id": "123", "text": "test", "created_at": "2021-07-01T12:00:00Z", "author_id": "456"}}'])
    ]

    # Create a mock callback function
    mock_callback = mock.MagicMock()

    # Call start_stream with a short timeout and retry
    with mock.patch("src.ingest.twitter_stream.time.sleep") as mock_sleep:
        mock_twitter_streamer.start_stream(
            callback=mock_callback,
            timeout=0.1,  # Very short timeout
            max_retries=2
        )

    # Check that sleep was called for backoff
    mock_sleep.assert_called_once()

    # Check that get was called twice (retry after error)
    assert mock_get.call_count == 2


@mock.patch("src.ingest.twitter_stream.TweetStreamer._delete_all_rules")
def test_stop_stream(mock_delete_rules, mock_twitter_streamer):
    """Test stop_stream method."""
    mock_twitter_streamer.stop_stream()

    # Check that delete_all_rules was called
    mock_delete_rules.assert_called_once()
