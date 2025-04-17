"""
Unit tests for the News API client.
"""
import json
import os
from datetime import datetime, timedelta
from unittest import mock

import pytest
from requests.exceptions import RequestException

from src.ingest.news_client import NewsAPIClient, RavenPackClient, fetch_news


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
def mock_newsapi_response():
    """Sample response data from NewsAPI."""
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {
                    "id": "bloomberg",
                    "name": "Bloomberg"
                },
                "author": "John Doe",
                "title": "Stock Market Surges on Economic Data",
                "description": "Markets respond positively to latest economic indicators.",
                "url": "https://www.bloomberg.com/news/articles/2021-07-01/stock-market-surges",
                "urlToImage": "https://www.bloomberg.com/image.jpg",
                "publishedAt": "2021-07-01T12:00:00Z",
                "content": "Full article content here..."
            },
            {
                "source": {
                    "id": "financial-times",
                    "name": "Financial Times"
                },
                "author": "Jane Smith",
                "title": "Tech Stocks Lead Market Rally",
                "description": "Technology sector outperforms broader market.",
                "url": "https://www.ft.com/content/tech-stocks-rally",
                "urlToImage": "https://www.ft.com/image.jpg",
                "publishedAt": "2021-07-01T14:30:00Z",
                "content": "Full article content here..."
            }
        ]
    }


@pytest.fixture
def mock_ravenpack_response():
    """Sample response data from RavenPack API."""
    return {
        "status": "ok",
        "data": [
            {
                "headline": "Stock Market Surges on Economic Data",
                "url": "https://www.bloomberg.com/news/articles/2021-07-01/stock-market-surges",
                "timestamp": "2021-07-01T12:00:00Z",
                "source_name": "Bloomberg"
            },
            {
                "headline": "Tech Stocks Lead Market Rally",
                "url": "https://www.ft.com/content/tech-stocks-rally",
                "timestamp": "2021-07-01T14:30:00Z",
                "source_name": "Financial Times"
            }
        ]
    }


@pytest.fixture
def mock_newsapi_client():
    """Create a mock NewsAPI client with a mocked API key."""
    with mock.patch.dict(os.environ, {"NEWS_API_KEY": "mock_api_key"}):
        return NewsAPIClient()


@pytest.fixture
def mock_ravenpack_client():
    """Create a mock RavenPack client with a mocked API key."""
    with mock.patch.dict(os.environ, {"RAVENPACK_API_KEY": "mock_api_key"}):
        return RavenPackClient()


def test_newsapi_client_init():
    """Test NewsAPIClient initialization."""
    # Test with explicit API key
    client = NewsAPIClient(api_key="test_key")
    assert client.api_key == "test_key"
    
    # Test with environment variable
    with mock.patch.dict(os.environ, {"NEWS_API_KEY": "env_key"}):
        client = NewsAPIClient()
        assert client.api_key == "env_key"
    
    # Test with missing API key
    with mock.patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError):
            NewsAPIClient()


def test_ravenpack_client_init():
    """Test RavenPackClient initialization."""
    # Test with explicit API key
    client = RavenPackClient(api_key="test_key")
    assert client.api_key == "test_key"
    
    # Test with environment variable
    with mock.patch.dict(os.environ, {"RAVENPACK_API_KEY": "env_key"}):
        client = RavenPackClient()
        assert client.api_key == "env_key"
    
    # Test with missing API key
    with mock.patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError):
            RavenPackClient()


@mock.patch("src.ingest.news_client.requests.Session.get")
def test_newsapi_fetch_news_success(mock_get, mock_newsapi_response, mock_newsapi_client):
    """Test successful fetch_news method for NewsAPI."""
    mock_get.return_value = MockResponse(mock_newsapi_response)
    
    since = datetime(2021, 7, 1)
    keywords = ["stocks", "market"]
    
    articles = mock_newsapi_client.fetch_news(keywords, since)
    
    # Check that the API was called with correct parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "everything" in args[0]
    assert "stocks OR market" in kwargs["params"]["q"]
    assert "2021-07-01" in kwargs["params"]["from"]
    
    # Check articles structure and content
    assert len(articles) == 2
    assert articles[0]["title"] == "Stock Market Surges on Economic Data"
    assert articles[0]["url"] == "https://www.bloomberg.com/news/articles/2021-07-01/stock-market-surges"
    assert articles[0]["published_at"] == "2021-07-01T12:00:00Z"
    assert articles[0]["source"] == "Bloomberg"


@mock.patch("src.ingest.news_client.requests.Session.get")
def test_newsapi_fetch_news_pagination(mock_get, mock_newsapi_client):
    """Test pagination in fetch_news method for NewsAPI."""
    # Create responses for pagination
    first_page = {
        "status": "ok",
        "totalResults": 101,  # More than one page
        "articles": [{"title": "Article 1", "url": "url1", "publishedAt": "2021-07-01T12:00:00Z", "source": {"name": "Source1"}}]
    }
    
    second_page = {
        "status": "ok",
        "totalResults": 101,
        "articles": [{"title": "Article 2", "url": "url2", "publishedAt": "2021-07-01T13:00:00Z", "source": {"name": "Source2"}}]
    }
    
    # Set up mock to return different responses on consecutive calls
    mock_get.side_effect = [
        MockResponse(first_page),
        MockResponse(second_page),
        MockResponse({"status": "ok", "articles": []})  # Empty third page to end pagination
    ]
    
    since = datetime(2021, 7, 1)
    keywords = ["stocks"]
    
    articles = mock_newsapi_client.fetch_news(keywords, since, page_size=1, max_pages=3)
    
    # Check that multiple API calls were made with different page numbers
    assert mock_get.call_count == 3
    
    # Check combined results
    assert len(articles) == 2
    assert articles[0]["title"] == "Article 1"
    assert articles[1]["title"] == "Article 2"


@mock.patch("src.ingest.news_client.requests.Session.get")
def test_newsapi_fetch_news_empty_results(mock_get, mock_newsapi_client):
    """Test fetch_news with empty results for NewsAPI."""
    mock_get.return_value = MockResponse({"status": "ok", "totalResults": 0, "articles": []})
    
    since = datetime(2021, 7, 1)
    keywords = ["nonexistent"]
    
    articles = mock_newsapi_client.fetch_news(keywords, since)
    
    # Check that an empty list is returned
    assert articles == []


@mock.patch("src.ingest.news_client.requests.Session.get")
def test_newsapi_fetch_news_api_error(mock_get, mock_newsapi_client):
    """Test fetch_news with API error for NewsAPI."""
    mock_get.return_value = MockResponse({}, raise_error=True)
    
    since = datetime(2021, 7, 1)
    keywords = ["stocks"]
    
    with pytest.raises(Exception) as excinfo:
        mock_newsapi_client.fetch_news(keywords, since)
    
    assert "Error fetching data from NewsAPI" in str(excinfo.value)


@mock.patch("src.ingest.news_client.requests.Session.get")
def test_ravenpack_fetch_news_success(mock_get, mock_ravenpack_response, mock_ravenpack_client):
    """Test successful fetch_news method for RavenPack."""
    mock_get.return_value = MockResponse(mock_ravenpack_response)
    
    since = datetime(2021, 7, 1)
    keywords = ["stocks", "market"]
    
    articles = mock_ravenpack_client.fetch_news(keywords, since)
    
    # Check that the API was called with correct parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert "news" in args[0]
    assert 'keyword:"stocks" OR keyword:"market"' in kwargs["params"]["having"]
    
    # Check articles structure and content
    assert len(articles) == 2
    assert articles[0]["title"] == "Stock Market Surges on Economic Data"
    assert articles[0]["url"] == "https://www.bloomberg.com/news/articles/2021-07-01/stock-market-surges"
    assert articles[0]["published_at"] == "2021-07-01T12:00:00Z"
    assert articles[0]["source"] == "Bloomberg"


@mock.patch("src.ingest.news_client.NewsAPIClient")
@mock.patch("src.ingest.news_client.RavenPackClient")
def test_fetch_news_function_newsapi(mock_ravenpack_class, mock_newsapi_class):
    """Test the convenience function fetch_news with NewsAPI."""
    # Create mock instances
    mock_newsapi_instance = mock.MagicMock()
    mock_newsapi_class.return_value = mock_newsapi_instance
    
    # Set up the mock return value
    expected_articles = [{"title": "Test Article"}]
    mock_newsapi_instance.fetch_news.return_value = expected_articles
    
    # Call the function with default (NewsAPI)
    since = datetime(2021, 7, 1)
    keywords = ["stocks"]
    result = fetch_news(keywords, since)
    
    # Verify the client was created and method was called
    mock_newsapi_class.assert_called_once()
    mock_newsapi_instance.fetch_news.assert_called_once_with(keywords, since)
    mock_ravenpack_class.assert_not_called()
    
    # Check result
    assert result == expected_articles


@mock.patch("src.ingest.news_client.NewsAPIClient")
@mock.patch("src.ingest.news_client.RavenPackClient")
def test_fetch_news_function_ravenpack(mock_ravenpack_class, mock_newsapi_class):
    """Test the convenience function fetch_news with RavenPack."""
    # Create mock instances
    mock_ravenpack_instance = mock.MagicMock()
    mock_ravenpack_class.return_value = mock_ravenpack_instance
    
    # Set up the mock return value
    expected_articles = [{"title": "Test Article"}]
    mock_ravenpack_instance.fetch_news.return_value = expected_articles
    
    # Call the function with RavenPack
    since = datetime(2021, 7, 1)
    keywords = ["stocks"]
    result = fetch_news(keywords, since, use_ravenpack=True)
    
    # Verify the client was created and method was called
    mock_ravenpack_class.assert_called_once()
    mock_ravenpack_instance.fetch_news.assert_called_once_with(keywords, since)
    mock_newsapi_class.assert_not_called()
    
    # Check result
    assert result == expected_articles
