"""
News API client for fetching financial news.
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests
from requests.exceptions import RequestException

from src.ingest.base_client import BaseAPIClient

# Configure logging
logger = logging.getLogger(__name__)


class NewsAPIClient(BaseAPIClient):
    """Client for interacting with the NewsAPI Pro."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, backoff_factor: float = 0.5):
        """
        Initialize the NewsAPI client.

        Args:
            api_key: NewsAPI key. If not provided, will look for NEWS_API_KEY env var.
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key must be provided or set as NEWS_API_KEY env var")

        # Initialize the base client
        super().__init__(
            base_url="https://newsapi.org/v2",
            api_key=self.api_key,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )

        # Update session headers for NewsAPI authentication
        self.session.headers.update({"X-Api-Key": self.api_key})

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5)
    def fetch_news(
        self,
        keywords: List[str],
        since: datetime,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
        max_pages: int = 5
    ) -> List[Dict]:
        """
        Fetch news articles related to the given keywords.

        Args:
            keywords: List of keywords to search for
            since: Fetch articles published since this datetime
            language: Language of articles (default: English)
            sort_by: Sorting order (relevancy, popularity, publishedAt)
            page_size: Number of results per page (max 100)
            max_pages: Maximum number of pages to fetch

        Returns:
            List of dictionaries containing article information
        """
        logger.info(f"Fetching news for keywords: {keywords} since {since}")

        # Validate inputs
        if not keywords:
            logger.warning("No keywords provided for news search")
            return []

        if page_size > 100:
            logger.warning(f"Page size {page_size} exceeds maximum of 100, setting to 100")
            page_size = 100

        if sort_by not in ["relevancy", "popularity", "publishedAt"]:
            logger.warning(f"Invalid sort_by value: {sort_by}, defaulting to publishedAt")
            sort_by = "publishedAt"

        # Format the query string (keywords joined with OR)
        query = " OR ".join(keywords)

        # Format date for NewsAPI
        from_date = since.strftime("%Y-%m-%d")

        # Initialize results and pagination variables
        all_articles = []
        page = 1

        # Paginate through results
        while page <= max_pages:
            try:
                params = {
                    "q": query,
                    "from": from_date,
                    "language": language,
                    "sortBy": sort_by,
                    "pageSize": page_size,
                    "page": page
                }

                logger.debug(f"Fetching news page {page} with params: {params}")

                response = self.session.get(f"{self.base_url}/everything", params=params, timeout=self.timeout)
                data = self.handle_response(response)

                # Check if we have articles
                articles = data.get("articles", [])
                if not articles:
                    logger.info(f"No more articles found after page {page-1}")
                    break

                logger.info(f"Retrieved {len(articles)} articles on page {page}")
                all_articles.extend(articles)

                # Check if we've reached the last page
                total_results = data.get("totalResults", 0)
                if page * page_size >= total_results:
                    logger.info(f"Reached end of results (total: {total_results})")
                    break

                page += 1

            except Exception as e:
                logger.error(f"Error fetching data from NewsAPI: {str(e)}")
                raise

        logger.info(f"Total articles retrieved: {len(all_articles)}")

        # Extract and normalize the required fields
        normalized_articles = []
        for article in all_articles:
            try:
                normalized_articles.append({
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name"),
                    "description": article.get("description"),
                    "content": article.get("content")
                })
            except Exception as e:
                logger.warning(f"Error processing article: {str(e)}")

        return normalized_articles


class RavenPackClient(BaseAPIClient):
    """Client for interacting with the RavenPack API."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3, backoff_factor: float = 0.5):
        """
        Initialize the RavenPack API client.

        Args:
            api_key: RavenPack API key. If not provided, will look for RAVENPACK_API_KEY env var.
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        self.api_key = api_key or os.environ.get("RAVENPACK_API_KEY")
        if not self.api_key:
            raise ValueError("RavenPack API key must be provided or set as RAVENPACK_API_KEY env var")

        # Initialize the base client
        super().__init__(
            base_url="https://api.ravenpack.com/1.0",
            api_key=self.api_key,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )

        # Update session headers for RavenPack authentication
        self.session.headers.update({"API-KEY": self.api_key})

    @BaseAPIClient.with_retry(max_retries=3, backoff_factor=0.5)
    def fetch_news(
        self,
        keywords: List[str],
        since: datetime,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        Fetch news articles from RavenPack related to the given keywords.

        Args:
            keywords: List of keywords to search for
            since: Fetch articles published since this datetime
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing article information
        """
        logger.info(f"Fetching news from RavenPack for keywords: {keywords} since {since}")

        # Validate inputs
        if not keywords:
            logger.warning("No keywords provided for RavenPack search")
            return []

        # Format the query string for RavenPack
        keyword_filter = " OR ".join([f'keyword:"{k}"' for k in keywords])

        # Format date for RavenPack API
        from_timestamp = int(since.timestamp() * 1000)  # Convert to milliseconds

        try:
            params = {
                "having": keyword_filter,
                "timestamp": f">{from_timestamp}",
                "limit": max_results
            }

            logger.debug(f"Fetching RavenPack news with params: {params}")

            response = self.session.get(f"{self.base_url}/json/news", params=params, timeout=self.timeout)
            data = self.handle_response(response)

            articles = data.get("data", [])
            logger.info(f"Retrieved {len(articles)} articles from RavenPack")

            # Extract and normalize the required fields
            normalized_articles = []
            for article in articles:
                try:
                    normalized_articles.append({
                        "title": article.get("headline"),
                        "url": article.get("url"),
                        "published_at": article.get("timestamp"),
                        "source": article.get("source_name"),
                        "sentiment": article.get("sentiment_score"),
                        "relevance": article.get("relevance")
                    })
                except Exception as e:
                    logger.warning(f"Error processing RavenPack article: {str(e)}")

            return normalized_articles

        except Exception as e:
            logger.error(f"Error fetching data from RavenPack API: {str(e)}")
            raise


def fetch_news(keywords: List[str], since: datetime, use_ravenpack: bool = False) -> List[Dict]:
    """
    Convenience function to fetch news from either NewsAPI or RavenPack.

    Args:
        keywords: List of keywords to search for
        since: Fetch articles published since this datetime
        use_ravenpack: Whether to use RavenPack instead of NewsAPI

    Returns:
        List of dictionaries containing article information
    """
    if use_ravenpack:
        client = RavenPackClient()
    else:
        client = NewsAPIClient()

    return client.fetch_news(keywords, since)
