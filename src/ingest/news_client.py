"""
News API client for fetching financial news.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import requests
from requests.exceptions import RequestException


class NewsAPIClient:
    """Client for interacting with the NewsAPI Pro."""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NewsAPI client.
        
        Args:
            api_key: NewsAPI key. If not provided, will look for NEWS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key must be provided or set as NEWS_API_KEY env var")
        
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": self.api_key})
    
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
                
                response = self.session.get(f"{self.BASE_URL}/everything", params=params)
                response.raise_for_status()
                data = response.json()
                
                # Check if we have articles
                articles = data.get("articles", [])
                if not articles:
                    break
                
                all_articles.extend(articles)
                
                # Check if we've reached the last page
                total_results = data.get("totalResults", 0)
                if page * page_size >= total_results:
                    break
                
                page += 1
                
            except RequestException as e:
                raise Exception(f"Error fetching data from NewsAPI: {str(e)}")
        
        # Extract and normalize the required fields
        normalized_articles = []
        for article in all_articles:
            normalized_articles.append({
                "title": article.get("title"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "source": article.get("source", {}).get("name")
            })
        
        return normalized_articles


class RavenPackClient:
    """Client for interacting with the RavenPack API."""
    
    BASE_URL = "https://api.ravenpack.com/1.0"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RavenPack API client.
        
        Args:
            api_key: RavenPack API key. If not provided, will look for RAVENPACK_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("RAVENPACK_API_KEY")
        if not self.api_key:
            raise ValueError("RavenPack API key must be provided or set as RAVENPACK_API_KEY env var")
        
        self.session = requests.Session()
        self.session.headers.update({"API-KEY": self.api_key})
    
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
            
            response = self.session.get(f"{self.BASE_URL}/json/news", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract and normalize the required fields
            normalized_articles = []
            for article in data.get("data", []):
                normalized_articles.append({
                    "title": article.get("headline"),
                    "url": article.get("url"),
                    "published_at": article.get("timestamp"),
                    "source": article.get("source_name")
                })
            
            return normalized_articles
            
        except RequestException as e:
            raise Exception(f"Error fetching data from RavenPack API: {str(e)}")


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
