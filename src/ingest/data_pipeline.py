"""
Data ingestion pipeline module.

This module provides a pipeline for ingesting data from various sources,
validating it, and storing it in Delta Lake.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from src.ingest.polygon_client import PolygonClient, fetch_ticks, fetch_fundamentals, fetch_news as fetch_polygon_news
from src.ingest.yahoo_client import YahooFinanceClient
from src.ingest.news_client import NewsAPIClient, fetch_news as fetch_news_api
from src.ingest.data_validator import validate_and_clean_tick_data, validate_and_clean_news, validate_and_clean_fundamentals
from src.etl.delta_storage import save_ticks_to_delta, save_news_to_delta, save_fundamentals_to_delta

# Configure logging
logger = logging.getLogger(__name__)


class DataPipeline:
    """Data ingestion pipeline for fetching, validating, and storing data."""
    
    def __init__(
        self,
        use_polygon: bool = True,
        use_yahoo: bool = True,
        use_newsapi: bool = True,
        use_cache: bool = True,
        max_retries: int = 3,
        backoff_factor: float = 0.5
    ):
        """
        Initialize the data pipeline.
        
        Args:
            use_polygon: Whether to use Polygon.io API
            use_yahoo: Whether to use Yahoo Finance API
            use_newsapi: Whether to use NewsAPI
            use_cache: Whether to use caching for API requests
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
        """
        logger.info("Initializing data pipeline")
        
        self.use_polygon = use_polygon
        self.use_yahoo = use_yahoo
        self.use_newsapi = use_newsapi
        
        # Initialize API clients
        if use_polygon:
            try:
                self.polygon_client = PolygonClient(max_retries=max_retries, backoff_factor=backoff_factor)
                logger.info("Polygon.io client initialized")
            except Exception as e:
                logger.error(f"Error initializing Polygon.io client: {str(e)}")
                self.polygon_client = None
                self.use_polygon = False
        else:
            self.polygon_client = None
        
        if use_yahoo:
            try:
                self.yahoo_client = YahooFinanceClient(use_cache=use_cache, max_retries=max_retries, backoff_factor=backoff_factor)
                logger.info("Yahoo Finance client initialized")
            except Exception as e:
                logger.error(f"Error initializing Yahoo Finance client: {str(e)}")
                self.yahoo_client = None
                self.use_yahoo = False
        else:
            self.yahoo_client = None
        
        if use_newsapi:
            try:
                self.news_client = NewsAPIClient(max_retries=max_retries, backoff_factor=backoff_factor)
                logger.info("NewsAPI client initialized")
            except Exception as e:
                logger.error(f"Error initializing NewsAPI client: {str(e)}")
                self.news_client = None
                self.use_newsapi = False
        else:
            self.news_client = None
    
    def fetch_stock_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        prefer_source: str = "polygon"
    ) -> pd.DataFrame:
        """
        Fetch stock price data for a given symbol and time range.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            interval: Time interval (e.g., "1d", "1h")
            prefer_source: Preferred data source ("polygon" or "yahoo")
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching stock data for {symbol} from {start} to {end}")
        
        # Try preferred source first
        if prefer_source == "polygon" and self.use_polygon:
            try:
                logger.info(f"Trying Polygon.io for {symbol}")
                # Convert interval to Polygon format
                polygon_interval = "minute" if interval in ["1m", "5m", "15m", "30m", "60m"] else "day"
                df = self.polygon_client.fetch_ticks(symbol, start, end, polygon_interval)
                
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} records from Polygon.io")
                    return df
                logger.warning(f"No data returned from Polygon.io for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data from Polygon.io: {str(e)}")
        
        # Try Yahoo Finance if Polygon failed or wasn't preferred
        if self.use_yahoo:
            try:
                logger.info(f"Trying Yahoo Finance for {symbol}")
                df = self.yahoo_client.fetch_ticks(symbol, start, end, interval)
                
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance")
                    return df
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        
        # If we preferred Yahoo but it failed, try Polygon as fallback
        if prefer_source == "yahoo" and self.use_polygon:
            try:
                logger.info(f"Trying Polygon.io as fallback for {symbol}")
                # Convert interval to Polygon format
                polygon_interval = "minute" if interval in ["1m", "5m", "15m", "30m", "60m"] else "day"
                df = self.polygon_client.fetch_ticks(symbol, start, end, polygon_interval)
                
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} records from Polygon.io")
                    return df
                logger.warning(f"No data returned from Polygon.io for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data from Polygon.io: {str(e)}")
        
        # If all sources failed, return empty DataFrame
        logger.error(f"All data sources failed for {symbol}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            Dictionary with fundamental data
        """
        logger.info(f"Fetching fundamental data for {symbol}")
        
        # Try Polygon.io first
        if self.use_polygon:
            try:
                logger.info(f"Trying Polygon.io for {symbol} fundamentals")
                data = self.polygon_client.fetch_fundamentals(symbol)
                
                if data and len(data) > 1:  # More than just the symbol
                    logger.info(f"Successfully fetched fundamentals from Polygon.io")
                    return data
                logger.warning(f"No fundamental data returned from Polygon.io for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching fundamentals from Polygon.io: {str(e)}")
        
        # If Polygon failed, return empty data
        logger.error(f"All data sources failed for {symbol} fundamentals")
        return {"symbol": symbol}
    
    def fetch_news(
        self,
        keywords: List[str],
        since: datetime,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles related to the given keywords.
        
        Args:
            keywords: List of keywords to search for
            since: Fetch articles published since this datetime
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing article information
        """
        logger.info(f"Fetching news for keywords: {keywords} since {since}")
        
        all_articles = []
        
        # Try NewsAPI
        if self.use_newsapi:
            try:
                logger.info(f"Trying NewsAPI for news")
                articles = self.news_client.fetch_news(keywords, since)
                
                if articles:
                    logger.info(f"Successfully fetched {len(articles)} articles from NewsAPI")
                    all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching news from NewsAPI: {str(e)}")
        
        # Try Polygon.io news
        if self.use_polygon:
            try:
                logger.info(f"Trying Polygon.io for news")
                for keyword in keywords:
                    articles = self.polygon_client.fetch_news(keyword)
                    
                    if articles:
                        logger.info(f"Successfully fetched {len(articles)} articles from Polygon.io for {keyword}")
                        all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching news from Polygon.io: {str(e)}")
        
        # Deduplicate articles by URL
        if all_articles:
            unique_articles = []
            urls_seen = set()
            
            for article in all_articles:
                url = article.get("url")
                if url and url not in urls_seen:
                    unique_articles.append(article)
                    urls_seen.add(url)
            
            logger.info(f"Deduplicated to {len(unique_articles)} unique articles")
            
            # Limit to max_results
            if len(unique_articles) > max_results:
                logger.info(f"Limiting to {max_results} articles")
                unique_articles = unique_articles[:max_results]
            
            return unique_articles
        
        logger.warning("No news articles found")
        return []
    
    def ingest_stock_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        prefer_source: str = "polygon",
        save_to_delta: bool = True
    ) -> pd.DataFrame:
        """
        Ingest stock price data for a given symbol and time range.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            interval: Time interval (e.g., "1d", "1h")
            prefer_source: Preferred data source ("polygon" or "yahoo")
            save_to_delta: Whether to save the data to Delta Lake
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Fetch data
        df = self.fetch_stock_data(symbol, start, end, interval, prefer_source)
        
        if df.empty:
            logger.warning(f"No data to ingest for {symbol}")
            return df
        
        # Validate and clean data
        df_clean = validate_and_clean_tick_data(df, symbol)
        
        # Save to Delta Lake if requested
        if save_to_delta:
            try:
                success = save_ticks_to_delta(df_clean, symbol)
                if success:
                    logger.info(f"Successfully saved {len(df_clean)} records to Delta Lake")
                else:
                    logger.error("Failed to save data to Delta Lake")
            except Exception as e:
                logger.error(f"Error saving data to Delta Lake: {str(e)}")
        
        return df_clean
    
    def ingest_fundamentals(
        self,
        symbol: str,
        save_to_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest fundamental data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            save_to_delta: Whether to save the data to Delta Lake
            
        Returns:
            Dictionary with fundamental data
        """
        # Fetch data
        data = self.fetch_fundamentals(symbol)
        
        if not data or (len(data) == 1 and "symbol" in data):
            logger.warning(f"No fundamental data to ingest for {symbol}")
            return data
        
        # Validate and clean data
        data_clean = validate_and_clean_fundamentals(data, symbol)
        
        # Save to Delta Lake if requested
        if save_to_delta:
            try:
                success = save_fundamentals_to_delta(data_clean)
                if success:
                    logger.info(f"Successfully saved fundamental data to Delta Lake")
                else:
                    logger.error("Failed to save fundamental data to Delta Lake")
            except Exception as e:
                logger.error(f"Error saving fundamental data to Delta Lake: {str(e)}")
        
        return data_clean
    
    def ingest_news(
        self,
        keywords: List[str],
        since: datetime,
        max_results: int = 100,
        save_to_delta: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ingest news articles related to the given keywords.
        
        Args:
            keywords: List of keywords to search for
            since: Fetch articles published since this datetime
            max_results: Maximum number of results to return
            save_to_delta: Whether to save the data to Delta Lake
            
        Returns:
            List of dictionaries containing article information
        """
        # Fetch data
        articles = self.fetch_news(keywords, since, max_results)
        
        if not articles:
            logger.warning(f"No news articles to ingest")
            return articles
        
        # Validate and clean data
        articles_clean = validate_and_clean_news(articles)
        
        # Save to Delta Lake if requested
        if save_to_delta:
            try:
                success = save_news_to_delta(articles_clean)
                if success:
                    logger.info(f"Successfully saved {len(articles_clean)} articles to Delta Lake")
                else:
                    logger.error("Failed to save news articles to Delta Lake")
            except Exception as e:
                logger.error(f"Error saving news articles to Delta Lake: {str(e)}")
        
        return articles_clean
    
    def ingest_all_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
        news_keywords: Optional[List[str]] = None,
        save_to_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest all data (stock prices, fundamentals, news) for the given symbols.
        
        Args:
            symbols: List of stock ticker symbols
            start: Start datetime
            end: End datetime
            interval: Time interval for stock data
            news_keywords: List of keywords for news search (defaults to symbols)
            save_to_delta: Whether to save the data to Delta Lake
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting all data for symbols: {symbols}")
        
        results = {
            "stock_data": {},
            "fundamentals": {},
            "news": None
        }
        
        # Ingest stock data for each symbol
        for symbol in symbols:
            try:
                df = self.ingest_stock_data(symbol, start, end, interval, save_to_delta=save_to_delta)
                results["stock_data"][symbol] = {
                    "success": not df.empty,
                    "records": len(df)
                }
            except Exception as e:
                logger.error(f"Error ingesting stock data for {symbol}: {str(e)}")
                results["stock_data"][symbol] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Ingest fundamentals for each symbol
        for symbol in symbols:
            try:
                data = self.ingest_fundamentals(symbol, save_to_delta=save_to_delta)
                results["fundamentals"][symbol] = {
                    "success": bool(data and len(data) > 1),
                    "fields": list(data.keys()) if data else []
                }
            except Exception as e:
                logger.error(f"Error ingesting fundamentals for {symbol}: {str(e)}")
                results["fundamentals"][symbol] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Ingest news
        if news_keywords is None:
            news_keywords = symbols
        
        try:
            articles = self.ingest_news(news_keywords, start, save_to_delta=save_to_delta)
            results["news"] = {
                "success": bool(articles),
                "articles": len(articles)
            }
        except Exception as e:
            logger.error(f"Error ingesting news: {str(e)}")
            results["news"] = {
                "success": False,
                "error": str(e)
            }
        
        return results


# Singleton instance for easy access
_data_pipeline = None

def get_data_pipeline() -> DataPipeline:
    """
    Get the singleton instance of DataPipeline.
    
    Returns:
        DataPipeline instance
    """
    global _data_pipeline
    if _data_pipeline is None:
        _data_pipeline = DataPipeline()
    return _data_pipeline


def ingest_stock_data(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1d",
    prefer_source: str = "polygon",
    save_to_delta: bool = True
) -> pd.DataFrame:
    """
    Convenience function to ingest stock price data.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        start: Start datetime
        end: End datetime
        interval: Time interval (e.g., "1d", "1h")
        prefer_source: Preferred data source ("polygon" or "yahoo")
        save_to_delta: Whether to save the data to Delta Lake
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    return get_data_pipeline().ingest_stock_data(symbol, start, end, interval, prefer_source, save_to_delta)


def ingest_fundamentals(symbol: str, save_to_delta: bool = True) -> Dict[str, Any]:
    """
    Convenience function to ingest fundamental data.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        save_to_delta: Whether to save the data to Delta Lake
        
    Returns:
        Dictionary with fundamental data
    """
    return get_data_pipeline().ingest_fundamentals(symbol, save_to_delta)


def ingest_news(
    keywords: List[str],
    since: datetime,
    max_results: int = 100,
    save_to_delta: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest news articles.
    
    Args:
        keywords: List of keywords to search for
        since: Fetch articles published since this datetime
        max_results: Maximum number of results to return
        save_to_delta: Whether to save the data to Delta Lake
        
    Returns:
        List of dictionaries containing article information
    """
    return get_data_pipeline().ingest_news(keywords, since, max_results, save_to_delta)


def ingest_all_data(
    symbols: List[str],
    start: datetime,
    end: datetime,
    interval: str = "1d",
    news_keywords: Optional[List[str]] = None,
    save_to_delta: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to ingest all data.
    
    Args:
        symbols: List of stock ticker symbols
        start: Start datetime
        end: End datetime
        interval: Time interval for stock data
        news_keywords: List of keywords for news search (defaults to symbols)
        save_to_delta: Whether to save the data to Delta Lake
        
    Returns:
        Dictionary with ingestion results
    """
    return get_data_pipeline().ingest_all_data(symbols, start, end, interval, news_keywords, save_to_delta)
