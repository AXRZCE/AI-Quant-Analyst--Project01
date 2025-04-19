"""
Tests for the data pipeline module.

This module contains tests for the data ingestion pipeline, including
API clients, data validation, and Delta Lake storage.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging

import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest.polygon_client import PolygonClient
from src.ingest.yahoo_client import YahooFinanceClient
from src.ingest.news_client import NewsAPIClient
from src.ingest.data_validator import DataValidator, validate_and_clean_tick_data
from src.ingest.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataPipeline(unittest.TestCase):
    """Test cases for the data pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_ticks = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        self.sample_news = [
            {
                'title': 'Test Article 1',
                'url': 'https://example.com/article1',
                'published_at': '2023-01-01T12:00:00Z',
                'source': 'Test Source'
            },
            {
                'title': 'Test Article 2',
                'url': 'https://example.com/article2',
                'published_at': '2023-01-02T12:00:00Z',
                'source': 'Test Source'
            }
        ]
        
        self.sample_fundamentals = {
            'symbol': 'AAPL',
            'report_date': '2023-01-01',
            'revenue': 100000000.0,
            'net_income': 20000000.0,
            'eps': 1.5
        }
    
    @patch('src.ingest.polygon_client.PolygonClient')
    @patch('src.ingest.yahoo_client.YahooFinanceClient')
    def test_fetch_stock_data_polygon_success(self, mock_yahoo, mock_polygon):
        """Test fetching stock data from Polygon.io."""
        # Mock the Polygon client
        mock_polygon_instance = mock_polygon.return_value
        mock_polygon_instance.fetch_ticks.return_value = self.sample_ticks
        
        # Create pipeline with mocked clients
        pipeline = DataPipeline()
        pipeline.polygon_client = mock_polygon_instance
        pipeline.yahoo_client = mock_yahoo.return_value
        
        # Test fetching data
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        result = pipeline.fetch_stock_data(symbol, start, end, prefer_source='polygon')
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['close'].iloc[-1], 106.0)
        
        # Verify Polygon client was called
        mock_polygon_instance.fetch_ticks.assert_called_once()
        
        # Verify Yahoo client was not called
        mock_yahoo.return_value.fetch_ticks.assert_not_called()
    
    @patch('src.ingest.polygon_client.PolygonClient')
    @patch('src.ingest.yahoo_client.YahooFinanceClient')
    def test_fetch_stock_data_polygon_failure_yahoo_fallback(self, mock_yahoo, mock_polygon):
        """Test fallback to Yahoo Finance when Polygon.io fails."""
        # Mock the Polygon client to fail
        mock_polygon_instance = mock_polygon.return_value
        mock_polygon_instance.fetch_ticks.return_value = pd.DataFrame()
        
        # Mock the Yahoo client to succeed
        mock_yahoo_instance = mock_yahoo.return_value
        mock_yahoo_instance.fetch_ticks.return_value = self.sample_ticks
        
        # Create pipeline with mocked clients
        pipeline = DataPipeline()
        pipeline.polygon_client = mock_polygon_instance
        pipeline.yahoo_client = mock_yahoo_instance
        
        # Test fetching data
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        result = pipeline.fetch_stock_data(symbol, start, end, prefer_source='polygon')
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 5)
        
        # Verify both clients were called
        mock_polygon_instance.fetch_ticks.assert_called_once()
        mock_yahoo_instance.fetch_ticks.assert_called_once()
    
    @patch('src.ingest.polygon_client.PolygonClient')
    def test_fetch_fundamentals(self, mock_polygon):
        """Test fetching fundamental data."""
        # Mock the Polygon client
        mock_polygon_instance = mock_polygon.return_value
        mock_polygon_instance.fetch_fundamentals.return_value = self.sample_fundamentals
        
        # Create pipeline with mocked client
        pipeline = DataPipeline()
        pipeline.polygon_client = mock_polygon_instance
        
        # Test fetching data
        symbol = 'AAPL'
        
        result = pipeline.fetch_fundamentals(symbol)
        
        # Verify the result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['eps'], 1.5)
        
        # Verify client was called
        mock_polygon_instance.fetch_fundamentals.assert_called_once_with(symbol)
    
    @patch('src.ingest.news_client.NewsAPIClient')
    def test_fetch_news(self, mock_news):
        """Test fetching news articles."""
        # Mock the NewsAPI client
        mock_news_instance = mock_news.return_value
        mock_news_instance.fetch_news.return_value = self.sample_news
        
        # Create pipeline with mocked client
        pipeline = DataPipeline()
        pipeline.news_client = mock_news_instance
        pipeline.polygon_client = None  # Disable Polygon for this test
        
        # Test fetching data
        keywords = ['AAPL', 'Apple']
        since = datetime(2023, 1, 1)
        
        result = pipeline.fetch_news(keywords, since)
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'Test Article 1')
        
        # Verify client was called
        mock_news_instance.fetch_news.assert_called_once()
    
    def test_validate_tick_data(self):
        """Test validating tick data."""
        # Create sample data with issues
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'open': [100.0, 101.0, -102.0, 103.0, 104.0],  # Negative value
            'high': [95.0, 106.0, 107.0, 108.0, 109.0],  # High < Low
            'low': [105.0, 96.0, 97.0, 98.0, 99.0],  # Low > High
            'close': [102.0, 103.0, None, 105.0, 106.0],  # NaN value
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Add duplicate timestamp
        df = pd.concat([df, df.iloc[[0]]])
        
        # Validate data
        df_clean, report = DataValidator.validate_tick_data(df, 'AAPL')
        
        # Verify the result
        self.assertEqual(len(df_clean), 5)  # Duplicates removed
        self.assertFalse(df_clean['open'].lt(0).any())  # No negative values
        self.assertFalse(df_clean['high'].lt(df_clean['low']).any())  # No high < low
        self.assertFalse(df_clean['close'].isna().any())  # No NaN values
        
        # Verify the report
        self.assertEqual(report['status'], 'issues_fixed')
        self.assertEqual(len(report['issues']), 4)  # 4 types of issues
    
    @patch('src.ingest.data_pipeline.validate_and_clean_tick_data')
    @patch('src.ingest.data_pipeline.save_ticks_to_delta')
    def test_ingest_stock_data(self, mock_save, mock_validate):
        """Test ingesting stock data."""
        # Mock the validation and storage functions
        mock_validate.return_value = self.sample_ticks
        mock_save.return_value = True
        
        # Create pipeline with mocked fetch method
        pipeline = DataPipeline()
        pipeline.fetch_stock_data = MagicMock(return_value=self.sample_ticks)
        
        # Test ingesting data
        symbol = 'AAPL'
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 5)
        
        result = pipeline.ingest_stock_data(symbol, start, end)
        
        # Verify the result
        self.assertEqual(len(result), 5)
        
        # Verify mocks were called
        pipeline.fetch_stock_data.assert_called_once_with(symbol, start, end, '1d', 'polygon')
        mock_validate.assert_called_once()
        mock_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
