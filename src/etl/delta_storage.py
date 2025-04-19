"""
Delta Lake storage module for data persistence.

This module provides functions for storing and retrieving data from Delta Lake,
ensuring proper partitioning and organization.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, hour

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DELTA_ROOT = os.getenv('DELTA_ROOT', 'data/delta')
DEFAULT_RAW_PATH = f"{DEFAULT_DELTA_ROOT}/raw"
DEFAULT_BRONZE_PATH = f"{DEFAULT_DELTA_ROOT}/bronze"
DEFAULT_SILVER_PATH = f"{DEFAULT_DELTA_ROOT}/silver"
DEFAULT_GOLD_PATH = f"{DEFAULT_DELTA_ROOT}/gold"

# Ensure directories exist
for path in [DEFAULT_RAW_PATH, DEFAULT_BRONZE_PATH, DEFAULT_SILVER_PATH, DEFAULT_GOLD_PATH]:
    os.makedirs(path, exist_ok=True)


class DeltaLakeStorage:
    """Delta Lake storage class for data persistence."""
    
    def __init__(
        self,
        app_name: str = "QuantAnalyst",
        raw_path: str = DEFAULT_RAW_PATH,
        bronze_path: str = DEFAULT_BRONZE_PATH,
        silver_path: str = DEFAULT_SILVER_PATH,
        gold_path: str = DEFAULT_GOLD_PATH
    ):
        """
        Initialize the Delta Lake storage.
        
        Args:
            app_name: Spark application name
            raw_path: Path to raw data
            bronze_path: Path to bronze data
            silver_path: Path to silver data
            gold_path: Path to gold data
        """
        logger.info("Initializing Delta Lake storage")
        
        self.raw_path = raw_path
        self.bronze_path = bronze_path
        self.silver_path = silver_path
        self.gold_path = gold_path
        
        # Initialize Spark session with Delta Lake
        builder = SparkSession.builder.appName(app_name) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info("Spark session initialized with Delta Lake")
    
    def close(self):
        """Close the Spark session."""
        if hasattr(self, 'spark') and self.spark:
            self.spark.stop()
            logger.info("Spark session closed")
    
    def _get_table_path(self, table_name: str, layer: str = "bronze") -> str:
        """
        Get the path for a Delta table.
        
        Args:
            table_name: Name of the table
            layer: Data layer (raw, bronze, silver, gold)
            
        Returns:
            Path to the Delta table
        """
        if layer == "raw":
            return os.path.join(self.raw_path, table_name)
        elif layer == "bronze":
            return os.path.join(self.bronze_path, table_name)
        elif layer == "silver":
            return os.path.join(self.silver_path, table_name)
        elif layer == "gold":
            return os.path.join(self.gold_path, table_name)
        else:
            raise ValueError(f"Invalid layer: {layer}")
    
    def _get_tick_schema(self) -> StructType:
        """
        Get the schema for tick data.
        
        Returns:
            Spark schema for tick data
        """
        return StructType([
            StructField("symbol", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("vwap", DoubleType(), True),
            StructField("transactions", IntegerType(), True)
        ])
    
    def _get_news_schema(self) -> StructType:
        """
        Get the schema for news data.
        
        Returns:
            Spark schema for news data
        """
        return StructType([
            StructField("title", StringType(), True),
            StructField("url", StringType(), False),
            StructField("published_at", TimestampType(), False),
            StructField("source", StringType(), True),
            StructField("description", StringType(), True),
            StructField("content", StringType(), True),
            StructField("sentiment", DoubleType(), True),
            StructField("relevance", DoubleType(), True)
        ])
    
    def _get_fundamentals_schema(self) -> StructType:
        """
        Get the schema for fundamental data.
        
        Returns:
            Spark schema for fundamental data
        """
        return StructType([
            StructField("symbol", StringType(), False),
            StructField("report_date", StringType(), False),
            StructField("revenue", DoubleType(), True),
            StructField("net_income", DoubleType(), True),
            StructField("eps", DoubleType(), True),
            StructField("assets", DoubleType(), True),
            StructField("liabilities", DoubleType(), True),
            StructField("equity", DoubleType(), True)
        ])
    
    def save_ticks(
        self,
        df: pd.DataFrame,
        symbol: str,
        mode: str = "append",
        partition_cols: List[str] = None
    ) -> bool:
        """
        Save tick data to Delta Lake.
        
        Args:
            df: DataFrame with tick data
            symbol: Stock ticker symbol
            mode: Write mode (append, overwrite)
            partition_cols: Columns to partition by
            
        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, not saving to Delta Lake")
            return False
        
        try:
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Default partition columns if not provided
            if partition_cols is None:
                partition_cols = ['symbol', 'year', 'month', 'day']
                
                # Add date parts for partitioning
                df['year'] = df['timestamp'].dt.year
                df['month'] = df['timestamp'].dt.month
                df['day'] = df['timestamp'].dt.day
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Save to Delta Lake
            table_path = self._get_table_path("ticks", "bronze")
            
            spark_df.write.format("delta") \
                .mode(mode) \
                .partitionBy(*partition_cols) \
                .save(table_path)
            
            logger.info(f"Saved {len(df)} tick records for {symbol} to Delta Lake")
            return True
            
        except Exception as e:
            logger.error(f"Error saving tick data to Delta Lake: {str(e)}")
            return False
    
    def save_news(
        self,
        articles: List[Dict[str, Any]],
        mode: str = "append",
        partition_cols: List[str] = None
    ) -> bool:
        """
        Save news articles to Delta Lake.
        
        Args:
            articles: List of news article dictionaries
            mode: Write mode (append, overwrite)
            partition_cols: Columns to partition by
            
        Returns:
            True if successful, False otherwise
        """
        if not articles:
            logger.warning("Empty articles list, not saving to Delta Lake")
            return False
        
        try:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(articles)
            
            # Convert published_at to datetime if it's not already
            if 'published_at' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['published_at']):
                df['published_at'] = pd.to_datetime(df['published_at'])
            
            # Default partition columns if not provided
            if partition_cols is None:
                partition_cols = ['year', 'month', 'day']
                
                # Add date parts for partitioning
                if 'published_at' in df.columns:
                    df['year'] = df['published_at'].dt.year
                    df['month'] = df['published_at'].dt.month
                    df['day'] = df['published_at'].dt.day
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Save to Delta Lake
            table_path = self._get_table_path("news", "bronze")
            
            spark_df.write.format("delta") \
                .mode(mode) \
                .partitionBy(*partition_cols) \
                .save(table_path)
            
            logger.info(f"Saved {len(articles)} news articles to Delta Lake")
            return True
            
        except Exception as e:
            logger.error(f"Error saving news data to Delta Lake: {str(e)}")
            return False
    
    def save_fundamentals(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        mode: str = "append",
        partition_cols: List[str] = None
    ) -> bool:
        """
        Save fundamental data to Delta Lake.
        
        Args:
            data: Dictionary or list of dictionaries with fundamental data
            mode: Write mode (append, overwrite)
            partition_cols: Columns to partition by
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to list if it's a single dictionary
            if isinstance(data, dict):
                data = [data]
            
            if not data:
                logger.warning("Empty fundamentals data, not saving to Delta Lake")
                return False
            
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)
            
            # Default partition columns if not provided
            if partition_cols is None:
                partition_cols = ['symbol']
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Save to Delta Lake
            table_path = self._get_table_path("fundamentals", "bronze")
            
            spark_df.write.format("delta") \
                .mode(mode) \
                .partitionBy(*partition_cols) \
                .save(table_path)
            
            logger.info(f"Saved {len(data)} fundamental records to Delta Lake")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fundamental data to Delta Lake: {str(e)}")
            return False
    
    def load_ticks(
        self,
        symbol: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Load tick data from Delta Lake.
        
        Args:
            symbol: Stock ticker symbol to filter by
            start_date: Start date to filter by
            end_date: End date to filter by
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with tick data
        """
        try:
            table_path = self._get_table_path("ticks", "bronze")
            
            # Check if table exists
            if not os.path.exists(table_path):
                logger.warning(f"Ticks table does not exist at {table_path}")
                return pd.DataFrame()
            
            # Build query
            query = self.spark.read.format("delta").load(table_path)
            
            # Apply filters
            if symbol:
                query = query.filter(col("symbol") == symbol)
            
            if start_date:
                query = query.filter(col("timestamp") >= start_date)
            
            if end_date:
                query = query.filter(col("timestamp") <= end_date)
            
            # Order by timestamp
            query = query.orderBy("timestamp")
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Convert to pandas DataFrame
            df = query.toPandas()
            
            logger.info(f"Loaded {len(df)} tick records from Delta Lake")
            return df
            
        except Exception as e:
            logger.error(f"Error loading tick data from Delta Lake: {str(e)}")
            return pd.DataFrame()
    
    def load_news(
        self,
        keywords: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Load news articles from Delta Lake.
        
        Args:
            keywords: Keywords to filter by (searches in title)
            start_date: Start date to filter by
            end_date: End date to filter by
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with news articles
        """
        try:
            table_path = self._get_table_path("news", "bronze")
            
            # Check if table exists
            if not os.path.exists(table_path):
                logger.warning(f"News table does not exist at {table_path}")
                return pd.DataFrame()
            
            # Build query
            query = self.spark.read.format("delta").load(table_path)
            
            # Apply filters
            if keywords:
                # Filter by any of the keywords in the title
                keyword_conditions = [col("title").contains(keyword) for keyword in keywords]
                combined_condition = keyword_conditions[0]
                for condition in keyword_conditions[1:]:
                    combined_condition = combined_condition | condition
                query = query.filter(combined_condition)
            
            if start_date:
                query = query.filter(col("published_at") >= start_date)
            
            if end_date:
                query = query.filter(col("published_at") <= end_date)
            
            # Order by published_at
            query = query.orderBy("published_at", ascending=False)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Convert to pandas DataFrame
            df = query.toPandas()
            
            logger.info(f"Loaded {len(df)} news articles from Delta Lake")
            return df
            
        except Exception as e:
            logger.error(f"Error loading news data from Delta Lake: {str(e)}")
            return pd.DataFrame()
    
    def load_fundamentals(
        self,
        symbol: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Load fundamental data from Delta Lake.
        
        Args:
            symbol: Stock ticker symbol to filter by
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with fundamental data
        """
        try:
            table_path = self._get_table_path("fundamentals", "bronze")
            
            # Check if table exists
            if not os.path.exists(table_path):
                logger.warning(f"Fundamentals table does not exist at {table_path}")
                return pd.DataFrame()
            
            # Build query
            query = self.spark.read.format("delta").load(table_path)
            
            # Apply filters
            if symbol:
                query = query.filter(col("symbol") == symbol)
            
            # Order by report_date
            query = query.orderBy("report_date", ascending=False)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Convert to pandas DataFrame
            df = query.toPandas()
            
            logger.info(f"Loaded {len(df)} fundamental records from Delta Lake")
            return df
            
        except Exception as e:
            logger.error(f"Error loading fundamental data from Delta Lake: {str(e)}")
            return pd.DataFrame()


# Singleton instance for easy access
_delta_storage = None

def get_delta_storage() -> DeltaLakeStorage:
    """
    Get the singleton instance of DeltaLakeStorage.
    
    Returns:
        DeltaLakeStorage instance
    """
    global _delta_storage
    if _delta_storage is None:
        _delta_storage = DeltaLakeStorage()
    return _delta_storage


def save_ticks_to_delta(df: pd.DataFrame, symbol: str) -> bool:
    """
    Convenience function to save tick data to Delta Lake.
    
    Args:
        df: DataFrame with tick data
        symbol: Stock ticker symbol
        
    Returns:
        True if successful, False otherwise
    """
    return get_delta_storage().save_ticks(df, symbol)


def save_news_to_delta(articles: List[Dict[str, Any]]) -> bool:
    """
    Convenience function to save news articles to Delta Lake.
    
    Args:
        articles: List of news article dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    return get_delta_storage().save_news(articles)


def save_fundamentals_to_delta(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
    """
    Convenience function to save fundamental data to Delta Lake.
    
    Args:
        data: Dictionary or list of dictionaries with fundamental data
        
    Returns:
        True if successful, False otherwise
    """
    return get_delta_storage().save_fundamentals(data)


def load_ticks_from_delta(symbol: str = None, start_date: datetime = None, end_date: datetime = None, limit: int = None) -> pd.DataFrame:
    """
    Convenience function to load tick data from Delta Lake.
    
    Args:
        symbol: Stock ticker symbol to filter by
        start_date: Start date to filter by
        end_date: End date to filter by
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with tick data
    """
    return get_delta_storage().load_ticks(symbol, start_date, end_date, limit)


def load_news_from_delta(keywords: List[str] = None, start_date: datetime = None, end_date: datetime = None, limit: int = None) -> pd.DataFrame:
    """
    Convenience function to load news articles from Delta Lake.
    
    Args:
        keywords: Keywords to filter by (searches in title)
        start_date: Start date to filter by
        end_date: End date to filter by
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with news articles
    """
    return get_delta_storage().load_news(keywords, start_date, end_date, limit)


def load_fundamentals_from_delta(symbol: str = None, limit: int = None) -> pd.DataFrame:
    """
    Convenience function to load fundamental data from Delta Lake.
    
    Args:
        symbol: Stock ticker symbol to filter by
        limit: Maximum number of records to return
        
    Returns:
        DataFrame with fundamental data
    """
    return get_delta_storage().load_fundamentals(symbol, limit)
