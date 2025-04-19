"""
Data source classes for backtesting.

This module provides classes for loading and preprocessing data for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str = "base_data_source"):
        """
        Initialize the data source.
        
        Args:
            name: Data source name
        """
        self.name = name
    
    @abstractmethod
    def get_data(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get data for the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with market data
        """
        pass


class CSVDataSource(DataSource):
    """Data source that loads data from CSV files."""
    
    def __init__(
        self,
        name: str = "csv_data_source",
        data_dir: str = "data",
        filename_template: str = "{symbol}.csv",
        date_column: str = "timestamp",
        price_column: str = "close",
        symbols: List[str] = None
    ):
        """
        Initialize the CSV data source.
        
        Args:
            name: Data source name
            data_dir: Directory containing CSV files
            filename_template: Template for CSV filenames
            date_column: Name of the date column
            price_column: Name of the price column
            symbols: List of symbols to load
        """
        super().__init__(name)
        self.data_dir = data_dir
        self.filename_template = filename_template
        self.date_column = date_column
        self.price_column = price_column
        self.symbols = symbols or []
        self.data_cache = {}  # symbol -> DataFrame
    
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a single symbol.
        
        Args:
            symbol: Symbol to load
            
        Returns:
            DataFrame with market data for the symbol
        """
        # Check if data is already cached
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        # Construct filename
        filename = self.filename_template.format(symbol=symbol)
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # Convert date column to datetime
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df = df.set_index(self.date_column)
            
            # Cache data
            self.data_cache[symbol] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for symbol {symbol}: {e}")
            return pd.DataFrame()
    
    def get_data(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get data for the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with market data
        """
        # Override symbols if provided
        symbols = kwargs.get('symbols', self.symbols)
        
        if not symbols:
            logger.warning("No symbols specified")
            return pd.DataFrame()
        
        # Load data for each symbol
        symbol_data = {}
        
        for symbol in symbols:
            df = self.load_symbol_data(symbol)
            
            if df.empty:
                continue
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                logger.warning(f"No data for symbol {symbol} in date range {start_date} to {end_date}")
                continue
            
            symbol_data[symbol] = df
        
        if not symbol_data:
            logger.warning("No data available for any symbol")
            return pd.DataFrame()
        
        # Get common dates
        common_dates = None
        
        for symbol, df in symbol_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        if not common_dates:
            logger.warning("No common dates across symbols")
            return pd.DataFrame()
        
        # Create combined DataFrame
        combined_data = pd.DataFrame(index=sorted(common_dates))
        
        for symbol, df in symbol_data.items():
            # Filter by common dates
            df = df.loc[df.index.isin(common_dates)]
            
            # Add price column
            if self.price_column in df.columns:
                combined_data[f"{symbol}_price"] = df[self.price_column]
            
            # Add other columns
            for col in df.columns:
                if col != self.price_column:
                    combined_data[f"{symbol}_{col}"] = df[col]
        
        return combined_data


class YahooFinanceDataSource(DataSource):
    """Data source that loads data from Yahoo Finance."""
    
    def __init__(
        self,
        name: str = "yahoo_finance_data_source",
        symbols: List[str] = None,
        cache_dir: Optional[str] = "data/cache"
    ):
        """
        Initialize the Yahoo Finance data source.
        
        Args:
            name: Data source name
            symbols: List of symbols to load
            cache_dir: Directory for caching data (None to disable caching)
        """
        super().__init__(name)
        self.symbols = symbols or []
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_data(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get data for the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with market data
        """
        # Override symbols if provided
        symbols = kwargs.get('symbols', self.symbols)
        
        if not symbols:
            logger.warning("No symbols specified")
            return pd.DataFrame()
        
        try:
            # Import yfinance
            import yfinance as yf
        except ImportError:
            logger.error("yfinance package not installed")
            return pd.DataFrame()
        
        # Check if data is cached
        cached_data = None
        
        if self.cache_dir:
            cache_file = os.path.join(
                self.cache_dir,
                f"yahoo_finance_{'-'.join(symbols)}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
            )
            
            if os.path.exists(cache_file):
                try:
                    cached_data = pd.read_pickle(cache_file)
                    logger.info(f"Loaded cached data from {cache_file}")
                except Exception as e:
                    logger.warning(f"Error loading cached data: {e}")
        
        if cached_data is not None:
            return cached_data
        
        # Download data
        try:
            # Download data for all symbols
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                logger.warning("No data downloaded")
                return pd.DataFrame()
            
            # Reshape data
            if len(symbols) == 1:
                # Single symbol
                symbol = symbols[0]
                data = data.rename(columns={col: f"{symbol}_{col.lower()}" for col in data.columns})
            else:
                # Multiple symbols
                data = data.stack().reset_index()
                data.columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # Reshape to wide format
                data_wide = pd.DataFrame(index=data['Date'].unique())
                
                for symbol in symbols:
                    symbol_data = data[data['Symbol'] == symbol]
                    
                    if not symbol_data.empty:
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            data_wide[f"{symbol}_{col.lower()}"] = symbol_data.set_index('Date')[col]
                
                data = data_wide
            
            # Add price columns
            for symbol in symbols:
                if f"{symbol}_close" in data.columns:
                    data[f"{symbol}_price"] = data[f"{symbol}_close"]
            
            # Cache data
            if self.cache_dir:
                try:
                    data.to_pickle(cache_file)
                    logger.info(f"Cached data to {cache_file}")
                except Exception as e:
                    logger.warning(f"Error caching data: {e}")
            
            return data
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return pd.DataFrame()


class SQLDataSource(DataSource):
    """Data source that loads data from a SQL database."""
    
    def __init__(
        self,
        name: str = "sql_data_source",
        connection_string: str = "sqlite:///data/market_data.db",
        table_name: str = "market_data",
        symbol_column: str = "symbol",
        date_column: str = "timestamp",
        price_column: str = "close",
        symbols: List[str] = None
    ):
        """
        Initialize the SQL data source.
        
        Args:
            name: Data source name
            connection_string: Database connection string
            table_name: Name of the table containing market data
            symbol_column: Name of the symbol column
            date_column: Name of the date column
            price_column: Name of the price column
            symbols: List of symbols to load
        """
        super().__init__(name)
        self.connection_string = connection_string
        self.table_name = table_name
        self.symbol_column = symbol_column
        self.date_column = date_column
        self.price_column = price_column
        self.symbols = symbols or []
    
    def get_data(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get data for the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with market data
        """
        # Override symbols if provided
        symbols = kwargs.get('symbols', self.symbols)
        
        if not symbols:
            logger.warning("No symbols specified")
            return pd.DataFrame()
        
        try:
            # Import sqlalchemy
            from sqlalchemy import create_engine, text
        except ImportError:
            logger.error("sqlalchemy package not installed")
            return pd.DataFrame()
        
        try:
            # Create engine
            engine = create_engine(self.connection_string)
            
            # Build query
            query = f"""
            SELECT *
            FROM {self.table_name}
            WHERE {self.date_column} >= :start_date
            AND {self.date_column} <= :end_date
            AND {self.symbol_column} IN :symbols
            ORDER BY {self.date_column}
            """
            
            # Execute query
            data = pd.read_sql(
                text(query),
                engine,
                params={
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': tuple(symbols)
                }
            )
            
            if data.empty:
                logger.warning("No data returned from database")
                return pd.DataFrame()
            
            # Convert date column to datetime
            if self.date_column in data.columns:
                data[self.date_column] = pd.to_datetime(data[self.date_column])
            
            # Reshape data to wide format
            data_wide = pd.DataFrame()
            
            for symbol in symbols:
                symbol_data = data[data[self.symbol_column] == symbol]
                
                if symbol_data.empty:
                    continue
                
                # Set date as index
                symbol_data = symbol_data.set_index(self.date_column)
                
                # Add columns to wide DataFrame
                for col in symbol_data.columns:
                    if col != self.symbol_column:
                        data_wide[f"{symbol}_{col}"] = symbol_data[col]
                
                # Add price column
                if self.price_column in symbol_data.columns:
                    data_wide[f"{symbol}_price"] = symbol_data[self.price_column]
            
            return data_wide
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()


class CombinedDataSource(DataSource):
    """Data source that combines data from multiple sources."""
    
    def __init__(
        self,
        name: str = "combined_data_source",
        data_sources: List[DataSource] = None
    ):
        """
        Initialize the combined data source.
        
        Args:
            name: Data source name
            data_sources: List of data sources
        """
        super().__init__(name)
        self.data_sources = data_sources or []
    
    def add_data_source(self, data_source: DataSource) -> None:
        """
        Add a data source.
        
        Args:
            data_source: Data source to add
        """
        self.data_sources.append(data_source)
    
    def get_data(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """
        Get data for the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with market data
        """
        if not self.data_sources:
            logger.warning("No data sources specified")
            return pd.DataFrame()
        
        # Get data from each source
        all_data = []
        
        for data_source in self.data_sources:
            data = data_source.get_data(start_date, end_date, **kwargs)
            
            if not data.empty:
                all_data.append(data)
        
        if not all_data:
            logger.warning("No data available from any source")
            return pd.DataFrame()
        
        # Combine data
        combined_data = pd.concat(all_data, axis=1)
        
        # Remove duplicate columns
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        
        return combined_data
