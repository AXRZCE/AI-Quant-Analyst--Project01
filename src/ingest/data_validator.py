"""
Data validation module for ensuring data quality.

This module provides functions for validating data quality and integrity
before it is stored in the data lake or used for feature engineering.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation class for ensuring data quality."""
    
    @staticmethod
    def validate_tick_data(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate tick data for quality and completeness.
        
        Args:
            df: DataFrame with tick data
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (cleaned_dataframe, validation_report)
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}")
            return df, {"status": "empty", "symbol": symbol, "issues": []}
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        issues = []
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
            issues.append({"type": "missing_columns", "columns": missing_columns})
            
            # Add missing columns with NaN values
            for col in missing_columns:
                df_clean[col] = np.nan
        
        # Check for NaN values in critical columns
        nan_counts = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns and df_clean[col].isna().any():
                nan_count = df_clean[col].isna().sum()
                nan_counts[col] = int(nan_count)
                logger.warning(f"Found {nan_count} NaN values in {col} column for {symbol}")
        
        if nan_counts:
            issues.append({"type": "nan_values", "counts": nan_counts})
            
            # Interpolate NaN values for price columns
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Check for duplicate timestamps
        if df_clean['timestamp'].duplicated().any():
            dup_count = df_clean['timestamp'].duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate timestamps for {symbol}")
            issues.append({"type": "duplicate_timestamps", "count": int(dup_count)})
            
            # Remove duplicates
            df_clean.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        
        # Check for out-of-order timestamps
        if not df_clean['timestamp'].equals(df_clean['timestamp'].sort_values()):
            logger.warning(f"Timestamps are not in order for {symbol}")
            issues.append({"type": "unordered_timestamps"})
            
            # Sort by timestamp
            df_clean.sort_values('timestamp', inplace=True)
        
        # Check for price anomalies (e.g., negative prices)
        anomaly_counts = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns and (df_clean[col] < 0).any():
                neg_count = (df_clean[col] < 0).sum()
                anomaly_counts[col] = int(neg_count)
                logger.warning(f"Found {neg_count} negative values in {col} column for {symbol}")
                
                # Replace negative values with NaN and then interpolate
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        if anomaly_counts:
            issues.append({"type": "negative_prices", "counts": anomaly_counts})
        
        # Check for high-low inconsistency
        if 'high' in df_clean.columns and 'low' in df_clean.columns:
            inconsistent_mask = df_clean['high'] < df_clean['low']
            if inconsistent_mask.any():
                inconsistent_count = inconsistent_mask.sum()
                logger.warning(f"Found {inconsistent_count} rows where high < low for {symbol}")
                issues.append({"type": "high_low_inconsistency", "count": int(inconsistent_count)})
                
                # Fix high-low inconsistency by swapping values
                inconsistent_rows = df_clean.loc[inconsistent_mask].index
                high_vals = df_clean.loc[inconsistent_rows, 'high'].copy()
                df_clean.loc[inconsistent_rows, 'high'] = df_clean.loc[inconsistent_rows, 'low']
                df_clean.loc[inconsistent_rows, 'low'] = high_vals
        
        # Check for gaps in time series
        if len(df_clean) > 1:
            # Get the most common time delta
            time_diffs = df_clean['timestamp'].diff().dropna()
            if not time_diffs.empty:
                most_common_delta = time_diffs.mode().iloc[0]
                
                # Check for gaps larger than 2x the most common delta
                large_gaps = time_diffs[time_diffs > 2 * most_common_delta]
                if not large_gaps.empty:
                    logger.warning(f"Found {len(large_gaps)} large gaps in time series for {symbol}")
                    issues.append({"type": "time_gaps", "count": len(large_gaps)})
        
        # Create validation report
        validation_report = {
            "status": "valid" if not issues else "issues_fixed",
            "symbol": symbol,
            "original_rows": len(df),
            "cleaned_rows": len(df_clean),
            "issues": issues
        }
        
        return df_clean, validation_report
    
    @staticmethod
    def validate_news_data(articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate news articles for quality and completeness.
        
        Args:
            articles: List of news article dictionaries
            
        Returns:
            Tuple of (cleaned_articles, validation_report)
        """
        if not articles:
            logger.warning("Empty articles list")
            return articles, {"status": "empty", "issues": []}
        
        # Make a copy to avoid modifying the original
        cleaned_articles = []
        issues = []
        
        # Required fields for news articles
        required_fields = ['title', 'url', 'published_at']
        
        # Count articles with missing fields
        missing_fields_count = 0
        duplicate_urls = set()
        urls_seen = set()
        
        for article in articles:
            # Check for required fields
            missing = [field for field in required_fields if not article.get(field)]
            
            if missing:
                missing_fields_count += 1
                logger.debug(f"Article missing required fields: {missing}")
                continue  # Skip articles with missing required fields
            
            # Check for duplicate URLs
            url = article.get('url')
            if url:
                if url in urls_seen:
                    duplicate_urls.add(url)
                    continue  # Skip duplicate articles
                urls_seen.add(url)
            
            # Add to cleaned articles
            cleaned_articles.append(article)
        
        # Record issues
        if missing_fields_count > 0:
            logger.warning(f"Found {missing_fields_count} articles with missing required fields")
            issues.append({"type": "missing_fields", "count": missing_fields_count})
        
        if duplicate_urls:
            logger.warning(f"Found {len(duplicate_urls)} duplicate article URLs")
            issues.append({"type": "duplicate_urls", "count": len(duplicate_urls)})
        
        # Create validation report
        validation_report = {
            "status": "valid" if not issues else "issues_fixed",
            "original_count": len(articles),
            "cleaned_count": len(cleaned_articles),
            "issues": issues
        }
        
        return cleaned_articles, validation_report
    
    @staticmethod
    def validate_fundamentals(data: Dict[str, Any], symbol: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate fundamental data for quality and completeness.
        
        Args:
            data: Dictionary with fundamental data
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (cleaned_data, validation_report)
        """
        if not data or (len(data) == 1 and 'symbol' in data):
            logger.warning(f"Empty or minimal fundamental data for {symbol}")
            return data, {"status": "empty", "symbol": symbol, "issues": []}
        
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()
        issues = []
        
        # Required fields for fundamental data
        required_fields = ['symbol', 'report_date']
        
        # Check for required fields
        missing = [field for field in required_fields if field not in cleaned_data]
        if missing:
            logger.warning(f"Fundamental data missing required fields for {symbol}: {missing}")
            issues.append({"type": "missing_fields", "fields": missing})
            
            # Add missing fields with default values
            if 'symbol' not in cleaned_data:
                cleaned_data['symbol'] = symbol
            
            if 'report_date' not in cleaned_data:
                cleaned_data['report_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Check for numeric fields that should be numeric
        numeric_fields = ['revenue', 'net_income', 'eps', 'assets', 'liabilities', 'equity']
        non_numeric = []
        
        for field in numeric_fields:
            if field in cleaned_data:
                try:
                    # Try to convert to float if it's not already
                    if not isinstance(cleaned_data[field], (int, float)) and cleaned_data[field] is not None:
                        cleaned_data[field] = float(cleaned_data[field])
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric value in {field} for {symbol}: {cleaned_data[field]}")
                    non_numeric.append(field)
                    cleaned_data[field] = None
        
        if non_numeric:
            issues.append({"type": "non_numeric_fields", "fields": non_numeric})
        
        # Create validation report
        validation_report = {
            "status": "valid" if not issues else "issues_fixed",
            "symbol": symbol,
            "issues": issues
        }
        
        return cleaned_data, validation_report


def validate_and_clean_tick_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convenience function to validate and clean tick data.
    
    Args:
        df: DataFrame with tick data
        symbol: Stock ticker symbol
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df, report = DataValidator.validate_tick_data(df, symbol)
    if report["status"] != "valid":
        logger.info(f"Data validation for {symbol}: Fixed {len(report['issues'])} issues")
    return cleaned_df


def validate_and_clean_news(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to validate and clean news articles.
    
    Args:
        articles: List of news article dictionaries
        
    Returns:
        Cleaned list of articles
    """
    cleaned_articles, report = DataValidator.validate_news_data(articles)
    if report["status"] != "valid":
        logger.info(f"News validation: Fixed {len(report['issues'])} issues")
    return cleaned_articles


def validate_and_clean_fundamentals(data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Convenience function to validate and clean fundamental data.
    
    Args:
        data: Dictionary with fundamental data
        symbol: Stock ticker symbol
        
    Returns:
        Cleaned fundamental data
    """
    cleaned_data, report = DataValidator.validate_fundamentals(data, symbol)
    if report["status"] != "valid":
        logger.info(f"Fundamentals validation for {symbol}: Fixed {len(report['issues'])} issues")
    return cleaned_data
