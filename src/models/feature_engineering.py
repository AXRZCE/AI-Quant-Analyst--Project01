"""
Feature engineering module for financial time series.

This module provides functions for creating features from financial time series data,
including technical indicators, volatility measures, and trend features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA

# Configure logging
logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame, price_col: str = 'close', volume_col: Optional[str] = 'volume') -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        volume_col: Name of the volume column (optional)
        
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Adding technical indicators")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Simple Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df_features[f'sma_{window}'] = df_features[price_col].rolling(window=window).mean()
    
    # Exponential Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df_features[f'ema_{window}'] = df_features[price_col].ewm(span=window, adjust=False).mean()
    
    # Moving Average Convergence Divergence (MACD)
    df_features['macd'] = df_features[price_col].ewm(span=12, adjust=False).mean() - \
                          df_features[price_col].ewm(span=26, adjust=False).mean()
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']
    
    # Relative Strength Index (RSI)
    delta = df_features[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    df_features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for window in [20]:
        df_features[f'bollinger_middle_{window}'] = df_features[price_col].rolling(window=window).mean()
        df_features[f'bollinger_std_{window}'] = df_features[price_col].rolling(window=window).std()
        df_features[f'bollinger_upper_{window}'] = df_features[f'bollinger_middle_{window}'] + 2 * df_features[f'bollinger_std_{window}']
        df_features[f'bollinger_lower_{window}'] = df_features[f'bollinger_middle_{window}'] - 2 * df_features[f'bollinger_std_{window}']
        
        # Bollinger Band width
        df_features[f'bb_width_{window}'] = (df_features[f'bollinger_upper_{window}'] - df_features[f'bollinger_lower_{window}']) / df_features[f'bollinger_middle_{window}']
        
        # Bollinger Band %B
        df_features[f'bb_pct_b_{window}'] = (df_features[price_col] - df_features[f'bollinger_lower_{window}']) / \
                                           (df_features[f'bollinger_upper_{window}'] - df_features[f'bollinger_lower_{window}'])
    
    # Volatility
    for window in [5, 10, 20, 50]:
        df_features[f'volatility_{window}'] = df_features[price_col].pct_change().rolling(window=window).std()
    
    # Price Momentum
    for window in [1, 5, 10, 20, 50]:
        df_features[f'momentum_{window}'] = df_features[price_col].pct_change(window)
    
    # Rate of Change
    for window in [5, 10, 20, 50]:
        df_features[f'roc_{window}'] = (df_features[price_col] - df_features[price_col].shift(window)) / df_features[price_col].shift(window) * 100
    
    # Average True Range (ATR)
    high = df_features['high'] if 'high' in df_features.columns else df_features[price_col]
    low = df_features['low'] if 'low' in df_features.columns else df_features[price_col]
    close_prev = df_features[price_col].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_features['atr_14'] = true_range.rolling(window=14).mean()
    
    # Volume-based indicators (if volume is available)
    if volume_col and volume_col in df_features.columns:
        # On-Balance Volume (OBV)
        df_features['obv'] = (np.sign(df_features[price_col].diff()) * df_features[volume_col]).fillna(0).cumsum()
        
        # Volume Moving Averages
        for window in [5, 10, 20, 50]:
            df_features[f'volume_sma_{window}'] = df_features[volume_col].rolling(window=window).mean()
        
        # Price-Volume Trend
        df_features['pvt'] = (df_features[price_col].pct_change() * df_features[volume_col]).fillna(0).cumsum()
        
        # Accumulation/Distribution Line
        high = df_features['high'] if 'high' in df_features.columns else df_features[price_col]
        low = df_features['low'] if 'low' in df_features.columns else df_features[price_col]
        close = df_features[price_col]
        
        clv = ((close - low) - (high - close)) / (high - low).replace(0, 1e-10)
        df_features['adl'] = (clv * df_features[volume_col]).fillna(0).cumsum()
    
    # Trend indicators
    df_features['price_up'] = np.where(df_features[price_col] > df_features[price_col].shift(1), 1, 0)
    df_features['price_down'] = np.where(df_features[price_col] < df_features[price_col].shift(1), 1, 0)
    
    # Trend strength
    for window in [5, 10, 20]:
        df_features[f'up_trend_strength_{window}'] = df_features['price_up'].rolling(window=window).sum() / window
        df_features[f'down_trend_strength_{window}'] = df_features['price_down'].rolling(window=window).sum() / window
    
    # Distance from moving averages
    for window in [20, 50, 200]:
        df_features[f'dist_sma_{window}'] = (df_features[price_col] / df_features[f'sma_{window}'] - 1) * 100
    
    # Golden/Death Cross
    df_features['golden_cross'] = np.where(df_features['sma_50'] > df_features['sma_200'], 1, 0)
    df_features['death_cross'] = np.where(df_features['sma_50'] < df_features['sma_200'], 1, 0)
    
    # Drop temporary columns
    df_features.drop(['price_up', 'price_down'], axis=1, inplace=True, errors='ignore')
    
    logger.info(f"Added {len(df_features.columns) - len(df.columns)} technical indicators")
    
    return df_features


def add_date_features(df: pd.DataFrame, date_col: str = 'timestamp') -> pd.DataFrame:
    """
    Add date-based features to the dataframe.
    
    Args:
        df: DataFrame with date/timestamp column
        date_col: Name of the date/timestamp column
        
    Returns:
        DataFrame with added date features
    """
    logger.info("Adding date features")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_features[date_col]):
        df_features[date_col] = pd.to_datetime(df_features[date_col])
    
    # Extract date components
    df_features['day_of_week'] = df_features[date_col].dt.dayofweek
    df_features['day_of_month'] = df_features[date_col].dt.day
    df_features['week_of_year'] = df_features[date_col].dt.isocalendar().week
    df_features['month'] = df_features[date_col].dt.month
    df_features['quarter'] = df_features[date_col].dt.quarter
    df_features['year'] = df_features[date_col].dt.year
    
    # Is month/quarter end
    df_features['is_month_end'] = df_features[date_col].dt.is_month_end.astype(int)
    df_features['is_quarter_end'] = df_features[date_col].dt.is_quarter_end.astype(int)
    df_features['is_year_end'] = df_features[date_col].dt.is_year_end.astype(int)
    
    # Is month/quarter start
    df_features['is_month_start'] = df_features[date_col].dt.is_month_start.astype(int)
    df_features['is_quarter_start'] = df_features[date_col].dt.is_quarter_start.astype(int)
    df_features['is_year_start'] = df_features[date_col].dt.is_year_start.astype(int)
    
    # Day type
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding of day of week, month, etc.
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    logger.info(f"Added {len(df_features.columns) - len(df.columns)} date features")
    
    return df_features


def add_lagged_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Add lagged features to the dataframe.
    
    Args:
        df: DataFrame with time series data
        cols: List of column names to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with added lagged features
    """
    logger.info(f"Adding lagged features for {len(cols)} columns with {len(lags)} lag periods")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Add lagged features
    for col in cols:
        if col in df_features.columns:
            for lag in lags:
                df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
    
    logger.info(f"Added {len(cols) * len(lags)} lagged features")
    
    return df_features


def add_rolling_features(df: pd.DataFrame, cols: List[str], windows: List[int], 
                         functions: Dict[str, callable] = None) -> pd.DataFrame:
    """
    Add rolling window features to the dataframe.
    
    Args:
        df: DataFrame with time series data
        cols: List of column names to create rolling features for
        windows: List of window sizes
        functions: Dictionary of function names and functions to apply
        
    Returns:
        DataFrame with added rolling features
    """
    logger.info(f"Adding rolling features for {len(cols)} columns with {len(windows)} window sizes")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Default functions if none provided
    if functions is None:
        functions = {
            'mean': lambda x: x.mean(),
            'std': lambda x: x.std(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'median': lambda x: x.median()
        }
    
    # Add rolling features
    for col in cols:
        if col in df_features.columns:
            for window in windows:
                for func_name, func in functions.items():
                    df_features[f'{col}_roll_{window}_{func_name}'] = df_features[col].rolling(window=window).apply(func)
    
    logger.info(f"Added {len(cols) * len(windows) * len(functions)} rolling features")
    
    return df_features


def add_target_features(df: pd.DataFrame, price_col: str = 'close', horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Add target features (future returns) to the dataframe.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        horizons: List of prediction horizons (in days)
        
    Returns:
        DataFrame with added target features
    """
    logger.info(f"Adding target features for {len(horizons)} horizons")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Add future returns
    for horizon in horizons:
        # Future price
        df_features[f'future_price_{horizon}'] = df_features[price_col].shift(-horizon)
        
        # Future return (percentage)
        df_features[f'future_return_{horizon}'] = df_features[price_col].pct_change(-horizon)
        
        # Future direction (up/down)
        df_features[f'future_direction_{horizon}'] = np.where(df_features[f'future_return_{horizon}'] > 0, 1, 0)
        
        # Future volatility
        df_features[f'future_volatility_{horizon}'] = df_features[price_col].pct_change(-horizon).rolling(window=horizon).std()
    
    logger.info(f"Added {len(horizons) * 4} target features")
    
    return df_features


def select_features(X: pd.DataFrame, y: pd.Series, method: str = 'f_regression', k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most important features.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Feature selection method ('f_regression', 'mutual_info', 'pca')
        k: Number of features to select
        
    Returns:
        Tuple of (selected feature matrix, list of selected feature names)
    """
    logger.info(f"Selecting {k} features using {method}")
    
    # Adjust k if it's larger than the number of features
    k = min(k, X.shape[1])
    
    # Select features
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
    elif method == 'pca':
        # Standardize data for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=k)
        X_selected = pca.fit_transform(X_scaled)
        
        # For PCA, we don't have original feature names
        selected_features = [f'pca_component_{i+1}' for i in range(k)]
        
        # Convert back to DataFrame
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Convert to DataFrame if not already
    if not isinstance(X_selected, pd.DataFrame):
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    logger.info(f"Selected features: {selected_features}")
    
    return X_selected, selected_features


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def prepare_features(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume',
                    date_col: str = 'timestamp', target_horizon: int = 5,
                    add_technical: bool = True, add_date: bool = True,
                    add_lags: bool = True, add_rolling: bool = True) -> pd.DataFrame:
    """
    Prepare features for model training.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        volume_col: Name of the volume column
        date_col: Name of the date/timestamp column
        target_horizon: Prediction horizon (in days)
        add_technical: Whether to add technical indicators
        add_date: Whether to add date features
        add_lags: Whether to add lagged features
        add_rolling: Whether to add rolling features
        
    Returns:
        DataFrame with prepared features
    """
    logger.info("Preparing features for model training")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Add technical indicators
    if add_technical:
        df_features = add_technical_indicators(df_features, price_col, volume_col)
    
    # Add date features
    if add_date and date_col in df_features.columns:
        df_features = add_date_features(df_features, date_col)
    
    # Add target features
    df_features = add_target_features(df_features, price_col, horizons=[target_horizon])
    
    # Add lagged features
    if add_lags:
        # Select columns for lagging
        lag_cols = [price_col]
        if volume_col in df_features.columns:
            lag_cols.append(volume_col)
        
        # Add technical indicators to lag
        tech_cols = ['rsi_14', 'macd', 'volatility_20']
        lag_cols.extend([col for col in tech_cols if col in df_features.columns])
        
        # Add lags
        df_features = add_lagged_features(df_features, lag_cols, lags=[1, 2, 3, 5, 10])
    
    # Add rolling features
    if add_rolling:
        # Select columns for rolling features
        roll_cols = [price_col]
        if volume_col in df_features.columns:
            roll_cols.append(volume_col)
        
        # Add rolling features
        df_features = add_rolling_features(df_features, roll_cols, windows=[5, 10, 20])
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    logger.info(f"Prepared features dataframe with {len(df_features.columns)} columns")
    
    return df_features


def get_feature_target_split(df: pd.DataFrame, target_col: str = 'future_return_5',
                           exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        exclude_cols: List of columns to exclude from features
        
    Returns:
        Tuple of (feature matrix, target vector)
    """
    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # Add future columns to exclude
        exclude_cols.extend([col for col in df.columns if col.startswith('future_')])
    
    # Remove target from exclude_cols if it's there
    if target_col in exclude_cols:
        exclude_cols.remove(target_col)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
    
    # Split into features and target
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
    
    return X, y
