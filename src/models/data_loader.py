"""
Data loader for training and backtesting.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

# Try to import Feast, but don't fail if it's not available
try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
FEATURES_DIR = os.getenv('FEATURES_DIR', 'data/features')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', 'data/raw')

def load_training_data(
    start_date: str,
    end_date: str,
    symbols: List[str],
    use_feast: bool = False,
    feature_repo_path: str = "infra/feast/feature_repo",
    feature_refs: Optional[List[str]] = None,
    label_horizon: int = 1,  # Number of periods ahead for return calculation
    label_type: str = "return"  # "return" or "direction"
) -> pd.DataFrame:
    """
    Fetches features from Feast (or local files) and joins with future returns.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbols: List of ticker symbols
        use_feast: Whether to use Feast feature store
        feature_repo_path: Path to Feast feature repository
        feature_refs: List of feature references for Feast
        label_horizon: Number of periods ahead for return calculation
        label_type: Type of label to generate ("return" or "direction")
        
    Returns:
        DataFrame with features and labels
    """
    if use_feast and FEAST_AVAILABLE:
        return _load_from_feast(
            start_date, end_date, symbols, feature_repo_path, feature_refs, label_horizon, label_type
        )
    else:
        return _load_from_files(
            start_date, end_date, symbols, label_horizon, label_type
        )

def _load_from_feast(
    start_date: str,
    end_date: str,
    symbols: List[str],
    feature_repo_path: str,
    feature_refs: Optional[List[str]],
    label_horizon: int,
    label_type: str
) -> pd.DataFrame:
    """
    Load training data from Feast feature store.
    """
    logger.info(f"Loading data from Feast for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Initialize Feast feature store
    fs = FeatureStore(repo_path=feature_repo_path)
    
    # If feature_refs not provided, use all available features
    if feature_refs is None:
        # Get all feature views
        feature_views = fs.list_feature_views()
        feature_refs = []
        for fv in feature_views:
            for feature in fv.features:
                feature_refs.append(f"{fv.name}:{feature}")
    
    # Create entity DataFrame with timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate daily timestamps
    timestamps = []
    current_dt = start_dt
    while current_dt <= end_dt:
        timestamps.append(current_dt)
        current_dt += timedelta(days=1)
    
    # Create entity DataFrame
    entity_rows = []
    for symbol in symbols:
        for ts in timestamps:
            entity_rows.append({
                "symbol": symbol,
                "event_timestamp": ts
            })
    
    entity_df = pd.DataFrame(entity_rows)
    
    # Get historical features
    feature_data = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_refs
    ).to_df()
    
    # Compute label
    if "close" in feature_data.columns:
        feature_data = _add_label(feature_data, label_horizon, label_type)
    else:
        logger.warning("Close price not found in features, cannot compute label")
    
    return feature_data

def _load_from_files(
    start_date: str,
    end_date: str,
    symbols: List[str],
    label_horizon: int,
    label_type: str
) -> pd.DataFrame:
    """
    Load training data from local Parquet files.
    """
    logger.info(f"Loading data from files for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Try to load from batch features first
    try:
        # Check if technical features exist
        technical_path = f"{FEATURES_DIR}/batch/technical"
        if os.path.exists(technical_path):
            df = pd.read_parquet(technical_path)
            logger.info(f"Loaded {len(df)} records from batch technical features")
        else:
            # Fall back to raw data
            df = _load_raw_data(start_date, end_date, symbols)
            logger.info(f"Loaded {len(df)} records from raw data")
    except Exception as e:
        logger.warning(f"Error loading batch features: {e}")
        # Fall back to raw data
        df = _load_raw_data(start_date, end_date, symbols)
        logger.info(f"Loaded {len(df)} records from raw data")
    
    # Filter by date range
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= start_dt.date()) & (df["date"] <= end_dt.date())]
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df[(df["date"] >= start_dt.date()) & (df["date"] <= end_dt.date())]
    
    # Filter by symbols
    if "symbol" in df.columns and symbols:
        df = df[df["symbol"].isin(symbols)]
    
    # Add label
    df = _add_label(df, label_horizon, label_type)
    
    return df

def _load_raw_data(
    start_date: str,
    end_date: str,
    symbols: List[str]
) -> pd.DataFrame:
    """
    Load raw tick data and compute basic features.
    """
    logger.info(f"Loading raw data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate list of date folders to check
    date_folders = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        date_folders.append(date_str)
        current_dt += timedelta(days=1)
    
    # Load data from each date folder
    dfs = []
    for date_folder in date_folders:
        folder_path = f"{RAW_DATA_DIR}/ticks/{date_folder}"
        if os.path.exists(folder_path):
            try:
                df = pd.read_parquet(folder_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading data from {folder_path}: {e}")
    
    if not dfs:
        logger.warning(f"No data found for the specified date range")
        return pd.DataFrame()
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter by symbols
    if "symbol" in df.columns and symbols:
        df = df[df["symbol"].isin(symbols)]
    
    # Compute basic features if they don't exist
    if "ma_5" not in df.columns:
        df = _compute_basic_features(df)
    
    return df

def _compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic technical features from price data.
    """
    logger.info("Computing basic technical features")
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "timestamp"])
    
    # Compute moving averages
    for window in [5, 15, 60]:
        df[f"ma_{window}"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Compute price changes
    df["price_change"] = df.groupby("symbol")["close"].transform(
        lambda x: x.diff()
    )
    
    # Compute RSI
    df["gain"] = df["price_change"].clip(lower=0)
    df["loss"] = -df["price_change"].clip(upper=0)
    
    for window in [14]:
        df[f"avg_gain_{window}"] = df.groupby("symbol")["gain"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"avg_loss_{window}"] = df.groupby("symbol")["loss"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        df[f"rs_{window}"] = df[f"avg_gain_{window}"] / df[f"avg_loss_{window}"].replace(0, 1e-9)
        df[f"rsi_{window}"] = 100 - (100 / (1 + df[f"rs_{window}"]))
    
    # Compute ATR
    df["high_low"] = df["high"] - df["low"]
    df["high_close_prev"] = abs(df["high"] - df.groupby("symbol")["close"].shift(1))
    df["low_close_prev"] = abs(df["low"] - df.groupby("symbol")["close"].shift(1))
    
    df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
    
    for window in [14]:
        df[f"atr_{window}"] = df.groupby("symbol")["tr"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Drop intermediate columns
    drop_cols = [
        "price_change", "gain", "loss", 
        "avg_gain_14", "avg_loss_14", "rs_14",
        "high_low", "high_close_prev", "low_close_prev", "tr"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df

def _add_label(
    df: pd.DataFrame,
    horizon: int = 1,
    label_type: str = "return"
) -> pd.DataFrame:
    """
    Add label column to the DataFrame.
    
    Args:
        df: DataFrame with price data
        horizon: Number of periods ahead for return calculation
        label_type: Type of label to generate ("return" or "direction")
        
    Returns:
        DataFrame with label column added
    """
    logger.info(f"Adding {label_type} label with horizon {horizon}")
    
    # Ensure DataFrame is sorted
    if "timestamp" in df.columns:
        df = df.sort_values(["symbol", "timestamp"])
    
    # Compute future price
    future_price = df.groupby("symbol")["close"].shift(-horizon)
    
    if label_type == "return":
        # Compute return
        df["label"] = future_price / df["close"] - 1.0
    elif label_type == "direction":
        # Compute direction (1 for up, 0 for down)
        df["label"] = (future_price > df["close"]).astype(int)
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    
    return df

def prepare_features(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    """
    Prepare features for model training by handling missing values and dropping unnecessary columns.
    
    Args:
        df: DataFrame with features
        drop_columns: Columns to drop
        fill_method: Method to fill missing values
        
    Returns:
        DataFrame with prepared features
    """
    logger.info("Preparing features for model training")
    
    # Make a copy to avoid modifying the original
    df_prep = df.copy()
    
    # Default columns to drop
    if drop_columns is None:
        drop_columns = ["timestamp", "date", "symbol", "label"]
    
    # Drop unnecessary columns
    df_prep = df_prep.drop(columns=[col for col in drop_columns if col in df_prep.columns])
    
    # Fill missing values
    if fill_method == "ffill":
        df_prep = df_prep.groupby("symbol").ffill()
    elif fill_method == "mean":
        df_prep = df_prep.fillna(df_prep.mean())
    elif fill_method == "zero":
        df_prep = df_prep.fillna(0)
    
    # Drop any remaining rows with NaN
    df_prep = df_prep.dropna()
    
    return df_prep

def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with features and label
        test_size: Fraction of data to use for testing
        validation_size: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with train, validation, and test DataFrames
    """
    logger.info(f"Splitting data with test_size={test_size}, validation_size={validation_size}")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Sort by timestamp if available and not shuffling
    if "timestamp" in df_copy.columns and not shuffle:
        df_copy = df_copy.sort_values("timestamp")
    
    # Calculate split indices
    n = len(df_copy)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))
    
    if shuffle:
        # Shuffle the data
        df_copy = df_copy.sample(frac=1, random_state=random_state)
        
        # Split into train, validation, and test
        train_df = df_copy.iloc[:val_idx]
        val_df = df_copy.iloc[val_idx:test_idx]
        test_df = df_copy.iloc[test_idx:]
    else:
        # Split into train, validation, and test
        train_df = df_copy.iloc[:val_idx]
        val_df = df_copy.iloc[val_idx:test_idx]
        test_df = df_copy.iloc[test_idx:]
    
    logger.info(f"Split sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

def get_feature_label_split(
    df: pd.DataFrame,
    label_column: str = "label",
    drop_columns: Optional[List[str]] = None
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Split DataFrame into features (X) and label (y).
    
    Args:
        df: DataFrame with features and label
        label_column: Name of the label column
        drop_columns: Additional columns to drop from features
        
    Returns:
        Dictionary with X and y
    """
    if drop_columns is None:
        drop_columns = ["timestamp", "date", "symbol"]
    
    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    
    # Extract label
    y = df[label_column]
    
    # Extract features
    drop_cols = drop_columns + [label_column]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return {"X": X, "y": y}

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2025-04-01"
    end_date = "2025-04-17"
    
    # Load data
    df = load_training_data(start_date, end_date, symbols)
    
    # Print summary
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Symbols: {df['symbol'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Prepare features
    X = prepare_features(df)
    print(f"Prepared {len(X)} feature records")
