"""
Data preparation for uncertainty quantification.
"""
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_df(path: str) -> pd.DataFrame:
    """
    Load data from Parquet files and prepare it for uncertainty quantification.
    
    Args:
        path: Path to the data files (can include wildcards)
        
    Returns:
        DataFrame with features and target
    """
    logger.info(f"Loading data from {path}")
    
    # Find all files matching the pattern
    files = glob(path)
    
    if not files:
        raise ValueError(f"No files found matching {path}")
    
    logger.info(f"Found {len(files)} files")
    
    # Load all files
    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError(f"No data loaded from {path}")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} records")
    
    # Keep only numeric features + target
    if "label" not in df.columns:
        # Try to compute label if not present
        if "close" in df.columns:
            logger.info("Computing label as next-period return")
            df = df.sort_values(["symbol", "timestamp"])
            df["label"] = df.groupby("symbol")["close"].shift(-1) / df["close"] - 1
    
    # Filter out non-numeric columns except symbol and timestamp
    feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "date"]]
    df = df[["symbol", "timestamp"] + feature_cols].copy()
    
    # Drop rows with missing values
    df = df.dropna()
    
    logger.info(f"Prepared DataFrame with {len(df)} records and {len(feature_cols)} features")
    
    return df

def prepare_features_targets(
    df: pd.DataFrame,
    target_col: str = "label",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and targets for modeling.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X, y) arrays
    """
    if exclude_cols is None:
        exclude_cols = ["symbol", "timestamp", "date", target_col]
    else:
        exclude_cols = exclude_cols + [target_col]
    
    # Select feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Convert to numpy arrays
    X = df[feature_cols].values
    y = df[target_col].values
    
    logger.info(f"Prepared features with shape {X.shape} and targets with shape {y.shape}")
    
    return X, y

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_size: Fraction of data to use for training
        val_size: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with train, validation, and test sets
    """
    n_samples = len(X)
    
    if shuffle and random_state is not None:
        # Create a random permutation
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
        X = X[indices]
        y = y[indices]
    
    # Calculate split indices
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)
    
    # Split the data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split data into train ({len(X_train)}), validation ({len(X_val)}), and test ({len(X_test)}) sets")
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Normalize data using mean and standard deviation from training set.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Dictionary with normalized data and normalization parameters
    """
    # Calculate mean and std from training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Replace zeros in std to avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    # Normalize training data
    X_train_norm = (X_train - mean) / std
    
    result = {
        "X_train": X_train_norm,
        "params": {
            "mean": mean,
            "std": std
        }
    }
    
    # Normalize validation data if provided
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        result["X_val"] = X_val_norm
    
    # Normalize test data if provided
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        result["X_test"] = X_test_norm
    
    logger.info("Normalized data")
    
    return result

if __name__ == "__main__":
    # Example usage
    try:
        # Try to load from batch features
        df = load_df("data/features/batch/technical/*.parquet")
    except ValueError:
        # Try to load from processed data
        try:
            df = load_df("data/processed/training_data.parquet")
        except ValueError:
            # Try to load from raw data
            df = load_df("data/raw/ticks/*/*.parquet")
    
    print(df.head())
    
    # Prepare features and targets
    X, y = prepare_features_targets(df)
    
    # Split data
    splits = split_data(X, y)
    
    # Normalize data
    normalized = normalize_data(splits["X_train"], splits["X_val"], splits["X_test"])
    
    print(f"X_train shape: {normalized['X_train'].shape}")
    print(f"X_val shape: {normalized['X_val'].shape}")
    print(f"X_test shape: {normalized['X_test'].shape}")
