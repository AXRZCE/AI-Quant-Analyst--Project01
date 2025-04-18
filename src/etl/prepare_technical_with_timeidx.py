"""
Prepare technical data with time index for RL training.
"""
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_technical_with_timeidx(
    input_path: str,
    output_path: str,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare technical data with time index for RL training.
    
    Args:
        input_path: Path to input data (can include wildcards)
        output_path: Path to output data
        feature_cols: List of feature columns to include (if None, include all)
        
    Returns:
        DataFrame with technical data and time index
    """
    logger.info(f"Preparing technical data with time index from {input_path}")
    
    # Find all input files
    input_files = glob(input_path)
    if not input_files:
        raise ValueError(f"No files found matching {input_path}")
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Load all data
    dfs = []
    for file in input_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError(f"No data loaded from {input_path}")
    
    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} records")
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort by symbol and timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    # Add time_idx column
    if "time_idx" not in df.columns:
        # Create time_idx for each symbol
        df["time_idx"] = df.groupby("symbol").cumcount()
        logger.info(f"Added time_idx column with range {df['time_idx'].min()} to {df['time_idx'].max()}")
    
    # Filter columns if feature_cols is provided
    if feature_cols is not None:
        # Ensure required columns are included
        required_cols = ["symbol", "timestamp", "time_idx", "close"]
        for col in required_cols:
            if col not in feature_cols and col in df.columns:
                feature_cols.append(col)
        
        # Filter columns
        df = df[feature_cols]
        logger.info(f"Filtered to {len(feature_cols)} columns: {feature_cols}")
    
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare technical data with time index for RL training")
    parser.add_argument("--input-path", type=str, required=True, help="Path to input data (can include wildcards)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output data")
    parser.add_argument("--feature-cols", type=str, nargs="+", help="List of feature columns to include")
    
    args = parser.parse_args()
    
    prepare_technical_with_timeidx(args.input_path, args.output_path, args.feature_cols)
