"""
Prepare training data from raw tick data.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(
    input_dir='data/raw/ticks',
    output_path='data/processed/training_data.parquet',
    label_horizon=1
):
    """
    Prepare training data from raw tick data.

    Args:
        input_dir: Input directory with raw tick data
        output_path: Output path for training data
        label_horizon: Number of periods ahead for return calculation
    """
    logger.info(f"Preparing training data from {input_dir}")

    # Find all Parquet files
    parquet_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))

    logger.info(f"Found {len(parquet_files)} Parquet files")

    # Load all data
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")

    if not dfs:
        logger.error("No data loaded")
        return

    # Combine all data
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} records total")

    # Ensure timestamp is datetime and handle timezone issues
    if 'timestamp' in df.columns:
        # First convert all timestamps to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='ISO8601', utc=True)
        # Then convert to naive datetime by removing timezone info
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'timestamp'])

    # Compute basic features
    logger.info("Computing basic features")

    # Moving averages
    for window in [5, 15, 60]:
        df[f'ma_{window}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Price changes
    df['price_change'] = df.groupby('symbol')['close'].transform(
        lambda x: x.diff()
    )

    # Returns
    df['return'] = df.groupby('symbol')['close'].transform(
        lambda x: x.pct_change()
    )

    # Volatility (standard deviation of returns)
    for window in [5, 15, 60]:
        df[f'volatility_{window}'] = df.groupby('symbol')['return'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

    # Volume features
    df['volume_change'] = df.groupby('symbol')['volume'].transform(
        lambda x: x.pct_change()
    )

    for window in [5, 15, 60]:
        df[f'volume_ma_{window}'] = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # RSI
    df['gain'] = df['price_change'].clip(lower=0)
    df['loss'] = -df['price_change'].clip(upper=0)

    for window in [14]:
        df[f'avg_gain_{window}'] = df.groupby('symbol')['gain'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'avg_loss_{window}'] = df.groupby('symbol')['loss'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        df[f'rs_{window}'] = df[f'avg_gain_{window}'] / df[f'avg_loss_{window}'].replace(0, 1e-9)
        df[f'rsi_{window}'] = 100 - (100 / (1 + df[f'rs_{window}']))

    # MACD
    df['ema_12'] = df.groupby('symbol')['close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df['ema_26'] = df.groupby('symbol')['close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    for window in [20]:
        df[f'bb_middle_{window}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'bb_std_{window}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
        df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        df[f'bb_pct_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])

    # Add label (future return)
    logger.info(f"Adding label with horizon {label_horizon}")
    df['label'] = df.groupby('symbol')['close'].transform(
        lambda x: x.shift(-label_horizon) / x - 1
    )

    # Drop intermediate columns
    drop_cols = [
        'gain', 'loss', 'avg_gain_14', 'avg_loss_14', 'rs_14',
        'ema_12', 'ema_26', 'bb_std_20'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Drop rows with NaN in important columns
    df = df.dropna(subset=['label'])

    logger.info(f"Final dataset has {len(df)} records and {len(df.columns)} columns")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to Parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved training data to {output_path}")

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data from raw tick data")
    parser.add_argument("--input-dir", type=str, default="data/raw/ticks", help="Input directory with raw tick data")
    parser.add_argument("--output-path", type=str, default="data/processed/training_data.parquet", help="Output path for training data")
    parser.add_argument("--label-horizon", type=int, default=1, help="Number of periods ahead for return calculation")

    args = parser.parse_args()

    prepare_training_data(
        input_dir=args.input_dir,
        output_path=args.output_path,
        label_horizon=args.label_horizon
    )
