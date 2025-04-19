"""
Update model metrics in Prometheus.

This script calculates model performance metrics and sends them to Prometheus Pushgateway.
"""

import os
import sys
import logging
import argparse
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """
    Load a model from a file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from Parquet files.
    
    Args:
        data_path: Path to the data files (can include wildcards)
        
    Returns:
        DataFrame with data
    """
    logger.info(f"Loading data from {data_path}")
    
    # Check if path is a directory
    if os.path.isdir(data_path):
        # Find all Parquet files in the directory
        files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".parquet")]
    else:
        # Assume it's a single file
        files = [data_path]
    
    # Check if files exist
    if not files:
        raise ValueError(f"No Parquet files found at {data_path}")
    
    # Load data
    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    # Combine data
    if not dfs:
        raise ValueError(f"No data loaded from {data_path}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} records from {len(files)} files")
    
    return df


def calculate_metrics(model: Any, data: pd.DataFrame, target_column: str = "target") -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        model: Model to evaluate
        data: Data to evaluate on
        target_column: Name of the target column
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Calculating model performance metrics")
    
    # Check if target column exists
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        return {}
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            "model_rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "model_mae": mean_absolute_error(y, y_pred),
            "model_r2": r2_score(y, y_pred),
            "model_explained_variance": explained_variance_score(y, y_pred),
            "model_mean_prediction": np.mean(y_pred),
            "model_std_prediction": np.std(y_pred),
            "model_min_prediction": np.min(y_pred),
            "model_max_prediction": np.max(y_pred)
        }
        
        logger.info(f"Metrics calculated: {metrics}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}


def send_metrics_to_prometheus(
    metrics: Dict[str, float],
    prometheus_url: str,
    job_name: str = "model_metrics"
) -> bool:
    """
    Send metrics to Prometheus Pushgateway.
    
    Args:
        metrics: Dictionary with metrics
        prometheus_url: Prometheus Pushgateway URL
        job_name: Job name
        
    Returns:
        True if metrics were sent successfully, False otherwise
    """
    logger.info(f"Sending metrics to Prometheus: {prometheus_url}")
    
    try:
        # Prepare metrics
        prometheus_metrics = []
        
        for name, value in metrics.items():
            prometheus_metrics.append(f"{name}{{job=\"{job_name}\"}} {value}")
        
        # Add timestamp
        prometheus_metrics.append(f"model_metrics_timestamp{{job=\"{job_name}\"}} {int(time.time())}")
        
        # Send metrics
        response = requests.post(
            f"{prometheus_url}/metrics/job/{job_name}",
            data="\n".join(prometheus_metrics),
            headers={"Content-Type": "text/plain"}
        )
        
        if response.status_code == 200:
            logger.info("Metrics sent to Prometheus successfully")
            return True
        else:
            logger.error(f"Error sending metrics to Prometheus: {response.status_code} {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error sending metrics to Prometheus: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update model metrics in Prometheus")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data files")
    parser.add_argument("--target-column", type=str, default="target", help="Name of the target column")
    parser.add_argument("--prometheus-url", type=str, required=True, help="Prometheus Pushgateway URL")
    parser.add_argument("--job-name", type=str, default="model_metrics", help="Job name for Prometheus")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Load data
        data = load_data(args.data_path)
        
        # Calculate metrics
        metrics = calculate_metrics(model, data, args.target_column)
        
        # Send metrics to Prometheus
        if metrics:
            success = send_metrics_to_prometheus(
                metrics=metrics,
                prometheus_url=args.prometheus_url,
                job_name=args.job_name
            )
            
            if success:
                print("Model metrics updated successfully in Prometheus")
            else:
                print("Failed to update model metrics in Prometheus")
        else:
            print("No metrics to send to Prometheus")
    
    except Exception as e:
        logger.error(f"Error updating model metrics: {e}")
        print(f"Error updating model metrics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
