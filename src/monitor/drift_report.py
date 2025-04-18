"""
Generate data drift report using Evidently.
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPerformancePreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_drift_report(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    target_column: Optional[str] = "label",
    output_path: Optional[str] = None
) -> Report:
    """
    Generate data drift report using Evidently.
    
    Args:
        reference_data: Reference data (e.g., training data)
        production_data: Production data (e.g., new data)
        target_column: Name of the target column
        output_path: Path to save the report
        
    Returns:
        Evidently report
    """
    logger.info("Generating drift report")
    
    # Create report
    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name=col) for col in reference_data.columns if col not in ["symbol", "timestamp", "date"]
        ]
    )
    
    # Add regression performance metrics if target column is available
    if target_column in reference_data.columns and target_column in production_data.columns:
        report.add_metric(RegressionPerformancePreset())
    
    # Run report
    report.run(reference_data=reference_data, current_data=production_data)
    
    # Save report if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Report saved to {output_path}")
    
    return report

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

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Generate data drift report")
    parser.add_argument("--reference-data", type=str, required=True, help="Path to reference data")
    parser.add_argument("--production-data", type=str, required=True, help="Path to production data")
    parser.add_argument("--target-column", type=str, default="label", help="Name of the target column")
    parser.add_argument("--output-path", type=str, help="Path to save the report")
    
    args = parser.parse_args()
    
    try:
        # Load data
        reference_data = load_data(args.reference_data)
        production_data = load_data(args.production_data)
        
        # Generate report
        report = generate_drift_report(
            reference_data=reference_data,
            production_data=production_data,
            target_column=args.target_column,
            output_path=args.output_path
        )
        
        # Print summary
        print("Drift report generated successfully.")
        print(f"Dataset drift detected: {report.as_dict()['metrics'][0]['result']['dataset_drift']}")
        
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        raise

if __name__ == "__main__":
    main()
