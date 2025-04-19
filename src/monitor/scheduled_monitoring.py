"""
Scheduled monitoring for data drift and data quality.

This module provides functionality for scheduling regular monitoring of data drift
and data quality, with support for alerting and reporting.
"""

import os
import sys
import logging
import argparse
import time
import json
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.monitor.drift_report import (
    load_data,
    generate_drift_report,
    generate_data_quality_report,
    generate_test_suite,
    check_and_alert
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "monitoring.log"))
    ]
)
logger = logging.getLogger(__name__)


class MonitoringJob:
    """Monitoring job for data drift and data quality."""
    
    def __init__(
        self,
        reference_data_path: str,
        production_data_path: str,
        output_dir: str = "reports",
        target_column: str = "label",
        alert: bool = True,
        email_recipients: Optional[List[str]] = None,
        slack_webhook_url: Optional[str] = None,
        prometheus_url: Optional[str] = None,
        drift_threshold: float = 0.05,
        missing_threshold: float = 0.1,
        quantile_threshold: float = 0.1
    ):
        """
        Initialize the monitoring job.
        
        Args:
            reference_data_path: Path to reference data
            production_data_path: Path to production data
            output_dir: Directory to save reports
            target_column: Name of the target column
            alert: Whether to send alerts if issues are detected
            email_recipients: List of email recipients
            slack_webhook_url: Slack webhook URL
            prometheus_url: Prometheus Pushgateway URL
            drift_threshold: Threshold for drift tests
            missing_threshold: Threshold for missing values tests
            quantile_threshold: Threshold for quantile tests
        """
        self.reference_data_path = reference_data_path
        self.production_data_path = production_data_path
        self.output_dir = output_dir
        self.target_column = target_column
        self.alert = alert
        self.email_recipients = email_recipients
        self.slack_webhook_url = slack_webhook_url
        self.prometheus_url = prometheus_url
        self.drift_threshold = drift_threshold
        self.missing_threshold = missing_threshold
        self.quantile_threshold = quantile_threshold
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logs directory
        os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the monitoring job.
        
        Returns:
            Dictionary with job results
        """
        logger.info("Running monitoring job")
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define output paths
            drift_report_path = os.path.join(self.output_dir, f"drift_report_{timestamp}.html")
            quality_report_path = os.path.join(self.output_dir, f"quality_report_{timestamp}.html")
            test_suite_path = os.path.join(self.output_dir, f"test_suite_{timestamp}.html")
            
            # Load data
            reference_data = load_data(self.reference_data_path)
            production_data = load_data(self.production_data_path)
            
            # Generate drift report
            drift_report = generate_drift_report(
                reference_data=reference_data,
                production_data=production_data,
                target_column=self.target_column,
                output_path=drift_report_path,
                include_data_quality=True,
                include_correlations=True
            )
            
            # Generate data quality report
            quality_report = generate_data_quality_report(
                data=production_data,
                output_path=quality_report_path
            )
            
            # Generate test suite
            test_suite = generate_test_suite(
                reference_data=reference_data,
                production_data=production_data,
                target_column=self.target_column,
                output_path=test_suite_path,
                drift_threshold=self.drift_threshold,
                missing_threshold=self.missing_threshold,
                quantile_threshold=self.quantile_threshold
            )
            
            # Check test results and send alerts if needed
            alert_result = None
            if self.alert:
                alert_result = check_and_alert(
                    test_suite=test_suite,
                    email_recipients=self.email_recipients,
                    slack_webhook_url=self.slack_webhook_url,
                    prometheus_url=self.prometheus_url,
                    report_path=test_suite_path
                )
            
            # Prepare result
            result = {
                "timestamp": timestamp,
                "drift_report_path": drift_report_path,
                "quality_report_path": quality_report_path,
                "test_suite_path": test_suite_path,
                "dataset_drift": drift_report.as_dict()['metrics'][0]['result']['dataset_drift'],
                "alert_result": alert_result
            }
            
            logger.info(f"Monitoring job completed successfully: {result}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error in monitoring job: {e}", exc_info=True)
            
            return {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "error": str(e)
            }


def schedule_monitoring_job(
    job: MonitoringJob,
    schedule_type: str = "daily",
    hour: int = 0,
    minute: int = 0,
    interval: int = 1
) -> None:
    """
    Schedule a monitoring job.
    
    Args:
        job: Monitoring job
        schedule_type: Type of schedule (daily, hourly, or interval)
        hour: Hour to run the job (for daily schedule)
        minute: Minute to run the job (for daily and hourly schedules)
        interval: Interval in minutes (for interval schedule)
    """
    logger.info(f"Scheduling monitoring job: {schedule_type}")
    
    if schedule_type == "daily":
        schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(job.run)
        logger.info(f"Monitoring job scheduled to run daily at {hour:02d}:{minute:02d}")
    
    elif schedule_type == "hourly":
        schedule.every().hour.at(f":{minute:02d}").do(job.run)
        logger.info(f"Monitoring job scheduled to run hourly at minute {minute:02d}")
    
    elif schedule_type == "interval":
        schedule.every(interval).minutes.do(job.run)
        logger.info(f"Monitoring job scheduled to run every {interval} minutes")
    
    else:
        logger.error(f"Unknown schedule type: {schedule_type}")
        return
    
    # Run the job immediately
    job.run()
    
    # Run the scheduler
    logger.info("Starting scheduler")
    while True:
        schedule.run_pending()
        time.sleep(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Schedule monitoring job")
    parser.add_argument("--reference-data", type=str, required=True, help="Path to reference data")
    parser.add_argument("--production-data", type=str, required=True, help="Path to production data")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--target-column", type=str, default="label", help="Name of the target column")
    parser.add_argument("--alert", action="store_true", help="Send alerts if issues are detected")
    parser.add_argument("--email", type=str, help="Comma-separated list of email recipients")
    parser.add_argument("--slack-webhook", type=str, help="Slack webhook URL")
    parser.add_argument("--prometheus-url", type=str, help="Prometheus Pushgateway URL")
    parser.add_argument("--drift-threshold", type=float, default=0.05, help="Threshold for drift tests")
    parser.add_argument("--missing-threshold", type=float, default=0.1, help="Threshold for missing values tests")
    parser.add_argument("--quantile-threshold", type=float, default=0.1, help="Threshold for quantile tests")
    parser.add_argument("--schedule-type", type=str, default="daily", choices=["daily", "hourly", "interval"], help="Type of schedule")
    parser.add_argument("--hour", type=int, default=0, help="Hour to run the job (for daily schedule)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run the job (for daily and hourly schedules)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes (for interval schedule)")
    
    args = parser.parse_args()
    
    try:
        # Parse email recipients
        email_recipients = args.email.split(",") if args.email else None
        
        # Create monitoring job
        job = MonitoringJob(
            reference_data_path=args.reference_data,
            production_data_path=args.production_data,
            output_dir=args.output_dir,
            target_column=args.target_column,
            alert=args.alert,
            email_recipients=email_recipients,
            slack_webhook_url=args.slack_webhook,
            prometheus_url=args.prometheus_url,
            drift_threshold=args.drift_threshold,
            missing_threshold=args.missing_threshold,
            quantile_threshold=args.quantile_threshold
        )
        
        # Schedule job
        schedule_monitoring_job(
            job=job,
            schedule_type=args.schedule_type,
            hour=args.hour,
            minute=args.minute,
            interval=args.interval
        )
    
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
