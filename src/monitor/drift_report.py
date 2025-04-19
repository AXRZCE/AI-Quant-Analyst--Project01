"""
Generate data drift and data quality reports using Evidently.

This module provides functionality for monitoring data drift and data quality
using Evidently AI, with support for alerting and scheduled monitoring.
"""
import os
import logging
import argparse
import json
import time
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    RegressionPerformancePreset,
    DataQualityPreset,
    ClassificationPerformancePreset
)
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnDistributionMetric,
    ColumnCorrelationsMetric,
    DatasetCorrelationsMetric
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import (
    TestColumnDrift,
    TestShareOfMissingValues,
    TestColumnQuantile,
    TestColumnMean,
    TestColumnMedian,
    TestColumnMin,
    TestColumnMax
)

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
    output_path: Optional[str] = None,
    include_data_quality: bool = True,
    include_correlations: bool = True
) -> Report:
    """
    Generate data drift report using Evidently.

    Args:
        reference_data: Reference data (e.g., training data)
        production_data: Production data (e.g., new data)
        target_column: Name of the target column
        output_path: Path to save the report
        include_data_quality: Whether to include data quality metrics
        include_correlations: Whether to include correlation metrics

    Returns:
        Evidently report
    """
    logger.info("Generating drift report")

    # Define metrics
    metrics = [
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        *[ColumnDriftMetric(column_name=col) for col in reference_data.columns
          if col not in ["symbol", "timestamp", "date"]]
    ]

    # Add data quality metrics if requested
    if include_data_quality:
        metrics.extend([
            ColumnQuantileMetric(column_name=col)
            for col in reference_data.select_dtypes(include=[np.number]).columns
            if col not in ["symbol", "timestamp", "date"]
        ])

    # Add correlation metrics if requested
    if include_correlations:
        metrics.append(DatasetCorrelationsMetric())

    # Create report
    report = Report(metrics=metrics)

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


def generate_data_quality_report(
    data: pd.DataFrame,
    output_path: Optional[str] = None
) -> Report:
    """
    Generate data quality report using Evidently.

    Args:
        data: Data to analyze
        output_path: Path to save the report

    Returns:
        Evidently report
    """
    logger.info("Generating data quality report")

    # Create report
    report = Report(metrics=[DataQualityPreset()])

    # Run report
    report.run(current_data=data)

    # Save report if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Report saved to {output_path}")

    return report


def generate_test_suite(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    target_column: Optional[str] = "label",
    output_path: Optional[str] = None,
    drift_threshold: float = 0.05,
    missing_threshold: float = 0.1,
    quantile_threshold: float = 0.1
) -> TestSuite:
    """
    Generate test suite using Evidently.

    Args:
        reference_data: Reference data (e.g., training data)
        production_data: Production data (e.g., new data)
        target_column: Name of the target column
        output_path: Path to save the report
        drift_threshold: Threshold for drift tests
        missing_threshold: Threshold for missing values tests
        quantile_threshold: Threshold for quantile tests

    Returns:
        Evidently test suite
    """
    logger.info("Generating test suite")

    # Create test suite
    test_suite = TestSuite(
        tests=[
            # Dataset-level tests
            TestShareOfMissingValues(missing_percent_threshold=missing_threshold*100),

            # Column-level tests
            *[TestColumnDrift(column_name=col, threshold=drift_threshold)
              for col in reference_data.columns
              if col not in ["symbol", "timestamp", "date"]],

            # Numeric column tests
            *[TestColumnQuantile(column_name=col, quantile=0.5, threshold=quantile_threshold)
              for col in reference_data.select_dtypes(include=[np.number]).columns
              if col not in ["symbol", "timestamp", "date"]],

            *[TestColumnMean(column_name=col, threshold=quantile_threshold)
              for col in reference_data.select_dtypes(include=[np.number]).columns
              if col not in ["symbol", "timestamp", "date"]],

            *[TestColumnMin(column_name=col, threshold=quantile_threshold)
              for col in reference_data.select_dtypes(include=[np.number]).columns
              if col not in ["symbol", "timestamp", "date"]],

            *[TestColumnMax(column_name=col, threshold=quantile_threshold)
              for col in reference_data.select_dtypes(include=[np.number]).columns
              if col not in ["symbol", "timestamp", "date"]]
        ]
    )

    # Run test suite
    test_suite.run(reference_data=reference_data, current_data=production_data)

    # Save test suite if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        test_suite.save_html(output_path)
        logger.info(f"Test suite saved to {output_path}")

    return test_suite

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

def send_email_alert(
    subject: str,
    body: str,
    recipients: List[str],
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_username: Optional[str] = None,
    smtp_password: Optional[str] = None,
    attachment_path: Optional[str] = None
) -> bool:
    """
    Send an email alert.

    Args:
        subject: Email subject
        body: Email body
        recipients: List of email recipients
        smtp_server: SMTP server
        smtp_port: SMTP port
        smtp_username: SMTP username
        smtp_password: SMTP password
        attachment_path: Path to attachment

    Returns:
        True if email was sent successfully, False otherwise
    """
    logger.info(f"Sending email alert to {recipients}")

    # Get SMTP credentials from environment variables if not provided
    smtp_username = smtp_username or os.environ.get("SMTP_USERNAME")
    smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")

    if not smtp_username or not smtp_password:
        logger.error("SMTP credentials not provided")
        return False

    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = smtp_username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        # Add body
        msg.attach(MIMEText(body, "html"))

        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                attachment = MIMEText(f.read(), "html")
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(attachment_path)}"
                )
                msg.attach(attachment)

        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        # Send email
        server.send_message(msg)
        server.quit()

        logger.info("Email alert sent successfully")
        return True

    except Exception as e:
        logger.error(f"Error sending email alert: {e}")
        return False


def send_slack_alert(
    message: str,
    webhook_url: Optional[str] = None,
    channel: Optional[str] = None,
    username: str = "Monitoring Bot",
    icon_emoji: str = ":robot_face:"
) -> bool:
    """
    Send a Slack alert.

    Args:
        message: Message to send
        webhook_url: Slack webhook URL
        channel: Slack channel
        username: Bot username
        icon_emoji: Bot icon emoji

    Returns:
        True if alert was sent successfully, False otherwise
    """
    logger.info("Sending Slack alert")

    # Get webhook URL from environment variable if not provided
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        logger.error("Slack webhook URL not provided")
        return False

    try:
        # Prepare payload
        payload = {
            "text": message,
            "username": username,
            "icon_emoji": icon_emoji
        }

        if channel:
            payload["channel"] = channel

        # Send request
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            logger.info("Slack alert sent successfully")
            return True
        else:
            logger.error(f"Error sending Slack alert: {response.status_code} {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error sending Slack alert: {e}")
        return False


def send_prometheus_alert(
    test_results: Dict[str, Any],
    prometheus_url: Optional[str] = None,
    job_name: str = "data_monitoring"
) -> bool:
    """
    Send metrics to Prometheus Pushgateway.

    Args:
        test_results: Test results
        prometheus_url: Prometheus Pushgateway URL
        job_name: Job name

    Returns:
        True if metrics were sent successfully, False otherwise
    """
    logger.info("Sending metrics to Prometheus")

    # Get Prometheus URL from environment variable if not provided
    prometheus_url = prometheus_url or os.environ.get("PROMETHEUS_PUSHGATEWAY_URL")

    if not prometheus_url:
        logger.error("Prometheus Pushgateway URL not provided")
        return False

    try:
        # Prepare metrics
        metrics = []

        # Add dataset drift metric
        if "dataset_drift" in test_results:
            metrics.append(f"dataset_drift{{job=\"{job_name}\"}} {1 if test_results['dataset_drift'] else 0}")

        # Add test results metrics
        if "tests" in test_results:
            for test in test_results["tests"]:
                test_name = test.get("name", "unknown")
                test_status = test.get("status", "unknown")
                metrics.append(f"test_status{{job=\"{job_name}\",test=\"{test_name}\"}} {1 if test_status == 'SUCCESS' else 0}")

        # Send metrics
        response = requests.post(
            f"{prometheus_url}/metrics/job/{job_name}",
            data="\n".join(metrics),
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


def check_and_alert(
    test_suite: TestSuite,
    email_recipients: Optional[List[str]] = None,
    slack_webhook_url: Optional[str] = None,
    prometheus_url: Optional[str] = None,
    report_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check test results and send alerts if needed.

    Args:
        test_suite: Test suite
        email_recipients: List of email recipients
        slack_webhook_url: Slack webhook URL
        prometheus_url: Prometheus Pushgateway URL
        report_path: Path to the report

    Returns:
        Dictionary with alert status
    """
    logger.info("Checking test results and sending alerts if needed")

    # Get test results
    test_results = test_suite.as_dict()

    # Check if any tests failed
    failed_tests = []
    for test in test_results.get("tests", []):
        if test.get("status", "") != "SUCCESS":
            failed_tests.append(test)

    # If no tests failed, return
    if not failed_tests:
        logger.info("No failed tests, no alerts needed")
        return {"status": "ok", "message": "No failed tests"}

    # Prepare alert message
    alert_subject = f"[ALERT] Data quality issues detected in {len(failed_tests)} tests"

    alert_body = f"""
    <h2>Data Quality Alert</h2>
    <p>Data quality issues were detected in {len(failed_tests)} tests.</p>

    <h3>Failed Tests:</h3>
    <ul>
    """

    for test in failed_tests:
        test_name = test.get("name", "Unknown test")
        test_status = test.get("status", "Unknown status")
        test_message = test.get("message", "No message")

        alert_body += f"<li><strong>{test_name}</strong>: {test_status} - {test_message}</li>\n"

    alert_body += "</ul>"

    if report_path:
        alert_body += f"<p>See the full report at: {report_path}</p>"

    # Send email alert
    email_sent = False
    if email_recipients:
        email_sent = send_email_alert(
            subject=alert_subject,
            body=alert_body,
            recipients=email_recipients,
            attachment_path=report_path
        )

    # Send Slack alert
    slack_sent = False
    if slack_webhook_url:
        slack_message = f"*{alert_subject}*\n\n{len(failed_tests)} tests failed. "
        if report_path:
            slack_message += f"See the full report at: {report_path}"

        slack_sent = send_slack_alert(
            message=slack_message,
            webhook_url=slack_webhook_url
        )

    # Send Prometheus metrics
    prometheus_sent = False
    if prometheus_url:
        prometheus_sent = send_prometheus_alert(
            test_results=test_results,
            prometheus_url=prometheus_url
        )

    return {
        "status": "alert",
        "failed_tests": len(failed_tests),
        "email_sent": email_sent,
        "slack_sent": slack_sent,
        "prometheus_sent": prometheus_sent
    }


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Generate data drift report and alerts")
    parser.add_argument("--reference-data", type=str, required=True, help="Path to reference data")
    parser.add_argument("--production-data", type=str, required=True, help="Path to production data")
    parser.add_argument("--target-column", type=str, default="label", help="Name of the target column")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save reports")
    parser.add_argument("--alert", action="store_true", help="Send alerts if issues are detected")
    parser.add_argument("--email", type=str, help="Comma-separated list of email recipients")
    parser.add_argument("--slack-webhook", type=str, help="Slack webhook URL")
    parser.add_argument("--prometheus-url", type=str, help="Prometheus Pushgateway URL")
    parser.add_argument("--drift-threshold", type=float, default=0.05, help="Threshold for drift tests")
    parser.add_argument("--missing-threshold", type=float, default=0.1, help="Threshold for missing values tests")
    parser.add_argument("--quantile-threshold", type=float, default=0.1, help="Threshold for quantile tests")

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define output paths
        drift_report_path = os.path.join(args.output_dir, f"drift_report_{timestamp}.html")
        quality_report_path = os.path.join(args.output_dir, f"quality_report_{timestamp}.html")
        test_suite_path = os.path.join(args.output_dir, f"test_suite_{timestamp}.html")

        # Load data
        reference_data = load_data(args.reference_data)
        production_data = load_data(args.production_data)

        # Generate drift report
        drift_report = generate_drift_report(
            reference_data=reference_data,
            production_data=production_data,
            target_column=args.target_column,
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
            target_column=args.target_column,
            output_path=test_suite_path,
            drift_threshold=args.drift_threshold,
            missing_threshold=args.missing_threshold,
            quantile_threshold=args.quantile_threshold
        )

        # Print summary
        print("\nMonitoring reports generated successfully.")
        print(f"Drift report saved to: {drift_report_path}")
        print(f"Quality report saved to: {quality_report_path}")
        print(f"Test suite saved to: {test_suite_path}")
        print(f"\nDataset drift detected: {drift_report.as_dict()['metrics'][0]['result']['dataset_drift']}")

        # Check test results and send alerts if needed
        if args.alert:
            email_recipients = args.email.split(",") if args.email else None

            alert_result = check_and_alert(
                test_suite=test_suite,
                email_recipients=email_recipients,
                slack_webhook_url=args.slack_webhook,
                prometheus_url=args.prometheus_url,
                report_path=test_suite_path
            )

            print("\nAlert status:")
            for key, value in alert_result.items():
                print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error in monitoring pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
