# Monitoring and Alerting System

This document provides an overview of the monitoring and alerting system for the AI Quant Analyst project.

## Overview

The monitoring and alerting system is designed to track the health and performance of the AI Quant Analyst platform, including:

1. **Data Quality Monitoring**: Detecting data drift and data quality issues
2. **Model Performance Monitoring**: Tracking model accuracy and performance metrics
3. **System Monitoring**: Monitoring system resources and infrastructure
4. **Alerting**: Sending notifications when issues are detected
5. **Logging**: Centralized logging and log analysis

## Components

### Data Quality Monitoring

The data quality monitoring system uses EvidentlyAI to detect data drift and data quality issues:

- **Drift Detection**: Monitors changes in data distributions over time
- **Data Quality Checks**: Validates data against quality rules
- **Test Suites**: Runs automated tests on data
- **Scheduled Monitoring**: Regularly checks data quality

Key files:
- `src/monitor/drift_report.py`: Core drift detection functionality
- `src/monitor/scheduled_monitoring.py`: Scheduled monitoring jobs

### Model Performance Monitoring

The model performance monitoring system tracks the accuracy and performance of models:

- **Accuracy Metrics**: RMSE, MAE, R², etc.
- **Latency Monitoring**: Response time tracking
- **Throughput Monitoring**: Requests per second
- **Error Rate Monitoring**: Failed predictions

Key metrics:
- `model_rmse`: Root Mean Squared Error
- `model_mae`: Mean Absolute Error
- `request_latency_seconds`: Request latency
- `request_count`: Request count
- `request_errors_total`: Error count

### System Monitoring

The system monitoring tracks the health and performance of the infrastructure:

- **CPU Usage**: CPU utilization by container
- **Memory Usage**: Memory utilization by container
- **Disk Usage**: Disk space utilization
- **Network Traffic**: Network I/O

### Alerting

The alerting system sends notifications when issues are detected:

- **Email Alerts**: Sends email notifications
- **Slack Alerts**: Sends Slack messages
- **Prometheus Alerts**: Configurable alert rules

Key files:
- `infra/prometheus/alert_rules.yml`: Prometheus alert rules
- `infra/prometheus/alertmanager.yml`: Alertmanager configuration

### Logging

The logging system collects and analyzes logs from all components:

- **Centralized Logging**: All logs in one place
- **Log Analysis**: Search and analyze logs
- **Log Visualization**: Visualize log patterns
- **Log Alerting**: Alerts based on log patterns

Key components:
- **Elasticsearch**: Log storage and indexing
- **Kibana**: Log visualization and analysis
- **Filebeat**: Log collection

## Setup and Configuration

### Prerequisites

- Docker and Docker Compose
- SMTP credentials for email alerts
- Slack webhook URL for Slack alerts

### Installation

1. Run the monitoring stack:

```bash
./run_monitoring.bat
```

2. Access the monitoring dashboards:

- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000 (admin/admin)
- Kibana: http://localhost:5601

### Configuration

#### Data Quality Monitoring

To configure data quality monitoring:

1. Edit `src/monitor/drift_report.py` to adjust thresholds and tests
2. Schedule monitoring jobs in `src/monitor/scheduled_monitoring.py`

Example:

```bash
python src/monitor/scheduled_monitoring.py \
  --reference-data data/features/batch/technical/2023-01-01 \
  --production-data data/features/batch/technical/latest \
  --output-dir reports \
  --alert \
  --email your-email@example.com \
  --slack-webhook https://hooks.slack.com/services/your-webhook-url \
  --schedule-type daily \
  --hour 0 \
  --minute 0
```

#### Alerting

To configure alerting:

1. Edit `infra/prometheus/alert_rules.yml` to adjust alert rules
2. Edit `infra/prometheus/alertmanager.yml` to configure notification channels

## Dashboards

### Data Quality Dashboard

The data quality dashboard provides an overview of data quality metrics:

- Dataset Drift Status
- Failed Tests Count
- Missing Values Percentage
- Test Status by Test

### Model Performance Dashboard

The model performance dashboard tracks model performance metrics:

- Model Error Metrics (RMSE, MAE)
- Request Latency
- Request Rate
- Error Rate

### System Dashboard

The system dashboard monitors system resources:

- CPU Usage by Container
- Memory Usage by Container
- Disk Usage
- Network Traffic

## Alerting Rules

### Data Quality Alerts

- **DatasetDriftDetected**: Alerts when dataset drift is detected
- **DataQualityTestsFailed**: Alerts when data quality tests fail
- **HighMissingValues**: Alerts when missing values exceed threshold

### Model Performance Alerts

- **ModelPerformanceDegraded**: Alerts when model performance degrades
- **ModelLatencyHigh**: Alerts when model latency is high
- **ModelErrorRateHigh**: Alerts when model error rate is high

### System Alerts

- **HighCPUUsage**: Alerts when CPU usage is high
- **HighMemoryUsage**: Alerts when memory usage is high
- **PodCrashLooping**: Alerts when pods are crash looping

## Logging

### Log Collection

Logs are collected from:

- Docker containers
- System logs
- Application logs

### Log Analysis

Logs can be analyzed in Kibana:

1. Go to http://localhost:5601
2. Navigate to "Discover"
3. Select the appropriate index pattern
4. Search and filter logs

### Log Visualization

Create visualizations in Kibana:

1. Go to http://localhost:5601
2. Navigate to "Visualize"
3. Create visualizations based on log data
4. Add visualizations to dashboards

## Best Practices

1. **Regular Monitoring**: Schedule regular monitoring jobs
2. **Appropriate Thresholds**: Set appropriate thresholds for alerts
3. **Alert Fatigue**: Avoid alert fatigue by tuning alert sensitivity
4. **Log Rotation**: Implement log rotation to manage disk space
5. **Dashboard Organization**: Organize dashboards by function
6. **Documentation**: Document all monitoring and alerting configurations

## Troubleshooting

### Common Issues

#### Monitoring Stack Not Starting

```bash
# Check Docker Compose logs
docker-compose -f infra/docker-compose-monitoring.yml logs
```

#### Alerts Not Being Sent

1. Check Alertmanager configuration
2. Verify SMTP and Slack webhook credentials
3. Check Alertmanager logs:

```bash
docker logs alertmanager
```

#### Data Quality Monitoring Failing

1. Check data paths
2. Verify data format
3. Check logs:

```bash
tail -f logs/monitoring.log
```

## Conclusion

The monitoring and alerting system provides comprehensive visibility into the health and performance of the AI Quant Analyst platform. By proactively monitoring data quality, model performance, and system resources, issues can be detected and resolved before they impact users.
