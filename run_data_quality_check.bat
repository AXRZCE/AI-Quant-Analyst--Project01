@echo off
echo Running Data Quality Check...

REM Create necessary directories
mkdir reports 2>nul
mkdir logs 2>nul

REM Set environment variables
set REFERENCE_DATA=data/features/batch/technical/reference
set PRODUCTION_DATA=data/features/batch/technical/latest
set OUTPUT_DIR=reports
set TARGET_COLUMN=target
set SMTP_USERNAME=your-email@gmail.com
set SMTP_PASSWORD=your-app-password
set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your-webhook-url
set PROMETHEUS_PUSHGATEWAY_URL=http://localhost:9091

REM Run the data quality check
python src/monitor/drift_report.py ^
  --reference-data %REFERENCE_DATA% ^
  --production-data %PRODUCTION_DATA% ^
  --target-column %TARGET_COLUMN% ^
  --output-dir %OUTPUT_DIR% ^
  --alert ^
  --email %SMTP_USERNAME% ^
  --slack-webhook %SLACK_WEBHOOK_URL% ^
  --prometheus-url %PROMETHEUS_PUSHGATEWAY_URL% ^
  --drift-threshold 0.05 ^
  --missing-threshold 0.1 ^
  --quantile-threshold 0.1

echo.
echo Data Quality Check completed!
echo Reports saved to %OUTPUT_DIR%
echo.
