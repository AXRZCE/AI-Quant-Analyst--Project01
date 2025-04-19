@echo off
echo Starting Project01 Monitoring Stack...

REM Create necessary directories
mkdir logs 2>nul
mkdir reports 2>nul
mkdir infra\grafana\provisioning\dashboards 2>nul
mkdir infra\grafana\provisioning\datasources 2>nul
mkdir infra\filebeat 2>nul

REM Set environment variables
set SMTP_USERNAME=your-email@gmail.com
set SMTP_PASSWORD=your-app-password
set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your-webhook-url
set GRAFANA_ADMIN_USER=admin
set GRAFANA_ADMIN_PASSWORD=admin

REM Start the monitoring stack
cd infra
docker-compose -f docker-compose-monitoring.yml up -d

echo.
echo Monitoring stack started successfully!
echo.
echo Access the following services:
echo - Prometheus: http://localhost:9090
echo - Alertmanager: http://localhost:9093
echo - Grafana: http://localhost:3000 (admin/admin)
echo - Kibana: http://localhost:5601
echo.
echo To stop the monitoring stack, run: docker-compose -f infra/docker-compose-monitoring.yml down
echo.
