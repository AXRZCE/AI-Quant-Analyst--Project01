# Project01: AI-Quant-Analyst Runbook

## Overview

This runbook provides step-by-step instructions for operating and maintaining the Project01 AI-Quant-Analyst platform. It covers common tasks such as deploying the system, monitoring its health, troubleshooting issues, and performing routine maintenance.

## Prerequisites

- Access to the Kubernetes cluster
- Access to the GitHub repository
- Access to the Docker registry
- Access to the monitoring dashboards
- Access to the Airflow UI

## Deployment

### Initial Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/AXRZCE/AI-Quant-Analyst--Project01.git
   cd AI-Quant-Analyst--Project01
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Deploy the infrastructure:
   ```bash
   kubectl apply -f infra/k8s/
   ```

4. Verify the deployment:
   ```bash
   kubectl get pods
   kubectl get services
   kubectl get deployments
   ```

### Updating the Deployment

1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Update the infrastructure:
   ```bash
   kubectl apply -f infra/k8s/
   ```

3. Restart the deployments:
   ```bash
   kubectl rollout restart deployment/baseline-service
   ```

4. Verify the update:
   ```bash
   kubectl get pods
   kubectl rollout status deployment/baseline-service
   ```

## Monitoring

### Accessing Dashboards

1. Access the Grafana dashboard:
   ```bash
   kubectl port-forward svc/grafana 3000:80
   ```
   Then open http://localhost:3000 in your browser.

2. Access the Prometheus dashboard:
   ```bash
   kubectl port-forward svc/prometheus 9090:9090
   ```
   Then open http://localhost:9090 in your browser.

3. Access the Airflow UI:
   ```bash
   kubectl port-forward svc/airflow-webserver 8080:8080
   ```
   Then open http://localhost:8080 in your browser.

### Checking System Health

1. Check the status of all pods:
   ```bash
   kubectl get pods
   ```

2. Check the logs of a specific pod:
   ```bash
   kubectl logs <pod-name>
   ```

3. Check the status of all services:
   ```bash
   kubectl get services
   ```

4. Check the status of all deployments:
   ```bash
   kubectl get deployments
   ```

5. Check the status of all Airflow DAGs:
   ```bash
   kubectl exec -it <airflow-webserver-pod> -- airflow dags list
   ```

### Alerts

1. View active alerts in Prometheus:
   ```bash
   kubectl port-forward svc/prometheus 9090:9090
   ```
   Then open http://localhost:9090/alerts in your browser.

2. View alert history in Grafana:
   ```bash
   kubectl port-forward svc/grafana 3000:80
   ```
   Then open http://localhost:3000/alerting/history in your browser.

## Troubleshooting

### Common Issues

#### Pod Crashes

1. Check the pod status:
   ```bash
   kubectl get pods
   ```

2. Check the pod logs:
   ```bash
   kubectl logs <pod-name>
   ```

3. Describe the pod:
   ```bash
   kubectl describe pod <pod-name>
   ```

4. Restart the pod:
   ```bash
   kubectl delete pod <pod-name>
   ```

#### Service Unavailable

1. Check the service status:
   ```bash
   kubectl get services
   ```

2. Check the endpoints:
   ```bash
   kubectl get endpoints <service-name>
   ```

3. Check the pod selector:
   ```bash
   kubectl describe service <service-name>
   ```

4. Check the pod labels:
   ```bash
   kubectl get pods --show-labels
   ```

#### Airflow DAG Failures

1. Check the DAG status:
   ```bash
   kubectl exec -it <airflow-webserver-pod> -- airflow dags list
   ```

2. Check the DAG runs:
   ```bash
   kubectl exec -it <airflow-webserver-pod> -- airflow dags list-runs -d <dag-id>
   ```

3. Check the task logs:
   ```bash
   kubectl exec -it <airflow-webserver-pod> -- airflow tasks logs <dag-id> <task-id> <execution-date>
   ```

4. Clear the failed task:
   ```bash
   kubectl exec -it <airflow-webserver-pod> -- airflow tasks clear <dag-id> -t <task-id> -s <execution-date>
   ```

### Debugging

1. Shell into a pod:
   ```bash
   kubectl exec -it <pod-name> -- /bin/bash
   ```

2. Check the environment variables:
   ```bash
   kubectl exec -it <pod-name> -- env
   ```

3. Check the filesystem:
   ```bash
   kubectl exec -it <pod-name> -- ls -la
   ```

4. Check the network connectivity:
   ```bash
   kubectl exec -it <pod-name> -- curl <service-name>
   ```

## Routine Maintenance

### Backup and Restore

1. Backup the database:
   ```bash
   kubectl exec -it <database-pod> -- pg_dump -U <username> <database> > backup.sql
   ```

2. Restore the database:
   ```bash
   kubectl exec -it <database-pod> -- psql -U <username> <database> < backup.sql
   ```

### Scaling

1. Scale a deployment:
   ```bash
   kubectl scale deployment <deployment-name> --replicas=<number>
   ```

2. Enable autoscaling:
   ```bash
   kubectl autoscale deployment <deployment-name> --min=<min> --max=<max> --cpu-percent=<percent>
   ```

### Updating Models

1. Train a new model:
   ```bash
   python src/models/train_baseline.py --data-path <data-path> --output-path <output-path>
   ```

2. Save the model to BentoML:
   ```bash
   python src/serving/save_model.py --model-path <model-path> --model-name <model-name>
   ```

3. Build the BentoML service:
   ```bash
   cd src/serving
   bentoml build
   ```

4. Containerize the BentoML service:
   ```bash
   bentoml containerize <service-name>:<tag> --tag <registry>/<service-name>:<tag> --push
   ```

5. Update the deployment:
   ```bash
   kubectl set image deployment/baseline-service baseline=<registry>/<service-name>:<tag>
   ```

### Updating Data

1. Ingest new data:
   ```bash
   python src/ingest/ingest_data.py --date <date>
   ```

2. Generate features:
   ```bash
   python src/etl/batch_features.py --date <date>
   ```

3. Run drift detection:
   ```bash
   python src/monitor/drift_report.py --reference-data <reference-data> --production-data <production-data> --output-path <output-path>
   ```

## Disaster Recovery

### Kubernetes Cluster Failure

1. Recreate the cluster:
   ```bash
   # Follow the cloud provider's instructions for creating a new cluster
   ```

2. Redeploy the infrastructure:
   ```bash
   kubectl apply -f infra/k8s/
   ```

3. Restore the database:
   ```bash
   kubectl exec -it <database-pod> -- psql -U <username> <database> < backup.sql
   ```

### Data Loss

1. Restore from backup:
   ```bash
   # Follow the backup and restore instructions
   ```

2. Regenerate features:
   ```bash
   python src/etl/batch_features.py --date <date>
   ```

3. Retrain models:
   ```bash
   python src/models/train_baseline.py --data-path <data-path> --output-path <output-path>
   ```

## Security Procedures

### Rotating Credentials

1. Update API keys:
   ```bash
   kubectl create secret generic api-keys --from-literal=polygon-api-key=<new-key> --dry-run=client -o yaml | kubectl apply -f -
   ```

2. Update database credentials:
   ```bash
   kubectl create secret generic db-credentials --from-literal=username=<new-username> --from-literal=password=<new-password> --dry-run=client -o yaml | kubectl apply -f -
   ```

3. Restart the affected deployments:
   ```bash
   kubectl rollout restart deployment/<deployment-name>
   ```

### Security Incidents

1. Isolate the affected components:
   ```bash
   kubectl scale deployment <deployment-name> --replicas=0
   ```

2. Investigate the logs:
   ```bash
   kubectl logs <pod-name>
   ```

3. Restore from a known good state:
   ```bash
   # Follow the backup and restore instructions
   ```

4. Update security measures:
   ```bash
   # Update firewall rules, access controls, etc.
   ```

5. Restore service:
   ```bash
   kubectl scale deployment <deployment-name> --replicas=<number>
   ```

## Contact Information

- **Project Owner**: Aksharajsinh Parmar (aksharaj.asp.15@gmail.com)
- **DevOps Team**: devops@project01.com
- **Data Science Team**: datascience@project01.com
- **On-Call Rotation**: oncall@project01.com
