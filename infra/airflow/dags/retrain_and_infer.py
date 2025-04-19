"""
Airflow DAG for retraining and inference.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.dates import days_ago
import json
import os

# Define default arguments
default_args = {
    "owner": "project01",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id="daily_retrain_and_infer",
    default_args=default_args,
    description="Daily retraining and inference pipeline",
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["project01", "trading"],
) as dag:

    # Task 1: Ingest new data
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command="python {{ var.value.project_dir }}/src/ingest/ingest_data.py --date $(date +\\%Y-\\%m-\\%d)",
    )

    # Task 2: Generate features
    generate_features = BashOperator(
        task_id="generate_features",
        bash_command="python {{ var.value.project_dir }}/src/etl/batch_features.py --date $(date +\\%Y-\\%m-\\%d)",
    )

    # Task 3: Train baseline model
    train_baseline = BashOperator(
        task_id="train_baseline",
        bash_command=(
            "python {{ var.value.project_dir }}/src/models/train_baseline.py "
            "--data-path {{ var.value.project_dir }}/data/features/batch/technical/$(date +\\%Y-\\%m-\\%d) "
            "--output-path {{ var.value.project_dir }}/models/baseline_$(date +\\%Y-\\%m-\\%d).pkl"
        ),
    )

    # Task 4: Save model to BentoML
    save_to_bentoml = BashOperator(
        task_id="save_to_bentoml",
        bash_command=(
            "python {{ var.value.project_dir }}/src/serving/save_model.py "
            "--model-path {{ var.value.project_dir }}/models/baseline_$(date +\\%Y-\\%m-\\%d).pkl "
            "--model-name baseline_xgb"
        ),
    )

    # Task 5: Build BentoML service
    build_bento = BashOperator(
        task_id="build_bento",
        bash_command=(
            "cd {{ var.value.project_dir }}/src/serving && "
            "bentoml build"
        ),
    )

    # Task 6: Containerize BentoML service
    containerize_bento = BashOperator(
        task_id="containerize_bento",
        bash_command=(
            "bentoml containerize baseline_xgb_service:latest "
            "--tag {{ var.value.docker_registry }}/baseline_xgb_service:$(date +\\%Y-\\%m-\\%d) "
            "--push"
        ),
    )

    # Task 7: Deploy to Kubernetes
    deploy_to_k8s = BashOperator(
        task_id="deploy_to_k8s",
        bash_command=(
            "kubectl set image deployment/baseline-service "
            "baseline={{ var.value.docker_registry }}/baseline_xgb_service:$(date +\\%Y-\\%m-\\%d)"
        ),
    )

    # Task 8: Check service health
    check_service_health = HttpSensor(
        task_id="check_service_health",
        http_conn_id="baseline_service",
        endpoint="/healthcheck",
        response_check=lambda response: json.loads(response.text)["status"] == "ok",
        poke_interval=30,
        timeout=300,
    )

    # Task 9: Run enhanced drift detection with alerting
    run_drift_detection = BashOperator(
        task_id="run_drift_detection",
        bash_command=(
            "python {{ var.value.project_dir }}/src/monitor/drift_report.py "
            "--reference-data {{ var.value.project_dir }}/data/features/batch/technical/$(date -d 'yesterday' +\\%Y-\\%m-\\%d) "
            "--production-data {{ var.value.project_dir }}/data/features/batch/technical/$(date +\\%Y-\\%m-\\%d) "
            "--output-dir {{ var.value.project_dir }}/reports "
            "--target-column target "
            "--alert "
            "--email {{ var.value.alert_email }} "
            "--slack-webhook {{ var.value.slack_webhook }} "
            "--prometheus-url {{ var.value.prometheus_pushgateway_url }} "
            "--drift-threshold 0.05 "
            "--missing-threshold 0.1 "
            "--quantile-threshold 0.1"
        ),
    )

    # Task 10: Generate data quality report
    generate_quality_report = BashOperator(
        task_id="generate_quality_report",
        bash_command=(
            "python {{ var.value.project_dir }}/src/monitor/drift_report.py "
            "--reference-data {{ var.value.project_dir }}/data/features/batch/technical/reference "
            "--production-data {{ var.value.project_dir }}/data/features/batch/technical/$(date +\\%Y-\\%m-\\%d) "
            "--output-dir {{ var.value.project_dir }}/reports "
            "--target-column target"
        ),
    )

    # Task 11: Update model metrics in Prometheus
    update_model_metrics = BashOperator(
        task_id="update_model_metrics",
        bash_command=(
            "python {{ var.value.project_dir }}/src/monitor/update_model_metrics.py "
            "--model-path {{ var.value.project_dir }}/models/baseline_$(date +\\%Y-\\%m-\\%d).pkl "
            "--data-path {{ var.value.project_dir }}/data/features/batch/technical/$(date +\\%Y-\\%m-\\%d) "
            "--prometheus-url {{ var.value.prometheus_pushgateway_url }}"
        ),
    )

    # Define task dependencies
    ingest_data >> generate_features >> train_baseline >> save_to_bentoml >> build_bento >> containerize_bento >> deploy_to_k8s >> check_service_health
    check_service_health >> run_drift_detection
    check_service_health >> generate_quality_report
    train_baseline >> update_model_metrics
