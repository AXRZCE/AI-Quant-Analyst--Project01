# Project01 - Project Details & Plan

## 1. Project Purpose

Project01 aims to build a private AI-driven quantitative analyst for stock market prediction. It will ingest market tick data, news, social sentiment, and macroeconomic indicators; compute robust features; train baseline and advanced models; serve predictions via APIs and interactive dashboards; and maintain production-grade MLOps with automated retraining and monitoring.

## 2. Requirements

### Data Sources:
- Tick-level OHLCV from Polygon.io (1 s granularity)
- News streams from Bloomberg API, RavenPack, NewsAPI Pro
- Social sentiment from Twitter API v2
- Fundamental & macro data from Refinitiv/Eikon or Quandl/FRED

### Data Pipeline:
- Ingest via Apache Kafka topics: ticks, news, tweets, fundamentals
- Store raw data in AWS S3 + Delta Lake (Parquet, ACID)

### Feature Engineering:
- Batch: Spark jobs compute technical indicators (MA, RSI, ATR), fundamental ratios
- Streaming: Flink or Spark Structured Streaming for rolling features
- Feature Store: Feast for consistent batch & online features

### Modeling:
- Baseline: XGBoost/LightGBM on Feast features
- Time-Series SOTA: Temporal Fusion Transformer, DeepAR (PyTorch Lightning)
- NLP: FinBERT/RoBERTa sentiment extractor
- Uncertainty: Bayesian intervals via PyMC3/NumPyro, Monte Carlo simulations
- Reinforcement Learning: PPO/SAC via Ray RLlib for portfolio allocation

### MLOps & Serving:
- CI/CD: GitHub Actions + MLflow
- Orchestration: Apache Airflow or Prefect DAGs
- Model Serving: BentoML (Docker + Kubernetes) or TorchServe
- Online Features: Feast Online Store
- Dashboard: Streamlit or Dash for visualization

### Monitoring & Logging:
- Drift detection: EvidentlyAI
- Metrics & Alerts: Prometheus + Grafana
- Logs & Traces: ELK Stack (Elasticsearch + Kibana)

### Infrastructure:
- Kubernetes (EKS, GKE, or AKS)
- GPU Compute: NVIDIA T4/V100/A100
- Databricks Unity Catalog for governance

## 3. Sprint Breakdown

| Sprint | Weeks | Objective | Deliverables |
|--------|-------|-----------|--------------|
| 1 | 1–2 | Kickoff & Environment Setup | Repo structure, virtualenv, requirements.txt, CI skeleton, docs/architecture.md |
| 2 | 3–4 | Data Ingestion & Raw Storage | Ingestion clients, Kafka topics, raw S3 + Delta landing |
| 3 | 5–6 | Feature Engineering & Feast | Spark batch/stream jobs, Feast feature repo, retrieval demos |
| 4 | 7–8 | Baseline Modeling & Backtesting | XGBoost baseline, backtester, W&B integration, evaluation report |
| 5 | 9–10 | Advanced TS & NLP | Temporal Fusion & DeepAR models, FinBERT fine-tune, sentiment features |
| 6 | 11–12 | Uncertainty & RL Prep | Bayesian forecasting modules, Monte Carlo sims, RL env scaffold |
| 7 | 13–14 | RL Strategy & Evaluation | Trained PPO/SAC policy, backtest comparison, integration tests |
| 8 | 15–16 | Productionize & Monitoring | Model API, Airflow DAGs, CI/CD, monitoring dashboards, final docs |

## 4. Folder Structure

```
Project01/
├── data/              # git-ignored raw & processed data        
├── docs/              # architecture, runbooks, diagrams      
│   └── architecture.md                                 
├── infra/             # infra-as-code: CI, Airflow, Feast     
├── notebooks/         # exploratory analysis & plots        
├── src/               # application source code             
│   ├── ingest/        # data ingestion clients             
│   ├── etl/           # feature pipelines                  
│   ├── models/        # model definitions & training scripts
│   └── backtest/      # backtesting harness                
├── tests/             # unit & integration tests            
├── .github/           # GitHub workflows (CI/CD)           
├── requirements.txt                            
└── README.md                                  
```

## 5. Tech Stack Summary

Refer to architecture.md for the full stack. Key components:

- **Data**: Kafka, S3 + Delta Lake, Feast
- **Modeling**: PyTorch Lightning, TensorFlow, XGBoost, Ray Tune, FinBERT
- **MLOps**: GitHub Actions, MLflow, Airflow, BentoML, Prometheus
- **Infra**: Kubernetes, NVIDIA GPUs, Databricks

## 6. Next Steps

✅ Sprint 1: finalize environment & CI → IN PROGRESS

🔜 Sprint 2: implement ingestion clients & raw storage jobs

Document last updated: Week 1, Day 1 of Sprint 1
