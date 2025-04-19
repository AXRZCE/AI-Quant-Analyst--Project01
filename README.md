# AI-Quant-Analyst--Project01

![CI](https://github.com/AXRZCE/AI-Quant-Analyst--Project01/actions/workflows/ci.yml/badge.svg)

## Overview

AI-powered quantitative analysis platform for financial markets. This project combines machine learning with financial data to generate trading signals and portfolio optimization strategies.

## Features

- Data ingestion from multiple financial sources
- Feature engineering pipeline
- Machine learning model training and evaluation
- Backtesting framework
- Web-based dashboard for visualization

## Setup

### Prerequisites

- Python 3.10
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/AXRZCE/AI-Quant-Analyst--Project01.git
cd Project01

# Create and activate virtual environment
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Project01/
├── data/                             # raw & processed data (git-ignored)
├── docs/                             # architecture, runbooks, diagrams
├── infra/                           # infra as code (CI, Feast, Airflow, etc.)
├── notebooks/                   # exploratory work, plots
├── src/                               # application code
│   ├── ingest/                 # data ingestion scripts
│   ├── etl/                       # feature pipelines
│   ├── models/                 # model definitions & trainers
│   ├── backtest/             # backtester code
│   ├── rl/                        # reinforcement learning components
├── tests/                           # unit & integration tests
├── .github/                       # GitHub workflows
├── backend/                       # FastAPI backend for UI
├── frontend/                      # React frontend for UI
```

## Components

### 1. Data Ingestion

See [docs/data_ingestion.md](docs/data_ingestion.md) for details on the data ingestion components.

### 2. Feature Engineering

- Feature pipelines using Spark and Delta Lake
- Feature registry using Feast
- Time-series features and technical indicators

### 3. Models

- Baseline models using XGBoost
- Deep learning models using PyTorch Lightning
- NLP sentiment analysis using FinBERT
- Reinforcement learning trading agents

### 4. UI

See [docs/ui.md](docs/ui.md) for details on the UI components.

## Usage

### Running the UI

#### Docker-based Setup (Recommended)

```bash
# On Windows
run_ui.bat

# On Linux/macOS
./run_ui.sh
```

#### Local Development Setup (Windows)

```bash
run_ui_local.bat
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

_Coming soon_
