# Data Pipeline Documentation

This document provides an overview of the data ingestion pipeline for the AI Quant Analyst project.

## Overview

The data pipeline is responsible for:

1. Fetching data from various sources (Polygon.io, Yahoo Finance, NewsAPI)
2. Validating and cleaning the data
3. Storing the data in Delta Lake
4. Registering features with Feast

## Components

### API Clients

- **BaseAPIClient**: Base class with retry logic and error handling
- **PolygonClient**: Client for Polygon.io API
- **YahooFinanceClient**: Client for Yahoo Finance API
- **NewsAPIClient**: Client for NewsAPI

### Data Validation

- **DataValidator**: Class for validating and cleaning data
  - Validates tick data (stock prices)
  - Validates news articles
  - Validates fundamental data

### Data Storage

- **DeltaLakeStorage**: Class for storing data in Delta Lake
  - Supports partitioning by date and symbol
  - Provides methods for saving and loading data

### Feature Registry

- **FeastRegistry**: Class for registering features with Feast
  - Creates feature views for different data types
  - Registers features with Feast

### Data Pipeline

- **DataPipeline**: Main class that orchestrates the entire pipeline
  - Fetches data from multiple sources with fallback
  - Validates and cleans the data
  - Stores the data in Delta Lake

## Usage

### Basic Usage

```python
from datetime import datetime, timedelta
from src.ingest.data_pipeline import ingest_all_data

# Define symbols and date range
symbols = ['AAPL', 'MSFT', 'GOOGL']
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Ingest all data
results = ingest_all_data(
    symbols=symbols,
    start=start_date,
    end=end_date,
    interval='1d',
    save_to_delta=True
)
```

### Using the Script

```bash
python scripts/run_data_pipeline.py --symbols AAPL MSFT GOOGL --days 30 --interval 1d
```

## Data Flow

1. **Data Ingestion**:
   - Fetch stock data from Polygon.io or Yahoo Finance
   - Fetch fundamental data from Polygon.io
   - Fetch news articles from NewsAPI

2. **Data Validation**:
   - Check for missing values
   - Check for anomalies (e.g., negative prices)
   - Check for inconsistencies (e.g., high < low)
   - Clean and normalize the data

3. **Data Storage**:
   - Store raw data in Delta Lake
   - Partition data by date and symbol
   - Organize data into bronze, silver, and gold layers

4. **Feature Registry**:
   - Register features with Feast
   - Create feature views for different data types
   - Enable consistent feature access for training and inference

## Error Handling

The pipeline includes robust error handling:

- Retry logic for API requests
- Fallback to alternative data sources
- Logging of errors and warnings
- Validation and cleaning of data

## Configuration

The pipeline can be configured through environment variables:

- `POLYGON_API_KEY`: API key for Polygon.io
- `NEWS_API_KEY`: API key for NewsAPI
- `CACHE_DIR`: Directory for caching API responses
- `DELTA_ROOT`: Root directory for Delta Lake storage
- `FEAST_REPO_PATH`: Path to the Feast feature repository

## Extending the Pipeline

To add a new data source:

1. Create a new client class that inherits from `BaseAPIClient`
2. Implement the necessary methods for fetching data
3. Add the client to the `DataPipeline` class
4. Update the data validation and storage components as needed

## Testing

The pipeline includes unit tests for all components:

```bash
python -m unittest tests/test_data_pipeline.py
```
