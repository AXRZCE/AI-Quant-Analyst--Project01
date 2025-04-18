# Project01: AI-Quant-Analyst API Documentation

## Overview

This document provides documentation for the API endpoints exposed by the Project01 AI-Quant-Analyst platform. The API allows users to make predictions, run backtests, and access model information.

## Base URL

The base URL for all API endpoints is:

```
https://baseline-service.project01.com
```

## Authentication

All API requests require authentication using an API key. The API key should be included in the `Authorization` header of the request:

```
Authorization: Bearer <api-key>
```

## Endpoints

### Prediction API

#### Single Prediction

```
POST /predict
```

Make a prediction for a single symbol.

**Request Body**:

```json
{
  "symbol": "AAPL",
  "features": {
    "ma_5": 0.1,
    "rsi_14": 50,
    "close": 150,
    "feature1": 0.5,
    "feature2": -0.2,
    ...
  }
}
```

**Response**:

```json
{
  "symbol": "AAPL",
  "predictions": [0.02],
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

#### Batch Prediction

```
POST /batch_predict
```

Make predictions for multiple symbols.

**Request Body**:

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "features": [
    {
      "ma_5": 0.1,
      "rsi_14": 50,
      "close": 150,
      "feature1": 0.5,
      "feature2": -0.2,
      ...
    },
    {
      "ma_5": 0.2,
      "rsi_14": 60,
      "close": 250,
      "feature1": 0.6,
      "feature2": -0.1,
      ...
    },
    {
      "ma_5": 0.3,
      "rsi_14": 70,
      "close": 2000,
      "feature1": 0.7,
      "feature2": 0.0,
      ...
    }
  ]
}
```

**Response**:

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "predictions": [0.02, 0.01, -0.01],
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

### Backtesting API

#### Run Backtest

```
POST /backtest
```

Run a backtest for a specific model and time period.

**Request Body**:

```json
{
  "model_id": "baseline_xgb",
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "start_date": "2023-01-01",
  "end_date": "2023-03-31",
  "initial_capital": 100000,
  "transaction_cost": 0.001
}
```

**Response**:

```json
{
  "backtest_id": "bt-123456",
  "status": "running",
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

#### Get Backtest Results

```
GET /backtest/{backtest_id}
```

Get the results of a backtest.

**Response**:

```json
{
  "backtest_id": "bt-123456",
  "status": "completed",
  "results": {
    "final_capital": 120000,
    "total_return": 0.2,
    "sharpe_ratio": 1.5,
    "max_drawdown": 0.1,
    "win_rate": 0.6,
    "num_trades": 50
  },
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

### Model API

#### List Models

```
GET /models
```

List all available models.

**Response**:

```json
{
  "models": [
    {
      "id": "baseline_xgb",
      "name": "Baseline XGBoost",
      "version": "1.0.0",
      "type": "regression",
      "created_at": "2023-04-01T00:00:00.000Z"
    },
    {
      "id": "tft_model",
      "name": "Temporal Fusion Transformer",
      "version": "1.0.0",
      "type": "time-series",
      "created_at": "2023-04-15T00:00:00.000Z"
    },
    {
      "id": "rl_policy",
      "name": "RL Policy",
      "version": "1.0.0",
      "type": "reinforcement-learning",
      "created_at": "2023-04-30T00:00:00.000Z"
    }
  ],
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

#### Get Model Details

```
GET /models/{model_id}
```

Get details about a specific model.

**Response**:

```json
{
  "id": "baseline_xgb",
  "name": "Baseline XGBoost",
  "version": "1.0.0",
  "type": "regression",
  "created_at": "2023-04-01T00:00:00.000Z",
  "features": ["ma_5", "rsi_14", "close", "feature1", "feature2"],
  "metrics": {
    "accuracy": 0.65,
    "precision": 0.7,
    "recall": 0.6,
    "f1_score": 0.65
  },
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

### Health Check API

#### Health Check

```
GET /healthcheck
```

Check the health of the API.

**Response**:

```json
{
  "status": "ok",
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request. In case of an error, the response body will contain an error message:

```json
{
  "error": "Invalid request: missing required field 'symbol'",
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

## Rate Limiting

The API is rate-limited to 100 requests per minute per API key. If you exceed this limit, you will receive a 429 Too Many Requests response:

```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "timestamp": "2023-04-18T12:34:56.789Z"
}
```

## Versioning

The API is versioned using the URL path. The current version is v1:

```
https://baseline-service.project01.com/v1/predict
```

## Examples

### cURL

```bash
curl -X POST https://baseline-service.project01.com/predict \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "features": {"ma_5": 0.1, "rsi_14": 50, "close": 150}}'
```

### Python

```python
import requests
import json

url = "https://baseline-service.project01.com/predict"
headers = {
    "Authorization": "Bearer <api-key>",
    "Content-Type": "application/json"
}
data = {
    "symbol": "AAPL",
    "features": {
        "ma_5": 0.1,
        "rsi_14": 50,
        "close": 150
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## Support

For API support, please contact api-support@project01.com.
