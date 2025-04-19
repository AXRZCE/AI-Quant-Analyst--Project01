# Project01 API Documentation

## Overview

This document provides comprehensive documentation for the Project01 API, which serves as the interface for the AI-Quant-Analyst platform. The API allows users to make predictions, access model information, and manage the system.

## Base URL

The base URL for all API endpoints is:

```
http://localhost:8000/api
```

In production, this would be replaced with your domain.

## Authentication

The API supports two authentication methods:

### API Key Authentication

Most endpoints require an API key for authentication. The API key should be included in the `X-API-Key` header of the request:

```
X-API-Key: your_api_key
```

API keys can be obtained from the system administrator.

### JWT Token Authentication

For user-specific operations, JWT token authentication is also supported. To obtain a token, use the `/api/token` endpoint with your username and password. The token should be included in the `Authorization` header of the request:

```
Authorization: Bearer your_token
```

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, clients are limited to 60 requests per minute. When the rate limit is exceeded, the API will return a 429 Too Many Requests response.

Rate limit information is included in the response headers:

- `X-RateLimit-Limit`: The maximum number of requests allowed per minute
- `X-RateLimit-Remaining`: The number of requests remaining in the current window
- `X-RateLimit-Reset`: The time at which the rate limit window resets (Unix timestamp)

## Caching

The API implements caching to improve performance. GET requests are cached by default for 5 minutes. Cache status is included in the response headers:

- `X-Cache`: "HIT" if the response was served from cache, "MISS" if it was generated

## Endpoints

### Health Check

```
GET /api/health
```

Check the health of the API.

**Response**:

```json
{
  "status": "ok",
  "timestamp": 1625097600,
  "version": "1.0.0",
  "environment": "development"
}
```

### Authentication

#### Get Access Token

```
POST /api/token
```

Get an access token using username and password.

**Request Body**:

```json
{
  "username": "user",
  "password": "password"
}
```

**Response**:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": 1625097600
}
```

#### Get Current User

```
GET /api/users/me
```

Get information about the current user.

**Authentication**: Bearer token

**Response**:

```json
{
  "username": "user",
  "email": "user@example.com",
  "full_name": "Regular User",
  "disabled": false,
  "scopes": ["read"]
}
```

### Prediction

#### Make Prediction (Legacy)

```
POST /api/predict
```

Make a prediction using the loaded model (legacy endpoint, no authentication required).

**Request Body**:

```json
{
  "symbol": "AAPL",
  "days": 7
}
```

**Response**:

```json
{
  "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
  "prices": [150.25, 152.30, 151.80],
  "ma_5": [149.50, 150.20, 151.10],
  "rsi_14": [65.2, 68.7, 62.3],
  "sentiment": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10
  },
  "prediction": 155.75
}
```

#### Make Prediction (Authenticated)

```
POST /api/predict/v2
```

Make a prediction using the loaded model with authentication.

**Authentication**: API key

**Request Body**:

```json
{
  "symbol": "AAPL",
  "days": 7,
  "features": {
    "feature_1": 0.5,
    "feature_2": 0.3
  }
}
```

**Response**:

```json
{
  "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
  "prices": [150.25, 152.30, 151.80],
  "ma_5": [149.50, 150.20, 151.10],
  "rsi_14": [65.2, 68.7, 62.3],
  "sentiment": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10
  },
  "prediction": 155.75,
  "confidence_interval": {
    "lower": 152.50,
    "upper": 158.90
  },
  "feature_importance": [
    {
      "feature": "feature_1",
      "importance": 0.7
    },
    {
      "feature": "feature_2",
      "importance": 0.3
    }
  ]
}
```

### Model Information

#### Get Model Info

```
GET /api/model/info
```

Get information about the loaded model.

**Authentication**: API key

**Response**:

```json
{
  "model_path": "models/baseline_xgb.pkl",
  "model_loaded": true,
  "model_type": "XGBRegressor",
  "features": ["feature_1", "feature_2", "feature_3"],
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85
  },
  "fallback_enabled": true
}
```

### Cache Management

#### Get Cache Statistics

```
GET /api/cache/stats
```

Get cache statistics.

**Authentication**: API key

**Response**:

```json
{
  "size": 100,
  "max_size": 1000,
  "hits": 500,
  "misses": 100,
  "hit_ratio": 0.83
}
```

#### Clear Cache

```
POST /api/cache/clear
```

Clear the cache.

**Authentication**: API key

**Response**:

```json
{
  "status": "ok",
  "message": "Cache cleared"
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request. In case of an error, the response body will contain an error object with the following structure:

```json
{
  "error": {
    "code": "error_code",
    "message": "Error message",
    "errors": [
      {
        "loc": ["field_name"],
        "msg": "Error message",
        "type": "error_type"
      }
    ]
  }
}
```

### Common Error Codes

- `validation_error`: The request data failed validation
- `authentication_error`: Authentication failed
- `authorization_error`: The user does not have permission to access the resource
- `not_found`: The requested resource was not found
- `rate_limit_exceeded`: The rate limit has been exceeded
- `server_error`: An internal server error occurred

## Examples

### cURL

```bash
# Health check
curl -X GET http://localhost:8000/api/health

# Make prediction (legacy)
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'

# Make prediction (authenticated)
curl -X POST http://localhost:8000/api/predict/v2 \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'

# Get model info
curl -X GET http://localhost:8000/api/model/info \
  -H "X-API-Key: your_api_key"
```

### Python

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000/api"

# API key
api_key = "your_api_key"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Make prediction (legacy)
payload = {
    "symbol": "AAPL",
    "days": 7
}
response = requests.post(f"{base_url}/predict", json=payload)
print(response.json())

# Make prediction (authenticated)
headers = {
    "X-API-Key": api_key
}
payload = {
    "symbol": "AAPL",
    "days": 7
}
response = requests.post(f"{base_url}/predict/v2", headers=headers, json=payload)
print(response.json())

# Get model info
response = requests.get(f"{base_url}/model/info", headers=headers)
print(response.json())
```

## Versioning

The API is versioned using endpoint paths. For example, `/api/predict` is the legacy endpoint, while `/api/predict/v2` is the newer, authenticated version.

## Rate Limiting Strategies

The API supports three rate limiting strategies:

1. **Fixed Window**: Limits requests based on a fixed time window (e.g., 60 requests per minute)
2. **Sliding Window**: Limits requests based on a sliding time window, providing smoother rate limiting
3. **Token Bucket**: Uses a token bucket algorithm, allowing for bursts of traffic while maintaining a long-term rate limit

The current strategy is configured in the `.env` file using the `RATE_LIMIT_STRATEGY` variable.

## Caching Backends

The API supports two caching backends:

1. **Memory**: In-memory cache (default)
2. **Redis**: Redis-based cache for distributed deployments

The current backend is configured in the `.env` file using the `CACHE_BACKEND` variable.

## Environment Variables

The API can be configured using the following environment variables:

### API Settings
- `API_TITLE`: API title
- `API_DESCRIPTION`: API description
- `API_VERSION`: API version
- `API_PREFIX`: API prefix
- `API_DEBUG`: Enable debug mode

### Security Settings
- `SECURITY_API_KEY_HEADER`: Header name for API key authentication
- `SECURITY_API_KEYS`: Comma-separated list of valid API keys
- `SECURITY_JWT_SECRET_KEY`: Secret key for JWT token generation
- `SECURITY_JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: JWT token expiration time in minutes
- `SECURITY_CORS_ORIGINS`: Comma-separated list of allowed origins for CORS
- `SECURITY_CORS_ALLOW_CREDENTIALS`: Allow credentials for CORS
- `SECURITY_CORS_ALLOW_METHODS`: Comma-separated list of allowed methods for CORS
- `SECURITY_CORS_ALLOW_HEADERS`: Comma-separated list of allowed headers for CORS

### Rate Limit Settings
- `RATE_LIMIT_ENABLED`: Enable rate limiting
- `RATE_LIMIT_PER_MINUTE`: Maximum number of requests per minute
- `RATE_LIMIT_STRATEGY`: Rate limiting strategy (fixed-window, sliding-window, token-bucket)
- `RATE_LIMIT_REDIS_URL`: Redis URL for distributed rate limiting

### Cache Settings
- `CACHE_ENABLED`: Enable caching
- `CACHE_TTL_SECONDS`: Cache TTL in seconds
- `CACHE_REDIS_URL`: Redis URL for distributed caching
- `CACHE_BACKEND`: Cache backend (memory, redis)
- `CACHE_MAX_SIZE`: Maximum number of items in the memory cache

### Logging Settings
- `LOGGING_LEVEL`: Logging level
- `LOGGING_FORMAT`: Logging format
- `LOGGING_FILE`: Log file path

### Model Settings
- `MODEL_PATH`: Path to the model file
- `FALLBACK_MODEL`: Enable fallback model

## Conclusion

This API provides a robust interface for the AI-Quant-Analyst platform, with features including authentication, rate limiting, caching, and comprehensive error handling. For any questions or issues, please contact the system administrator.
