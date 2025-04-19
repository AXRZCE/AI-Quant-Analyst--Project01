# Real-Time Data Usage in Project01

This document explains how Project01 uses real-time data from APIs instead of sample/mock data for its quantitative models.

## Data Sources

Project01 uses multiple real-time data sources with fallback mechanisms to ensure reliable data availability:

### 1. Market Data

**Primary Source**: Polygon.io API
- Provides real-time and historical market data
- Used for fetching OHLCV (Open, High, Low, Close, Volume) data
- Requires API key in `.env` file: `POLYGON_API_KEY`

**Fallback Source**: Yahoo Finance API
- Used when Polygon.io is unavailable or returns no data
- Provides free access to market data (with some limitations)
- No API key required
- Implemented in `src/ingest/yahoo_client.py`

### 2. News Data

**Primary Source**: NewsAPI
- Provides news articles from various sources
- Used for sentiment analysis
- Requires API key in `.env` file: `NEWS_API_KEY`

**Fallback Mechanism**:
- If no news is found for a ticker symbol, the system tries:
  1. Broader search terms (e.g., "AAPL stock", "market AAPL")
  2. Company name instead of ticker (e.g., "Apple" instead of "AAPL")
  3. Extended time range (from 48 hours to 5 days)

### 3. Social Media Data

**Primary Source**: Twitter API
- Provides social media sentiment data
- Used for social sentiment analysis
- Requires Bearer Token in `.env` file: `TWITTER_BEARER_TOKEN`

## Fallback Mechanisms

The application implements multiple fallback mechanisms to ensure it uses real data whenever possible:

1. **API Fallback Chain**: If one API fails, the application tries alternative APIs
2. **Extended Time Ranges**: If recent data is unavailable, the application tries longer time ranges
3. **Alternative Search Terms**: For news, the application tries different search terms
4. **Caching**: Data is cached to reduce API calls and provide data even when APIs are temporarily unavailable

## Synthetic Data Generation

As a last resort, if all attempts to fetch real data fail, the application will generate synthetic data. This is clearly logged with warning messages:

```
ERROR: All attempts to fetch real data for {symbol} failed, generating synthetic data
ERROR: WARNING: Using synthetic data for predictions - results will not be accurate!
```

The synthetic data generation is designed to be somewhat realistic, using:
- Realistic price movements with appropriate volatility
- Proper time series characteristics
- Reasonable price ranges

However, predictions based on synthetic data should not be used for actual trading decisions.

## Model Training

Models are trained using real data from the APIs:

1. **Initial Training**: The `src/models/train_model.py` script fetches real data for multiple symbols and trains a model
2. **On-demand Training**: If no model is found, the application will train a new model using real data
3. **Feature Engineering**: Features are computed from real market data (moving averages, RSI, etc.)

## Sentiment Analysis

Sentiment analysis is performed using:

1. **FinBERT**: A BERT model fine-tuned for financial sentiment analysis
2. **Real News Data**: Fetched from NewsAPI with multiple fallback mechanisms
3. **Fallback Sentiment**: Only if all attempts to get real news fail, a neutral sentiment is used

## Ensuring Real Data Usage

To ensure your application is using real data:

1. **Check Logs**: Look for messages indicating successful API calls
2. **Verify API Keys**: Ensure all API keys in `.env` are valid
3. **Monitor Data Quality**: Check that predictions are based on real market movements
4. **Check Cache Directory**: Examine cached data files to verify real data is being stored

## Troubleshooting

If you suspect the application is using synthetic data:

1. **Check API Keys**: Verify all API keys are valid and not expired
2. **Check API Quotas**: Ensure you haven't exceeded API rate limits
3. **Examine Logs**: Look for error messages related to API calls
4. **Test API Directly**: Use tools like curl or Postman to test API endpoints directly
5. **Clear Cache**: Remove cached data to force fresh API calls
