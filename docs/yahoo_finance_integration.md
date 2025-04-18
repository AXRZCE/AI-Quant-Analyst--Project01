# Yahoo Finance Integration

This document describes the Yahoo Finance integration in Project01, which serves as a backup/alternative to the Polygon.io API for market data.

## Overview

The Yahoo Finance integration uses the `yfinance` Python package to fetch market data, fundamentals, dividends, earnings, and analyst recommendations. It complements the Polygon.io API and provides a fallback mechanism in case Polygon.io is unavailable or rate-limited.

## Features

- **Historical Price Data**: Fetch OHLCV (Open, High, Low, Close, Volume) data for any ticker
- **Fundamental Data**: Company information, financial ratios, and key metrics
- **Dividends**: Historical dividend payments
- **Earnings**: Earnings history and surprises
- **Analyst Recommendations**: Latest analyst recommendations
- **Caching**: Local caching to reduce API calls and improve performance

## Usage

### Basic Usage

```python
from datetime import datetime, timedelta
from src.ingest.yahoo_client import fetch_ticks, fetch_fundamentals

# Fetch historical price data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
df = fetch_ticks("AAPL", start_date, end_date)

# Fetch fundamental data
fundamentals = fetch_fundamentals("AAPL")
```

### Advanced Usage with Caching

```python
from src.ingest.yahoo_client import YahooFinanceClient

# Create client with custom cache settings
client = YahooFinanceClient(use_cache=True, cache_expiry=48)  # 48 hours cache expiry

# Fetch data (will be cached)
df = client.fetch_ticks("MSFT", start_date, end_date)

# Fetch additional data types
dividends = client.fetch_dividends("MSFT", start_date, end_date)
earnings = client.fetch_earnings("MSFT")
recommendations = client.fetch_recommendations("MSFT")
```

## Fallback Mechanism

The application is configured to use Polygon.io as the primary data source and Yahoo Finance as a fallback. The fallback mechanism works as follows:

1. The application attempts to fetch data from Polygon.io
2. If Polygon.io returns no data or fails, the application automatically tries Yahoo Finance
3. If both APIs fail, the application generates synthetic data for testing purposes

This fallback mechanism is implemented in `backend/app/predict.py`.

## Data Format Differences

While we've tried to normalize the data formats between Polygon.io and Yahoo Finance, there are some differences to be aware of:

| Feature | Polygon.io | Yahoo Finance |
|---------|------------|--------------|
| Price Granularity | Second, minute, hour, day | Minute, hour, day, week, month |
| Extended Hours | Separate API calls | Included in some data |
| Real-time Data | Available (paid tier) | 15-min delayed |
| Historical Range | Extensive | Limited for intraday data |
| Rate Limits | Strict, tier-based | More lenient |

## Caching

The Yahoo Finance client includes a caching mechanism to reduce API calls and improve performance:

- Cache files are stored in `data/cache/yahoo/` by default
- Cache expiry is set to 24 hours by default but can be configured
- Each data type (ticks, fundamentals, etc.) is cached separately
- Cache files use the Parquet format for efficient storage

## Error Handling

The client includes robust error handling:

- All API calls are wrapped in try/except blocks
- Errors are logged with appropriate context
- Empty DataFrames with the correct columns are returned on error
- Cache read/write errors are handled gracefully

## Limitations

- Yahoo Finance data is not official and may contain inaccuracies
- The API is unofficial and may change without notice
- Intraday data (1m, 5m, etc.) is limited to recent dates
- Some data may be delayed or incomplete

## Future Improvements

- Add support for options data
- Implement parallel fetching for multiple symbols
- Add support for international markets
- Improve error recovery mechanisms
- Add more advanced caching strategies
