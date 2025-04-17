# Data Ingestion Components

This directory contains the data ingestion components for the AI-Quant-Analyst project.

## Setup

1. **Environment Variables**

Create a `.env` file in the project root with the following variables:

```
KAFKA_BOOTSTRAP=localhost:9092
RAW_DATA_DIR=data/raw
BATCH_SIZE=500

# API Keys
POLYGON_API_KEY=your_polygon_api_key
NEWS_API_KEY=your_newsapi_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Start Kafka**

```bash
docker compose -f infra/kafka-docker-compose.yml up -d
```

## Usage

### 1. Polygon Client

The Polygon client fetches market data and publishes it to the 'ticks' Kafka topic.

```bash
python src/ingest/polygon_client.py
```

### 2. NewsAPI Client

The NewsAPI client fetches financial news and publishes it to the 'news' Kafka topic.

```bash
python src/ingest/news_api_client.py
```

### 3. Twitter Streamer

The Twitter streamer streams tweets related to finance and publishes them to the 'tweets' Kafka topic.

```bash
python src/ingest/tweet_streamer.py
```

### 4. Kafka to Parquet Converter

This script consumes messages from Kafka topics and saves them as Parquet files.

```bash
python src/ingest/kafka_to_parquet.py
```

## Data Flow

1. Data is fetched from external APIs (Polygon, NewsAPI, Twitter)
2. Data is published to Kafka topics ('ticks', 'news', 'tweets')
3. Data is consumed from Kafka and saved as Parquet files in the 'data/raw/{topic}' directories

## Directory Structure

```
data/
  raw/
    ticks/      # Parquet files with market data
    news/       # Parquet files with news articles
    tweets/     # Parquet files with tweets

src/
  ingest/
    polygon_client.py      # Polygon API client
    news_api_client.py     # NewsAPI client
    tweet_streamer.py      # Twitter streaming client
    kafka_to_parquet.py    # Kafka to Parquet converter
```

## Next Steps

After ingesting the data, you can proceed to:

1. Feature engineering
2. Model training
3. Backtesting
4. Deployment
