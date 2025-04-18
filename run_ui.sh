#!/usr/bin/env bash
set -e

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Load environment vars
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "⚠️  .env file not found—creating example file"
  cat > .env << EOL
# API Keys
POLYGON_API_KEY=your_polygon_api_key
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
EOL
  echo ".env file created. Please edit it to add your API keys."
  exit 1
fi

echo "🛑 Stopping any existing services..."
docker-compose -f infra/docker-compose.yml down

echo "📦 Building all services..."
docker-compose -f infra/docker-compose.yml build

echo "▶️  Starting Kafka, Backend, and Frontend..."
docker-compose -f infra/docker-compose.yml up
