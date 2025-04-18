@echo off
SETLOCAL EnableDelayedExpansion

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker is not installed. Please install Docker and try again.
    exit /b 1
)

if exist .env (
  for /f "usebackq tokens=* delims=" %%A in (".env") do (
    set "%%A"
  )
) else (
  echo ⚠️  .env file not found—creating example file
  (
    echo # API Keys
    echo POLYGON_API_KEY=your_polygon_api_key
    echo NEWS_API_KEY=your_news_api_key
    echo TWITTER_BEARER_TOKEN=your_twitter_bearer_token
    echo.
    echo # Kafka Configuration
    echo KAFKA_BOOTSTRAP_SERVERS=kafka:9092
  ) > .env
  echo .env file created. Please edit it to add your API keys.
  exit /b 1
)

echo 🛑 Stopping any existing services...
docker-compose -f infra\docker-compose.yml down

echo 📦 Building all services...
docker-compose -f infra\docker-compose.yml build

echo ▶️  Starting Kafka, Backend, and Frontend...
docker-compose -f infra\docker-compose.yml up
