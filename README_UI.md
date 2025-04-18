# Project01 UI

This folder contains the React + FastAPI UI for your private AI Quant Analyst.

## Prerequisites

- Docker & Docker Compose installed
- A `.env` file in the project root with:
  ```bash
  POLYGON_API_KEY=your_polygon_key
  NEWS_API_KEY=your_newsapi_key
  TWITTER_BEARER_TOKEN=your_twitter_token
  ```
- (Optional) Adjust ports or VITE_API_URL in `infra/docker-compose.yml`

## Running Locally

### Windows

```bat
run_ui.bat
```

### Linux / macOS

```bash
./run_ui.sh
```

Once up:

- **Frontend** → http://localhost:3000
- **Backend API** → http://localhost:8000/api/predict

## Usage

1. Enter a stock symbol (e.g. `AAPL`) and a look‑back in days.
2. Click **Predict**.
3. Watch:
   - Price chart with MA & RSI
   - Sentiment gauges for recent news
   - Next‑return prediction

## Architecture

The UI consists of two main components:

1. **Backend API (FastAPI)**: Provides endpoints for model predictions and data retrieval
2. **Frontend (React + TypeScript)**: User interface for interacting with the models

## Directory Structure

```
PROJECT01/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── models.py           # Pydantic schemas
│   │   ├── predict.py          # Orchestrator (ingest→features→model)
│   │   └── requirements.txt    # Backend‐only deps
│   └── Dockerfile
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx             # Main React component
│   │   ├── index.tsx           # React bootstrap
│   │   ├── api.ts              # Axios client
│   │   └── components/         # SymbolForm, PriceChart, etc.
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── infra/
│   └── docker-compose.yml      # Includes UI services
```

## API Endpoints

### `/api/predict`

Predicts future stock price movement based on historical data and sentiment analysis.

**Request:**

```json
{
  "symbol": "AAPL",
  "days": 7
}
```

**Response:**

```json
{
  "timestamps": ["2023-01-01T00:00:00", ...],
  "prices": [150.0, 151.2, ...],
  "ma_5": [149.5, 150.1, ...],
  "rsi_14": [65.2, 67.8, ...],
  "sentiment": {
    "positive": 0.65,
    "neutral": 0.25,
    "negative": 0.10
  },
  "prediction": 0.023
}
```

## Next Steps

- **Test** with different symbols
- **Customize** styling or components
- **Add Authentication** if you share internally
- **Extend** to other models (TFT, RL) via new endpoints

---

Enjoy your private, full‑stack AI Quant Analyst UI! 🚀
