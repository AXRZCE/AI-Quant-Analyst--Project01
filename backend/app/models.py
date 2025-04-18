from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    days: int = Field(1, description="Number of days of historical data to use", ge=1, le=30)

class PredictionResponse(BaseModel):
    timestamps: List[str] = Field(..., description="Timestamps for the historical data")
    prices: List[float] = Field(..., description="Historical closing prices")
    ma_5: List[float] = Field(..., description="5-day moving average")
    rsi_14: List[float] = Field(..., description="14-day relative strength index")
    sentiment: Dict[str, float] = Field(..., description="Sentiment analysis results")
    prediction: float = Field(..., description="Predicted return")
