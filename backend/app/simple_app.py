from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    symbol: str
    days: int

app = FastAPI(title="Project01 API")

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "Welcome to Project01 API"}

@app.get("/api")
def api_root():
    """API root endpoint"""
    return {"message": "Project01 API is running"}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/predict")
def predict(payload: PredictionRequest):
    """Prediction endpoint"""
    return {
        "timestamps": ["2025-04-01", "2025-04-02", "2025-04-03"],
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
