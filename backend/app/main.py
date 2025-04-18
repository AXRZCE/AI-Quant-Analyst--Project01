from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PredictionRequest, PredictionResponse
from .predict import run_prediction

app = FastAPI(title="Project01 API")

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Run prediction for a given stock symbol"""
    try:
        return run_prediction(payload.symbol, payload.days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
