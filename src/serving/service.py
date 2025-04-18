"""
BentoML service for serving the baseline XGBoost model.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import bentoml
from bentoml.io import JSON
from joblib import load
from prometheus_client import Counter, Summary, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total prediction requests")
REQUEST_LATENCY = Summary("request_latency_seconds", "Request latency in seconds")

# Load the model
try:
    # Try to load from BentoML model store
    MODEL = bentoml.sklearn.get("baseline_xgb:latest").to_runner()
    logger.info("Loaded model from BentoML model store")
except Exception as e:
    logger.warning(f"Failed to load model from BentoML model store: {e}")
    # Load from file
    model_path = os.environ.get("MODEL_PATH", "models/baseline_xgb.pkl")
    if os.path.exists(model_path):
        MODEL = load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

# Create BentoML service
svc = bentoml.Service("baseline_xgb_service", runners=[MODEL])

@svc.api(input=JSON(), output=JSON())
@REQUEST_LATENCY.time()
def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions using the baseline XGBoost model.
    
    Args:
        input_data: Dictionary with the following structure:
            {
                "symbol": str,
                "features": {
                    "feature1": value1,
                    "feature2": value2,
                    ...
                }
            }
            
    Returns:
        Dictionary with predictions:
            {
                "symbol": str,
                "predictions": [float, ...],
                "timestamp": str
            }
    """
    # Increment request counter
    REQUEST_COUNT.inc()
    
    try:
        # Extract symbol and features
        symbol = input_data.get("symbol", "UNKNOWN")
        features = input_data.get("features", {})
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Make predictions
        preds = MODEL.predict.run(df).tolist()
        
        # Return predictions
        return {
            "symbol": symbol,
            "predictions": preds,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return {
            "symbol": input_data.get("symbol", "UNKNOWN"),
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@svc.api(input=JSON(), output=JSON())
def batch_predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make batch predictions using the baseline XGBoost model.
    
    Args:
        input_data: Dictionary with the following structure:
            {
                "symbols": [str, ...],
                "features": [
                    {
                        "feature1": value1,
                        "feature2": value2,
                        ...
                    },
                    ...
                ]
            }
            
    Returns:
        Dictionary with predictions:
            {
                "symbols": [str, ...],
                "predictions": [float, ...],
                "timestamp": str
            }
    """
    # Increment request counter
    REQUEST_COUNT.inc()
    
    try:
        # Extract symbols and features
        symbols = input_data.get("symbols", ["UNKNOWN"] * len(input_data.get("features", [])))
        features = input_data.get("features", [])
        
        # Convert features to DataFrame
        df = pd.DataFrame(features)
        
        # Make predictions
        preds = MODEL.predict.run(df).tolist()
        
        # Return predictions
        return {
            "symbols": symbols,
            "predictions": preds,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        return {
            "symbols": input_data.get("symbols", ["UNKNOWN"]),
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@svc.api(input=JSON(), output=JSON())
def healthcheck() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status:
            {
                "status": "ok",
                "timestamp": str
            }
    """
    return {
        "status": "ok",
        "timestamp": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Run BentoML service
    svc.run()
