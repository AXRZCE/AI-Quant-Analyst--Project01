"""
Save the baseline XGBoost model to BentoML.
"""
import os
import logging
import argparse
import bentoml
from joblib import load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model_to_bentoml(model_path: str, model_name: str = "baseline_xgb") -> str:
    """
    Save a model to BentoML.
    
    Args:
        model_path: Path to the model file
        model_name: Name of the model in BentoML
        
    Returns:
        Tag of the saved model
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    logger.info(f"Loading model from {model_path}")
    model = load(model_path)
    
    # Save the model to BentoML
    logger.info(f"Saving model to BentoML as {model_name}")
    model_tag = bentoml.sklearn.save_model(
        model_name,
        model,
        signatures={
            "predict": {"batchable": True}
        },
        metadata={
            "source": model_path,
            "description": "Baseline XGBoost model for trading"
        }
    )
    
    logger.info(f"Model saved to BentoML with tag: {model_tag}")
    
    return str(model_tag)

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Save a model to BentoML")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model-name", type=str, default="baseline_xgb", help="Name of the model in BentoML")
    
    args = parser.parse_args()
    
    try:
        model_tag = save_model_to_bentoml(args.model_path, args.model_name)
        print(f"Model saved to BentoML with tag: {model_tag}")
    except Exception as e:
        logger.error(f"Error saving model to BentoML: {e}")
        raise

if __name__ == "__main__":
    main()
