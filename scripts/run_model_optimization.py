"""
Script to run the entire model optimization pipeline.

This script runs the following steps:
1. Train the baseline model
2. Tune hyperparameters
3. Train with MLflow tracking
4. Evaluate the final model
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """
    Run a command and log the output.
    
    Args:
        command: Command to run
        description: Description of the command
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running {description}...")
    
    try:
        # Run the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.strip())
        
        # Wait for the process to complete
        process.wait()
        
        # Check if successful
        if process.returncode == 0:
            logger.info(f"{description} completed successfully")
            return True
        else:
            logger.error(f"{description} failed with return code {process.returncode}")
            return False
        
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return False


def main():
    """Main function to run the model optimization pipeline."""
    start_time = time.time()
    
    # Create directories
    os.makedirs(project_root / "models", exist_ok=True)
    os.makedirs(project_root / "models/tuning", exist_ok=True)
    os.makedirs(project_root / "models/mlflow", exist_ok=True)
    os.makedirs(project_root / "models/reports", exist_ok=True)
    
    # Step 1: Train the baseline model
    baseline_success = run_command(
        f"python {project_root}/src/models/train_model.py",
        "baseline model training"
    )
    
    if not baseline_success:
        logger.error("Baseline model training failed, stopping pipeline")
        sys.exit(1)
    
    # Step 2: Tune hyperparameters
    tuning_success = run_command(
        f"python {project_root}/src/models/tune_model.py",
        "hyperparameter tuning"
    )
    
    if not tuning_success:
        logger.warning("Hyperparameter tuning failed, continuing with default parameters")
    
    # Step 3: Train with MLflow tracking
    mlflow_success = run_command(
        f"python {project_root}/src/models/train_with_mlflow.py",
        "MLflow training"
    )
    
    if not mlflow_success:
        logger.error("MLflow training failed, stopping pipeline")
        sys.exit(1)
    
    # Step 4: Evaluate the final model
    # Find the most recent model file
    model_dir = project_root / "models"
    model_files = list(model_dir.glob("xgboost_model_*.pkl"))
    
    if model_files:
        # Sort by modification time (newest first)
        latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        evaluation_success = run_command(
            f"python {project_root}/src/models/evaluate_model.py --model {latest_model}",
            "model evaluation"
        )
        
        if not evaluation_success:
            logger.error("Model evaluation failed")
    else:
        logger.error("No model files found for evaluation")
    
    # Calculate total time
    elapsed_time = time.time() - start_time
    logger.info(f"Model optimization pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
