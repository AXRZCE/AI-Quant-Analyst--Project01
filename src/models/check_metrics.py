"""
Check model metrics against thresholds for CI/CD pipeline.

This script checks if model metrics meet the required thresholds for deployment.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# Define metric thresholds
METRIC_THRESHOLDS = {
    'xgboost': {
        'accuracy': 0.65,
        'precision': 0.65,
        'recall': 0.60,
        'f1': 0.65,
        'mse': 0.0050,
        'mae': 0.0180,
        'r2': 0.55,
        'sharpe': 1.50
    },
    'lstm': {
        'accuracy': 0.60,
        'precision': 0.65,
        'recall': 0.55,
        'f1': 0.60,
        'mse': 0.0055,
        'mae': 0.0185,
        'r2': 0.50,
        'sharpe': 1.40
    },
    'tft': {
        'accuracy': 0.65,
        'precision': 0.70,
        'recall': 0.60,
        'f1': 0.65,
        'mse': 0.0045,
        'mae': 0.0170,
        'r2': 0.55,
        'sharpe': 1.60
    },
    'ensemble': {
        'accuracy': 0.68,
        'precision': 0.72,
        'recall': 0.65,
        'f1': 0.68,
        'mse': 0.0042,
        'mae': 0.0165,
        'r2': 0.60,
        'sharpe': 1.70
    }
}


def load_metrics(metrics_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from a JSON file.
    
    Args:
        metrics_path: Path to metrics file
        
    Returns:
        Dictionary of model metrics
    """
    logger.info(f"Loading metrics from {metrics_path}")
    
    # Check if file exists
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def check_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Check if metrics meet thresholds.
    
    Args:
        metrics: Dictionary of model metrics
        
    Returns:
        Dictionary of check results
    """
    logger.info("Checking metrics against thresholds")
    
    results = {}
    
    for model_name, model_metrics in metrics.items():
        model_results = {
            'passed': True,
            'metrics': {}
        }
        
        # Get thresholds for model
        thresholds = METRIC_THRESHOLDS.get(model_name, {})
        
        # Check each metric
        for metric_name, metric_value in model_metrics.items():
            threshold = thresholds.get(metric_name)
            
            if threshold is None:
                # Skip metrics without thresholds
                continue
            
            # Check if metric meets threshold
            if metric_name in ['mse', 'mae']:
                # Lower is better for these metrics
                passed = metric_value <= threshold
            else:
                # Higher is better for these metrics
                passed = metric_value >= threshold
            
            # Store result
            model_results['metrics'][metric_name] = {
                'value': metric_value,
                'threshold': threshold,
                'passed': passed
            }
            
            # Update overall result
            if not passed:
                model_results['passed'] = False
        
        # Store model results
        results[model_name] = model_results
    
    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print check results.
    
    Args:
        results: Dictionary of check results
    """
    logger.info("Check results:")
    
    for model_name, model_results in results.items():
        passed = model_results['passed']
        status = "PASSED" if passed else "FAILED"
        
        logger.info(f"{model_name}: {status}")
        
        for metric_name, metric_result in model_results['metrics'].items():
            value = metric_result['value']
            threshold = metric_result['threshold']
            passed = metric_result['passed']
            
            status = "PASSED" if passed else "FAILED"
            
            if metric_name in ['mse', 'mae']:
                # Lower is better
                comparison = "<="
            else:
                # Higher is better
                comparison = ">="
            
            logger.info(f"  {metric_name}: {value:.4f} {comparison} {threshold:.4f} - {status}")


def main():
    """Main function."""
    try:
        # Load metrics
        metrics = load_metrics('models/metrics.json')
        
        # Check metrics
        results = check_metrics(metrics)
        
        # Print results
        print_results(results)
        
        # Check if all models passed
        all_passed = all(model_results['passed'] for model_results in results.values())
        
        if all_passed:
            logger.info("All models passed checks")
            return 0
        else:
            logger.error("Some models failed checks")
            return 1
    
    except Exception as e:
        logger.error(f"Error checking metrics: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
