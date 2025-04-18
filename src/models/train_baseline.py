"""
Train a baseline XGBoost model for predicting stock returns.
"""
import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple

# Import local modules
from data_loader import load_training_data, prepare_features, get_feature_label_split

# Try to import wandb, but don't fail if it's not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    use_wandb: bool = False
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters
        use_wandb: Whether to log to Weights & Biases

    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training XGBoost model")

    # Default parameters
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }

    # Initialize model
    model = xgb.XGBRegressor(**params)

    # Prepare evaluation set
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )

    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators

    # Evaluate model
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    metrics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "best_iteration": best_iteration
    }

    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)

        metrics.update({
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2
        })

    # Log metrics
    logger.info(f"Training RMSE: {train_rmse:.5f}")
    logger.info(f"Training MAE: {train_mae:.5f}")
    logger.info(f"Training R²: {train_r2:.5f}")

    if "val_rmse" in metrics:
        logger.info(f"Validation RMSE: {metrics['val_rmse']:.5f}")
        logger.info(f"Validation MAE: {metrics['val_mae']:.5f}")
        logger.info(f"Validation R²: {metrics['val_r2']:.5f}")

    # Log to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(metrics)

        # Log feature importance
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values("Importance", ascending=False)

        wandb.log({
            "feature_importance": wandb.Table(dataframe=importance_df)
        })

    return model, metrics

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        use_wandb: Whether to log to Weights & Biases

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model on test data")

    # Make predictions
    preds = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Calculate directional accuracy
    direction_actual = (y_test > 0).astype(int)
    direction_pred = (preds > 0).astype(int)
    directional_accuracy = (direction_actual == direction_pred).mean()

    # Calculate Sharpe ratio (assuming daily returns)
    pred_returns = preds
    sharpe_ratio = pred_returns.mean() / pred_returns.std() * np.sqrt(252)  # Annualized

    metrics = {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "directional_accuracy": directional_accuracy,
        "sharpe_ratio": sharpe_ratio
    }

    # Log metrics
    logger.info(f"Test RMSE: {rmse:.5f}")
    logger.info(f"Test MAE: {mae:.5f}")
    logger.info(f"Test R²: {r2:.5f}")
    logger.info(f"Directional Accuracy: {directional_accuracy:.5f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.5f}")

    # Log to wandb if available
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(metrics)

        # Create scatter plot of actual vs predicted
        if len(y_test) <= 1000:  # Limit to 1000 points to avoid overloading
            results_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": preds
            })
            wandb.log({
                "actual_vs_predicted": wandb.Table(dataframe=results_df)
            })

    return metrics

def save_model(
    model: xgb.XGBRegressor,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save the trained model and metadata.

    Args:
        model: Trained model
        output_path: Path to save the model
        metadata: Additional metadata to save with the model
    """
    logger.info(f"Saving model to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare model package
    model_package = {
        "model": model,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }

    # Save model package
    joblib.dump(model_package, output_path)

    logger.info(f"Model saved successfully")

def main(args):
    """
    Main training function.

    Args:
        args: Command-line arguments
    """
    logger.info("Starting baseline model training")

    # Initialize wandb if available
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        wandb.config.update(vars(args))

    # Load data
    if args.data_path:
        # Load from Parquet file
        logger.info(f"Loading data from {args.data_path}")
        df = pd.read_parquet(args.data_path)
    else:
        # Load from feature store or raw data
        logger.info(f"Loading data for symbols: {args.symbols}")
        df = load_training_data(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols.split(","),
            use_feast=args.use_feast,
            feature_repo_path=args.feature_repo_path,
            label_horizon=args.label_horizon,
            label_type=args.label_type
        )

    logger.info(f"Loaded {len(df)} records")

    # Split data into train, validation, and test sets
    train_ratio = 1.0 - args.test_size - args.val_size

    if "timestamp" in df.columns:
        # Time-based split
        df = df.sort_values("timestamp")
        train_end_idx = int(len(df) * train_ratio)
        val_end_idx = int(len(df) * (train_ratio + args.val_size))

        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:val_end_idx]
        test_df = df.iloc[val_end_idx:]
    else:
        # Random split
        train_df, temp_df = train_test_split(df, test_size=args.test_size + args.val_size, random_state=args.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=args.test_size / (args.test_size + args.val_size), random_state=args.random_state)

    logger.info(f"Train set: {len(train_df)} records")
    logger.info(f"Validation set: {len(val_df)} records")
    logger.info(f"Test set: {len(test_df)} records")

    # Prepare features and labels
    train_data = get_feature_label_split(train_df, label_column="label")
    val_data = get_feature_label_split(val_df, label_column="label")
    test_data = get_feature_label_split(test_df, label_column="label")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    logger.info(f"Features: {X_train.columns.tolist()}")

    # Set up XGBoost parameters
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": args.random_state
    }

    # Train model
    model, train_metrics = train_xgboost_model(
        X_train, y_train,
        X_val, y_val,
        params=params,
        use_wandb=args.use_wandb
    )

    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, use_wandb=args.use_wandb)

    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}

    # Save model
    metadata = {
        "features": X_train.columns.tolist(),
        "metrics": all_metrics,
        "params": params,
        "data_info": {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "symbols": df["symbol"].unique().tolist() if "symbol" in df.columns else None,
            "date_range": [df["timestamp"].min(), df["timestamp"].max()] if "timestamp" in df.columns else None
        }
    }

    save_model(model, args.output_path, metadata)

    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("Training completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline XGBoost model")

    # Data loading arguments
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument("--data-path", type=str, help="Path to Parquet file or directory")
    data_group.add_argument("--start-date", type=str, default="2025-04-01", help="Start date (YYYY-MM-DD)")
    data_group.add_argument("--end-date", type=str, default="2025-04-17", help="End date (YYYY-MM-DD)")
    data_group.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols")
    data_group.add_argument("--use-feast", action="store_true", help="Use Feast feature store")
    data_group.add_argument("--feature-repo-path", type=str, default="infra/feast/feature_repo", help="Path to Feast feature repository")
    data_group.add_argument("--label-horizon", type=int, default=1, help="Number of periods ahead for return calculation")
    data_group.add_argument("--label-type", type=str, default="return", choices=["return", "direction"], help="Type of label to generate")

    # Model arguments
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--n-estimators", type=int, default=200, help="Number of boosting rounds")
    model_group.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth")
    model_group.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    model_group.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    model_group.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample ratio")

    # Training arguments
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    train_group.add_argument("--val-size", type=float, default=0.1, help="Validation set size")
    train_group.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-path", type=str, default="models/baseline_xgb.pkl", help="Path to save the model")
    output_group.add_argument("--use-wandb", action="store_true", help="Log metrics to Weights & Biases")
    output_group.add_argument("--wandb-project", type=str, default="Project01-baseline", help="Weights & Biases project name")
    output_group.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")

    args = parser.parse_args()
    main(args)
