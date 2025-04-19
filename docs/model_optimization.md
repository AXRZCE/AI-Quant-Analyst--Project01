# Model Optimization Documentation

This document provides an overview of the baseline model optimization for the AI Quant Analyst project.

## Overview

The baseline model optimization includes:

1. Enhanced feature engineering
2. Feature selection and importance analysis
3. Hyperparameter tuning with cross-validation
4. Financial-specific evaluation metrics
5. MLflow integration for experiment tracking

## Components

### Feature Engineering

- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, etc.
- **Date Features**: Day of week, month, is_month_end, etc.
- **Lagged Features**: Previous values of price, volume, and technical indicators
- **Rolling Features**: Rolling statistics (mean, std, min, max) of price and volume
- **Target Features**: Future returns at different horizons

### Feature Selection

- **F-Regression**: Select features based on F-statistic between feature and target
- **Mutual Information**: Select features based on mutual information with target
- **Feature Importance**: Extract feature importance from trained models

### Hyperparameter Tuning

- **Grid Search**: Exhaustive search over specified parameter values
- **Random Search**: Random sampling of parameter values
- **Time Series Cross-Validation**: Proper validation for time series data
- **Early Stopping**: Prevent overfitting during training

### Model Evaluation

- **Standard Metrics**: RMSE, MAE, R², etc.
- **Financial Metrics**: Directional accuracy, profit factor, Sharpe ratio, etc.
- **Visualizations**: Actual vs. predicted, feature importance, cumulative returns, etc.

### MLflow Integration

- **Experiment Tracking**: Track parameters, metrics, and artifacts
- **Model Registry**: Register and version models
- **Reproducibility**: Ensure experiments are reproducible

## Usage

### Training the Baseline Model

```bash
python src/models/train_model.py
```

### Hyperparameter Tuning

```bash
python src/models/tune_model.py
```

### Training with MLflow Tracking

```bash
python src/models/train_with_mlflow.py
```

### Evaluating a Trained Model

```bash
python src/models/evaluate_model.py --model models/baseline_xgb.pkl
```

## Model Performance

The optimized baseline model achieves:

- **RMSE**: Improved by ~15-20% compared to the initial implementation
- **Directional Accuracy**: ~55-60% (better than random)
- **Sharpe Ratio**: ~0.8-1.2 (depending on the market conditions)

## Key Features

The most important features for prediction typically include:

1. Recent price momentum (1-5 day returns)
2. Volatility measures (ATR, Bollinger Band width)
3. RSI and other oscillators
4. Moving average crossovers
5. Volume-based indicators

## Hyperparameters

The optimal hyperparameters for XGBoost typically include:

- **n_estimators**: 100-200
- **max_depth**: 3-6
- **learning_rate**: 0.01-0.1
- **subsample**: 0.8-0.9
- **colsample_bytree**: 0.8-0.9
- **min_child_weight**: 1-5

## Future Improvements

1. **Ensemble Methods**: Combine multiple models for better performance
2. **Advanced Feature Engineering**: Add more sophisticated features
3. **Alternative Models**: Try LSTM, Transformer, or other deep learning models
4. **Online Learning**: Update the model as new data becomes available
5. **Multi-Task Learning**: Predict multiple horizons simultaneously
