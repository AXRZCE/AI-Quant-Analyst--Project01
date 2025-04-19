"""
Advanced models integration module.

This module provides functionality for integrating advanced models like
Temporal Fusion Transformer (TFT) and FinBERT into the prediction pipeline.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TFTWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for Temporal Fusion Transformer model."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_encoder_length: int = 60,
        max_prediction_length: int = 1,
        target_col: str = "target",
        time_idx_col: str = "time_idx",
        group_ids: List[str] = None
    ):
        """
        Initialize the TFT wrapper.
        
        Args:
            model_path: Path to the saved model
            max_encoder_length: Maximum encoder length
            max_prediction_length: Maximum prediction length
            target_col: Target column name
            time_idx_col: Time index column name
            group_ids: Group ID columns
        """
        self.model_path = model_path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.target_col = target_col
        self.time_idx_col = time_idx_col
        self.group_ids = group_ids or ["symbol"]
        self.model = None
        self.training_dataset = None
        
        # Try to load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained TFT model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Import required modules
            from pytorch_forecasting import TemporalFusionTransformer
            
            # Load model
            self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
            logger.info(f"Loaded TFT model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading TFT model: {e}")
            self.model = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        time_idx_col: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None
    ) -> None:
        """
        Prepare data for TFT model.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_idx_col: Time index column name
            group_ids: Group ID columns
            static_categoricals: Static categorical columns
            time_varying_known_reals: Time-varying known real columns
            time_varying_unknown_reals: Time-varying unknown real columns
        """
        try:
            # Import required modules
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            
            # Use provided values or defaults
            target = target_col or self.target_col
            time_idx = time_idx_col or self.time_idx_col
            group_ids = group_ids or self.group_ids
            
            # Infer columns if not provided
            if static_categoricals is None:
                static_categoricals = [col for col in df.columns if col in group_ids]
            
            if time_varying_known_reals is None:
                time_varying_known_reals = [time_idx]
            
            if time_varying_unknown_reals is None:
                time_varying_unknown_reals = [target] + [
                    col for col in df.columns 
                    if col not in static_categoricals + time_varying_known_reals + group_ids + [time_idx]
                ]
            
            # Create dataset
            self.training_dataset = TimeSeriesDataSet(
                data=df,
                time_idx=time_idx,
                target=target,
                group_ids=group_ids,
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=static_categoricals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=group_ids),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            
            logger.info("Prepared data for TFT model")
        except Exception as e:
            logger.error(f"Error preparing data for TFT model: {e}")
            self.training_dataset = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TFTWrapper':
        """
        Fit the TFT model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted model
        """
        try:
            # Import required modules
            import pytorch_lightning as pl
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import QuantileLoss
            from torch.utils.data import DataLoader
            
            # Prepare data
            df = X.copy()
            df[self.target_col] = y
            
            # Ensure time_idx column exists
            if self.time_idx_col not in df.columns:
                if 'timestamp' in df.columns:
                    # Create time index from timestamp
                    df[self.time_idx_col] = (df['timestamp'] - df['timestamp'].min()).dt.days
                else:
                    # Create sequential time index
                    df[self.time_idx_col] = np.arange(len(df))
            
            # Prepare data
            self.prepare_data(df)
            
            if self.training_dataset is None:
                logger.error("Failed to prepare data for TFT model")
                return self
            
            # Create data loader
            train_dataloader = self.training_dataset.to_dataloader(
                batch_size=64,
                shuffle=True,
                num_workers=0
            )
            
            # Create model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_dataset,
                learning_rate=0.03,
                hidden_size=32,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=16,
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=30,
                gradient_clip_val=0.1,
                limit_train_batches=30,
                callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
                enable_model_summary=True,
            )
            
            # Train model
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader
            )
            
            logger.info("Fitted TFT model")
            
            return self
        except Exception as e:
            logger.error(f"Error fitting TFT model: {e}")
            return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the TFT model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            logger.error("TFT model not loaded")
            return np.zeros(len(X))
        
        try:
            # Prepare data
            df = X.copy()
            
            # Add dummy target column if needed
            if self.target_col not in df.columns:
                df[self.target_col] = 0.0
            
            # Ensure time_idx column exists
            if self.time_idx_col not in df.columns:
                if 'timestamp' in df.columns:
                    # Create time index from timestamp
                    df[self.time_idx_col] = (df['timestamp'] - df['timestamp'].min()).dt.days
                else:
                    # Create sequential time index
                    df[self.time_idx_col] = np.arange(len(df))
            
            # Create dataset
            if self.training_dataset is None:
                self.prepare_data(df)
                
                if self.training_dataset is None:
                    logger.error("Failed to prepare data for TFT model")
                    return np.zeros(len(X))
            
            # Create data loader
            dataloader = self.training_dataset.to_dataloader(
                batch_size=64,
                shuffle=False,
                num_workers=0
            )
            
            # Make predictions
            predictions = self.model.predict(dataloader)
            
            # Return median predictions (quantile 0.5)
            return predictions.numpy()
        except Exception as e:
            logger.error(f"Error making predictions with TFT model: {e}")
            return np.zeros(len(X))
    
    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if self.model is None:
            logger.error("TFT model not loaded")
            return np.zeros(len(X)), np.zeros(len(X))
        
        try:
            # Prepare data
            df = X.copy()
            
            # Add dummy target column if needed
            if self.target_col not in df.columns:
                df[self.target_col] = 0.0
            
            # Ensure time_idx column exists
            if self.time_idx_col not in df.columns:
                if 'timestamp' in df.columns:
                    # Create time index from timestamp
                    df[self.time_idx_col] = (df['timestamp'] - df['timestamp'].min()).dt.days
                else:
                    # Create sequential time index
                    df[self.time_idx_col] = np.arange(len(df))
            
            # Create dataset
            if self.training_dataset is None:
                self.prepare_data(df)
                
                if self.training_dataset is None:
                    logger.error("Failed to prepare data for TFT model")
                    return np.zeros(len(X)), np.zeros(len(X))
            
            # Create data loader
            dataloader = self.training_dataset.to_dataloader(
                batch_size=64,
                shuffle=False,
                num_workers=0
            )
            
            # Make predictions for all quantiles
            predictions = self.model.predict(dataloader, return_x=True, return_y=True)
            
            # Get quantiles
            quantiles = self.model.loss.quantiles
            
            # Find closest quantiles to alpha/2 and 1-alpha/2
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            lower_idx = np.argmin(np.abs(np.array(quantiles) - lower_q))
            upper_idx = np.argmin(np.abs(np.array(quantiles) - upper_q))
            
            # Get predictions for these quantiles
            lower_bound = predictions[..., lower_idx].numpy()
            upper_bound = predictions[..., upper_idx].numpy()
            
            return lower_bound, upper_bound
        except Exception as e:
            logger.error(f"Error making interval predictions with TFT model: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the TFT model.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            logger.error("TFT model not loaded")
            return ""
        
        try:
            # Use provided path or default
            if model_path is None:
                model_path = "models/tft_model.ckpt"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            logger.info(f"Saved TFT model to {model_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Error saving TFT model: {e}")
            return ""


class FinBERTWrapper:
    """Wrapper for FinBERT sentiment analysis model."""
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize the FinBERT wrapper.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to use for inference (None for auto-detection)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.id2label = None
        
        # Try to load model
        self.load_model()
    
    def load_model(self) -> None:
        """Load the FinBERT model."""
        try:
            # Import required modules
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer and model
            logger.info(f"Loading FinBERT model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping
            if hasattr(self.model.config, "id2label"):
                self.id2label = self.model.config.id2label
            else:
                # Default FinBERT labels
                self.id2label = {0: "positive", 1: "neutral", 2: "negative"}
            
            logger.info(f"Loaded FinBERT model with labels: {self.id2label}")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        if self.model is None or self.tokenizer is None:
            logger.error("FinBERT model not loaded")
            return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]
        
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Convert to probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert to numpy
                probs = probs.cpu().numpy()
                
                # Convert to dictionaries
                for prob in probs:
                    result = {self.id2label[i]: float(p) for i, p in enumerate(prob)}
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]
    
    def analyze_with_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of texts and include the text in the result.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores and text
        """
        results = self.analyze(texts)
        
        # Add text to results
        for i, result in enumerate(results):
            result["text"] = texts[i]
        
        return results
    
    def get_sentiment_features(
        self,
        texts: List[str],
        include_scores: bool = True,
        include_label: bool = True
    ) -> pd.DataFrame:
        """
        Get sentiment features for texts.
        
        Args:
            texts: List of texts to analyze
            include_scores: Whether to include sentiment scores
            include_label: Whether to include sentiment label
            
        Returns:
            DataFrame with sentiment features
        """
        # Analyze sentiment
        results = self.analyze(texts)
        
        # Create DataFrame
        df = pd.DataFrame()
        
        if include_scores:
            # Add sentiment scores
            for label in self.id2label.values():
                df[f"sentiment_{label}"] = [result.get(label, 0.0) for result in results]
        
        if include_label:
            # Add sentiment label
            df["sentiment_label"] = [
                max(result.items(), key=lambda x: x[1])[0] for result in results
            ]
            
            # Add sentiment score (positive - negative)
            df["sentiment_score"] = [
                result.get("positive", 0.0) - result.get("negative", 0.0) for result in results
            ]
        
        return df


class ModelIntegrator:
    """Integrator for advanced models into the prediction pipeline."""
    
    def __init__(
        self,
        use_tft: bool = True,
        use_finbert: bool = True,
        tft_model_path: Optional[str] = None,
        finbert_model_name: str = "yiyanghkust/finbert-tone"
    ):
        """
        Initialize the model integrator.
        
        Args:
            use_tft: Whether to use TFT model
            use_finbert: Whether to use FinBERT model
            tft_model_path: Path to the TFT model
            finbert_model_name: Name of the FinBERT model
        """
        self.use_tft = use_tft
        self.use_finbert = use_finbert
        self.tft_model_path = tft_model_path
        self.finbert_model_name = finbert_model_name
        
        # Initialize models
        self.tft_model = None
        self.finbert_model = None
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load the models."""
        # Load TFT model
        if self.use_tft:
            try:
                self.tft_model = TFTWrapper(model_path=self.tft_model_path)
                logger.info("Initialized TFT model")
            except Exception as e:
                logger.error(f"Error initializing TFT model: {e}")
                self.tft_model = None
        
        # Load FinBERT model
        if self.use_finbert:
            try:
                self.finbert_model = FinBERTWrapper(model_name=self.finbert_model_name)
                logger.info("Initialized FinBERT model")
            except Exception as e:
                logger.error(f"Error initializing FinBERT model: {e}")
                self.finbert_model = None
    
    def process_data(
        self,
        df: pd.DataFrame,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process data with advanced models.
        
        Args:
            df: DataFrame with data
            text_column: Column with text data for sentiment analysis
            
        Returns:
            DataFrame with additional features
        """
        result_df = df.copy()
        
        # Add TFT features
        if self.use_tft and self.tft_model is not None:
            try:
                logger.info("Adding TFT features")
                
                # Make predictions
                tft_predictions = self.tft_model.predict(df)
                
                # Add to DataFrame
                result_df["tft_prediction"] = tft_predictions
                
                # Add confidence intervals
                lower, upper = self.tft_model.predict_interval(df)
                result_df["tft_lower"] = lower
                result_df["tft_upper"] = upper
                
                logger.info("Added TFT features")
            except Exception as e:
                logger.error(f"Error adding TFT features: {e}")
        
        # Add FinBERT features
        if self.use_finbert and self.finbert_model is not None and text_column is not None:
            try:
                logger.info("Adding FinBERT features")
                
                # Get texts
                texts = df[text_column].fillna("").tolist()
                
                # Get sentiment features
                sentiment_df = self.finbert_model.get_sentiment_features(texts)
                
                # Add to DataFrame
                for col in sentiment_df.columns:
                    result_df[col] = sentiment_df[col].values
                
                logger.info("Added FinBERT features")
            except Exception as e:
                logger.error(f"Error adding FinBERT features: {e}")
        
        return result_df
    
    def get_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Get sentiment for texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores
        """
        if not self.use_finbert or self.finbert_model is None:
            logger.error("FinBERT model not available")
            return [{"positive": 0.33, "neutral": 0.34, "negative": 0.33} for _ in texts]
        
        return self.finbert_model.analyze(texts)
    
    def get_time_series_forecast(
        self,
        df: pd.DataFrame,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Get time series forecast.
        
        Args:
            df: DataFrame with time series data
            horizon: Forecast horizon
            
        Returns:
            DataFrame with forecasts
        """
        if not self.use_tft or self.tft_model is None:
            logger.error("TFT model not available")
            return pd.DataFrame()
        
        try:
            # Set prediction length
            self.tft_model.max_prediction_length = horizon
            
            # Prepare data
            self.tft_model.prepare_data(df)
            
            # Make predictions
            predictions = self.tft_model.predict(df)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame()
            
            # Add predictions
            forecast_df["forecast"] = predictions
            
            # Add confidence intervals
            lower, upper = self.tft_model.predict_interval(df)
            forecast_df["lower"] = lower
            forecast_df["upper"] = upper
            
            return forecast_df
        except Exception as e:
            logger.error(f"Error getting time series forecast: {e}")
            return pd.DataFrame()


def get_model_integrator(
    use_tft: bool = True,
    use_finbert: bool = True,
    tft_model_path: Optional[str] = None,
    finbert_model_name: str = "yiyanghkust/finbert-tone"
) -> ModelIntegrator:
    """
    Get a model integrator.
    
    Args:
        use_tft: Whether to use TFT model
        use_finbert: Whether to use FinBERT model
        tft_model_path: Path to the TFT model
        finbert_model_name: Name of the FinBERT model
        
    Returns:
        ModelIntegrator instance
    """
    return ModelIntegrator(
        use_tft=use_tft,
        use_finbert=use_finbert,
        tft_model_path=tft_model_path,
        finbert_model_name=finbert_model_name
    )
