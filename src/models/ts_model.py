"""
Time-series model using Temporal Fusion Transformer.
"""
import os
import logging
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TFTModel:
    """
    Temporal Fusion Transformer model for time-series forecasting.
    """
    
    def __init__(
        self,
        data_path: str = None,
        max_encoder_length: int = 60,
        max_prediction_length: int = 1,
        batch_size: int = 64,
        max_epochs: int = 30,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        output_size: int = 7,  # quantiles
        log_dir: str = "logs/ts",
        model_dir: str = "models",
        model_name: str = "tft.ckpt"
    ):
        """
        Initialize the TFT model.
        
        Args:
            data_path: Path to the data file
            max_encoder_length: Maximum length of the encoder (lookback window)
            max_prediction_length: Maximum length of the prediction (forecast horizon)
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs for training
            learning_rate: Learning rate for training
            hidden_size: Hidden size of the model
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
            output_size: Number of quantiles to predict
            log_dir: Directory for logs
            model_dir: Directory for saving the model
            model_name: Name of the model file
        """
        self.data_path = data_path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.output_size = output_size
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model_name = model_name
        
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
    
    def prepare_data(
        self,
        df: Optional[pd.DataFrame] = None,
        target: str = "close",
        group_ids: List[str] = ["symbol"],
        static_categoricals: List[str] = ["symbol"],
        time_varying_known_reals: List[str] = ["time_idx"],
        time_varying_unknown_reals: Optional[List[str]] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare the data for training, validation, and testing.
        
        Args:
            df: DataFrame with the data (if None, load from data_path)
            target: Target variable to predict
            group_ids: List of column names identifying a time series
            static_categoricals: List of categorical variables that do not change over time
            time_varying_known_reals: List of continuous variables that are known in the future
            time_varying_unknown_reals: List of continuous variables that are not known in the future
            train_val_test_split: Tuple with train, validation, and test split ratios
            
        Returns:
            Tuple of (training_dataset, validation_dataset, test_dataset)
        """
        logger.info("Preparing data for TFT model")
        
        # Load data if not provided
        if df is None:
            if self.data_path is None:
                raise ValueError("Either df or data_path must be provided")
            
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_parquet(self.data_path)
        
        # Ensure required columns exist
        required_columns = ["time_idx", target] + group_ids
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the data")
        
        # Set default time_varying_unknown_reals if not provided
        if time_varying_unknown_reals is None:
            # Use all numeric columns except time_idx and group_ids
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            time_varying_unknown_reals = [col for col in numeric_cols 
                                         if col not in time_varying_known_reals 
                                         and col not in group_ids 
                                         and col != "time_idx"]
            
            # Always include the target variable
            if target not in time_varying_unknown_reals:
                time_varying_unknown_reals.append(target)
        
        logger.info(f"Using {len(time_varying_unknown_reals)} time-varying unknown reals: {time_varying_unknown_reals}")
        
        # Create training dataset
        training_cutoff = int(len(df) * train_val_test_split[0])
        validation_cutoff = int(len(df) * (train_val_test_split[0] + train_val_test_split[1]))
        
        # Sort by group_ids and time_idx
        df = df.sort_values(by=group_ids + ["time_idx"])
        
        # Create the training dataset
        self.training_dataset = TimeSeriesDataSet(
            data=df.iloc[:training_cutoff],
            time_idx="time_idx",
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
        
        # Create the validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df.iloc[training_cutoff:validation_cutoff], 
            predict=False
        )
        
        # Create the test dataset
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df.iloc[validation_cutoff:], 
            predict=False
        )
        
        logger.info(f"Created datasets with {len(self.training_dataset)} training, "
                   f"{len(self.validation_dataset)} validation, and {len(self.test_dataset)} test samples")
        
        return self.training_dataset, self.validation_dataset, self.test_dataset
    
    def create_dataloaders(
        self,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.
        
        Args:
            train_batch_size: Batch size for training (if None, use self.batch_size)
            val_batch_size: Batch size for validation (if None, use self.batch_size)
            test_batch_size: Batch size for testing (if None, use self.batch_size)
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
        """
        if self.training_dataset is None:
            raise ValueError("Datasets not prepared. Call prepare_data() first.")
        
        # Use default batch size if not provided
        if train_batch_size is None:
            train_batch_size = self.batch_size
        if val_batch_size is None:
            val_batch_size = self.batch_size
        if test_batch_size is None:
            test_batch_size = self.batch_size
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            self.training_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        val_dataloader = DataLoader(
            self.validation_dataset, 
            batch_size=val_batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def build_model(self) -> TemporalFusionTransformer:
        """
        Build the Temporal Fusion Transformer model.
        
        Returns:
            TemporalFusionTransformer model
        """
        if self.training_dataset is None:
            raise ValueError("Datasets not prepared. Call prepare_data() first.")
        
        logger.info("Building TFT model")
        
        # Create the loss function
        loss = QuantileLoss()
        
        # Create the model
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.output_size,
            loss=loss,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        logger.info(f"Created TFT model with {self.model.hparams}")
        
        return self.model
    
    def train(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        limit_train_batches: int = 30,
        gradient_clip_val: float = 0.1
    ) -> pl.Trainer:
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training (if None, create using create_dataloaders())
            val_dataloader: DataLoader for validation (if None, create using create_dataloaders())
            limit_train_batches: Number of batches per epoch (for quick iterations)
            gradient_clip_val: Gradient clipping value
            
        Returns:
            PyTorch Lightning Trainer
        """
        if self.model is None:
            self.build_model()
        
        # Create DataLoaders if not provided
        if train_dataloader is None or val_dataloader is None:
            train_dataloader, val_dataloader, _ = self.create_dataloaders()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create the trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=gradient_clip_val,
            limit_train_batches=limit_train_batches,
            callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
            logger=pl.loggers.CSVLogger(self.log_dir),
        )
        
        logger.info(f"Training TFT model for {self.max_epochs} epochs")
        
        # Train the model
        self.trainer.fit(
            self.model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader
        )
        
        logger.info(f"Finished training after {self.trainer.current_epoch} epochs")
        
        return self.trainer
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model (if None, use self.model_dir/self.model_name)
            
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Use default path if not provided
        if path is None:
            path = os.path.join(self.model_dir, self.model_name)
        
        # Save the model
        self.trainer.save_checkpoint(path)
        
        logger.info(f"Saved model to {path}")
        
        return path
    
    def load_model(self, path: Optional[str] = None) -> TemporalFusionTransformer:
        """
        Load a trained model.
        
        Args:
            path: Path to the model (if None, use self.model_dir/self.model_name)
            
        Returns:
            Loaded model
        """
        # Use default path if not provided
        if path is None:
            path = os.path.join(self.model_dir, self.model_name)
        
        # Check if the model file exists
        if not os.path.exists(path):
            raise ValueError(f"Model file {path} not found")
        
        # Load the model
        self.model = TemporalFusionTransformer.load_from_checkpoint(path)
        
        logger.info(f"Loaded model from {path}")
        
        return self.model
    
    def predict(
        self,
        df: pd.DataFrame,
        return_x: bool = False,
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        mode: str = "prediction"
    ) -> Union[pd.DataFrame, Tuple]:
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame with the data
            return_x: Whether to return the input data
            return_index: Whether to return the index
            return_decoder_lengths: Whether to return the decoder lengths
            mode: Prediction mode ("prediction", "quantiles", or "raw")
            
        Returns:
            Predictions as a DataFrame or tuple
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.training_dataset is None:
            raise ValueError("Datasets not prepared. Call prepare_data() first.")
        
        # Create a dataset from the input data
        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            df, 
            predict=True
        )
        
        # Create a DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Make predictions
        predictions = self.model.predict(
            dataloader,
            return_x=return_x,
            return_index=return_index,
            return_decoder_lengths=return_decoder_lengths,
            mode=mode
        )
        
        return predictions
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation (if None, use test_dataloader)
            return_predictions: Whether to return the predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use test dataloader if not provided
        if dataloader is None:
            _, _, dataloader = self.create_dataloaders()
        
        # Evaluate the model
        results = self.trainer.test(self.model, dataloaders=dataloader)
        
        # Get predictions if requested
        predictions = None
        if return_predictions:
            predictions = self.model.predict(dataloader)
        
        # Return results and predictions
        if return_predictions:
            return results[0], predictions
        else:
            return results[0]

def train_tft(
    data_path: str,
    max_epochs: int = 30,
    max_encoder_length: int = 60,
    max_prediction_length: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.03,
    hidden_size: int = 16,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,
    output_size: int = 7,
    log_dir: str = "logs/ts",
    model_dir: str = "models",
    model_name: str = "tft.ckpt"
) -> TFTModel:
    """
    Train a Temporal Fusion Transformer model.
    
    Args:
        data_path: Path to the data file
        max_epochs: Maximum number of epochs for training
        max_encoder_length: Maximum length of the encoder (lookback window)
        max_prediction_length: Maximum length of the prediction (forecast horizon)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        hidden_size: Hidden size of the model
        attention_head_size: Number of attention heads
        dropout: Dropout rate
        hidden_continuous_size: Hidden size for continuous variables
        output_size: Number of quantiles to predict
        log_dir: Directory for logs
        model_dir: Directory for saving the model
        model_name: Name of the model file
        
    Returns:
        Trained TFTModel
    """
    # Create the model
    model = TFTModel(
        data_path=data_path,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=output_size,
        log_dir=log_dir,
        model_dir=model_dir,
        model_name=model_name
    )
    
    # Prepare the data
    model.prepare_data()
    
    # Build the model
    model.build_model()
    
    # Train the model
    model.train()
    
    # Save the model
    model.save_model()
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Temporal Fusion Transformer model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum number of epochs for training")
    parser.add_argument("--max-encoder-length", type=int, default=60, help="Maximum length of the encoder (lookback window)")
    parser.add_argument("--max-prediction-length", type=int, default=1, help="Maximum length of the prediction (forecast horizon)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate for training")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden size of the model")
    parser.add_argument("--attention-head-size", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--hidden-continuous-size", type=int, default=8, help="Hidden size for continuous variables")
    parser.add_argument("--output-size", type=int, default=7, help="Number of quantiles to predict")
    parser.add_argument("--log-dir", type=str, default="logs/ts", help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for saving the model")
    parser.add_argument("--model-name", type=str, default="tft.ckpt", help="Name of the model file")
    
    args = parser.parse_args()
    
    train_tft(
        data_path=args.data_path,
        max_epochs=args.max_epochs,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=args.output_size,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        model_name=args.model_name
    )
