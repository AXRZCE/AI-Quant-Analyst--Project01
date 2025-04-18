"""
Bayesian regression model for uncertainty quantification.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import numpyro, but don't fail if it's not available
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    import jax.numpy as jnp
    import jax.random as random
    NUMPYRO_AVAILABLE = True
except ImportError:
    logger.warning("NumPyro not available. Using dummy implementation.")
    NUMPYRO_AVAILABLE = False

# Import local modules
from uncertainty.data_prep import load_df, prepare_features_targets, split_data, normalize_data

class BayesianRegression:
    """
    Bayesian regression model for uncertainty quantification.
    """
    
    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        random_seed: int = 0
    ):
        """
        Initialize the Bayesian regression model.
        
        Args:
            num_warmup: Number of warmup steps for MCMC
            num_samples: Number of samples to draw from the posterior
            num_chains: Number of MCMC chains
            random_seed: Random seed for reproducibility
        """
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.random_seed = random_seed
        self.mcmc = None
        self.feature_names = None
        self.normalization_params = None
    
    def model(self, X, y=None):
        """
        Define the Bayesian regression model.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
        """
        if not NUMPYRO_AVAILABLE:
            raise ImportError("NumPyro is required for Bayesian regression")
        
        n_features = X.shape[1]
        
        # Priors
        coef = numpyro.sample("coef", dist.Normal(0, 1).expand([n_features]))
        intercept = numpyro.sample("intercept", dist.Normal(0, 1))
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        
        # Likelihood
        mu = numpyro.deterministic("mu", intercept + (X * coef).sum(-1))
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit the Bayesian regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of the features
            
        Returns:
            Dictionary with model information
        """
        if not NUMPYRO_AVAILABLE:
            logger.warning("NumPyro not available. Using dummy implementation.")
            return {"status": "error", "message": "NumPyro not available"}
        
        logger.info("Fitting Bayesian regression model")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Initialize random key
        rng_key = random.PRNGKey(self.random_seed)
        
        # Set up the MCMC sampler
        kernel = NUTS(self.model)
        self.mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains
        )
        
        # Run MCMC
        self.mcmc.run(rng_key, X, y)
        
        # Get summary statistics
        summary = self.get_summary()
        
        logger.info("Bayesian regression model fitted successfully")
        
        return {
            "status": "success",
            "summary": summary
        }
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of the posterior distribution.
        
        Returns:
            DataFrame with summary statistics
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return pd.DataFrame()
        
        # Get samples
        samples = self.mcmc.get_samples()
        
        # Create summary DataFrame
        summary_dict = {
            "mean": {},
            "std": {},
            "5%": {},
            "50%": {},
            "95%": {}
        }
        
        # Add intercept
        intercept = samples["intercept"]
        summary_dict["mean"]["intercept"] = np.mean(intercept)
        summary_dict["std"]["intercept"] = np.std(intercept)
        summary_dict["5%"]["intercept"] = np.percentile(intercept, 5)
        summary_dict["50%"]["intercept"] = np.percentile(intercept, 50)
        summary_dict["95%"]["intercept"] = np.percentile(intercept, 95)
        
        # Add coefficients
        coef = samples["coef"]
        for i in range(coef.shape[1]):
            name = f"coef_{i}" if self.feature_names is None else self.feature_names[i]
            summary_dict["mean"][name] = np.mean(coef[:, i])
            summary_dict["std"][name] = np.std(coef[:, i])
            summary_dict["5%"][name] = np.percentile(coef[:, i], 5)
            summary_dict["50%"][name] = np.percentile(coef[:, i], 50)
            summary_dict["95%"][name] = np.percentile(coef[:, i], 95)
        
        # Add sigma
        sigma = samples["sigma"]
        summary_dict["mean"]["sigma"] = np.mean(sigma)
        summary_dict["std"]["sigma"] = np.std(sigma)
        summary_dict["5%"]["sigma"] = np.percentile(sigma, 5)
        summary_dict["50%"]["sigma"] = np.percentile(sigma, 50)
        summary_dict["95%"]["sigma"] = np.percentile(sigma, 95)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_dict)
        
        return summary_df
    
    def predict(
        self,
        X: np.ndarray,
        return_samples: bool = False,
        n_samples: Optional[int] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions with the Bayesian regression model.
        
        Args:
            X: Feature matrix
            return_samples: Whether to return samples from the posterior predictive distribution
            n_samples: Number of samples to return (if None, use all samples)
            
        Returns:
            Array with predictions or dictionary with prediction samples
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return np.zeros(len(X))
        
        logger.info(f"Making predictions for {len(X)} samples")
        
        # Get samples
        samples = self.mcmc.get_samples()
        
        # Subsample if requested
        if n_samples is not None and n_samples < len(samples["intercept"]):
            indices = np.random.choice(len(samples["intercept"]), n_samples, replace=False)
            samples = {k: v[indices] for k, v in samples.items()}
        
        # Set up the predictive distribution
        predictive = Predictive(self.model, samples)
        
        # Make predictions
        rng_key = random.PRNGKey(self.random_seed + 1)
        predictions = predictive(rng_key, X)
        
        if return_samples:
            return {
                "mu": np.array(predictions["mu"]),
                "obs": np.array(predictions["obs"]) if "obs" in predictions else None
            }
        else:
            # Return mean predictions
            return np.mean(predictions["mu"], axis=0)
    
    def predict_interval(
        self,
        X: np.ndarray,
        interval: float = 0.9
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with credible intervals.
        
        Args:
            X: Feature matrix
            interval: Credible interval (between 0 and 1)
            
        Returns:
            Dictionary with mean predictions and credible intervals
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return {
                "mean": np.zeros(len(X)),
                "lower": np.zeros(len(X)),
                "upper": np.zeros(len(X))
            }
        
        # Get prediction samples
        predictions = self.predict(X, return_samples=True)
        
        # Calculate mean and credible intervals
        mean = np.mean(predictions["mu"], axis=0)
        lower = np.percentile(predictions["mu"], (1 - interval) * 100 / 2, axis=0)
        upper = np.percentile(predictions["mu"], 100 - (1 - interval) * 100 / 2, axis=0)
        
        return {
            "mean": mean,
            "lower": lower,
            "upper": upper
        }
    
    def plot_trace(
        self,
        params: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot trace plots for model parameters.
        
        Args:
            params: List of parameters to plot (if None, plot all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return plt.figure()
        
        # Get samples
        samples = self.mcmc.get_samples()
        
        # Determine parameters to plot
        if params is None:
            params = ["intercept", "sigma"]
            if self.feature_names is not None:
                params += self.feature_names
            else:
                params += [f"coef_{i}" for i in range(samples["coef"].shape[1])]
        
        # Create figure
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 2, figsize=figsize)
        
        # Plot each parameter
        for i, param in enumerate(params):
            if param == "intercept":
                trace = samples["intercept"]
            elif param == "sigma":
                trace = samples["sigma"]
            elif param.startswith("coef_"):
                idx = int(param.split("_")[1])
                trace = samples["coef"][:, idx]
            elif self.feature_names is not None and param in self.feature_names:
                idx = self.feature_names.index(param)
                trace = samples["coef"][:, idx]
            else:
                continue
            
            # Trace plot
            axes[i, 0].plot(trace)
            axes[i, 0].set_title(f"Trace: {param}")
            
            # Histogram
            axes[i, 1].hist(trace, bins=30, density=True)
            axes[i, 1].set_title(f"Posterior: {param}")
        
        plt.tight_layout()
        
        return fig
    
    def plot_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_new: Optional[np.ndarray] = None,
        interval: float = 0.9,
        n_samples: int = 100,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot predictions with credible intervals.
        
        Args:
            X: Feature matrix used for training
            y: Target vector used for training
            X_new: New feature matrix for predictions (if None, use X)
            interval: Credible interval (between 0 and 1)
            n_samples: Number of samples to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return plt.figure()
        
        # Use training data if X_new is not provided
        if X_new is None:
            X_new = X
        
        # Get prediction samples
        predictions = self.predict(X_new, return_samples=True, n_samples=n_samples)
        
        # Calculate mean and credible intervals
        mean = np.mean(predictions["mu"], axis=0)
        lower = np.percentile(predictions["mu"], (1 - interval) * 100 / 2, axis=0)
        upper = np.percentile(predictions["mu"], 100 - (1 - interval) * 100 / 2, axis=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data
        ax.scatter(y, self.predict(X), alpha=0.5, label="Training data")
        
        # Plot predictions
        ax.scatter(mean, mean, alpha=0.5, color="red", label="Predictions")
        
        # Plot credible intervals
        for i in range(len(mean)):
            ax.plot([mean[i], mean[i]], [lower[i], upper[i]], color="red", alpha=0.3)
        
        # Add diagonal line
        min_val = min(np.min(y), np.min(mean))
        max_val = max(np.max(y), np.max(mean))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
        
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.set_title("Predictions with credible intervals")
        ax.legend()
        
        return fig
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        if not NUMPYRO_AVAILABLE or self.mcmc is None:
            logger.warning("Model not fitted or NumPyro not available")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Get samples
        samples = self.mcmc.get_samples()
        
        # Save samples and metadata
        np.savez(
            path,
            **samples,
            feature_names=self.feature_names,
            normalization_params=self.normalization_params
        )
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BayesianRegression":
        """
        Load a model from a file.
        
        Args:
            path: Path to the model file
            
        Returns:
            Loaded model
        """
        if not NUMPYRO_AVAILABLE:
            logger.warning("NumPyro not available")
            return cls()
        
        # Load samples and metadata
        data = np.load(path, allow_pickle=True)
        
        # Create model
        model = cls()
        
        # Set feature names
        model.feature_names = data["feature_names"].tolist() if "feature_names" in data else None
        
        # Set normalization parameters
        model.normalization_params = data["normalization_params"].item() if "normalization_params" in data else None
        
        # Create dummy MCMC object
        model.mcmc = type("DummyMCMC", (), {"get_samples": lambda: {}})()
        
        # Add method to get samples
        def get_samples():
            samples = {}
            for key in data.keys():
                if key not in ["feature_names", "normalization_params"]:
                    samples[key] = data[key]
            return samples
        
        model.mcmc.get_samples = get_samples
        
        logger.info(f"Model loaded from {path}")
        
        return model

def run_inference(
    data_path: str,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    random_seed: int = 0
) -> BayesianRegression:
    """
    Run Bayesian inference on data.
    
    Args:
        data_path: Path to the data files
        num_warmup: Number of warmup steps for MCMC
        num_samples: Number of samples to draw from the posterior
        num_chains: Number of MCMC chains
        random_seed: Random seed for reproducibility
        
    Returns:
        Fitted BayesianRegression model
    """
    if not NUMPYRO_AVAILABLE:
        logger.warning("NumPyro not available. Using dummy implementation.")
        return BayesianRegression()
    
    # Load data
    df = load_df(data_path)
    
    # Prepare features and targets
    X, y = prepare_features_targets(df)
    
    # Split data
    splits = split_data(X, y)
    
    # Normalize data
    normalized = normalize_data(splits["X_train"], splits["X_val"], splits["X_test"])
    
    # Get feature names
    feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "date", "label"]]
    
    # Create and fit model
    model = BayesianRegression(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        random_seed=random_seed
    )
    
    # Store normalization parameters
    model.normalization_params = normalized["params"]
    
    # Fit model
    model.fit(normalized["X_train"], splits["y_train"], feature_names=feature_cols)
    
    return model

def sample_posterior(
    model: BayesianRegression,
    X_new: np.ndarray,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Sample from the posterior predictive distribution.
    
    Args:
        model: Fitted BayesianRegression model
        X_new: New feature matrix
        n_samples: Number of samples to draw
        
    Returns:
        Array of samples from the posterior predictive distribution
    """
    if not NUMPYRO_AVAILABLE:
        logger.warning("NumPyro not available. Using dummy implementation.")
        return np.zeros((n_samples, len(X_new)))
    
    # Normalize X_new if normalization parameters are available
    if model.normalization_params is not None:
        mean = model.normalization_params["mean"]
        std = model.normalization_params["std"]
        X_new = (X_new - mean) / std
    
    # Get prediction samples
    predictions = model.predict(X_new, return_samples=True, n_samples=n_samples)
    
    return predictions["mu"]

if __name__ == "__main__":
    # Check if NumPyro is available
    if not NUMPYRO_AVAILABLE:
        print("NumPyro is not available. Please install it with 'pip install numpyro jax jaxlib'.")
        exit(1)
    
    # Example usage
    try:
        # Try to load from batch features
        data_path = "data/features/batch/technical/*.parquet"
        model = run_inference(data_path, num_warmup=100, num_samples=200)
    except ValueError:
        # Try to load from processed data
        try:
            data_path = "data/processed/training_data.parquet"
            model = run_inference(data_path, num_warmup=100, num_samples=200)
        except ValueError:
            # Try to load from raw data
            data_path = "data/raw/ticks/*/*.parquet"
            model = run_inference(data_path, num_warmup=100, num_samples=200)
    
    # Print summary
    summary = model.get_summary()
    print(summary)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/bayesian_regression.npz")
