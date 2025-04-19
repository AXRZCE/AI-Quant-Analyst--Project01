"""
Uncertainty quantification module.

This module provides functionality for quantifying uncertainty in model predictions,
including Bayesian methods, bootstrap methods, and conformal prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample
from sklearn.model_selection import KFold, TimeSeriesSplit
from joblib import Parallel, delayed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BootstrapUncertainty:
    """Bootstrap uncertainty quantification."""
    
    def __init__(
        self,
        base_model: BaseEstimator,
        n_estimators: int = 100,
        subsample: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the bootstrap uncertainty quantifier.
        
        Args:
            base_model: Base model to bootstrap
            n_estimators: Number of bootstrap estimators
            subsample: Subsample ratio
            random_state: Random state
            n_jobs: Number of parallel jobs
        """
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BootstrapUncertainty':
        """
        Fit the bootstrap models.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted bootstrap uncertainty quantifier
        """
        logger.info(f"Fitting {self.n_estimators} bootstrap models")
        
        # Set random state
        np.random.seed(self.random_state)
        
        # Create random states for each estimator
        random_states = np.random.randint(0, 10000, size=self.n_estimators)
        
        # Fit models in parallel
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_bootstrap)(X, y, i, random_states[i])
            for i in range(self.n_estimators)
        )
        
        self.fitted = True
        
        logger.info(f"Fitted {len(self.models)} bootstrap models")
        
        return self
    
    def _fit_bootstrap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idx: int,
        random_state: int
    ) -> BaseEstimator:
        """
        Fit a single bootstrap model.
        
        Args:
            X: Feature matrix
            y: Target vector
            idx: Model index
            random_state: Random state
            
        Returns:
            Fitted model
        """
        # Create bootstrap sample
        n_samples = int(len(X) * self.subsample)
        indices = resample(
            np.arange(len(X)),
            replace=True,
            n_samples=n_samples,
            random_state=random_state
        )
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Clone and fit model
        model = clone(self.base_model)
        model.fit(X_boot, y_boot)
        
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the bootstrap models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Mean predictions
        """
        if not self.fitted:
            logger.error("Models not fitted")
            return np.zeros(len(X))
        
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Return mean predictions
        return np.mean(predictions, axis=1)
    
    def predict_interval(
        self,
        X: np.ndarray,
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
        if not self.fitted:
            logger.error("Models not fitted")
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Calculate quantiles
        lower_bound = np.quantile(predictions, alpha / 2, axis=1)
        upper_bound = np.quantile(predictions, 1 - alpha / 2, axis=1)
        
        return lower_bound, upper_bound
    
    def predict_distribution(self, X: np.ndarray) -> np.ndarray:
        """
        Get the full predictive distribution.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_estimators) with all predictions
        """
        if not self.fitted:
            logger.error("Models not fitted")
            return np.zeros((len(X), 1))
        
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        return predictions
    
    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Get the standard deviation of predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Standard deviation of predictions
        """
        if not self.fitted:
            logger.error("Models not fitted")
            return np.zeros(len(X))
        
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        # Calculate standard deviation
        return np.std(predictions, axis=1)
    
    def plot_prediction_intervals(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot prediction intervals.
        
        Args:
            X: Feature matrix
            y: True values (optional)
            alpha: Significance level
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.fitted:
            logger.error("Models not fitted")
            return plt.figure()
        
        # Get predictions and intervals
        y_pred = self.predict(X)
        lower, upper = self.predict_interval(X, alpha)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot predictions
        ax.plot(y_pred, 'b-', label='Prediction')
        
        # Plot intervals
        ax.fill_between(
            np.arange(len(y_pred)),
            lower, upper,
            alpha=0.2, color='b',
            label=f'{int((1-alpha)*100)}% Confidence Interval'
        )
        
        # Plot true values if provided
        if y is not None:
            ax.plot(y, 'r.', label='True Value')
        
        # Add labels and legend
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title('Prediction Intervals')
        ax.legend()
        
        return fig


class ConformalPrediction:
    """Conformal prediction for uncertainty quantification."""
    
    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        cv: int = 5,
        use_time_series_cv: bool = False
    ):
        """
        Initialize the conformal predictor.
        
        Args:
            model: Model to use
            alpha: Significance level
            cv: Number of cross-validation folds
            use_time_series_cv: Whether to use time series cross-validation
        """
        self.model = model
        self.alpha = alpha
        self.cv = cv
        self.use_time_series_cv = use_time_series_cv
        self.calibration_scores = None
        self.fitted_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConformalPrediction':
        """
        Fit the conformal predictor.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted conformal predictor
        """
        logger.info("Fitting conformal predictor")
        
        # Create cross-validation object
        if self.use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.cv)
        else:
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Split data into training and calibration sets
        train_indices, cal_indices = next(cv.split(X))
        X_train, X_cal = X[train_indices], X[cal_indices]
        y_train, y_cal = y[train_indices], y[cal_indices]
        
        # Fit model on training data
        self.fitted_model = clone(self.model)
        self.fitted_model.fit(X_train, y_train)
        
        # Calculate calibration scores
        y_pred_cal = self.fitted_model.predict(X_cal)
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        
        logger.info("Fitted conformal predictor")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the conformal predictor.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.fitted_model is None:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        return self.fitted_model.predict(X)
    
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level (if None, use self.alpha)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if self.fitted_model is None or self.calibration_scores is None:
            logger.error("Model not fitted")
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Use provided alpha or default
        alpha = alpha or self.alpha
        
        # Get predictions
        y_pred = self.fitted_model.predict(X)
        
        # Calculate quantile
        q = np.quantile(self.calibration_scores, 1 - alpha)
        
        # Calculate intervals
        lower_bound = y_pred - q
        upper_bound = y_pred + q
        
        return lower_bound, upper_bound


class BayesianModelWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for Bayesian models."""
    
    def __init__(
        self,
        model_type: str = "numpyro",
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        random_seed: int = 42
    ):
        """
        Initialize the Bayesian model wrapper.
        
        Args:
            model_type: Type of Bayesian model ('numpyro' or 'pymc')
            num_warmup: Number of warmup steps
            num_samples: Number of samples
            num_chains: Number of chains
            random_seed: Random seed
        """
        self.model_type = model_type
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.random_seed = random_seed
        self.model = None
        self.trace = None
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianModelWrapper':
        """
        Fit the Bayesian model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Fitted Bayesian model wrapper
        """
        logger.info(f"Fitting Bayesian model ({self.model_type})")
        
        # Get feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy arrays
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        if self.model_type == "numpyro":
            self._fit_numpyro(X_np, y_np)
        elif self.model_type == "pymc":
            self._fit_pymc(X_np, y_np)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
        
        return self
    
    def _fit_numpyro(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit a NumPyro model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        try:
            # Import NumPyro
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
            import jax.random as random
            
            # Define model
            def model(X, y=None):
                n_features = X.shape[1]
                
                # Priors
                coef = numpyro.sample("coef", dist.Normal(0, 1).expand([n_features]))
                intercept = numpyro.sample("intercept", dist.Normal(0, 1))
                sigma = numpyro.sample("sigma", dist.Exponential(1.0))
                
                # Likelihood
                mu = numpyro.deterministic("mu", intercept + (X * coef).sum(-1))
                numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            
            # Initialize random key
            rng_key = random.PRNGKey(self.random_seed)
            
            # Set up the MCMC sampler
            kernel = NUTS(model)
            mcmc = MCMC(
                kernel,
                num_warmup=self.num_warmup,
                num_samples=self.num_samples,
                num_chains=self.num_chains
            )
            
            # Run MCMC
            mcmc.run(rng_key, X, y)
            
            # Store model and trace
            self.model = model
            self.trace = mcmc.get_samples()
            
            logger.info("Fitted NumPyro model")
        except ImportError:
            logger.error("NumPyro not available. Install with 'pip install numpyro jax jaxlib'")
        except Exception as e:
            logger.error(f"Error fitting NumPyro model: {e}")
    
    def _fit_pymc(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit a PyMC model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        try:
            # Import PyMC
            import pymc as pm
            import arviz as az
            
            # Create model
            with pm.Model() as model:
                # Priors
                coef = pm.Normal("coef", mu=0, sigma=1, shape=X.shape[1])
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                sigma = pm.Exponential("sigma", lam=1.0)
                
                # Likelihood
                mu = intercept + pm.math.dot(X, coef)
                likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)
                
                # Sample
                trace = pm.sample(
                    draws=self.num_samples,
                    tune=self.num_warmup,
                    chains=self.num_chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True
                )
            
            # Store model and trace
            self.model = model
            self.trace = trace
            
            logger.info("Fitted PyMC model")
        except ImportError:
            logger.error("PyMC not available. Install with 'pip install pymc arviz'")
        except Exception as e:
            logger.error(f"Error fitting PyMC model: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the Bayesian model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Mean predictions
        """
        if self.model is None or self.trace is None:
            logger.error("Model not fitted")
            return np.zeros(len(X))
        
        # Convert to numpy array
        X_np = X.values if hasattr(X, 'values') else X
        
        if self.model_type == "numpyro":
            return self._predict_numpyro(X_np)
        elif self.model_type == "pymc":
            return self._predict_pymc(X_np)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return np.zeros(len(X))
    
    def _predict_numpyro(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a NumPyro model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Mean predictions
        """
        try:
            # Import NumPyro
            from numpyro.infer import Predictive
            import jax.random as random
            
            # Set up the predictive distribution
            predictive = Predictive(self.model, self.trace)
            
            # Make predictions
            rng_key = random.PRNGKey(self.random_seed + 1)
            predictions = predictive(rng_key, X)
            
            # Return mean predictions
            return np.mean(predictions["mu"], axis=0)
        except Exception as e:
            logger.error(f"Error making predictions with NumPyro model: {e}")
            return np.zeros(len(X))
    
    def _predict_pymc(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a PyMC model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Mean predictions
        """
        try:
            # Import PyMC
            import pymc as pm
            import arviz as az
            
            # Get posterior samples
            coef_samples = az.extract(self.trace, var_names=["coef"]).values
            intercept_samples = az.extract(self.trace, var_names=["intercept"]).values
            
            # Calculate predictions for each sample
            predictions = intercept_samples[:, np.newaxis] + np.dot(coef_samples, X.T)
            
            # Return mean predictions
            return np.mean(predictions, axis=0)
        except Exception as e:
            logger.error(f"Error making predictions with PyMC model: {e}")
            return np.zeros(len(X))
    
    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with credible intervals.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        if self.model is None or self.trace is None:
            logger.error("Model not fitted")
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Convert to numpy array
        X_np = X.values if hasattr(X, 'values') else X
        
        if self.model_type == "numpyro":
            return self._predict_interval_numpyro(X_np, alpha)
        elif self.model_type == "pymc":
            return self._predict_interval_pymc(X_np, alpha)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def _predict_interval_numpyro(
        self,
        X: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with credible intervals using a NumPyro model.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        try:
            # Import NumPyro
            from numpyro.infer import Predictive
            import jax.random as random
            
            # Set up the predictive distribution
            predictive = Predictive(self.model, self.trace)
            
            # Make predictions
            rng_key = random.PRNGKey(self.random_seed + 1)
            predictions = predictive(rng_key, X)
            
            # Calculate credible intervals
            lower = np.quantile(predictions["mu"], alpha / 2, axis=0)
            upper = np.quantile(predictions["mu"], 1 - alpha / 2, axis=0)
            
            return lower, upper
        except Exception as e:
            logger.error(f"Error making interval predictions with NumPyro model: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def _predict_interval_pymc(
        self,
        X: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with credible intervals using a PyMC model.
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        try:
            # Import PyMC
            import pymc as pm
            import arviz as az
            
            # Get posterior samples
            coef_samples = az.extract(self.trace, var_names=["coef"]).values
            intercept_samples = az.extract(self.trace, var_names=["intercept"]).values
            
            # Calculate predictions for each sample
            predictions = intercept_samples[:, np.newaxis] + np.dot(coef_samples, X.T)
            
            # Calculate credible intervals
            lower = np.quantile(predictions, alpha / 2, axis=0)
            upper = np.quantile(predictions, 1 - alpha / 2, axis=0)
            
            return lower, upper
        except Exception as e:
            logger.error(f"Error making interval predictions with PyMC model: {e}")
            return np.zeros(len(X)), np.zeros(len(X))
    
    def get_posterior_samples(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples.
        
        Returns:
            Dictionary of parameter names and samples
        """
        if self.model is None or self.trace is None:
            logger.error("Model not fitted")
            return {}
        
        if self.model_type == "numpyro":
            return self._get_posterior_samples_numpyro()
        elif self.model_type == "pymc":
            return self._get_posterior_samples_pymc()
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return {}
    
    def _get_posterior_samples_numpyro(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples from a NumPyro model.
        
        Returns:
            Dictionary of parameter names and samples
        """
        try:
            # Get samples
            samples = self.trace
            
            # Create dictionary with named parameters
            result = {
                "intercept": samples["intercept"],
                "sigma": samples["sigma"]
            }
            
            # Add coefficients with feature names
            coef_samples = samples["coef"]
            for i, name in enumerate(self.feature_names):
                result[f"coef_{name}"] = coef_samples[:, i]
            
            return result
        except Exception as e:
            logger.error(f"Error getting posterior samples from NumPyro model: {e}")
            return {}
    
    def _get_posterior_samples_pymc(self) -> Dict[str, np.ndarray]:
        """
        Get posterior samples from a PyMC model.
        
        Returns:
            Dictionary of parameter names and samples
        """
        try:
            # Import PyMC
            import arviz as az
            
            # Get samples
            intercept_samples = az.extract(self.trace, var_names=["intercept"]).values
            coef_samples = az.extract(self.trace, var_names=["coef"]).values
            sigma_samples = az.extract(self.trace, var_names=["sigma"]).values
            
            # Create dictionary with named parameters
            result = {
                "intercept": intercept_samples,
                "sigma": sigma_samples
            }
            
            # Add coefficients with feature names
            for i, name in enumerate(self.feature_names):
                result[f"coef_{name}"] = coef_samples[:, i]
            
            return result
        except Exception as e:
            logger.error(f"Error getting posterior samples from PyMC model: {e}")
            return {}
    
    def plot_posterior(
        self,
        params: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot posterior distributions.
        
        Args:
            params: List of parameters to plot (if None, plot all)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.model is None or self.trace is None:
            logger.error("Model not fitted")
            return plt.figure()
        
        # Get posterior samples
        samples = self.get_posterior_samples()
        
        if not samples:
            logger.error("No posterior samples available")
            return plt.figure()
        
        # Select parameters to plot
        if params is None:
            params = list(samples.keys())
        else:
            # Filter to available parameters
            params = [p for p in params if p in samples]
        
        if not params:
            logger.error("No valid parameters to plot")
            return plt.figure()
        
        # Create figure
        n_params = len(params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single parameter case
        if n_params == 1:
            axes = np.array([axes])
        
        # Flatten axes
        axes = axes.flatten()
        
        # Plot each parameter
        for i, param in enumerate(params):
            if i < len(axes):
                ax = axes[i]
                
                # Get samples
                param_samples = samples[param]
                
                # Plot histogram
                ax.hist(param_samples, bins=30, alpha=0.7, density=True)
                
                # Add mean and credible interval
                mean = np.mean(param_samples)
                lower = np.quantile(param_samples, 0.025)
                upper = np.quantile(param_samples, 0.975)
                
                ax.axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.3f}')
                ax.axvline(lower, color='r', linestyle='--', label=f'95% CI: [{lower:.3f}, {upper:.3f}]')
                ax.axvline(upper, color='r', linestyle='--')
                
                # Add labels
                ax.set_title(param)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize='small')
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


def create_uncertainty_quantifier(
    model: BaseEstimator,
    method: str = "bootstrap",
    **kwargs
) -> Union[BootstrapUncertainty, ConformalPrediction, BayesianModelWrapper]:
    """
    Create an uncertainty quantifier.
    
    Args:
        model: Base model
        method: Uncertainty quantification method
        **kwargs: Additional arguments for the uncertainty quantifier
        
    Returns:
        Uncertainty quantifier
    """
    logger.info(f"Creating uncertainty quantifier using {method} method")
    
    if method == "bootstrap":
        return BootstrapUncertainty(model, **kwargs)
    elif method == "conformal":
        return ConformalPrediction(model, **kwargs)
    elif method == "bayesian":
        return BayesianModelWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown uncertainty quantification method: {method}")


def add_uncertainty_to_predictions(
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> pd.DataFrame:
    """
    Add uncertainty to predictions.
    
    Args:
        predictions: Mean predictions
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        
    Returns:
        DataFrame with predictions and uncertainty
    """
    # Create DataFrame
    df = pd.DataFrame({
        "prediction": predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "uncertainty": (upper_bound - lower_bound) / 2
    })
    
    return df
