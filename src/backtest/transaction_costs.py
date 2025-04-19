"""
Transaction cost models for backtesting.

This module provides various transaction cost models to simulate realistic trading costs
including commissions, slippage, market impact, and bid-ask spread.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TransactionCostModel:
    """Base class for transaction cost models."""
    
    def __init__(self, name: str = "base"):
        """
        Initialize the transaction cost model.
        
        Args:
            name: Name of the model
        """
        self.name = name
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate transaction cost.
        
        Args:
            price: Execution price
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters
            
        Returns:
            Transaction cost (positive value)
        """
        return 0.0
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust execution price based on transaction costs.
        
        Args:
            price: Original price
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters
            
        Returns:
            Adjusted execution price
        """
        return price


class FixedCommissionModel(TransactionCostModel):
    """Fixed commission model (e.g., $5 per trade)."""
    
    def __init__(self, commission: float = 5.0):
        """
        Initialize the fixed commission model.
        
        Args:
            commission: Fixed commission per trade
        """
        super().__init__(name="fixed_commission")
        self.commission = commission
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate fixed commission.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Fixed commission
        """
        if abs(quantity) > 0:
            return self.commission
        return 0.0


class PercentageCommissionModel(TransactionCostModel):
    """Percentage commission model (e.g., 0.1% of trade value)."""
    
    def __init__(self, commission_rate: float = 0.001, min_commission: float = 0.0):
        """
        Initialize the percentage commission model.
        
        Args:
            commission_rate: Commission rate as a percentage of trade value
            min_commission: Minimum commission per trade
        """
        super().__init__(name="percentage_commission")
        self.commission_rate = commission_rate
        self.min_commission = min_commission
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate percentage commission.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Commission amount
        """
        if abs(quantity) > 0:
            commission = abs(price * quantity) * self.commission_rate
            return max(commission, self.min_commission)
        return 0.0


class TieredCommissionModel(TransactionCostModel):
    """Tiered commission model based on trade value or volume."""
    
    def __init__(self, tiers: List[Tuple[float, float]], base_rate: float = 0.0):
        """
        Initialize the tiered commission model.
        
        Args:
            tiers: List of (threshold, rate) tuples, sorted by threshold
                  For trade value > threshold, rate is applied
            base_rate: Base commission rate
        """
        super().__init__(name="tiered_commission")
        self.tiers = sorted(tiers, key=lambda x: x[0])
        self.base_rate = base_rate
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate tiered commission.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Commission amount
        """
        if abs(quantity) == 0:
            return 0.0
            
        trade_value = abs(price * quantity)
        
        # Find applicable tier
        rate = self.base_rate
        for threshold, tier_rate in self.tiers:
            if trade_value > threshold:
                rate = tier_rate
        
        return trade_value * rate


class FixedPlusPercentageModel(TransactionCostModel):
    """Combined fixed plus percentage commission model."""
    
    def __init__(self, fixed_commission: float = 1.0, percentage_rate: float = 0.0005):
        """
        Initialize the fixed plus percentage commission model.
        
        Args:
            fixed_commission: Fixed commission per trade
            percentage_rate: Percentage commission rate
        """
        super().__init__(name="fixed_plus_percentage")
        self.fixed_commission = fixed_commission
        self.percentage_rate = percentage_rate
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate fixed plus percentage commission.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Commission amount
        """
        if abs(quantity) == 0:
            return 0.0
            
        trade_value = abs(price * quantity)
        return self.fixed_commission + (trade_value * self.percentage_rate)


class ConstantSlippageModel(TransactionCostModel):
    """Constant slippage model (e.g., 0.1% of price)."""
    
    def __init__(self, slippage_rate: float = 0.001):
        """
        Initialize the constant slippage model.
        
        Args:
            slippage_rate: Slippage rate as a percentage of price
        """
        super().__init__(name="constant_slippage")
        self.slippage_rate = slippage_rate
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust price based on constant slippage.
        
        Args:
            price: Original price
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters
            
        Returns:
            Adjusted price with slippage
        """
        if quantity > 0:  # Buy
            return price * (1 + self.slippage_rate)
        elif quantity < 0:  # Sell
            return price * (1 - self.slippage_rate)
        return price


class VolumeBasedSlippageModel(TransactionCostModel):
    """Volume-based slippage model."""
    
    def __init__(self, volume_impact_coef: float = 0.1, market_volume: Optional[Callable] = None):
        """
        Initialize the volume-based slippage model.
        
        Args:
            volume_impact_coef: Coefficient for volume impact
            market_volume: Function to get market volume for a symbol
        """
        super().__init__(name="volume_based_slippage")
        self.volume_impact_coef = volume_impact_coef
        self.market_volume = market_volume
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust price based on volume-based slippage.
        
        Args:
            price: Original price
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters including 'symbol' and 'timestamp'
            
        Returns:
            Adjusted price with slippage
        """
        if abs(quantity) == 0:
            return price
            
        # Get market volume if available
        market_vol = 100000  # Default volume
        if self.market_volume and 'symbol' in kwargs and 'timestamp' in kwargs:
            try:
                market_vol = self.market_volume(kwargs['symbol'], kwargs['timestamp'])
            except Exception as e:
                logger.warning(f"Error getting market volume: {e}")
        
        # Calculate volume ratio and impact
        volume_ratio = abs(quantity) / market_vol
        impact = self.volume_impact_coef * volume_ratio
        
        # Apply impact to price
        if quantity > 0:  # Buy
            return price * (1 + impact)
        else:  # Sell
            return price * (1 - impact)


class BidAskSpreadModel(TransactionCostModel):
    """Bid-ask spread model."""
    
    def __init__(self, spread_percentage: float = 0.001, spread_function: Optional[Callable] = None):
        """
        Initialize the bid-ask spread model.
        
        Args:
            spread_percentage: Bid-ask spread as a percentage of price
            spread_function: Optional function to get spread for a symbol
        """
        super().__init__(name="bid_ask_spread")
        self.spread_percentage = spread_percentage
        self.spread_function = spread_function
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust price based on bid-ask spread.
        
        Args:
            price: Original price (mid price)
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters including 'symbol' and 'timestamp'
            
        Returns:
            Adjusted price with spread
        """
        if abs(quantity) == 0:
            return price
            
        # Get spread if available
        spread = self.spread_percentage
        if self.spread_function and 'symbol' in kwargs and 'timestamp' in kwargs:
            try:
                spread = self.spread_function(kwargs['symbol'], kwargs['timestamp'])
            except Exception as e:
                logger.warning(f"Error getting spread: {e}")
        
        # Apply spread to price
        if quantity > 0:  # Buy at ask
            return price * (1 + spread / 2)
        else:  # Sell at bid
            return price * (1 - spread / 2)


class MarketImpactModel(TransactionCostModel):
    """Market impact model based on square root law."""
    
    def __init__(self, impact_coef: float = 0.1, market_cap_function: Optional[Callable] = None):
        """
        Initialize the market impact model.
        
        Args:
            impact_coef: Coefficient for market impact
            market_cap_function: Function to get market cap for a symbol
        """
        super().__init__(name="market_impact")
        self.impact_coef = impact_coef
        self.market_cap_function = market_cap_function
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust price based on market impact.
        
        Args:
            price: Original price
            quantity: Quantity to trade (positive for buy, negative for sell)
            **kwargs: Additional parameters including 'symbol' and 'timestamp'
            
        Returns:
            Adjusted price with market impact
        """
        if abs(quantity) == 0:
            return price
            
        # Get market cap if available
        market_cap = 1e9  # Default market cap (1 billion)
        if self.market_cap_function and 'symbol' in kwargs:
            try:
                market_cap = self.market_cap_function(kwargs['symbol'])
            except Exception as e:
                logger.warning(f"Error getting market cap: {e}")
        
        # Calculate trade value and impact
        trade_value = abs(price * quantity)
        impact = self.impact_coef * np.sqrt(trade_value / market_cap)
        
        # Apply impact to price
        if quantity > 0:  # Buy
            return price * (1 + impact)
        else:  # Sell
            return price * (1 - impact)


class CompositeCostModel(TransactionCostModel):
    """Composite transaction cost model combining multiple models."""
    
    def __init__(self, models: List[TransactionCostModel]):
        """
        Initialize the composite cost model.
        
        Args:
            models: List of transaction cost models
        """
        super().__init__(name="composite")
        self.models = models
    
    def calculate_cost(self, price: float, quantity: float, **kwargs) -> float:
        """
        Calculate total transaction cost from all models.
        
        Args:
            price: Execution price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Total transaction cost
        """
        return sum(model.calculate_cost(price, quantity, **kwargs) for model in self.models)
    
    def adjust_price(self, price: float, quantity: float, **kwargs) -> float:
        """
        Adjust price based on all models.
        
        Args:
            price: Original price
            quantity: Quantity to trade
            **kwargs: Additional parameters
            
        Returns:
            Final adjusted price
        """
        adjusted_price = price
        for model in self.models:
            adjusted_price = model.adjust_price(adjusted_price, quantity, **kwargs)
        return adjusted_price


# Factory function to create common transaction cost models
def create_transaction_cost_model(
    model_type: str = "realistic",
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    fixed_commission: float = 0.0,
    market_volume_function: Optional[Callable] = None
) -> TransactionCostModel:
    """
    Create a transaction cost model based on the specified type.
    
    Args:
        model_type: Type of model ('none', 'simple', 'realistic', 'custom')
        commission_rate: Commission rate as a percentage of trade value
        slippage_rate: Slippage rate as a percentage of price
        fixed_commission: Fixed commission per trade
        market_volume_function: Function to get market volume for a symbol
        
    Returns:
        Transaction cost model
    """
    if model_type == "none":
        return TransactionCostModel(name="none")
    
    elif model_type == "simple":
        return CompositeCostModel([
            PercentageCommissionModel(commission_rate=commission_rate),
            ConstantSlippageModel(slippage_rate=slippage_rate)
        ])
    
    elif model_type == "realistic":
        return CompositeCostModel([
            FixedPlusPercentageModel(fixed_commission=fixed_commission, percentage_rate=commission_rate),
            VolumeBasedSlippageModel(volume_impact_coef=0.1, market_volume=market_volume_function),
            BidAskSpreadModel(spread_percentage=slippage_rate)
        ])
    
    elif model_type == "custom":
        # Return a custom model based on the provided parameters
        return CompositeCostModel([
            FixedPlusPercentageModel(fixed_commission=fixed_commission, percentage_rate=commission_rate),
            ConstantSlippageModel(slippage_rate=slippage_rate)
        ])
    
    else:
        logger.warning(f"Unknown model type: {model_type}, using 'simple' instead")
        return create_transaction_cost_model("simple", commission_rate, slippage_rate)
