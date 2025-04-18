"""
Reinforcement learning environment for trading.
"""
import os
import logging
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    A simple trading environment:
    - State: [feature_vector..., cash, position]
    - Actions: 0=hold, 1=buy all-in, 2=sell all-out
    - Reward: PnL change
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 1,
        symbol_col: str = "symbol",
        price_col: str = "close",
        date_col: Optional[str] = "timestamp"
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with price and feature data
            feature_cols: List of feature column names
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction of trade value
            reward_scaling: Scaling factor for rewards
            window_size: Number of time steps to include in the observation
            symbol_col: Name of the symbol column
            price_col: Name of the price column
            date_col: Name of the date column
        """
        super().__init__()
        
        # Store parameters
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.date_col = date_col
        
        # Check if required columns exist
        required_cols = [symbol_col, price_col] + feature_cols
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # Get unique symbols
        self.symbols = self.df[symbol_col].unique()
        self.n_symbols = len(self.symbols)
        
        # Create symbol to index mapping
        self.symbol_to_idx = {symbol: i for i, symbol in enumerate(self.symbols)}
        
        # Observation space: features + [cash_ratio, position_ratio] for each symbol
        obs_dim = len(feature_cols) + 2 * self.n_symbols
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: 0=hold, 1=buy all-in, 2=sell all-out for each symbol
        self.action_space = spaces.Discrete(3 ** self.n_symbols)
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        # Reset current step
        self.current_step = 0
        
        # Reset portfolio
        self.cash = self.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        
        # Reset history
        self.history = []
        
        # Get initial observation
        obs = self._get_obs()
        
        return obs
    
    def _get_obs(self):
        """
        Get the current observation.
        
        Returns:
            Current observation
        """
        # Get current row for each symbol
        symbol_rows = {}
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                symbol_rows[symbol] = symbol_df.iloc[self.current_step]
            else:
                # If we've reached the end of data for this symbol, use the last row
                symbol_rows[symbol] = symbol_df.iloc[-1]
        
        # Extract features
        features = []
        for symbol in self.symbols:
            row = symbol_rows[symbol]
            features.extend(row[self.feature_cols].values)
        
        # Add portfolio state
        cash_ratio = self.cash / self.initial_capital
        
        # Calculate position values
        position_values = []
        for symbol in self.symbols:
            price = symbol_rows[symbol][self.price_col]
            position_value = self.positions[symbol] * price
            position_ratio = position_value / self.initial_capital
            position_values.append(position_ratio)
        
        # Combine features and portfolio state
        obs = np.concatenate([
            features,
            [cash_ratio],
            position_values
        ]).astype(np.float32)
        
        return obs
    
    def _decode_action(self, action: int) -> Dict[str, int]:
        """
        Decode action from action space to symbol-specific actions.
        
        Args:
            action: Action from action space
            
        Returns:
            Dictionary mapping symbols to actions (0=hold, 1=buy, 2=sell)
        """
        symbol_actions = {}
        
        # Convert action to base-3 representation
        action_base3 = []
        a = action
        for _ in range(self.n_symbols):
            action_base3.append(a % 3)
            a //= 3
        
        # Assign actions to symbols
        for i, symbol in enumerate(self.symbols):
            if i < len(action_base3):
                symbol_actions[symbol] = action_base3[i]
            else:
                symbol_actions[symbol] = 0  # Default to hold
        
        return symbol_actions
    
    def step(self, action: int):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current prices
        prices = {}
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                prices[symbol] = symbol_df.iloc[self.current_step][self.price_col]
            else:
                # If we've reached the end of data for this symbol, use the last price
                prices[symbol] = symbol_df.iloc[-1][self.price_col]
        
        # Calculate portfolio value before action
        portfolio_value_before = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)
        
        # Decode action
        symbol_actions = self._decode_action(action)
        
        # Execute actions
        for symbol, symbol_action in symbol_actions.items():
            price = prices[symbol]
            
            if symbol_action == 1 and self.cash > 0:  # Buy
                # Calculate position size (all-in)
                position_size = self.cash / price
                
                # Apply transaction cost
                position_size *= (1 - self.transaction_cost)
                
                # Update positions and cash
                self.positions[symbol] += position_size
                self.cash = 0
                
            elif symbol_action == 2 and self.positions[symbol] > 0:  # Sell
                # Calculate sale value
                sale_value = self.positions[symbol] * price
                
                # Apply transaction cost
                sale_value *= (1 - self.transaction_cost)
                
                # Update positions and cash
                self.cash += sale_value
                self.positions[symbol] = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = True
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                done = False
                break
        
        # Calculate portfolio value after action
        portfolio_value_after = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)
        
        # Calculate reward (change in portfolio value)
        reward = (portfolio_value_after - portfolio_value_before) * self.reward_scaling
        
        # Get next observation
        obs = self._get_obs() if not done else None
        
        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "symbol_actions": symbol_actions,
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "reward": reward
        })
        
        # Create info dictionary
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "step": self.current_step
        }
        
        return obs, reward, done, info
    
    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != "human":
            raise NotImplementedError(f"Rendering mode {mode} not implemented")
        
        # Get latest history entry
        if not self.history:
            print("No history available")
            return
        
        latest = self.history[-1]
        
        # Print information
        print(f"Step: {latest['step']}")
        print(f"Portfolio Value: ${latest['portfolio_value']:.2f}")
        print(f"Cash: ${latest['cash']:.2f}")
        
        # Print positions
        print("Positions:")
        for symbol, position in latest['positions'].items():
            if position > 0:
                symbol_df = self.df[self.df[self.symbol_col] == symbol]
                if self.current_step - 1 < len(symbol_df):
                    price = symbol_df.iloc[self.current_step - 1][self.price_col]
                else:
                    price = symbol_df.iloc[-1][self.price_col]
                
                position_value = position * price
                print(f"  {symbol}: {position:.6f} shares (${position_value:.2f})")
        
        # Print actions
        print("Actions:")
        for symbol, action in latest['symbol_actions'].items():
            action_name = ["HOLD", "BUY", "SELL"][action]
            print(f"  {symbol}: {action_name}")
        
        print(f"Reward: {latest['reward']:.6f}")
        print()
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history as a DataFrame.
        
        Returns:
            DataFrame with portfolio history
        """
        if not self.history:
            return pd.DataFrame()
        
        # Extract portfolio values
        history_df = pd.DataFrame([
            {
                "step": entry["step"],
                "portfolio_value": entry["portfolio_value"],
                "cash": entry["cash"],
                "reward": entry["reward"]
            }
            for entry in self.history
        ])
        
        # Add position columns
        for symbol in self.symbols:
            history_df[f"position_{symbol}"] = [entry["positions"][symbol] for entry in self.history]
        
        # Add action columns
        for symbol in self.symbols:
            history_df[f"action_{symbol}"] = [entry["symbol_actions"][symbol] for entry in self.history]
        
        # Add dates if available
        if self.date_col is not None:
            dates = []
            for step in history_df["step"]:
                # Find the date for this step (use the first symbol's data)
                symbol = self.symbols[0]
                symbol_df = self.df[self.df[self.symbol_col] == symbol]
                if step - 1 < len(symbol_df):
                    date = symbol_df.iloc[step - 1][self.date_col]
                else:
                    date = symbol_df.iloc[-1][self.date_col]
                dates.append(date)
            
            history_df["date"] = dates
        
        return history_df
    
    def plot_portfolio_history(self, figsize=(12, 8)):
        """
        Plot the portfolio history.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        # Get portfolio history
        history_df = self.get_portfolio_history()
        
        if len(history_df) == 0:
            return plt.figure()
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot portfolio value
        axes[0].plot(history_df["step"], history_df["portfolio_value"])
        axes[0].set_title("Portfolio Value")
        axes[0].set_ylabel("Value ($)")
        axes[0].grid(True)
        
        # Plot cash
        axes[1].plot(history_df["step"], history_df["cash"])
        axes[1].set_title("Cash")
        axes[1].set_ylabel("Value ($)")
        axes[1].grid(True)
        
        # Plot positions
        for symbol in self.symbols:
            axes[2].plot(history_df["step"], history_df[f"position_{symbol}"], label=symbol)
        
        axes[2].set_title("Positions")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Shares")
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        return fig

class MultiAssetTradingEnv(TradingEnv):
    """
    A trading environment with multiple assets and more complex actions.
    - State: [feature_vector..., cash, position_1, position_2, ...]
    - Actions: Continuous allocation of capital across assets
    - Reward: PnL change
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 1,
        symbol_col: str = "symbol",
        price_col: str = "close",
        date_col: Optional[str] = "timestamp",
        max_position_size: float = 1.0
    ):
        """
        Initialize the multi-asset trading environment.
        
        Args:
            df: DataFrame with price and feature data
            feature_cols: List of feature column names
            initial_capital: Initial capital
            transaction_cost: Transaction cost as a fraction of trade value
            reward_scaling: Scaling factor for rewards
            window_size: Number of time steps to include in the observation
            symbol_col: Name of the symbol column
            price_col: Name of the price column
            date_col: Name of the date column
            max_position_size: Maximum position size as a fraction of capital
        """
        # Initialize parent class
        super().__init__(
            df=df,
            feature_cols=feature_cols,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            reward_scaling=reward_scaling,
            window_size=window_size,
            symbol_col=symbol_col,
            price_col=price_col,
            date_col=date_col
        )
        
        # Store additional parameters
        self.max_position_size = max_position_size
        
        # Override action space: continuous allocation of capital across assets
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_symbols,),
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (allocation of capital across assets)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current prices
        prices = {}
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                prices[symbol] = symbol_df.iloc[self.current_step][self.price_col]
            else:
                # If we've reached the end of data for this symbol, use the last price
                prices[symbol] = symbol_df.iloc[-1][self.price_col]
        
        # Calculate portfolio value before action
        portfolio_value_before = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)
        
        # Normalize action to sum to 1
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate target allocation
        target_allocation = {}
        for i, symbol in enumerate(self.symbols):
            target_allocation[symbol] = action[i]
        
        # Calculate current allocation
        current_allocation = {}
        for symbol in self.symbols:
            current_allocation[symbol] = self.positions[symbol] * prices[symbol] / portfolio_value_before if portfolio_value_before > 0 else 0
        
        # Calculate trades
        trades = {}
        for symbol in self.symbols:
            # Calculate target position value
            target_value = target_allocation[symbol] * portfolio_value_before
            
            # Calculate current position value
            current_value = current_allocation[symbol] * portfolio_value_before
            
            # Calculate trade value
            trade_value = target_value - current_value
            
            # Store trade
            trades[symbol] = trade_value
        
        # Execute trades
        for symbol, trade_value in trades.items():
            price = prices[symbol]
            
            if trade_value > 0:  # Buy
                # Apply transaction cost
                trade_value *= (1 - self.transaction_cost)
                
                # Calculate position size
                position_size = trade_value / price
                
                # Update positions and cash
                self.positions[symbol] += position_size
                self.cash -= trade_value
                
            elif trade_value < 0:  # Sell
                # Calculate position size
                position_size = -trade_value / price
                
                # Apply transaction cost
                trade_value *= (1 - self.transaction_cost)
                
                # Update positions and cash
                self.positions[symbol] -= position_size
                self.cash += -trade_value
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = True
        for symbol in self.symbols:
            symbol_df = self.df[self.df[self.symbol_col] == symbol]
            if self.current_step < len(symbol_df):
                done = False
                break
        
        # Calculate portfolio value after action
        portfolio_value_after = self.cash + sum(self.positions[symbol] * prices[symbol] for symbol in self.symbols)
        
        # Calculate reward (change in portfolio value)
        reward = (portfolio_value_after - portfolio_value_before) * self.reward_scaling
        
        # Get next observation
        obs = self._get_obs() if not done else None
        
        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "symbol_actions": target_allocation,
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "reward": reward
        })
        
        # Create info dictionary
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "step": self.current_step
        }
        
        return obs, reward, done, info

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT', 'MSFT'],
        'timestamp': pd.date_range(start='2023-01-01', periods=4).tolist() * 2,
        'close': [150.0, 152.0, 151.0, 153.0, 250.0, 252.0, 251.0, 253.0],
        'ma_5': [0.1, 0.2, 0.15, 0.18, 0.2, 0.25, 0.22, 0.24],
        'rsi_14': [50, 55, 48, 52, 60, 65, 58, 62]
    }
    df = pd.DataFrame(data)
    
    # Create environment
    env = TradingEnv(
        df=df,
        feature_cols=['ma_5', 'rsi_14'],
        initial_capital=100_000,
        transaction_cost=0.001
    )
    
    # Reset environment
    obs = env.reset()
    print("Initial observation:", obs)
    
    # Take some actions
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            break
    
    # Get portfolio history
    history_df = env.get_portfolio_history()
    print("\nPortfolio History:")
    print(history_df)
