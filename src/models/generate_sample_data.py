"""
Generate sample data for testing the model and backtester.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2025-04-01',
    end_date='2025-04-17',
    output_dir='data/raw/ticks'
):
    """
    Generate sample tick data for testing.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
    """
    logger.info(f"Generating sample data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Convert dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate daily timestamps
    timestamps = []
    current_dt = start_dt
    while current_dt <= end_dt:
        # Generate 10 timestamps per day (hourly during trading hours)
        for hour in range(9, 19):
            timestamps.append(current_dt.replace(hour=hour, minute=0, second=0))
        
        current_dt += timedelta(days=1)
    
    # Generate data for each symbol
    all_data = []
    
    for symbol in symbols:
        # Set base price for each symbol
        if symbol == 'AAPL':
            base_price = 150.0
        elif symbol == 'MSFT':
            base_price = 250.0
        elif symbol == 'GOOGL':
            base_price = 2000.0
        else:
            base_price = 100.0
        
        # Generate price series with random walk
        prices = [base_price]
        for i in range(1, len(timestamps)):
            # Random price change with some momentum
            price_change = np.random.normal(0, 1) * base_price * 0.01  # 1% standard deviation
            
            # Add some momentum (autocorrelation)
            if i > 1:
                prev_change = prices[-1] - prices[-2]
                price_change += prev_change * 0.2  # 20% momentum
            
            # Add the change to the previous price
            new_price = max(0.1, prices[-1] + price_change)  # Ensure price is positive
            prices.append(new_price)
        
        # Generate OHLC data
        for i, timestamp in enumerate(timestamps):
            close_price = prices[i]
            
            # Generate high, low, open with some randomness
            high_price = close_price * (1 + random.random() * 0.02)  # Up to 2% higher
            low_price = close_price * (1 - random.random() * 0.02)   # Up to 2% lower
            open_price = low_price + random.random() * (high_price - low_price)
            
            # Generate volume
            volume = int(random.random() * 10000) + 1000
            
            # Create tick record
            tick = {
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            all_data.append(tick)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Add date column for partitioning
    df['date'] = df['timestamp'].dt.date
    
    # Save data by date
    for date, group in df.groupby('date'):
        date_str = date.strftime("%Y-%m-%d")
        date_dir = os.path.join(output_dir, date_str)
        
        # Create directory if it doesn't exist
        os.makedirs(date_dir, exist_ok=True)
        
        # Save to Parquet
        output_path = os.path.join(date_dir, f"ticks_{date_str.replace('-', '')}.parquet")
        group.to_parquet(output_path, index=False)
        
        logger.info(f"Saved {len(group)} records to {output_path}")
    
    logger.info(f"Generated {len(df)} records total")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for testing")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols")
    parser.add_argument("--start-date", type=str, default="2025-04-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-04-17", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/raw/ticks", help="Output directory")
    
    args = parser.parse_args()
    
    generate_sample_data(
        symbols=args.symbols.split(","),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir
    )
