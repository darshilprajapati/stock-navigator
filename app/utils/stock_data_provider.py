import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockDataProvider:
    @staticmethod
    def get_dummy_data(symbol, start_date="2024-01-01", end_date="2025-01-01"):
        base_prices = {
            'AAPL': 170.0,
            'AMZN': 130.0,
            'QCOM': 140.0,
            'META': 330.0,
            'NVDA': 450.0,
            'JPM': 140.0
        }
        
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        base_price = base_prices.get(symbol, 100.0)
        num_days = len(dates)
        
        # Generate price movements
        np.random.seed(hash(symbol) % 10000)  # Consistent but different for each symbol
        daily_returns = np.random.normal(0.0001, 0.02, size=num_days)
        prices = base_price * (1 + np.cumsum(daily_returns))
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, size=num_days)),
            'High': prices * (1 + abs(np.random.normal(0.005, 0.002, size=num_days))),
            'Low': prices * (1 - abs(np.random.normal(0.005, 0.002, size=num_days))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, size=num_days)
        }, index=dates)
        
        return df

dummy_provider = StockDataProvider()
