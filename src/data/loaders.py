import yfinance as yf
import numpy as np
from src.analytics.returns import compute_log_returns

def load_assets():
    #testing assets
    #ADD REAL ASSETS LATER
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'LMT', 'GLD', 'PLTR', 'TSLA', 'NVDA']
    data = yf.download(assets, start='2024-01-01')["Adj Close"]
    #getting log returns
    returns = compute_log_returns(data)
    return assets, returns