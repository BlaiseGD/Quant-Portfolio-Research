import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from src.analytics.metrics import portfolio_metrics


def maximize_sharpe(weights, returns):
    return -portfolio_metrics(weights, returns)[2]