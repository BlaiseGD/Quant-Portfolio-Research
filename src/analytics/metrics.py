import pandas as pd
import numpy as np
from analytics.returns import compute_log_returns
from analytics.risk import compute_annual_covariance

def portfolio_metrics(weights, returns, rf):
    mu = returns.mean()*252
    cov = returns.cov() * 252
    exp_ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (exp_ret - rf) / vol
    return exp_ret, vol, sharpe