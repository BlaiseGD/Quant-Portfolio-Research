import numpy as np

def compute_log_returns(prices):
    return np.log(prices).diff().dropna()