from src.data.loaders import load_assets
from src.models.optimization import maximize_sharpe
from src.models.constraints import weight_sum_constraint
from src.analytics.metrics import portfolio_metrics
from scipy.optimize import minimize
import numpy as np

rf = 0.037
assets, returns = load_assets()  # implement in loaders.py

bounds = [(0.03, 0.33) for _ in len(assets)]
constraints = {"type": "eq", "fun": weight_sum_constraint}
default_weights = np.full(len(assets), 1/len(assets))

result = minimize(maximize_sharpe, default_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = result.x

print(portfolio_metrics(opt_weights, returns, rf))