# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from covariance import returns_corr_cov
# %%

def risk_parity_portfolio(
        returns: pd.DataFrame, 
        lw: bool = True, 
        plot: bool = False
    ) -> pd.Series:
    '''
    Input: DataFrame of returns for stocks
    Output: Weights obtained via risk parity (individual risk = <w , Sigma w> / N)
    '''

    # Get covariance
    covariance_matrix, _ = returns_corr_cov(returns, lw=lw, plot=plot)

    # Get optimal weights
    optimized_weights = risk_parity_opt(covariance_matrix)

    return optimized_weights
# %%
def risk_parity_opt(covariance_matrix: pd.DataFrame) -> pd.Series:
    
    '''
    Input: DataFrame of covariance
    Output: Weights obtained via equal risk parity (individual risk = <w , Sigma w> / N), optimized with scipy
    '''

    n = len(covariance_matrix)

    def objective_func(w):
        portfolio_var = w.T @ covariance_matrix @ w
        individual_var = w * covariance_matrix @ w
        # risk_i / total_risk - 1/n instead of risk_i - total_risk /n for numerical stability, former is scale invariant
        objective = np.sum((individual_var / portfolio_var - 1 / n) ** 2) 
        return objective

    w0 = np.ones(n)/n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0,1)]*n

    result = minimize(objective_func, w0, constraints=constraints, bounds=bounds)
    return pd.Series(result.x, index=covariance_matrix.columns)

# %%

if __name__ == '__main__':
    from data import returns
    optimized_weights = risk_parity_portfolio(returns)    
    optimized_weights.sort_values().plot(kind='barh', title=f'Risk Parity Portfolio Weights', figsize=(8,5))
    print(optimized_weights)
