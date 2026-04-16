# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

from covariance import returns_corr_cov
# %%
def min_variance_portfolio(
        returns: pd.DataFrame, 
        lw: bool = True, 
        plot: bool = False
    ) -> pd.Series:
    '''
    Input: DataFrame of correlations, returns for stocks (option for ledoit-wolf shrinkage)
    Output: Weights obtained via markowitz min-variance optimization as per
    min_w   < w, Corr w > / 2
    '''
    
    # Get correlation
    cov_matrix, _ = returns_corr_cov(returns, lw=lw, plot=plot)

    # Get optimized weights 
    optimized_weights = min_variance(cov_matrix, plot=plot)

    return optimized_weights


# %%
def min_variance(cov_matrix: pd.DataFrame, plot: bool) -> pd.Series:
    '''
    Input: DataFrame of correlations, returns for stocks (option for ledoit-wolf shrinkage)
    Output: Weights obtained via convex optimization of
    max_w   < w, Corr w > / 2
    '''

    # Get objective function
    n = len(cov_matrix)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

    # Optimize
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    optimized_weights = pd.Series(w.value, index=cov_matrix.columns)
    optimized_weights[optimized_weights.abs() < 1e-6] = 0

    if plot:
        optimized_weights.sort_values().plot(kind='barh', title='Min Risk Portfolio weights', figsize=(8,5))
    
    return optimized_weights


# %%

if __name__ == '__main__':
    from src.data import returns
    optimized_weights = min_variance_portfolio(returns)
    print(optimized_weights)
    optimized_weights.sort_values().plot(kind='barh', title='Min Risk Portfolio weights', figsize=(8,5))