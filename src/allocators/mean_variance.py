# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

from covariance import returns_corr_cov, returns_mean
# %%
def mean_variance_portfolio(
        returns: pd.DataFrame, 
        lambda_risk: float, 
        lw: bool = True, 
        shrink: float = 0.3, 
        plot: bool = False
    ) -> pd.Series:
    '''
    Input: DataFrame of correlations, returns for stocks (option for ledoit-wolf and mean return shrinkage)
    Output: Weights obtained via markowitz mean-variance optimization as per
    max_w   < w, mu > - lambda < w, Corr w > / 2
    '''
    
    # Get correlation and expected returns
    cov_matrix, _ = returns_corr_cov(returns, lw=lw, plot=plot)
    exp_returns = returns_mean(returns, shrink=shrink, plot=plot)

    # Annualise so lambda_risk is on an interpretable scale
    cov_matrix_annual = cov_matrix * 252
    exp_returns_annual = exp_returns * 252

    # Get optimized weights 
    optimized_weights = mean_variance(cov_matrix_annual, exp_returns_annual, lambda_risk, plot=plot)

    return optimized_weights


# %%
def mean_variance(cov_matrix: pd.DataFrame, exp_returns: pd.DataFrame, lambda_risk: float, plot: bool) -> pd.Series:
    '''
    Input: DataFrame of correlations, returns for stocks (option for ledoit-wolf and mean return shrinkage)
    Output: Weights obtained via convex optimization of
    max_w   < w, mu > - lambda < w, Corr w > / 2
    '''

    # Get objective function
    n = len(cov_matrix)
    w = cp.Variable(n)

    objective = cp.Maximize(w @ exp_returns.values - lambda_risk * cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]

    # Optimize
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    optimized_weights = pd.Series(w.value, index=cov_matrix.columns)
    optimized_weights[optimized_weights.abs() < 1e-6] = 0

    if plot:
        optimized_weights.sort_values().plot(kind='barh', title=f'Mean-Variance Portfolio Weights, lambda = {lambda_risk}', figsize=(8,5))
    
    return optimized_weights


# %%

if __name__ == '__main__':
    from src.data import returns
    lambda_risk = 1
    optimized_weights = mean_variance_portfolio(returns, lambda_risk=lambda_risk)    
    optimized_weights.sort_values().plot(kind='barh', title=f'Mean-Variance Portfolio Weights, lambda = {lambda_risk}', figsize=(8,5))
    print(optimized_weights)
