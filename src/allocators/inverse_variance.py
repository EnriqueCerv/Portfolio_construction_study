# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from covariance import returns_corr_cov
# %%
# %%

def inverse_variance_portfolio(
        returns: pd.DataFrame, 
        lw: bool = True, 
        plot: bool = False
    ) -> pd.Series:
    '''
    Input: DataFrame of returns for stocks
    Output: Weights are simply 1/sigma_i^2 / sum_j 1/sigma_j^2
    '''

    # Get covariance
    covariance_matrix, _ = returns_corr_cov(returns, lw=lw, plot=plot)

    # Get optimal weights
    inv_variances = 1 / np.diag(covariance_matrix)
    total_inv_variance = inv_variances.sum()
    optimized_weights = inv_variances / total_inv_variance

    return pd.Series(optimized_weights, index=covariance_matrix.columns)

# %%

if __name__ == '__main__':
    from src.data import returns
    optimized_weights = inverse_variance_portfolio(returns)    
    optimized_weights.sort_values().plot(kind='barh', title=f'Inverse Variance Portfolio Weights', figsize=(8,5))
    print(optimized_weights)