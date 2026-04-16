# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn.covariance import LedoitWolf

# %%
# # # # # # # # # # # # #
# Get covariance and shrinked covariance
# # # # # # # # # # # # #

def returns_corr_cov(returns: pd.DataFrame, lw: bool, plot: bool):
    """
    Returns the daily covariance and correlation matrices, with or without Ledoit-Wolfe shrinkage
    """

    if not lw:
        returns_corr = returns.corr()
        sigma = returns.cov()
    
    else:
        sigma = pd.DataFrame(LedoitWolf().fit(returns.values).covariance_, index=returns.columns, columns=returns.columns)
        std = np.sqrt(np.diag(sigma))
        returns_corr = pd.DataFrame(sigma / np.outer(std, std), index=returns.columns, columns=returns.columns)

    if plot:
        plt.figure(figsize=(10,8))
        sns.heatmap(returns_corr, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title('Return correlation')
        plt.show()
    
    return sigma, returns_corr


def returns_mean(returns: pd.DataFrame, shrink: float, plot: bool):
    """
    Returns the average daily returns, with shrinkage in (0, 1) (1 is no shrinkage)
    """

    mu_raw = returns.mean()
    mu_benchmark = returns['VOO'].mean() # Skew towards benchmark, but can also be mu_raw.mean() instead
    mu = shrink * mu_raw + (1 - shrink) * mu_benchmark

    if plot:
        mu.sort_values().plot(kind="barh", figsize=(8,5), title=f"Expected returns, shrink {shrink}")
        plt.show()

    return mu

# %%
if __name__ == '__main__':

    from src.data import returns

    lw_sigma, lw_corr = returns_corr_cov(returns, lw=True, plot=True)
    sigma, corr = returns_corr_cov(returns, lw=False, plot=True)
    print('Conditional number of return covariance: ', np.linalg.cond(sigma))
    print('Conditional number of return LW covariance: ', np.linalg.cond(lw_sigma))
# %%
