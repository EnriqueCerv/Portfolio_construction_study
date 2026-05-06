# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

from covariance import returns_corr_cov
from mean_variance import mean_variance
# %%
def black_litterman_portfolio(
        returns: pd.DataFrame, 
        lambda_risk: float,
        tau: float,
        Q_view_vector: np.ndarray,
        P_view_matrix: np.ndarray,
        diagonal_Omega: bool = False,
        easy_Omega: bool = True,
        plot: bool = False
    ) -> pd.Series:

    '''
    Input: DataFrame of raw returns, risk tradeoff lambda, trust in prior tau, Kx1 matrix of views, KxN matrix of views picking,
    Output: Weights obtained via markowitz mean-variance optimization as per
            argmax_w   < w, mu > - lambda < w, Cov w > / 2 for posterior mu and cov
    '''
    
    # Get correlation and expected returns
    cov_matrix, _ = returns_corr_cov(returns, lw=False, plot=plot)
    tickers = list(cov_matrix.columns)

    # The prior should be the the normalised market cap per day, two issues with that:
    # 1: yfinance only stores last market cap, or last shares outsdanding (marketCap = shares * close) so we'd need to backtrack 
    # last available shares * df['Close'] to get an approximation of past market caps
    # 2: etfs like voo do not have shares outstanding so cannot get above
    # Since this is just a prior, we can have the prior be the equal weight portfolio --> Important thing is the views
    market_weights = pd.Series(np.ones(len(tickers)) / len(tickers), index=cov_matrix.columns)
    market_returns = get_equilibrium_returns(cov_matrix=cov_matrix, market_weights=market_weights, lambda_risk=lambda_risk)

    # Get posteriors
    cov_BL, exp_returns_BL = BL_posterior(
        market_returns=market_returns, 
        cov_matrix=cov_matrix, 
        tau=tau, 
        Q_view_vector=Q_view_vector, 
        P_view_matrix=P_view_matrix,
        easy_Omega=easy_Omega,
        diagonal_Omega=diagonal_Omega
    )
    cov_BL = pd.DataFrame(cov_BL, index=cov_matrix.index, columns=cov_matrix.columns)
    exp_returns_BL = pd.Series(exp_returns_BL, index=market_returns.index)

    # Annualise so lambda_risk is on an interpretable scale
    cov_BL_annual = cov_BL * 252
    exp_returns_BL_annual = exp_returns_BL * 252

    # Get optimized weights 
    optimized_weights = mean_variance(cov_BL_annual, exp_returns_BL_annual, lambda_risk, plot=plot)

    return optimized_weights
    
# %%
# # # # # # # # 
# Get posteriors
# # # # # # # #
def get_equilibrium_returns(
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series,
        lambda_risk: float
    ) -> pd.Series:
    '''
    Input: Covariance matrix, equilibrium portfolio weights, risk
    Output: Implied market equilibrium returns Pi = 2lambda @ Sigma @ w_market, solution to 
            max_w <w, mu> - lambda <w, Sigma w> 
    '''

    return 2 * lambda_risk * cov_matrix @ market_weights


def BL_posterior(
        market_returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        tau: float,
        Q_view_vector: np.ndarray,
        P_view_matrix: np.ndarray,
        easy_Omega: bool,
        diagonal_Omega: bool
    ) -> tuple:
    '''
    Input: Prior returns and covariance, trust in prior tau, views
    Output: Posterior covariance and returns
    '''

    # Get uncertainty in each view (and its inverse)
    Omega = views_variance_easy(
        cov_matrix=cov_matrix, 
        tau=tau, 
        P_view_matrix=P_view_matrix, 
        easy_Omega=easy_Omega,
        diagonal_Omega=diagonal_Omega)
    Omega_inv = np.linalg.inv(Omega)

    # Get posterior covariance
    cov_BL = np.linalg.inv(tau * cov_matrix) + P_view_matrix.T @ Omega_inv @ P_view_matrix
    cov_BL = np.linalg.inv(cov_BL)

    # Get posterior returns
    exp_returns_BL = cov_BL @ (np.linalg.inv(tau * cov_matrix) @ market_returns + P_view_matrix.T @ Omega_inv @ Q_view_vector)

    return cov_BL, exp_returns_BL


def views_variance_easy(
        cov_matrix: pd.DataFrame,
        tau: float,
        P_view_matrix: np.ndarray,
        easy_Omega: bool,
        diagonal_Omega: bool
) -> np.ndarray:
    
    '''
    Input: Prior covariance, trust in prior tau, views, Omega calculation method, diagonal omega
    Use diagonal omega if views are independent
    Output: Uncertainty in views matrix Omega as in Section 2.3 of Idzorek (2005) if easy, section 3 else
    '''
    if easy_Omega:
        Omega = tau * P_view_matrix @ cov_matrix @ P_view_matrix.T
        return Omega if not diagonal_Omega else np.diag(np.diag(Omega))