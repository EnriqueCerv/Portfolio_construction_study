# %%
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import seaborn as sns
# %%
# # # # # # # # # # # # #
# LOAD PORTFOLIO
# # # # # # # # # # # # #
def load_stock(ticker: str, period: str = 'max', plot: bool = True) -> pd.DataFrame:
    """
    Returns data frame of given ticker, also a plot of its close and returns if plot == True
    """
    df = yf.Ticker(ticker)
    df = df.history(period=period)
    df['Return'] = df['Close'].pct_change()
    df['Up/Down'] = (df['Return'].shift(-1) >= 0).astype(int) # Technically do not need for past portfolio optimization, but in case you want to have an ML algo

    if plot:
        ax = df.plot(y='Close', use_index=True, title=ticker, figsize = (10,5))
        df.plot(y='Return', ax= ax, secondary_y= True, alpha = 0.3, linewidth = 1, style = '--')
        ax.set_ylabel('Price')
        ax.right_ax.set_ylabel('Return')
    
    return df

# Enrique global tickers, need to separate by exchange
# tickers = ['VOO', 'CSP1.L', 'VWRA.L', 'AAPL', 'MSFT', 'SWRD.L', 'SMH', 'TSM', 'AMD', 'BOTZ', 'PWRD', 'NVDA', 'NUCL.L']
tickers = ['VOO', 'AAPL', 'SMH', 'TSM', 'NVDA', 'MSFT', 'AMD', 'BOTZ', 'PWRD', 'NLR']

# defense_tickers = ['LMT', 'RTX', 'GD', 'NOC']
# tickers += defense_tickers

# big_companies = ['AMZN', 'GOOG', 'TSLA', 'JPM', 'META']
# tickers += big_companies

# tickers = ['IVV', 'IBKR', 'BOTZ', 'PWRD', 'RPG', 'QQQ', 'VUG', 'XLK', 'VOO', 'SMH']
# tickers = ['IONQ', 'IBKR', 'MSFT', 'AAPL', 'NVDA', 'TSM', 'SMH', 'VOO', 'GOOG']

data = {ticker: load_stock(ticker, period='5y', plot=True) for ticker in tickers}
returns = pd.DataFrame({ticker: data[ticker]['Return'] for ticker in data.keys()}).dropna(how='any')
# %%
# # # # # # # # # # # # #
# Get covariance and shrinked covariance
# # # # # # # # # # # # #

def returns_corr_cov(returns: pd.DataFrame, lw: bool = True, plot: bool = True):
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
    
    return sigma.dropna(), returns_corr.dropna()

lw_sigma, lw_corr = returns_corr_cov(returns, lw=True)
sigma, corr = returns_corr_cov(returns, lw=False)
print('Conditional number of return covariance: ', np.linalg.cond(sigma))
print('Conditional number of return LW covariance: ', np.linalg.cond(lw_sigma))

# %%
# # # # # # # # # # # # #
# Get mean and shrinked mean
# # # # # # # # # # # # #

def returns_mean(returns: pd.DataFrame, shrink = None, plot: bool = True):
    """
    Returns the average daily returns, with or without shrinkage
    """

    if shrink is None:
        mu = returns.mean()
    
    else:
        mu_raw = returns.mean()
        mu_benchmark = returns['VOO'].mean() # Skew towards benchmark, but can also be mu_raw.mean() instead
        mu = shrink * mu_raw + (1 - shrink) * mu_benchmark

    if plot:
        mu.sort_values().plot(kind="barh", figsize=(8,5), title=f"Expected returns, shrink {shrink}")
        plt.show()

    return mu

shrunk_mu = returns_mean(returns, shrink=0.3)
mu = returns_mean(returns)
# %%
# # # # # # # # # # # # #
# Identify correlated tickers
# # # # # # # # # # # # #
upper_triang_mask = np.array([[j > i for j in range(len(corr))] for i in range(len(corr))])
high_corr = corr.where(upper_triang_mask & (corr >= 0.8))
high_corr = high_corr.stack().dropna()
high_corr
# %%
# # # # # # # # # # # # #
# Portfolio optimization (long only)
# # # # # # # # # # # # #

def portfolio_optimize(
        sigma: pd.DataFrame,
        mu: pd.DataFrame = None,
        lambda_risk: float = None,
        min_w:float = 0, 
        max_w: float = 1, 
        plot: bool = True
) -> pd.DataFrame:
    """
    Returns optimal portfolio weights for data in window as implied by sigma/mu
    """
    n = len(sigma)
    w = cp.Variable(n)

    if lambda_risk is not None:
        if mu is None:
            return 'Need expected return vector'
        else:
            objective = cp.Maximize(w @ mu.values - lambda_risk * cp.quad_form(w, sigma.values))
    else:
        objective = cp.Minimize(cp.quad_form(w, sigma.values))
    
    constraints = [cp.sum(w) == 1, w >= min_w, w <= max_w]
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    weights = pd.Series(w.value, index=sigma.columns)

    if plot:
        weights.sort_values().plot(kind='barh', title=f'Return-Risk tradeoff, lambda = {lambda_risk}', figsize=(8,5))
    
    return weights

portfolio_optimize(lw_sigma, shrunk_mu, lambda_risk=1)

# %%
# # # # # # # # # # # # #
# Rolling portfolio optimizer
# # # # # # # # # # # # #


def rolling_min_var_weights(
        returns: pd.DataFrame,
        lookback: int = 126,
        rebalance_freq: int = 21,
        shrink: float = 0.3,
        lambda_risk: float = None,
        min_w:float = 0, 
        max_w: float = 1, 
):
    """
    Returns DataFrame of optimal portfolio weights per each time window
    """

    weights = []
    dates = []

    for t in range(lookback, len(returns), rebalance_freq):
        window = returns.iloc[t - lookback : t].dropna()

        if len(window) < lookback:
            continue

        sigma, _ = returns_corr_cov(window, plot = False)
        mu = returns_mean(window, shrink=shrink, plot = False)
        w = portfolio_optimize(sigma, mu, lambda_risk=lambda_risk, min_w=min_w, max_w=max_w, plot=False)

        weights.append(w)
        dates.append(returns.index[t])

    return pd.DataFrame(weights, index = dates)

W = rolling_min_var_weights(
    returns=returns, shrink=0.3, lambda_risk=0.5
    )
# %%
# # # # # # # # # # # # #
# Backtest
# # # # # # # # # # # # #

def backtest_portfolio(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        rebalance_freq: int = 21
):
    """
    Returns DataFrame of optimized portfolio returns as given by optimized weights in each time period
    """
    portfolio_returns = []
    for i in range(len(weights) - 1):
        start = returns.index.get_loc(weights.index[i])
        end = start + rebalance_freq

        slice = returns.iloc[start:end]
        w = weights.iloc[i].values

        portfolio_returns.append(slice @ w)
    
    return pd.concat(portfolio_returns)

opt_portfolio_returns = backtest_portfolio(returns, W)
# equal_portfolio_returns = returns.mean(axis=1)
equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
equal_portfolio_returns = returns @ equal_weights
voo_returns = returns['VOO']
# %%
# # # # # # # # # # # # #
# Eval backtest
# # # # # # # # # # # # #
def performance_stats(r):
    return pd.Series({
        "Annual Return": (1 + r).prod() ** (252 / len(r)) - 1,
        "Annual Vol": r.std() * np.sqrt(252),
        "Sharpe": (r.mean() / r.std()) * np.sqrt(252),
        "Max Drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    })

pd.DataFrame({
    "Min Var (Shrunk)": performance_stats(opt_portfolio_returns),
    "Equal Weight": performance_stats(equal_portfolio_returns),
    "VOO": performance_stats(voo_returns)
})

# %%
plt.figure(figsize=(10,6))
(1 + opt_portfolio_returns).cumprod().plot(label="Min Var (Shrunk)")
(1 + equal_portfolio_returns).cumprod().plot(label="Equal Weight")
(1 + voo_returns).cumprod().plot(label="VOO")
plt.legend()
plt.title("Out-of-Sample Backtest")
plt.show()