# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

# %%
# # # # # # # # # # # # #
# LOAD MY PORTFOLIO (WELL PARTS OF IT)
# # # # # # # # # # # # #
def load_stock(ticker: str, period: str = '1y', graph: bool = True) -> pd.DataFrame:
    df = yf.Ticker(ticker)
    df = df.history(period=period)
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    df['Pct Change'] = df['Close'].pct_change()
    df['Up/Down'] = (df['Pct Change'].shift(-1) >= 0).astype(int)

    if graph:
        ax = df.plot(y='Close', use_index=True, title=ticker, figsize = (10,5))
        df.plot(y='Pct Change', ax= ax, secondary_y= True, alpha = 0.3, linewidth = 1, style = '--')
        ax.set_ylabel('Price')
        ax.right_ax.set_ylabel('Pct Change')
    
    return df

# Enrique's portfolio
tickers = ['AAPL', 'MSFT', 'VOO', 'SWRD', 'SMH', 'TSM', 'AMD', 'NVDA', 'BOTZ', 'PWRD']
# Hannah's portfolio
# tickers = ['IVV', 'SWRD', 'IBKR', 'BOTZ', 'PWRD', 'RPG', 'QQQ', 'VUG', 'XLK', 'VOO', 'SMH']
data = {ticker: load_stock(ticker) for ticker in tickers}

# %%
# # # # # # # # # # # # #
# COMPUTE CORRELATION MATRICES FOR CLOSING (USELESS) AND RETURNS (THE ONE THAT MATTERS)
# # # # # # # # # # # # #
closes = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in data.keys()})
closes_corr = closes.corr()

pct_changes = pd.DataFrame({ticker: data[ticker]['Pct Change'] for ticker in data.keys()})
pct_changes_corr = pct_changes.corr()

plt.figure(figsize=(10,8))
sns.heatmap(closes_corr, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Closing correlation')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(pct_changes_corr, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
plt.title('Pct Change correlation')
plt.show()
# %%
# # # # # # # # # # # # #
# IDENTIFY REDUNDANCY IN PORTFOLIO
# # # # # # # # # # # # #

high_corr = pct_changes_corr.where((pct_changes_corr > 0.8) & (pct_changes_corr < 1))
high_corr = high_corr.stack().dropna().sort_values(ascending=False)
high_corr
# %%
# # # # # # # # # # # # #
# Find portfolio with min returns risk (shorting allowed)
# # # # # # # # # # # # #

# Begin by vectorising
pct_changes = pct_changes.dropna(how = 'any')
returns = pct_changes.mean() * 252
sigma = pct_changes.cov() * 252

def portfolio_min_risk(sigma: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    ''' 
    Input is data frame of 
    covariances for a time period
    Output is min risk portfolio, long and short positions
    '''
    # Our portfolio is sum_i w_i stock_i with 
    # variance = wT Sigma w subject to 
    # [1,...,1] w = sum_i wi = 1
    ones = np.ones(len(sigma))
    sigma_inverse = np.linalg.inv(sigma)

    # Letting O = [1,...,1]^T, then the closed form
    # solution is: (Sigma^-1 * O) / (O^T Sigma^-1 O)
    min_weight = sigma_inverse @ ones
    min_weight /= ones @ sigma_inverse @ ones

    # Negative weights = short
    weights = pd.Series(min_weight, index=sigma.columns)
    if plot:
        weights.sort_values().plot(kind='barh',title='Min variance portfolio', figsize=(8,5))
    return weights


portfolio_min_risk(sigma)

# %%
# # # # # # # # # # # # #
# Find portfolio with minimum risk (long only)
# # # # # # # # # # # # #

# Begin by vectorising
pct_changes = pct_changes.dropna(how = 'any')
returns = pct_changes.mean() * 252
sigma = pct_changes.cov() * 252

def portfolio_min_risk_long_only(sigma: pd.DataFrame, min_val: float = 0, max_val: float = 1, plot: bool = True) -> pd.DataFrame:
    ''' 
    Input is data frame of 
    covariances for a time period
    Output is min risk portfolio, long only
    '''

    n = len(sigma)
    w = cp.Variable(n)

    # Set up the optimization problem. 
    # Quad_form(w, s) = w^T s w
    objective = cp.Minimize(cp.quad_form(w, sigma.values))
    constraints = [cp.sum(w) == 1, w >= min_val, w <= max_val]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = pd.Series(w.value, index=sigma.columns)
    if plot:    
        weights.sort_values().plot(kind='barh', title='Min variance portfolio, long only', figsize=(8,5))

    return weights

portfolio_min_risk_long_only(sigma)
# %%
# # # # # # # # # # # # #
# Portfolio with tradeoff of returns/risk
# # # # # # # # # # # # #

# Begin by vectorising
pct_changes = pct_changes.dropna(how = 'any')
returns = pct_changes.mean() * 252
sigma = pct_changes.cov() * 252

def portfolio_tradeoff(sigma: pd.DataFrame, lamb: float = 0.5, long_only: bool = True, min_val: float = 0, max_val: float = 1, plot: bool = True) -> pd.DataFrame:
    ''' 
    Input is data frame of 
    covariances for a time period
    Output is min risk portfolio, long only
    '''
    n = len(sigma)
    w = cp.Variable(n)

    # Optimization problem is max_w wT returns - lambda wT Sigma w
    # for constant lambda (low = prioritise returns, high = prioritise risk)
    objective = cp.Maximize(w @ returns.values - lamb * cp.quad_form(w, sigma.values))
    constraints = [cp.sum(w) == 1]
    
    if long_only:
        constraints += [w >= min_val, w <= max_val]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = pd.Series(w.value, index=sigma.columns)
    if plot:
        weights.sort_values().plot(kind='barh', title=f'Return Variance tradeoff, lambda = {lamb}', figsize=(8,5))

    return weights

portfolio_tradeoff(sigma, lamb = 0.5, long_only=False)
# %%
# # # # # # # # # # # # #
# Portfolio with tradeoff of returns/risk -- Get efficient frontier
# # # # # # # # # # # # #

# Begin by vectorising
pct_changes = pct_changes.dropna(how = 'any')
returns = pct_changes.mean() * 252
sigma = pct_changes.cov() * 252

lambdas = np.linspace(0, 5, 50)
frontier_returns = []
frontier_vols = []

for l in lambdas:
    w = portfolio_tradeoff(sigma, l, plot=False, long_only=True, max_val=0.5)
    frontier_returns.append(w @ returns.values)
    frontier_vols.append(np.sqrt(w.T @ sigma.values @ w))

import matplotlib.pyplot as plt
plt.plot(frontier_vols, frontier_returns, marker='o')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Expected Return")
plt.title("Efficient Frontier (Long-Only, Bounded)")
plt.show()

# %%
