# %% 
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# %%
# # # # # # # # # # # # #
# LOAD PORTFOLIO
# # # # # # # # # # # # #
def load_stock(ticker: str, start: str, end: str, plot: bool = False) -> pd.DataFrame:
    """
    Returns data frame of given ticker, also a plot of its close and returns if plot == True
    """
    df = yf.Ticker(ticker)
    df = df.history(start=start, end=end)
    df['Return'] = df['Close'].pct_change()
    
    if plot:
        ax = df.plot(y='Close', use_index=True, title=ticker, figsize=(10,5))
        df.plot(y='Return', ax=ax, secondary_y=True, alpha=0.3, linewidth=1, style='--')
        ax.set_ylabel('Price')
        ax.right_ax.set_ylabel('Return')
    
    return df

# Personal
tickers = ['VOO', 'AAPL', 'SMH', 'TSM', 'MSFT', 'AMD', 'BOTZ', 'NLR']
# Defense
tickers += ['LMT', 'RTX', 'GD', 'NOC']
# Other big tech
tickers += ['AMZN', 'GOOG', 'TSLA', 'JPM', 'META', 'ASML', 'V', 'NVDA']
# # Additional diversification
# # Commodities:
tickers += ['GLD', 'SLV']
# 'PDBC', 'XLE']
# # International equity:
# tickers += ['EWJ', 'EWZ', 'INDA']

# START = '2021-04-13'
START = '2016-04-29'
END   = '2026-04-29' 

RETURNS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'returns.csv')
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)


if os.path.exists(RETURNS_CSV):
    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)

else:
    data = {ticker: load_stock(ticker, start=START, end=END, plot=False) for ticker in tickers}
    returns = pd.DataFrame({ticker: data[ticker]['Return'] for ticker in tickers}).dropna(how='any')
    returns.index = returns.index.tz_localize(None)
    returns.to_csv(RETURNS_CSV)
    print(f"Saved {len(returns)} rows to {RETURNS_CSV}, ending {returns.index[-1].date()}")

# %%
if __name__ == '__main__':
    print(f"Returns shape: {returns.shape}")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    for ticker in tickers:
        load_stock(ticker, start=START, end=END, plot=True)
