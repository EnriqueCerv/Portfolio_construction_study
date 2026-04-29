# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from allocators.hrp import hrp_portfolio
from allocators.mean_variance import mean_variance_portfolio
from allocators.min_variance import min_variance_portfolio
from allocators.inverse_variance import inverse_variance_portfolio
from allocators.risk_parity import risk_parity_portfolio
from stats import summary_table
# %%

# The different portfolio allocator methods
portfolio_dict = {
    'hrp': hrp_portfolio,
    'risk_parity': risk_parity_portfolio,
    'mean_variance': mean_variance_portfolio,
    'min_variance': min_variance_portfolio,
    'inverse_variance': inverse_variance_portfolio
}


def backtest_returns(
        returns: pd.DataFrame,
        lookback: int = 126,
        rebalance_freq: int = 21,
        cost_bps: float = 5,
        portfolio: str = 'hrp',
        **kwargs
):
    """
    Input: Returns, lookback window size, rebalance freqeuency, portfolio type
    Output: DataFrame of cumulative returns over entire time series
    """
    
    # Get rolling weights for each time windo
    rolling_weights = compute_rolling_weights(returns, lookback, rebalance_freq, portfolio, **kwargs)
    
    # Get optimized portfolio returns
    optimized_returns, avg_turnover = backtest_portfolio(returns, rolling_weights, rebalance_freq=rebalance_freq, cost_bps=cost_bps)

    # Get portfolio cumulative returns
    cum_optimized_returns = (1 + optimized_returns).cumprod()

    return optimized_returns, avg_turnover, cum_optimized_returns


# %%

# # # # # # # # # # # # #
# Rolling portfolio optimizer
# # # # # # # # # # # # #
def compute_rolling_weights(
        returns: pd.DataFrame,
        lookback: int,
        rebalance_freq: int,
        portfolio: str,
        **kwargs
    ):
    """
    Input: Returns, lookback window size, rebalance freqeuency, portfolio type
    Output: DataFrame of optimal portfolio weights per time window
    """

    portfolio_function = portfolio_dict[portfolio]
    weights = []
    dates = []

    for t in range(lookback, len(returns), rebalance_freq):
        window = returns.iloc[t - lookback : t].dropna()

        if len(window) < lookback:
            continue

        w = portfolio_function(window, **kwargs)

        weights.append(w)
        dates.append(returns.index[t])

    return pd.DataFrame(weights, index = dates)

# # # # # # # # # # # # #
# Backtest with turnover
# # # # # # # # # # # # #
def backtest_portfolio(
        returns: pd.DataFrame,
        weights: pd.DataFrame,
        rebalance_freq: int,
        cost_bps: float
    ):
    """
    Input: Returns, weights of stock, rebalancing frequency and turnover cost 5-10 for moderate, 20-30 for extreme
    Output: DataFrame of optimized portfolio returns as given by optimized weights in each time period
            minus the turnover cost between rebalances
    """

    portfolio_returns = []
    turnovers = []
    old_weights = np.zeros(weights.shape[1])

    for i in range(len(weights) - 1):
        start = returns.index.get_loc(weights.index[i])
        end = start + rebalance_freq

        window_returns = returns.iloc[start:end].copy()
        new_weights = weights.iloc[i].values

        # Calculate rebalance cost
        turnover = np.sum(np.abs(new_weights - old_weights))
        cost = turnover * cost_bps / 10000
        window_returns.iloc[0] = window_returns.iloc[0] - cost # only deduct from first day 

        portfolio_returns.append(window_returns @ new_weights)
        turnovers.append(turnover)
        old_weights = new_weights
    
    return pd.concat(portfolio_returns), np.mean(turnovers)

# %%
if __name__ == '__main__':
    from data import returns

    # # # # # # # # # # # #
    # Get returns and turnover
    # # # # # # # # # # # #

    cost_bps = 10
    
    # HRP
    hrp_returns, hrp_avg_turnover, hrp_cum_returns = backtest_returns(
        returns,
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='hrp',
        lw = True, 
        plot = False
        )
    
    # Equal risk parity
    risk_parity_returns, risk_parity_avg_turnover, risk_parity_cum_returns = backtest_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='risk_parity',
        lw = True,
        plot = False
    )

    # Mean-variance
    mean_variance_returns, mean_variance_avg_turnover, mean_variance_cum_returns = backtest_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='mean_variance',
        lambda_risk = 1, 
        lw = True, 
        shrink = 0.3, 
        plot = False
        )
    
    # Min-variance
    min_variance_returns, min_variance_avg_turnover, min_variance_cum_returns = backtest_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='min_variance',
        lw = True,
        plot = False
    )
    
    # Inverse variance
    inverse_variance_returns, inverse_variance_avg_turnover, inverse_variance_cum_returns = backtest_returns(
        returns, 
        lookback=126, 
        rebalance_freq=21, 
        cost_bps = cost_bps,
        portfolio='inverse_variance',
        lw = True,
        plot = False
    )

    # Trim benchmarks to backtest start date
    backtest_start = hrp_cum_returns.index[0]
    backtest_end = hrp_cum_returns.index[-1]

    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
    equal_portfolio_returns = (returns @ equal_weights)[backtest_start:backtest_end]
    equal_portfolio_cum_returns = (1 + equal_portfolio_returns).cumprod()

    voo_returns = returns['VOO'][backtest_start:backtest_end]
    voo_cum_returns = (1 + voo_returns).cumprod()


    # # # # # # # # # # # #
    # Get summary table comparing strats
    # # # # # # # # # # # #
         
    # Compile {strat : strat_return } dictionary
    returns_dict = {
        'HRP': hrp_returns,
        'Risk Parity': risk_parity_returns,
        'Mean-Variance': mean_variance_returns,
        'Min-Variance': min_variance_returns,
        'Inverse-Variance': inverse_variance_returns,
        'Equal Weight': equal_portfolio_returns,
        'VOO': voo_returns,
    }

    # Compile {strat : turnover} dict
    turnovers_dict = {
        'HRP': hrp_avg_turnover,
        'Risk Parity': risk_parity_avg_turnover,
        'Mean-Variance': mean_variance_avg_turnover,
        'Min-Variance': min_variance_avg_turnover,
        'Inverse-Variance': inverse_variance_avg_turnover,
        'Equal Weight': 0,
        'VOO': 0,
    }

    # Get stats:
    stats_table, bootstrap_sharpes = summary_table(returns_dict=returns_dict, turnovers_dict=turnovers_dict)
    print(stats_table.T)

    # # # # # # # # # # # #
    # Plot and save comparisons
    # # # # # # # # # # # #
    plt.figure(figsize=(12, 6))
    hrp_cum_returns.plot(label='HRP')
    risk_parity_cum_returns.plot(label='Risk Parity')
    mean_variance_cum_returns.plot(label='Mean-variance')
    min_variance_cum_returns.plot(label='Min-variance')
    inverse_variance_cum_returns.plot(label='Inverse-variance')
    equal_portfolio_cum_returns.plot(label='Equal weight')
    voo_cum_returns.plot(label='VOO')
    plt.legend()
    plt.title('Out-of-sample backtest')
    plt.xticks(rotation=5)
    plt.grid()
    plt.tight_layout()
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'oos_backtest_comparison.png'))
    plt.show()



    # # # # # # # # # # # #
    # Plot bootstraps
    # # # # # # # # # # # #
    point_sharpes = {strat: row['Sharpe'] for strat, row in stats_table.iterrows()}

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    
    for (strat, sharpes), color in zip(bootstrap_sharpes.items(), colors):
        sns.kdeplot(sharpes, ax=ax, color=color, linewidth=2, label=strat)
        
        # Add vertical line for point estimate
        if point_sharpes and strat in point_sharpes:
            ax.axvline(point_sharpes[strat], color=color, linewidth=1.5, 
                      linestyle='--', alpha=0.7)
        
        # Shade 95% CI
        ci_lower = np.percentile(sharpes, 2.5)
        ci_upper = np.percentile(sharpes, 97.5)
        
        # Get KDE values for shading
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sharpes)
        x_range = np.linspace(ci_lower, ci_upper, 200)
        ax.fill_between(x_range, kde(x_range), alpha=0.1, color=color)

    ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bootstrap Distribution of Sharpe Ratios\n(dashed lines = point estimates, shaded = 95% CI)', fontsize=13)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'bootstrap_sharpe_comparison.png'))
    plt.show()