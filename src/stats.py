# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# %%
def summary_table(returns_dict: dict, turnovers_dict: dict = None,
                   benchmark: str = 'VOO',
                   n_boot: int = 10000) -> pd.DataFrame:
    '''
    The main entry point. Takes {'HRP': hrp_returns, 'MV': mv_returns, ...}
    Returns a DataFrame with all metrics + bootstrap CIs + DSR for each method.
    This replaces the manual stats assembly at the bottom of backtest.py.
    '''

    rows = {}
    bootstrap_sharpes = {}
    
    n = len(returns_dict)
    benchmark_returns = returns_dict[benchmark]

    for strat, strat_returns in returns_dict.items():
        avg_turnover = turnovers_dict[strat] if turnovers_dict else None

        # Get all the basic stats
        stats = performance_stats(strat_returns, avg_turnover)

        # Get bootstrap CI intervals for SR
        boot = block_bootstrap_sharpe(strat_returns, n_boot=n_boot)
        stats['Sharpe CI lower'] = boot['ci_lower']
        stats['Sharpe CI upper'] = boot['ci_upper']
        bootstrap_sharpes[strat] = boot['bootstrap_distribution']

        # Get Deflated Sharpe Ratio
        daily_sharpe = strat_returns.mean() / strat_returns.std()
        stats['DSR'] = deflated_sharpe_ratio(
            observed_sharpe=daily_sharpe,
            n_trials=n,
            t_days=len(strat_returns),
            skew=stats['Skew'],
            kurtosis=stats['Kurtosis']
        )

        # Get bootstrap CI intervals for SR difference against benchmark, for all strats != benchmark
        if strat == benchmark:
            stats['Mean Sharpe diff'] = 0
            stats['Sharpe diff CI lower'] = np.nan
            stats['Sharpe diff CI upper'] = np.nan
            stats['Pct wins vs Benchmark'] = np.nan
        else:
            boot_diff = sharpe_difference_test(strat_returns, benchmark_returns, n_boot=n_boot)
            stats['Mean Sharpe diff'] = boot_diff['mean_diff']
            stats['Sharpe diff CI lower'] = boot_diff['ci_lower']
            stats['Sharpe diff CI upper'] = boot_diff['ci_upper']
            stats['Pct wins vs Benchmark'] = boot_diff['pct_returns1_wins']

        rows[strat] = stats

    return pd.DataFrame(rows).T, bootstrap_sharpes

# %%
def performance_stats(returns: pd.Series, avg_turnover: float = None) -> pd.Series:
    '''
    Outputs: Annual Return, Annual Vol, Sharpe, Max Drawdown, Calmar,
             Total time underwater, Max time underwater, Avg Turnover (annualized), Skew, Kurtosis
    '''

    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    cum_returns = (1 + returns).cumprod()

    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()

    is_underwater = cum_returns < cum_returns.cummax()
    time_underwater = is_underwater.astype(int).groupby(
        (is_underwater != is_underwater.shift()).cumsum()
        ).sum().max()

    return pd.Series({
        'Annual Return': annual_return,
        'Annual Vol': returns.std() * np.sqrt(252),
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown,
        'Calmar': annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan,
        # 'Total Time Underwater': is_underwater.sum(), # removed because it overcounts
        'Time Underwater': time_underwater,
        'Avg Turnover': avg_turnover * 12 if avg_turnover else 0,
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })

def block_bootstrap_sharpe(returns: pd.Series, n_boot: int = 10000, 
                            block_size: int = 21, ci: float = 95, 
                            seed: int = 42) -> dict:
    '''
    Outputs: {'sharpe': float, 'ci_lower': float, 'ci_upper': float,
              'bootstrap_distribution': np.array}
    '''

    n = len(returns)
    returns_arr = returns.values
    sharpes = np.zeros(n_boot)
    np.random.seed(seed)

    # for n_sample in range(n_boot):
    #     sample = []
    #     while len(sample) < n:
    #         start = np.random.randint(0, n)
    #         block = [(start + i) % n for i in range(block_size)]
    #         sample.extend(block)
        
    #     idx = sample[:n]

    #     returns_sample = returns_arr[idx]
    #     returns_sharpe = (returns_sample.mean() / returns_sample.std()) * np.sqrt(252)

    #     sharpes[n_sample] = returns_sharpe

    # Vectorisation of the above -- a lot faster
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))            # (n_boot, n_blocks)
    offsets = np.arange(block_size)                                 # (block_size,)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n     # (n_boot, n_blocks, block_size)
    indices = indices.reshape(n_boot, -1)[:, :n]                    # (n_boot, n)
    samples = returns_arr[indices]                                  # (n_boot, n)
    sharpes = samples.mean(axis=1) / samples.std(axis=1) * np.sqrt(252)
    
    return {
        'Sharpe': (returns.mean() / returns.std()) * np.sqrt(252),
        'ci_lower': np.percentile(sharpes, (100 - ci) / 2),
        'ci_upper': np.percentile(sharpes, (100 + ci) / 2),
        'bootstrap_distribution': sharpes
    }

def sharpe_difference_test(returns1: pd.Series, returns2: pd.Series, 
                            n_boot: int = 10000, block_size: int = 21,
                            ci: float = 95, seed: int = 42) -> dict:
    '''
    Returns: {'mean_diff': float, 'ci_lower': float, 'ci_upper': float,
              'pct_r1_wins': float}.
    '''

    n = len(returns1)
    returns1_arr = returns1.values
    returns2_arr = returns2.values
    sharpes_diff = np.zeros(n_boot)
    np.random.seed(seed)

    # for n_sample in range(n_boot):
    #     sample = []
    #     while len(sample) < n:
    #         start = np.random.randint(0, n)
    #         block = [(start + i) % n for i in range(block_size)]
    #         sample.extend(block)
        
    #     idx = sample[:n]

    #     returns1_sample = returns1_arr[idx]
    #     returns1_sharpe = (returns1_sample.mean() / returns1_sample.std()) * np.sqrt(252)

    #     returns2_sample = returns2_arr[idx]
    #     returns2_sharpe = (returns2_sample.mean() / returns2_sample.std()) * np.sqrt(252)

    #     sharpes_diff[n_sample] = returns1_sharpe - returns2_sharpe
    
    # Vectorisation of the above -- a lot faster
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)

    starts = rng.integers(0, n, size=(n_boot, n_blocks))            # (n_boot, n_blocks)
    offsets = np.arange(block_size)                                 # (block_size,)

    indices = (starts[:, :, None] + offsets[None, None, :]) % n     # (n_boot, n_blocks, block_size)
    indices = indices.reshape(n_boot, -1)[:, :n]                    # (n_boot, n)

    samples1 = returns1_arr[indices]                                  # (n_boot, n)
    sharpes1 = samples1.mean(axis=1) / samples1.std(axis=1) * np.sqrt(252)

    samples2 = returns2_arr[indices]
    sharpes2 = samples2.mean(axis=1) / samples2.std(axis=1) * np.sqrt(252)

    sharpes_diff = sharpes1 - sharpes2
    
    return {
        'mean_diff': sharpes_diff.mean(),
        'ci_lower': np.percentile(sharpes_diff, (100 - ci) / 2),
        'ci_upper': np.percentile(sharpes_diff, (100 + ci) / 2),
        'pct_returns1_wins': (sharpes_diff > 0).sum() / n_boot
    }

def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int,
                           t_days: int, skew: float, 
                           kurtosis: float) -> float:
    '''
    Input: daily (not annualized) sharpe, number of allocators tried (5), t_days = len(returns)
    Returns: probability that the observed Sharpe is significant after
             multiple-testing correction (Bailey & López de Prado, 2014).
    '''

    em = 0.5772
    var_sharpe = (1 / t_days) * (1 - skew * observed_sharpe + (kurtosis + 2) / 4 * observed_sharpe ** 2)
    sharpe_null = np.sqrt(var_sharpe) * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * np.exp(1))))
    dsr = (observed_sharpe - sharpe_null) * np.sqrt(t_days)
    dsr /= np.sqrt(1 - skew * observed_sharpe + (kurtosis + 2) / 4 * observed_sharpe ** 2)

    return norm.cdf(dsr)


# %%
