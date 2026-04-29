# Portfolio Construction Study

Seven portfolio construction methods were compared on a 22-asset universe ---including equities, commodities (GLD, SLVR) and etfs--- over the last 10 years, with transaction costs and turnover penalties. 
We compared HRP, Risk Parity, Mean-Variance, Min-Variance, and Inverse-Variance against the benchmarks of an Equal Weight portfolio and the full S&P 500 (VOO).

---

## Out-of-Sample Backtest

Across six allocation methods on a 22-asset US equity universe (2015–2026, 10bps transaction costs), risk-parity-style methods (Risk Parity, HRP, Equal Weight) achieved the highest Sharpe ratios (1.1–1.3), while mean-variance optimization performed worst (Sharpe 0.7) due to high turnover consuming returns.

![Out-of-sample cumulative returns](results/oos_backtest_comparison.png)

---

## Turnover & Transaction Costs

Mean-Variance incurs turnover of ~10.1x per rebalance — an order of magnitude above every other method — which directly explains its poor risk-adjusted performance despite reasonable gross returns. Inverse-Variance and Risk Parity are the most turnover-efficient optimised methods, approaching the zero-turnover baseline of Equal Weight.

![Average turnover by strategy](results/turnover_comparison.png)

---

## Statistical Significance

With 10 years of data, Risk Parity, HRP, Equal Weight, and Inverse-Variance all achieved Sharpe ratios statistically distinguishable from VOO at 95% confidence (block bootstrap, B=10,000), with % wins vs benchmark exceeding 99% for the top three methods.

After multiple-testing correction (Deflated Sharpe Ratio), Risk Parity (DSR 0.978), Equal Weight (DSR 0.978), and HRP (DSR 0.973) all exceed the conventional 0.95 significance threshold — a meaningful improvement over the 5-year results, confirming that sample length is a binding constraint for strategy evaluation. Bootstrap confidence intervals on individual Sharpe ratios still span ~1.3 units, underscoring residual uncertainty in point estimates.

![Bootstrap Sharpe ratio distributions](results/bootstrap_sharpe_comparison.png)

---

## Key Finding

HRP did not outperform plain Risk Parity or Equal Weight on this small, highly-correlated universe, contrary to López de Prado (2016). We hypothesize that HRP's clustering advantages require larger and more diverse universes to manifest.

For personal equity portfolio, HRP attains returns (0.25) between those of equal weight (0.27) and risk parity (0.23) methods, with risk parity having much lower volatility (0.17) than hrp (0.22) and equal weight (0.21). 
Indicating that risk-parity is advantageous for risk averse personal investors.

---

> See `notebooks/method_comparison.ipynb` for full results and methodology.
