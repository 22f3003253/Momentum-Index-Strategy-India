# Momentum-Index-Strategy-India
Click here for streamlit dashboard

https://momentum-index-strategy-india-exkw4kxgrreacsym4agens.streamlit.app/

# India Momentum Index Replication

This project implements an end-to-end momentum index construction pipeline in Python, inspired by the methodology of the MSCI Momentum Index. The objective was to replicate the key principles of MSCI's approach within the Indian equity market using historical constituent data from the Nifty 500 universe.

The pipeline covers the complete index lifecycle:

* Universe selection from the Nifty 500
* Momentum score calculation using risk-adjusted price momentum
* Stock ranking and constituent selection
* Portfolio weighting based on momentum characteristics
* Periodic rebalancing and index maintenance
* Historical backtesting and performance evaluation

## Methodology

The strategy attempts to follow the MSCI Momentum Index methodology as closely as possible using publicly available data. Momentum scores are constructed using medium- and long-term price performance while incorporating volatility adjustment to reduce exposure to excessively risky stocks. Securities are ranked on their momentum characteristics and selected into the index, with weights determined by their relative momentum strength.

## Assumptions and Limitations

This is an educational replication rather than an exact recreation of MSCI's proprietary index. Several assumptions were required due to data availability and methodology details that are not fully disclosed, including:

* Use of Nifty 500 constituents as the investment universe
* Approximation of MSCI's momentum scoring framework
* Simplified treatment of corporate actions and index maintenance events
* Assumed rebalancing schedules and turnover controls
* Limited implementation of liquidity and investability screens

As a result, the backtest should be viewed as a methodology replication and research exercise rather than a precise representation of the official MSCI Momentum Index.

## Results

Despite these simplifications, the momentum strategy demonstrated substantial outperformance relative to both the Nifty 50 and Nifty 500 over the test period.

| Metric                | India Momentum | Nifty 50 | Nifty 500 |
| --------------------- | -------------- | -------- | --------- |
| Total Return          | 131.74%        | 48.00%   | 60.32%    |
| CAGR                  | 19.37%         | 8.61%    | 10.46%    |
| Annualized Volatility | 18.52%         | 13.61%   | 14.26%    |
| Sharpe Ratio          | 0.732          | 0.206    | 0.326     |
| Sortino Ratio         | 0.863          | 0.279    | 0.408     |
| Calmar Ratio          | 0.786          | 0.500    | 0.555     |
| Maximum Drawdown      | -24.66%        | -17.23%  | -18.84%   |
| Monthly Win Rate      | 66.1%          | 57.6%    | 62.7%     |

The backtest suggests that a systematic momentum strategy can capture persistent trends in Indian equities, generating significantly higher returns and superior risk-adjusted performance compared with broad market benchmarks. While the strategy experienced moderately higher volatility and drawdowns, the improvement in return, Sharpe ratio, and consistency of positive months indicates a meaningful momentum premium over the sample period.

## Disclaimer

This project is intended for educational and research purposes only. Past performance does not guarantee future results, and the methodology contains assumptions that may differ from official index construction processes.

