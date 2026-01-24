# Intraday Trading Strategy Results
Generated: 2026-01-24 00:39:56

## Performance Summary

| Strategy         | Metric         | Train   | Valid   | Test   |
|:-----------------|:---------------|:--------|:--------|:-------|
| MomentumBreakout | Ann. Return    | -0.28%  | -0.20%  | -0.33% |
| MomentumBreakout | Sharpe Ratio   | -0.08   | -0.06   | -0.10  |
| MomentumBreakout | Max Drawdown   | 30.70%  | 14.66%  | 16.14% |
| MomentumBreakout | Win Rate       | 52.5%   | 54.0%   | 52.4%  |
| MomentumBreakout | Num Trades     | 10649   | 1328    | 1747   |
| MomentumBreakout | Time in Market | 4.5%    | 3.6%    | 3.3%   |
| MeanReversion    | Ann. Return    | 3.93%   | 4.78%   | -1.50% |
| MeanReversion    | Sharpe Ratio   | 0.53    | 0.66    | -0.20  |
| MeanReversion    | Max Drawdown   | 31.56%  | 13.29%  | 33.22% |
| MeanReversion    | Win Rate       | 62.6%   | 65.1%   | 60.0%  |
| MeanReversion    | Num Trades     | 16862   | 2627    | 3479   |
| MeanReversion    | Time in Market | 15.6%   | 14.5%   | 15.5%  |
| ORB              | Ann. Return    | -0.05%  | -0.48%  | 1.83%  |
| ORB              | Sharpe Ratio   | -0.01   | -0.11   | 0.43   |
| ORB              | Max Drawdown   | 24.87%  | 22.79%  | 6.66%  |
| ORB              | Win Rate       | 52.6%   | 55.7%   | 60.0%  |
| ORB              | Num Trades     | 2405    | 377     | 532    |
| ORB              | Time in Market | 8.6%    | 8.4%    | 9.0%   |
| XGBoost_24bar    | Ann. Return    | 42.24%  | 26.39%  | 8.17%  |
| XGBoost_24bar    | Sharpe Ratio   | 2.73    | 1.96    | 0.57   |
| XGBoost_24bar    | Max Drawdown   | 18.69%  | 13.63%  | 23.59% |
| XGBoost_24bar    | Win Rate       | 57.0%   | 57.2%   | 54.5%  |
| XGBoost_24bar    | Num Trades     | 18097   | 2947    | 3891   |
| XGBoost_24bar    | Time in Market | 87.2%   | 86.2%   | 82.3%  |
| LightGBM_24bar   | Ann. Return    | 36.01%  | 25.71%  | 10.45% |
| LightGBM_24bar   | Sharpe Ratio   | 2.32    | 1.91    | 0.76   |
| LightGBM_24bar   | Max Drawdown   | 20.38%  | 16.72%  | 18.48% |
| LightGBM_24bar   | Win Rate       | 56.4%   | 57.4%   | 55.2%  |
| LightGBM_24bar   | Num Trades     | 18110   | 2941    | 3887   |
| LightGBM_24bar   | Time in Market | 87.2%   | 85.4%   | 80.6%  |
| XGBoost_36bar    | Ann. Return    | 42.72%  | 25.38%  | 6.57%  |
| XGBoost_36bar    | Sharpe Ratio   | 2.83    | 1.87    | 0.44   |
| XGBoost_36bar    | Max Drawdown   | 15.48%  | 14.82%  | 21.98% |
| XGBoost_36bar    | Win Rate       | 57.0%   | 57.1%   | 54.5%  |
| XGBoost_36bar    | Num Trades     | 17959   | 2946    | 3877   |
| XGBoost_36bar    | Time in Market | 86.9%   | 87.2%   | 83.4%  |
| LightGBM_36bar   | Ann. Return    | 37.75%  | 22.68%  | 10.01% |
| LightGBM_36bar   | Sharpe Ratio   | 2.49    | 1.71    | 0.69   |
| LightGBM_36bar   | Max Drawdown   | 16.72%  | 14.63%  | 19.82% |
| LightGBM_36bar   | Win Rate       | 56.4%   | 56.6%   | 54.9%  |
| LightGBM_36bar   | Num Trades     | 17936   | 2934    | 3866   |
| LightGBM_36bar   | Time in Market | 86.6%   | 85.7%   | 81.3%  |
| XGBoost_eod      | Ann. Return    | 46.23%  | 24.99%  | 7.86%  |
| XGBoost_eod      | Sharpe Ratio   | 3.29    | 1.83    | 0.53   |
| XGBoost_eod      | Max Drawdown   | 18.69%  | 16.75%  | 26.05% |
| XGBoost_eod      | Win Rate       | 57.3%   | 57.4%   | 54.3%  |
| XGBoost_eod      | Num Trades     | 17410   | 2886    | 3784   |
| XGBoost_eod      | Time in Market | 83.4%   | 84.8%   | 81.3%  |
| LightGBM_eod     | Ann. Return    | 38.98%  | 25.83%  | 9.30%  |
| LightGBM_eod     | Sharpe Ratio   | 2.74    | 1.93    | 0.64   |
| LightGBM_eod     | Max Drawdown   | 22.86%  | 18.02%  | 27.31% |
| LightGBM_eod     | Win Rate       | 56.6%   | 57.4%   | 56.3%  |
| LightGBM_eod     | Num Trades     | 17371   | 2866    | 3769   |
| LightGBM_eod     | Time in Market | 82.8%   | 82.6%   | 78.1%  |

## Strategy Details

### MomentumBreakout

**Train**

- Total Return: -18.90%
- Ann. Return: -0.28%
- Ann. Volatility: 3.70%
- Sharpe Ratio: -0.08
- Max Drawdown: 30.70%
- Calmar Ratio: -0.01
- Num Trades: 10649
- Win Rate: 52.5%
- Profit Factor: 0.99
- Avg Trade Return: -0.001%
- Avg Win: 0.200%
- Avg Loss: -0.224%
- Time in Market: 4.5%
- Avg Holding (bars): 6.1
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: -2.35%
- Ann. Return: -0.20%
- Ann. Volatility: 3.24%
- Sharpe Ratio: -0.06
- Max Drawdown: 14.66%
- Calmar Ratio: -0.01
- Num Trades: 1328
- Win Rate: 54.0%
- Profit Factor: 0.99
- Avg Trade Return: -0.001%
- Avg Win: 0.180%
- Avg Loss: -0.214%
- Time in Market: 3.6%
- Avg Holding (bars): 6.3
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: -5.19%
- Ann. Return: -0.33%
- Ann. Volatility: 3.29%
- Sharpe Ratio: -0.10
- Max Drawdown: 16.14%
- Calmar Ratio: -0.02
- Num Trades: 1747
- Win Rate: 52.4%
- Profit Factor: 0.98
- Avg Trade Return: -0.003%
- Avg Win: 0.206%
- Avg Loss: -0.232%
- Time in Market: 3.3%
- Avg Holding (bars): 6.0
- Excess Return: 0.00%
- Info Ratio: 0.00

### MeanReversion

**Train**

- Total Return: 1618.65%
- Ann. Return: 3.93%
- Ann. Volatility: 7.43%
- Sharpe Ratio: 0.53
- Max Drawdown: 31.56%
- Calmar Ratio: 0.12
- Num Trades: 16862
- Win Rate: 62.6%
- Profit Factor: 1.12
- Avg Trade Return: 0.018%
- Avg Win: 0.265%
- Avg Loss: -0.397%
- Time in Market: 15.6%
- Avg Holding (bars): 13.4
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 75.14%
- Ann. Return: 4.78%
- Ann. Volatility: 7.27%
- Sharpe Ratio: 0.66
- Max Drawdown: 13.29%
- Calmar Ratio: 0.36
- Num Trades: 2627
- Win Rate: 65.1%
- Profit Factor: 1.15
- Avg Trade Return: 0.023%
- Avg Win: 0.260%
- Avg Loss: -0.419%
- Time in Market: 14.5%
- Avg Holding (bars): 13.0
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: -21.43%
- Ann. Return: -1.50%
- Ann. Volatility: 7.38%
- Sharpe Ratio: -0.20
- Max Drawdown: 33.22%
- Calmar Ratio: -0.05
- Num Trades: 3479
- Win Rate: 60.0%
- Profit Factor: 0.96
- Avg Trade Return: -0.006%
- Avg Win: 0.262%
- Avg Loss: -0.407%
- Time in Market: 15.5%
- Avg Holding (bars): 13.9
- Excess Return: 0.00%
- Info Ratio: 0.00

### ORB

**Train**

- Total Return: -3.39%
- Ann. Return: -0.05%
- Ann. Volatility: 4.49%
- Sharpe Ratio: -0.01
- Max Drawdown: 24.87%
- Calmar Ratio: -0.00
- Num Trades: 2405
- Win Rate: 52.6%
- Profit Factor: 1.01
- Avg Trade Return: 0.002%
- Avg Win: 0.477%
- Avg Loss: -0.525%
- Time in Market: 8.6%
- Avg Holding (bars): 52.0
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: -5.66%
- Ann. Return: -0.48%
- Ann. Volatility: 4.31%
- Sharpe Ratio: -0.11
- Max Drawdown: 22.79%
- Calmar Ratio: -0.02
- Num Trades: 377
- Win Rate: 55.7%
- Profit Factor: 0.95
- Avg Trade Return: -0.012%
- Avg Win: 0.408%
- Avg Loss: -0.541%
- Time in Market: 8.4%
- Avg Holding (bars): 52.2
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 33.41%
- Ann. Return: 1.83%
- Ann. Volatility: 4.25%
- Sharpe Ratio: 0.43
- Max Drawdown: 6.66%
- Calmar Ratio: 0.27
- Num Trades: 532
- Win Rate: 60.0%
- Profit Factor: 1.28
- Avg Trade Return: 0.057%
- Avg Win: 0.436%
- Avg Loss: -0.510%
- Time in Market: 9.0%
- Avg Holding (bars): 52.6
- Excess Return: 0.00%
- Info Ratio: 0.00

### XGBoost_24bar

**Train**

- Total Return: 19952492047027.43%
- Ann. Return: 42.24%
- Ann. Volatility: 15.46%
- Sharpe Ratio: 2.73
- Max Drawdown: 18.69%
- Calmar Ratio: 2.26
- Num Trades: 18097
- Win Rate: 57.0%
- Profit Factor: 1.58
- Avg Trade Return: 0.149%
- Avg Win: 0.707%
- Avg Loss: -0.591%
- Time in Market: 87.2%
- Avg Holding (bars): 69.9
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1562.13%
- Ann. Return: 26.39%
- Ann. Volatility: 13.44%
- Sharpe Ratio: 1.96
- Max Drawdown: 13.63%
- Calmar Ratio: 1.94
- Num Trades: 2947
- Win Rate: 57.2%
- Profit Factor: 1.41
- Avg Trade Return: 0.099%
- Avg Win: 0.592%
- Avg Loss: -0.562%
- Time in Market: 86.2%
- Avg Holding (bars): 68.9
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 248.90%
- Ann. Return: 8.17%
- Ann. Volatility: 14.22%
- Sharpe Ratio: 0.57
- Max Drawdown: 23.59%
- Calmar Ratio: 0.35
- Num Trades: 3891
- Win Rate: 54.5%
- Profit Factor: 1.12
- Avg Trade Return: 0.036%
- Avg Win: 0.607%
- Avg Loss: -0.648%
- Time in Market: 82.3%
- Avg Holding (bars): 66.0
- Excess Return: 0.00%
- Info Ratio: 0.00

### LightGBM_24bar

**Train**

- Total Return: 729144672812.20%
- Ann. Return: 36.01%
- Ann. Volatility: 15.55%
- Sharpe Ratio: 2.32
- Max Drawdown: 20.38%
- Calmar Ratio: 1.77
- Num Trades: 18110
- Win Rate: 56.4%
- Profit Factor: 1.49
- Avg Trade Return: 0.130%
- Avg Win: 0.702%
- Avg Loss: -0.609%
- Time in Market: 87.2%
- Avg Holding (bars): 69.8
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1456.87%
- Ann. Return: 25.71%
- Ann. Volatility: 13.45%
- Sharpe Ratio: 1.91
- Max Drawdown: 16.72%
- Calmar Ratio: 1.54
- Num Trades: 2941
- Win Rate: 57.4%
- Profit Factor: 1.41
- Avg Trade Return: 0.097%
- Avg Win: 0.579%
- Avg Loss: -0.551%
- Time in Market: 85.4%
- Avg Holding (bars): 68.4
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 386.11%
- Ann. Return: 10.45%
- Ann. Volatility: 13.74%
- Sharpe Ratio: 0.76
- Max Drawdown: 18.48%
- Calmar Ratio: 0.57
- Num Trades: 3887
- Win Rate: 55.2%
- Profit Factor: 1.16
- Avg Trade Return: 0.045%
- Avg Win: 0.587%
- Avg Loss: -0.625%
- Time in Market: 80.6%
- Avg Holding (bars): 64.7
- Excess Return: 0.00%
- Info Ratio: 0.00

### XGBoost_36bar

**Train**

- Total Return: 25577826664829.31%
- Ann. Return: 42.72%
- Ann. Volatility: 15.08%
- Sharpe Ratio: 2.83
- Max Drawdown: 15.48%
- Calmar Ratio: 2.76
- Num Trades: 17959
- Win Rate: 57.0%
- Profit Factor: 1.61
- Avg Trade Return: 0.151%
- Avg Win: 0.700%
- Avg Loss: -0.578%
- Time in Market: 86.9%
- Avg Holding (bars): 70.1
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1408.91%
- Ann. Return: 25.38%
- Ann. Volatility: 13.58%
- Sharpe Ratio: 1.87
- Max Drawdown: 14.82%
- Calmar Ratio: 1.71
- Num Trades: 2946
- Win Rate: 57.1%
- Profit Factor: 1.39
- Avg Trade Return: 0.096%
- Avg Win: 0.598%
- Avg Loss: -0.573%
- Time in Market: 87.2%
- Avg Holding (bars): 69.7
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 174.99%
- Ann. Return: 6.57%
- Ann. Volatility: 14.78%
- Sharpe Ratio: 0.44
- Max Drawdown: 21.98%
- Calmar Ratio: 0.30
- Num Trades: 3877
- Win Rate: 54.5%
- Profit Factor: 1.10
- Avg Trade Return: 0.031%
- Avg Win: 0.613%
- Avg Loss: -0.667%
- Time in Market: 83.4%
- Avg Holding (bars): 67.2
- Excess Return: 0.00%
- Info Ratio: 0.00

### LightGBM_36bar

**Train**

- Total Return: 1864360018777.30%
- Ann. Return: 37.75%
- Ann. Volatility: 15.13%
- Sharpe Ratio: 2.49
- Max Drawdown: 16.72%
- Calmar Ratio: 2.26
- Num Trades: 17936
- Win Rate: 56.4%
- Profit Factor: 1.53
- Avg Trade Return: 0.137%
- Avg Win: 0.696%
- Avg Loss: -0.587%
- Time in Market: 86.6%
- Avg Holding (bars): 70.0
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1061.75%
- Ann. Return: 22.68%
- Ann. Volatility: 13.29%
- Sharpe Ratio: 1.71
- Max Drawdown: 14.63%
- Calmar Ratio: 1.55
- Num Trades: 2934
- Win Rate: 56.6%
- Profit Factor: 1.36
- Avg Trade Return: 0.087%
- Avg Win: 0.582%
- Avg Loss: -0.560%
- Time in Market: 85.7%
- Avg Holding (bars): 68.8
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 355.81%
- Ann. Return: 10.01%
- Ann. Volatility: 14.46%
- Sharpe Ratio: 0.69
- Max Drawdown: 19.82%
- Calmar Ratio: 0.51
- Num Trades: 3866
- Win Rate: 54.9%
- Profit Factor: 1.15
- Avg Trade Return: 0.044%
- Avg Win: 0.604%
- Avg Loss: -0.640%
- Time in Market: 81.3%
- Avg Holding (bars): 65.6
- Excess Return: 0.00%
- Info Ratio: 0.00

### XGBoost_eod

**Train**

- Total Return: 153619310986178.41%
- Ann. Return: 46.23%
- Ann. Volatility: 14.05%
- Sharpe Ratio: 3.29
- Max Drawdown: 18.69%
- Calmar Ratio: 2.47
- Num Trades: 17410
- Win Rate: 57.3%
- Profit Factor: 1.73
- Avg Trade Return: 0.165%
- Avg Win: 0.682%
- Avg Loss: -0.527%
- Time in Market: 83.4%
- Avg Holding (bars): 69.4
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1354.13%
- Ann. Return: 24.99%
- Ann. Volatility: 13.68%
- Sharpe Ratio: 1.83
- Max Drawdown: 16.75%
- Calmar Ratio: 1.49
- Num Trades: 2886
- Win Rate: 57.4%
- Profit Factor: 1.38
- Avg Trade Return: 0.097%
- Avg Win: 0.608%
- Avg Loss: -0.591%
- Time in Market: 84.8%
- Avg Holding (bars): 69.2
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 233.00%
- Ann. Return: 7.86%
- Ann. Volatility: 14.76%
- Sharpe Ratio: 0.53
- Max Drawdown: 26.05%
- Calmar Ratio: 0.30
- Num Trades: 3784
- Win Rate: 54.3%
- Profit Factor: 1.12
- Avg Trade Return: 0.036%
- Avg Win: 0.626%
- Avg Loss: -0.663%
- Time in Market: 81.3%
- Avg Holding (bars): 67.1
- Excess Return: 0.00%
- Info Ratio: 0.00

### LightGBM_eod

**Train**

- Total Return: 3596588003407.64%
- Ann. Return: 38.98%
- Ann. Volatility: 14.25%
- Sharpe Ratio: 2.74
- Max Drawdown: 22.86%
- Calmar Ratio: 1.71
- Num Trades: 17371
- Win Rate: 56.6%
- Profit Factor: 1.60
- Avg Trade Return: 0.144%
- Avg Win: 0.678%
- Avg Loss: -0.552%
- Time in Market: 82.8%
- Avg Holding (bars): 69.1
- Excess Return: 0.00%
- Info Ratio: 0.00

**Valid**

- Total Return: 1475.24%
- Ann. Return: 25.83%
- Ann. Volatility: 13.36%
- Sharpe Ratio: 1.93
- Max Drawdown: 18.02%
- Calmar Ratio: 1.43
- Num Trades: 2866
- Win Rate: 57.4%
- Profit Factor: 1.41
- Avg Trade Return: 0.100%
- Avg Win: 0.596%
- Avg Loss: -0.569%
- Time in Market: 82.6%
- Avg Holding (bars): 67.9
- Excess Return: 0.00%
- Info Ratio: 0.00

**Test**

- Total Return: 311.37%
- Ann. Return: 9.30%
- Ann. Volatility: 14.52%
- Sharpe Ratio: 0.64
- Max Drawdown: 27.31%
- Calmar Ratio: 0.34
- Num Trades: 3769
- Win Rate: 56.3%
- Profit Factor: 1.14
- Avg Trade Return: 0.042%
- Avg Win: 0.591%
- Avg Loss: -0.665%
- Time in Market: 78.1%
- Avg Holding (bars): 64.7
- Excess Return: 0.00%
- Info Ratio: 0.00

