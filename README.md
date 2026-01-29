# Qlib Intraday Strategy

A modular Python framework for developing, backtesting, and optimizing intraday ETF trading strategies using Machine Learning (LightGBM/XGBoost/RandomForest) and Mixture of Experts (MoE) ensembles.

## 🏆 Current Best Strategy
- **Framework**: Mixture of Experts (MoE)
- **Models**: Ensemble of LightGBM, XGBoost, and Random Forest.
- **Allocation**: "Max 2" Concentrated Portfolio.
- **Performance (Valid+Test)**: ~232% Total Return, 2.40 Sharpe (using Full Universe models, 9:40 start).

## 🚀 Getting Started

### Core Scripts
*   **Training**: `scripts/train_expanded_moe.py` 
    - Trains Global and Time-Period specific MoE models (LGB, XGB, RF) on the full ETF universe (SPY, QQQ, IWM, DIA).
*   **Simulation**: `scripts/run_full_day_strategy_v2.py` 
    - Loads the trained MoE models.
    - Runs the simulation for "Max 2" and "Dynamic Rebalance" strategies.
    - Generates the interactive dashboard.
*   **Dashboard**: `results/active/full_day_strategy_dashboard_v2.html`
    - View this file in a browser to analyze performance.

### Experiments
See `experiments.md` for details on side-projects:
- QQQ-Only Models (Specialized vs Generalized)
- Start Time Analysis (9:30 vs 9:40 - Found to be identical).

## 📂 Project Structure

```text
├── config/                 # Configuration (intraday_config.yaml)
├── src/
│   ├── data/               # Data loading
│   ├── features/           # Alpha factors & Seasonality
│   ├── strategies/         # ML Models & MoE Logic
│   └── backtest/           # Event-driven backtest engine
├── scripts/
│   ├── experiments/        # Side-experiments
│   ├── train_expanded_moe.py
│   └── run_full_day_strategy_v2.py
└── results/                # Models and Reports
```

## 🛠 Usage
1.  **Train Models:**
    ```bash
    python scripts/train_expanded_moe.py
    ```
2.  **Run Simulation:**
    ```bash
    python scripts/run_full_day_strategy_v2.py
    ```
3.  **View Results:**
    Open `results/active/full_day_strategy_dashboard_v2.html`.
