# Intraday Trading Strategy - Qlib & ML Framework

A modular Python framework for developing, backtesting, and optimizing intraday ETF trading strategies using Machine Learning (LightGBM/XGBoost) and Rule-Based approaches.

## 🏆 Champion Strategy
**Configuration**: `LightGBM_24bar_Combo_Trail0.5_Target2.0`
- **Model**: LightGBM (Gradient Boosting) trained on 24-bar (2hr) forward returns.
- **Exit Rule**: 
  - **Trailing Stop**: 0.5% (Protects gains, cuts losses fast)
  - **Profit Target**: 2.0% (Locks in alpha spikes)
  - **Time Exit**: 24 bars (EOD) if neither hit.
- **Performance (Test Period, 1bp cost)**:
  - **Annual Return**: **9.61%** (vs Base 5.18%)
  - **Sharpe Ratio**: 0.74
  - **Avg Holding**: 29.8 bars

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Optional] Qlib installed (framework uses custom data loaders compatible with Qlib format)

### Installation
```bash
pip install -r requirements.txt
```

### Data
Data is expected in the Qlib csv-provider format. Configure the path in `config/intraday_config.yaml`.

## 🛠 Usage

### 1. Train Models & Run Baselines
Runs all rule-based strategies and trains ML models (LightGBM/XGBoost) for multiple horizons (24, 36, EOD).
```bash
python scripts/run_all_strategies.py
```

### 2. Run Optimization Experiments
Runs the exit rule variations (Trailing Stops, Targets, etc.) on the trained Champion Model.
```bash
python scripts/run_exit_experiments.py
```

### 3. View Dashboard
Open the interactive HTML dashboard to visualize equity curves and metrics.
```bash
open results/dashboard.html
```

## 📂 Project Structure

```text
├── config/                 # Configuration files (YAML)
├── src/
│   ├── data/               # Data loading and processing
│   ├── features/           # Alpha factor generation (~115 features)
│   ├── strategies/         # Strategy logic (ML & Rule-based)
│   │   ├── ml_models/      # LightGBM, XGBoost wrappers
│   │   └── rule_based/     # Momentum, Mean Reversion, etc.
│   ├── backtest/           # Intraday backtest engine (vectors/events)
│   │   └── exit_rules.py   # Exit rule framework (StopLoss, Target, etc.)
│   └── reporting/          # Dashboard and CSV report generation
├── scripts/
│   ├── run_all_strategies.py    # Main training pipeline
│   ├── run_exit_experiments.py  # Optimization runner
│   └── generate_dashboard_from_csv.py
└── results/                # Output directory (metrics, models, plots)
```

## 📊 Results Summary
The project evolved through three phases:
1.  **Core Implementation**: Establishing the data pipeline, features, and ML models.
2.  **Horizon Expansion**: Testing 24-bar vs 36-bar vs EOD horizons. (24-bar proved best).
3.  **Exit Optimization**: Testing mechanical exits.
    - *Discovery*: A tight 0.5% trailing stop combined with a 2.0% profit target significantly outperformed the base model by solving the "churn" issue and capturing volatility.

See `walkthrough.md` for a detailed dev log.
