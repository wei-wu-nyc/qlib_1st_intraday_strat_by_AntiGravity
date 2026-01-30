# Walk-Forward Validation (Rolling Trainer) Plan

## Objective
Implement a robust walk-forward validation workflow to evaluate model performance to minimize look-ahead bias and simulates real-world production updates. For this experiment, we will use an **Expanding Window** training scheme (Train from Start to T-1, Predict T).

## Configuration
- **Validation Period:** 2019-01-01 to 2025-12-31 (Current Test Set)
- **Step Size:** 1 Year
- **Training Window:** Expanding (Start Date fixed at 2008-01-01 or config start, up to Validation Start - 1 day).
    - *Optionally configurable to Rolling Window (e.g., past 5 years).*
- **Models:** Global (LGB, XGB, RF) and MoE (LGB, XGB, RF) Ensembles.
- **Strategies:** Max 2 and Dynamic Allocation.

## Proposed Architecture

### 1. Config Updates (`config/intraday_config.yaml`)
Add a section for rolling validation settings:
```yaml
rolling:
  train_window_type: "expanding" # or "rolling"
  rolling_window_size: 5 # years (only if type is "rolling")
  validation_start_year: 2019
  validation_end_year: 2025
  step_size: 1 # year
```

### 2. New Script: `scripts/run_rolling_validation.py`
This script will orchestrate the entire process:
1.  **Iterate years** from 2019 to 2025.
2.  **Define Splits**:
    - `train_range`: (Start, Year-1)
    - `test_range`: (Year, Year)
3.  **Train Models**:
    - Call a modified training function that accepts custom date ranges.
    - Save models to `results/models_rolling/{year}/`.
4.  **Predict & Backtest**:
    - Load models for `{year}`.
    - Run backtest on `test_range`.
    - Save daily returns/trades to memory.
5.  **Aggregation**:
    - Combine all annual results into a single equity curve.
    - Compute Full Period Stats vs Annual Stats.
    - Generate `results/active/rolling_validation_dashboard.html`.

### 3. Refactoring Training Logic
- Extract the core training logic from `train_expanded_moe.py` into a reusable function/module `src/training/moe_trainer.py` that accepts:
    - `train_start_date`
    - `train_end_date`
    - `output_dir`
    - `model_config`
- This avoids code duplication and allows the rolling script to just call `train_moe(...)`.

## Dashboard
A specialized dashboard (horizontal layout) showing:
- **Top:** Full Period Cumulative Return (concatenated).
- **Middle:** Table of Annual Performance (Year, Return, Sharpe, Win Rate).
- **Bottom:** Charts/tables comparison.

## Execution Steps
1.  Refactor `train_expanded_moe.py` logic into `src/training/trainer.py`.
2.  Create `run_rolling_validation.py`.
3.  Run the validation (this will take time as it trains ~7 sets of models).
4.  Analyze results.
## Phase 2: Individual Model Breakdown (Planned)

### Objective
Evaluate and display the performance of the 3 individual MoE models (LGB, XGB, RF) separately using the "Max 2 Strategy" to compare against the Ensemble.

### Implementation Steps
1.  **Modify `scripts/run_rolling_validation.py`**:
    -   In the validation loop, after loading models, generate signals for each individual model (`lgb_moe`, `xgb_moe`, `rf_moe`) *in addition* to the ensemble.
    -   Run `MultiTradeStrategy` (Max 2) for each of the 3 individual signal sets.
    -   Collect daily returns and statistics for all 3 outcomes (LGB-Max2, XGB-Max2, RF-Max2).
2.  **Update Dashboard**:
    -   **Top Metrics**: Add summary cards for LGB, XGB, and RF (Total Return, Sharpe).
    -   **Chart**: Add 3 new lines (maybe dashed or thinner) for the individual models.
    -   **Bottom Table**: Add columns for LGB, XGB, and RF interactions (Return, Sharpe, Win Ratio).
    -   *Constraint:* Keep the "Max 2" focus; likely drop "Dynamic" details for these individual ones to avoid clutter, or just show Max 2 for everyone as requested.

### Estimated Effort
-   **Script Logic**: Low (looping backtest 3 more times).
-   **Dashboard HTML**: Low/Medium (adding table columns and chart traces).
-   **Execution Time**: Backtesting is fast (seconds per year). No re-training needed.

## Phase 3: Recency-Based Sample Weighting

### Objective
Implement `sample_weight` for training LGB, XGB, and RF models to prioritize recent data.

### Configuration
Update `intraday_config.yaml` with:
```yaml
training:
  sample_weighting:
    method: "linear" # or "exponential" or "none"
    linear_min_weight: 0.25 # Floor for oldest data
    exponential_half_life_years: 2.0 # Time to drop to 0.5 weight
```

### Implementation Steps
1.  **Refactor Model Wrappers**:
    -   Update `LightGBMIntradayStrategy`, `XGBoostIntradayStrategy`, `RandomForestIntradayStrategy` in `src/strategies/ml_models/`.
    -   Modify `.fit()` to accept `sample_weight` argument and pass it to the underlying `model.fit()`.
    -   *Specific Fix*: Ensure `RandomForest` wrapper passes kwargs correctly.
2.  **Update `moe_trainer.py`**:
    -   Calculate `Age` in years: `(MaxDate - CurrentDate).days / 365.25`.
    -   Implement weight formulas:
        -   **Linear**: $w = 1 - (Age / MaxAge) * (1 - MinWeight)$
        -   **Exponential**: $w = 2 ^ {-Age / HalfLife}$
    -   Pass calculated weights to model constructors/fit methods.
3.  **Verification**:
    -   Train a test model and verify weights are generated correctly (print stats: min, max, mean weights).

## Phase 4: Leverage & "All-Entry" Strategy
### Objective
Implement a leveraged trading strategy ("ALLENTRY") that takes every valid entry signal (up to 5/day) with a fixed percentage allocation, allowing total exposure to exceed 100%.

### Configuration (`intraday_config.yaml`)
Add borrowing rate (default 0).
```yaml
backtest:
  borrow_rate_annual: 0.0
```

### Strategy Logic (`MultiTradeStrategy`)
Enhance `src/backtest/strategies/multi_trade.py`:
-   **Allocation**: Add `fixed_pos_pct` (float). If set, ignore `max_positions` and allocate this fixed % of *Current Equity* per trade.
-   **Leverage**: Remove usage of `cash >= allocation` check. Allow negative cash.
-   **Cost**: In `_close_position` or daily update, if `cash < 0`, deduct interest: `interest = abs(cash) * (borrow_rate / 365)`.

### Validation Variants
Run rolling validation on the **Ensemble** model with:
1.  **Max 2** (Baseline): 2 positions max (approx 100% cap).
2.  **AE80**: 5 slots x 16% = 80% Max Exposure.
3.  **AE100**: 5 slots x 20% = 100% Max Exposure.
4.  **AE150**: 5 slots x 30% = 150% Max Exposure.
5.  **AE200**: 5 slots x 40% = 200% Max Exposure.

### Dashboard Requirements
-   **Snapshots**: Stats for Bench, Max2, AE80, AE100, AE150, AE200.
-   **Equity Curve**: All variants.
-   **Annual Table**: Focus on AE100 (or comparison).
-   **Time-of-Day Table**: Specific breakdown for AE100 (PnL by Entry Time).
    
## Phase 8: Quarterly Validation (Seasonality vs Degradation)
### Objective
Disentangle model degradation from market seasonality by retraining models every quarter (Jan, Apr, Jul, Oct).
If performance still decays within the quarter, it is degradation. If "Month 1" (Jan, Apr, Jul, Oct) always performs best, it is Degradation. If "December" always performs bad regardless of training, it is Seasonality.

### Experiment Scope
-   **Period**: 2013 - 2025 (13 Years).
-   **Frequency**: Quarterly Retraining (4x per year).
-   **Total Training Sets**: ~52 sets (vs 13 currently).
    -   Existing Annual Models (Q1) can be reused for 2019-2025.
    -   **New Models to Train**: ~45 sets.

### Estimated Runtime
-   ~10 minutes per model set (on local hardware).
-   **Total Time:** 45 * 10 min = **~7.5 hours**.
-   *Optimization:* We will rely on existing `train_moe` logic which is robust but single-threaded at the job level (internal parallelism used).

### Implementation (`scripts/run_quarterly_validation.py`)
1.  **Loop**: Year (2013-2025) -> Quarter (Q1, Q2, Q3, Q4).
2.  **Training Window**:
    -   Q1: 2008-01-01 to (Year-1)-12-31.
    -   Q2: 2008-01-01 to Year-03-31.
    -   Q3: 2008-01-01 to Year-06-30.
    -   Q4: 2008-01-01 to Year-09-30.
3.  **Testing Window**:
    -   Q1: Year-01-01 to Year-03-31.
    -   Q2: Year-04-01 to Year-06-30.
    -   Q3: Year-07-01 to Year-09-30.
    -   Q4: Year-10-01 to Year-12-31.

### Dashboard Enhancements (`comparison_dashboard_quarterly.html`)
-   **Legacy Tables**: Period comparison, Trade distribution, Risk tail.
-   **Seasonality Tables**:
    -   **Monthly**: Calendar Month 1-12 (Aggregated).
    -   **Day of Week**: Mon (0) - Fri (4).
    -   **Day of Month**: 1 - 31.

