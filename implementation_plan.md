# Model Refinement and Dashboard Enchancement Plan

## Goal
Reduce overfitting in current LightGBM models by training a "Robust" variant with constrained complexity (lower depth, higher estimators). Enhance the dashboard to include "Win Rate" for better performance evaluation.

## User Review Required
> [!IMPORTANT]
> **New Model Parameters**: I am proposing the following parameters for the "Robust" model to specifically target the "crazy returns" seen in the training period (overfitting):
> - `max_depth`: **3** (was 6) - *Limits tree complexity*
> - `num_leaves`: **8** (was 31) - *Drastically reduces leaf nodes*
> - `n_estimators`: **2000** (was 500) - *Increases ensemble size (weak learners)*
> - `learning_rate`: **0.01** (was 0.05) - *Slower likelihood improvement to prevent memorization*
> - `colsample_bytree`: **0.6** (was 0.8) - *More randomness in feature selection*

## Proposed Changes

### 1. Dashboard Enhancements
**File**: `scripts/run_moe_backtest.py`
- **Add Win Rate**: Extract `win_rate` from `BacktestResults` and display it in the HTML table as a percentage.
- **Update Table Layout**: Add columns for "Win Rate" under each model section.

### 2. Robust Model Training
**New File**: `scripts/train_robust_model.py`
- Create a script to train the `LightGBMIntradayStrategy` with the new robust parameters.
- Save the model to `results/models/lightgbm_robust`.

### 3. Backtest Comparison
**File**: `scripts/run_moe_backtest.py` (Renamed to `scripts/run_robust_comparison.py` or modified in place)
- Integrate the new "Robust" model into the backtest loop.
- **Comparison Visual**: Update the HTML dashboard to compare **Global (Baseline)** vs **Robust** vs **MoE**.
    - This allows us to see if the "Robust" model fixes the training set overfitting while maintaining/improving test set performance.

## Phase 12: Strategy Refinement & Dashboard Analysis (Current)

**Goal:** Address user feedback regarding chart resizing, Time-of-Day analysis, and add QQQ-only strategy variants.

#### [NEW] [run_full_day_strategy_v2.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/scripts/run_full_day_strategy_v2.py)
- Created a clean V2 script to implement extensive changes safely.
- **Key Features:**
    - **New Strategies:**
        - `MultiTradeStrategy (QQQ Only Max 2)`: Evaluates pure QQQ concentration.
        - `RebalanceStrategy (QQQ Only Dynamic)`: Evaluates pure QQQ dynamic allocation.
    - **Time-of-Day Analysis:** Aggregates performance by entry time (09:40, 10:00, etc.).
    - **Responsive Dashboard:** Updated Plotly config for auto-resizing.
    - **Corrected Metrics:** Trade stats sourced exclusively from "Concentrated (Max 2)" strategy for accuracy.

#### [MODIFY] [walkthrough.md](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/walkthrough.md)
- Will update with final analysis and screenshots from V2 run.

## Verification Plan

### Automated Tests
- [x] Run `run_full_day_strategy_v2.py` to generate `results/active/full_day_strategy_dashboard_v2.html`.
- [ ] Check console output for successful execution of all 3 periods.

### Manual Verification
- [ ] Open `full_day_strategy_dashboard_v2.html` in browser.
- [ ] Verify:
    - [ ] Charts resize when window changes.
    - [ ] "Time of Day" table is populated and sorted correctly.
    - [ ] "QQQ Only" strategy traces appear on the chart.
    - [ ] "Concentrated (Max 2)" metrics match expected values (e.g. 55% WR).
