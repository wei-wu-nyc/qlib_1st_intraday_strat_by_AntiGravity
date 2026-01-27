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

## Verification Plan

### Automated Tests
- Run `scripts/train_robust_model.py` to generate the new model.
- Run the updated backtest script to generate the `moe_full_dashboard.html`.

### Manual Verification
- **Check Overfitting**: Compare the "Total Return" in the **Train** period between Global and Robust models. The Robust model should have significantly *lower* (more realistic) training returns than the Global model.
- **Check Generalization**: Compare performance in the **Test** period.
- **Verify Win Rate**: Ensure the "Win Rate" column appears and looks correct (typically 50-55% for intraday).
