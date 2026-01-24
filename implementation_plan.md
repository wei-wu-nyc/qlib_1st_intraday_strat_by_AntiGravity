# Extended Horizons, Benchmark Fix & Exit Rule Experiments (COMPLETED)

## Status: All Phases Complete ✅
- **Phase 1**: Extended Horizons (24/36/EOD) -> Done.
- **Phase 2**: Benchmark Metrics -> Done.
- **Phase 3**: Exit Rule Experiments -> Done. (Winner: Combo Trail0.5% + Target2.0%)

## Phase 1: Extended Horizons (Immediate)

### Goal
Run ML models with horizons: 24-bar, 36-bar, and EOD (`ret_to_close`). Drop horizons < 24.

### Changes

#### [MODIFY] [intraday_config.yaml](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/config/intraday_config.yaml)
```yaml
trading:
  primary_label_horizon: 24
  alternative_horizons:
    - 36    # 3 hours
    # EOD will use ret_to_close label
```

#### [MODIFY] [run_all_strategies.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/scripts/run_all_strategies.py)
- Add special handling for EOD horizon using `ret_to_close` label
- Update horizon loop: `[24, 36, 'eod']`

---

## Phase 2: Benchmark Metrics Fix

### Goal
Show Sharpe Ratio and Max Drawdown for Benchmark (SPY) in Performance Comparison table.

#### [MODIFY] [dashboard.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/src/reporting/dashboard.py)
- Calculate benchmark Sharpe and MaxDD from equity curve data
- Populate all metrics columns instead of just Ann. Return

---

## Phase 3: Exit Rule Experimentation Framework

### Exit Rules to Implement

| Rule | Description | Parameters |
|------|-------------|------------|
| **Fixed Horizon** | Exit after N bars regardless of signal | `horizon_bars=12` |
| **Negative Return Gate** | Exit if return < 0 after N bars | `gate_bars=12` |
| **Stop Loss** | Exit if return drops below threshold | `stop_pct=-0.5%` |
| **Trailing Stop** | Exit if return drops from peak | `trail_pct=-0.3%` |
| **Profit Target** | Exit if return exceeds threshold | `target_pct=+0.5%` |

### Suggested Parameter Ranges

- **Stop Loss**: -0.1%, -0.3%, -0.5%, -1.0%
- **Profit Target**: +0.3%, +0.5%, +1.0%
- **Fixed Horizon**: 6, 12, 18, 24 bars
- **Gate Horizon**: 6, 12 bars

### Code Architecture for Efficient Experiments

#### [NEW] [exit_rules.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/src/backtest/exit_rules.py)
- Define `ExitRule` base class with `should_exit(position, current_bar, current_price)` method
- Implement: `FixedHorizonExit`, `StopLossExit`, `TrailingStopExit`, `ProfitTargetExit`, `NegativeGateExit`
- Make exit rules composable (multiple rules can be combined)

#### [MODIFY] [intraday_backtest.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/src/backtest/intraday_backtest.py)
- Accept `exit_rules: List[ExitRule]` in constructor
- Apply exit rules in position check loop

#### [NEW] [run_exit_experiments.py](file:///Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/scripts/run_exit_experiments.py)
- Load pre-trained models (no retraining)
- Loop through exit rule configurations
- Generate results with suffix: `LightGBM_24bar_sl0.5%`

### Efficient Experiment Flow

**Strategy**: To avoid dashboard clutter (3 horizons * 2 models * N exit rules = chaos), we will:
1.  **Phase 3a (Deep Dive)**: Run ALL exit rule variations on the **Champion Model (`LightGBM_24bar`)**.
2.  **Phase 3b (Validation)**: Apply the *best* performing exit rules to 36-bar and EOD models to verify robustness.

### Dashboard Organization
- Strategies will be named like `LightGBM_24bar_SL0.5` (Stop Loss 0.5%)
- Dashboard will show these as variations of the base strategy.

```
1. Train models ONCE (already done)
2. Save signals to CSV (precompute predictions)
3. For each exit rule config:
   - Load saved signals for LightGBM_24bar
   - Run backtest with new exit rules
   - Save metrics with config suffix
4. Generate comparison dashboard
```

---

## Verification

1. Run extended horizons (24, 36, EOD)
2. Verify benchmark has all metrics in table
3. Confirm exit rule framework works with one config
