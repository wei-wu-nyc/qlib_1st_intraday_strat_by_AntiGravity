# Rewrite Intraday Backtest Using Qlib

## Tasks

- [x] Implement equal-weight and best-per-bar backtests
- [x] Create dashboard with benchmark
- [x] Run 1bp cost comparison- [x] Run with 1bp transaction cost
- [x] Implement reduced-switching strategies
- [x] Optimize timing (10am entry / 24-bar exit found)
- [x] Refactor and Cleanup:
  - [x] Split qlib_backtest.py into engine + strategies
  - [x] Create run_best_strategy.py
  - [x] Extract dashboard generator
  - [x] Archive old files
  - [x] Verify final run
- [x] Time-of-Day MoE Experiment:
  - [x] Create specialized training script (train_moe.py)
  - [x] Train 6 models (9:30, 10:00, 10:30, 12:00, 14:00, 15:00)
  - [x] Implement MoE Strategy (time-based switching)
  - [x] Run backtest comparison (MoE vs Global)
  - [x] Create comparison dashboard (Test Period)
  - [x] Run Full Day Analysis (Train/Valid/Test)
  - [x] Update dashboard with full history
  - [x] Bug Fix: Resolved -100% return due to early market close entries

- [ ] Trading Model Refinement:
  - [ ] Plan robust model parameters and dashboard updates
  - [ ] Add "Win Rate" to dashboard
  - [ ] Train Robust LightGBM (max_depth=3, n_estimators=2000, lr=0.01)
  - [x] Analyze Robust Model Features
- [/] **Phase 6: Expanded ML Models & Ensemble**
    - [x] Train MoE XGBoost Models
    - [x] Train MoE Random Forest Models
    - [x] Implement Ensemble Strategy (Avg of LGB+XGB+RF)
    - [/] detailed Comparison Dashboard (LGB vs XGB vs RF vs Ensemble)
    - [ ] Analyze feature importance across model families (Skipped for now)

- [/] **Phase 7: Horizon Sensitivity Analysis**
    - [x] Plan experiment (Ensemble MoE on horizons 12, 18, 24, 30, 36)
    - [x] Create `run_horizon_analysis.py`
    - [x] Generate Sensitivity Dashboard
    - [x] Document optimal holding period insights

- [/] **Phase 8: Finalize Configuration**
    - [x] Update default holding period to **36 bars** in scripts/strategies
    - [x] Run final "Champion" Backtest
    - [x] Create Git Branch `feature/ensemble-moe-complete`
    - [x] Commit and Push changes
    - [x] Final cleanup and archival
    - [x] Finalized Documentation (Transaction Cost Audit)
    - [x] Pushed to Main

- [/] **Phase 9: Full Day Multi-Trade Strategy**
    - [x] Create `MultiTradeStrategy` in `src/backtest/strategies/multi_trade.py`
    - [x] Enhance Dashboard (Equity Curve + Detailed Stats + Benchmark + Controls)
    - [x] Run Validation (verify overlapping trades and limits)
    - [x] Add QQQ+IWM Only Strategy Variant
    - [x] Add Valid Only View & Daily Returns Stats
    - [x] Add Avg Invested Metric

- [x] **Phase 10: Capital Efficiency Experiment**
    - [x] Create Concentrated Strategy (Max 2 positions) in script
    - [x] Run Comparison (Full vs. QQQ+IWM vs. Concentrated)
    - [x] Update Dashboard with 3-way comparison
    - [x] Verify higher utilization and returns

- [x] **Phase 11: Dynamic Rebalancing (100% Invested)**
    - [x] Create `RebalanceStrategy` with internal netting logic
    - [x] Integrate into `run_full_day_strategy.py`
    - [x] Run Comparison (Full vs. QQQ+IWM vs. Concentrated vs. Rebalance)
    - [x] Update Dashboard with Correct Metrics Source

- [x] **Phase 12: Strategy Refinement & Analysis**
    - [x] Fix Dashboard Chart Resizing
    - [x] Add Time-of-Day Stats (Win%, Return, Trades)
    - [x] Add "Max 2 (QQQ Only)" Variant
    - [x] Add "Dyn Rebalancing (QQQ Only)" Variant
    - [x] Run Full Simulation & Verify (V2 Script)
    - [x] Change Dashboard Benchmark to QQQ
    - [x] Add Benchmark Summary Box to Dashboard Top
    - [x] Fix Stats Box Layout (Horizontal Alignment)
