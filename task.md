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
  - [x] Run Full Period Analysis (Train/Valid/Test)
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
    - [ ] Analyze feature importance across model families Feature Engineering / RFE (Recursive Feature Elimination)
