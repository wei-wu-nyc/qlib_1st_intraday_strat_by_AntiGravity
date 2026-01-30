# Qlib Walk-Forward Validation Tasks

- [x] **Research Qlib RollingTrainer**
    - [x] Search for documentation and examples of `RollingTrainer` and `RollingGen`.
    - [x] Investigate configuration options (train window size, step size, fixed vs expanding start).
    - [x] Determine how to aggregate statistics across periods.
- [x] **Design Rolling Workflow**
    - [x] Create implementation plan for `scripts/run_rolling_validation.py`.
    - [x] Define config structure for rolling parameters.
    - [x] Adapt MoE and Global models to fit the rolling interface.
- [x] **Implementation**
    - [x] Create rolling config in `config/intraday_config.yaml`.
    - [x] Implement the rolling training script (`scripts/run_rolling_validation.py` & `src/training/moe_trainer.py`).
    - [x] Implement result aggregation and dashboard generation.
- [x] **Verification**
    - [x] Run rolling validation simulation (Completed - Models Trained 2019-2025).
    - [x] Verify metrics match individual run logic (Dashboard Reviewed).
    - [x] Generate walk-forward dashboard (Screenshot captured).
- [x] **Individual Model Breakdown**
    - [x] Update `scripts/run_rolling_validation.py` to evaluate LGB, XGB, RF separately.
    - [x] Update dashboard to show statistics for individual models (Max 2 Strategy).
    - [x] Verify dashboard output.
- [x] **Sample Weighting**
    - [x] Implement linear weighing logic in `moe_trainer.py`.
    - [x] Run rolling validation (Linear, min=0.25) -> `rolling_validation_linear.html`.
    - [x] Run rolling validation (Exponential, half-life=2yr).
    - [x] Compare linear vs exponential results.
- [x] **Phase 4: Leverage Support (AE Variants)**
    - [x] Implement `borrow_rate` in config and `MultiTradeStrategy`.
    - [x] Implement `fixed_pos_pct` for "All-Entry" leverage logic.
    - [x] Run rolling validation for [Max2, AE80, AE100, AE150, AE200].
    - [x] Generate Leverage Dashboard (inc. Time-of-Day analysis for AE100).
    - [x] Document key findings (AE200 > Max2 > AE100).

    - [ ] Dynamic Entry Selection (Best of Market Nodes)
    - [ ] Parameter Optimization (Size, Exit) for new entries
- [x] **Phase 5: Entry Validation** <!-- id: 5 -->
    - [x] Experiment: Expanded Entry Times (Default vs Extended) <!-- id: 6 -->
    - [x] Experiment: Relaxed vs Fixed Entry <!-- id: 7 -->
    - [x] Experiment: Entry Delay Sensitivity <!-- id: 8 -->
    - [x] Experiment: Exact Single Bar Entry Map <!-- id: 9 -->
    - [x] Analysis: Sub-period Stability (2019-2022 vs 2023-2025) <!-- id: 10 -->

- [x] **Phase 6: Strategy Refinement (6-Slot)** <!-- id: 11 -->
    - [x] Run Leverage Analysis V2 (6 Slots: 9:45-12:20) <!-- id: 12 -->
    - [x] Analyze Impact of removing afternoon trades <!-- id: 13 -->
    - [x] **Bug Fix:** Short Trading Day (Early Close) Logic <!-- id: 14 -->
    
- [x] **Phase 7: Extended History Validation (2013-2025)** <!-- id: 15 -->
    - [x] Create `run_extended_history.py` (Train+Predict loop) <!-- id: 16 -->
    - [x] Generate Period Comparison Dashboard (13-15, 16-18, 19-22, 23-25) <!-- id: 17 -->
    - [x] Save full trade logs for analysis <!-- id: 18 -->

- [x] **Phase 8: Quarterly Validation (Seasonality vs Degradation)** <!-- id: 19 -->
    - [x] Create `run_quarterly_validation.py` (Loop 2013-2025 Q1-Q4) <!-- id: 20 -->
    - [x] Implement Dashboard with Seasonality (MoY, DoW, DoM) <!-- id: 21 -->
    - [x] Launch Long-Running Job <!-- id: 22 -->
