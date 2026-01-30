# Experiments Log

This document records the side-experiments run during the development of the intraday strategy.

## 1. QQQ-Only Models
**Date:** 2024-01-28
**Goal:** Determine if training models exclusively on QQQ data improves performance for QQQ trading.
**Script:** `scripts/experiments/run_qqq_simulation.py` & `scripts/experiments/train_qqq_models.py`
**Result:**
- **Sharpe Ratio:** Improved (2.62 vs 2.40) for QQQ-Only models.
- **Total Return:** Lower than models trained on full universe (4 ETFs).
- **Conclusion:** Specialization adds precision (Sharpe) but loses raw signal power (Return). Kept full universe models for production.

## 2. Start Time 9:30 vs 9:40
**Date:** 2024-01-28
**Goal:** Test if relaxing the entry constraint to allow trading at 9:30 (Market Open) improves returns.
**Script:** `scripts/experiments/run_start_time_experiment.py`
**Result:**
- **Performance:** Identical for both 9:30 and 9:40 starts.
- **Conclusion:** No signal/trade difference.
- **Decision:** Kept 9:40 start time as a safety buffer against opening volatility.

## 3. Entry Time Expansion (Adding 13:00 & 15:00)
**Date:** 2024-01-29
**Goal:** Verify if excluding 13:00 and 15:00 entries was correct, and test re-enabling them.
**Script:** `scripts/experiment_entry_times.py`
**Result:**
- **Detrimental:** Adding 13:00 and 15:00 **lowered Sharpe Ratio** (1.37 -> 1.10) and **Total Return** (173% -> 163%).
- **Leverage:** Increased Max Invested from 143% to ~200% without generating alpha.
- **Specifics:** 
    - **13:00:** -$13k PnL.
    - **15:00:** -$33k PnL (Worst performer).
- **Decision:** **Confirmed** that 13:00 and 15:00 should remain excluded. The 5-slot configuration (9:40, 10:00, 10:30, 12:00, 14:00) is optimal.

## 4. Relaxed Entry Timing ("Any Bar in Block")
**Date:** 2024-01-29
**Goal:** strict "Fixed" entry (e.g. exactly 12:00) vs "Relaxed" entry (anytime 12:00-14:00, once per block).
**Script:** `scripts/experiment_relaxed_mode.py`
**Result:**
- **Performance:** Relaxed mode performed **significantly worse**.
    - **Sharpe:** Dropped from **1.37** to **0.59**.
    - **Return:** Dropped from **173%** to **62%**.
- **Behavior:** The model still clustered trades at the start of blocks (9:40, 10:00, 12:00), but when it *did* delay entry, those trades were generally unprofitable.
- **Decision:** **Reject** relaxed timing. The "First Bar" constraint acts as a quality filter, likely aligning with the strongest alpha at the top of the hour.

## 5. Entry Delay Sensitivity (0-3 Bars)
**Date:** 2024-01-29
**Goal:** Test if the strategy is sensitive to exact entry timing by introducing delays (0, 5, 10, 15 mins).
**Script:** `scripts/experiment_entry_delay.py`
**Result:** **Linear Degradation**. Performance drops significantly with every 5-minute delay.
- **Delay 0 (Baseline):** 173.7% Return, 1.37 Sharpe.
- **Delay 1 (+5m):** 137.8% Return, 1.15 Sharpe.
- **Delay 2 (+10m):** 119.5% Return, 1.02 Sharpe.
- **Delay 3 (+15m):** 89.6% Return, 0.85 Sharpe.
- **Delay 3 (+15m):** 89.6% Return, 0.85 Sharpe.
- **Conclusion:** The signal alpha decays rapidly. The "top-of-hour" structure (9:40, 10:00, etc.) captures the peak information. Any delay misses the move.

## 6. Single Bar Entry Map (Exhaustive)
**Date:** 2024-01-29
**Goal:** Map performance if we traded at ONLY one specific bar (e.g. 09:35 only) for the entire history.
**Script:** `scripts/experiment_single_bar.py`
**Result:** Verified "Market Nodes" hypothesis.
- **Global Peak:** **10:00 AM** (4.7% Ann Return, 1.24 Sharpe).
- **Local Peaks:** 
    - **12:00 PM** (3.5%) > Neighbors (2.7%).
    - **14:00 PM** (1.4%) > Neighbors (-0.1%).
- **Surprise:** 09:35 (3.7%) outperformed 09:40 (2.9%), suggesting the "Open" might be tradeable with the right logic, though 10:00 remains much safer (higher Sharpe).
- **Dead Zones:** 13:00 (0.0%) and 15:00 (-0.3%) confirmed as non-productive.
- **Charts:** `results/active/single_bar_entry_dashboard.html`

## 7. Sub-Period Stability Analysis (2019-2022 vs 2023-2025)
**Date:** 2024-01-29
**Goal:** Assess if the "Market Node" alpha is robust or decaying by comparing Early (2019-2022) vs Late (2023-2025) periods.
**Script:** `scripts/experiment_single_bar.py` (Sub-period mode)
**Result:** Mixed bag of stability and shifts.
- **10:00 AM (Global Peak):** **Stable but weaker**. Returns dropped from 5.5% -> 3.5%, Sharpe 1.35 -> 1.08. Still a core alpha provider.
- **12:00 PM (Local Peak):** **Very Robust**. Performance slightly *improved* in recent years (Sharpe 1.11 -> 1.23).
- **14:00 PM (Afternoon):** **COLLAPSE**. 2019-2022 was solid (2.7% Ret, 0.82 SR), but 2023-2025 is negative (-0.3% Ret). This entry slot is likely dead.
- **New Emergence (11:55 AM):** The slots just before 12:00 are performing exceptionally well recently.
    - **11:55 Late:** 4.4% Return, **1.69 Sharpe** (Highest in recent period).
    - **11:55 Early:** Only 1.5% Return.
    - **Hypothesis:** Front-running the 12:00 flow?
- **Conclusion:** 
    - Keep 10:00 and 12:00.
    - **Remove 14:00**.
    - **Investigate 11:55** as a replacement or addition.

## 8. Strategy Refinement V2 (No Afternoon Trading)
**Date:** 2024-01-29
**Goal:** Implement specific 6-slot strategy based on user request: `09:45, 10:00, 10:40, 11:50, 12:00, 12:20`. Test leverage scaling.
**Script:** `scripts/run_leverage_analysis_v2.py`
**Results (AE 200 - 2.0x Max):**
- **Sharpe Ratio:** **1.56** (Benchmark QQQ: 0.96). Massive improvement in risk-adjusted return.
- **Max Drawdown:** **-13.1%** (Benchmark QQQ: -35.9%). Excellent downside protection (survived 2022 with only -3.9%).
- **Total Return:** 244.7% (19.4% Ann) vs QQQ 310% (22.5% Ann).
- **Time of Day:**
    - **12:20:** Highest Win Rate (56.9%).
    - **10:00:** Highest PnL ($142k).
    - **10:40:** Weakest slot ($77k).
- **Conclusion:** This configuration is highly defensive. It gives up some upside in crazy bull markets (2023) but avoids the crashes. Ideal for leveraged steady growth.
- **Charts:** `results/active/leverage_dashboard_v2.html`

## 9. Extended History Validation (2013-2025)
**Date:** 2024-01-29
**Goal:** Test for overfitting by extending the backtest window back to 2013. Compare performance regimes.
**Script:** `scripts/run_extended_history.py`
**Scope:**
- **Strategy:** AE200 (6-Slot, 2.0x Leverage, Short-Day Fix).
- **Training:** Walk-forward rolling (Train 2008-Y-1, Test Y).
- **Metric:** 4-Period Comparison (2013-2015, 2016-2018, 2019-2022, 2023-2025).
- **Output:** Dashboard + Individual Trade CSVs.

**Results (Strategy AE200):**
The strategy demonstrates **remarkable consistency** over 12 years (2013-2025), validating that the alpha is structural and not overfitted to the recent period.

| Period | Total Ret | Ann Ret | Sharpe | Max DD | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2013-2015** | 38.3% | 11.5% | **1.53** | -6.8% | Stable early performance. |
| **2016-2018** | 54.8% | 15.7% | **1.86** | -8.1% | Peak efficiency period. |
| **2019-2022** | 118.4% | 21.6% | **1.57** | -13.1% | High volatility, high return. |
| **2023-2025** | 57.8% | 16.6% | **1.57** | -4.7% | Excellent risk control (-4.7% DD). |

**Conclusion:** The Sharpe Ratio has remained effectively constant (~1.55) for over a decade. The "Short-Day Fix" and 6-slot logic are robust.

**Model Degradation Analysis (Seasonality/Decay):**
- **Month 1 (Jan):** Sharpe 2.91, Return +18.6%.
- **Month 6 (Jun):** Sharpe 1.35, Return +8.6%.
- **Month 12 (Dec):** Sharpe -0.95, Return -5.5%.
- **Finding:** Strong alpha decay throughout the year. Annual retraining is likely suboptimal. Moving to **Quarterly** or **Monthly** retraining could significantly boost performance.

**Artifacts:** `results/extended_history/comparison_dashboard_detailed.html`, `results/extended_history/all_trades_2013_2025.csv`
