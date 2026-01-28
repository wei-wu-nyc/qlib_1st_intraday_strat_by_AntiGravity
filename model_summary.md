# Strategy Model Summary: Once-Per-Day Ensemble MoE

**Latest Update**: 2026-01-27
**Status**: Finalized (Champion Configuration)

## 1. Core Logic: Once-Per-Day Limit
The strategy enforces a strict **single trade per day** limit to minimize overtrading and transaction costs.
*   **One Trade Limit**: Once a position is opened (`entered_today = True`), no further entries are permitted for that trading day, even if the position is closed early.
*   **Ranking**: At the specified entry time (e.g., 10:00 AM), the strategy evaluates the candidate pool (`SPY, QQQ, DIA, IWM`).
*   **Selection Criteria**: It selects the **single best instrument** that meets:
    1.  **Model Signal**: Must be **1** (Buy).
    2.  **Predicted Return**: Must be **> 0**.
*   **Tie-Breaker**: If scores are identical, priority is: `SPY > QQQ > IWM > DIA`.

## 2. Configuration Details
*   **Initial Equity**: $1,000,000
*   **Position Sizing**: **100% Equity**. The strategy goes "All In" on the selected instrument (minus estimated transaction costs).
*   **Transaction Costs**:
    *   **Default**: 1 basis point (0.01%) per trade side.
    *   **Round-Trip**: ~2 basis points total.
    *   *Note*: Some auxiliary scripts may override this for specific experiments (e.g., 0bp analysis).

## 3. Exit Rules
The trade is held until **one** of the following conditions is met:
1.  **Time Limit (Standardized)**: **36 bars** (3.0 hours).
    *   *Rationale*: Captures morning trend and afternoon persistence.
2.  **End of Day**: Market close (15:55 / Bar 77).
    *   Positions are forcibly closed before the session ends to avoid overnight risk.

## 4. Model Architecture (Champion)
*   **Type**: Ensemble of Mixture of Experts (MoE).
*   **Components**: Average of 3 MoE models:
    1.  **LightGBM MoE**
    2.  **XGBoost MoE**
    3.  **Random Forest MoE**
*   **MoE Logic**: Each "MoE" model consists of distinct sub-models specialized for specific time-of-day slots (Morning, Mid-Day, Afternoon), allowing the strategy to adapt to changing intraday volatility regimes.
