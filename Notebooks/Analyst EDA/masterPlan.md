# Oxygen Analytics: Master Investment Strategy Plan

## 1. Project Objective

To develop a multi-faceted, data-driven investment decision framework for AAPL, NVDA, and GOOGL. The framework will leverage sentiment analysis and machine learning to provide actionable recommendations based on an investor's chosen time horizon.

## 2. Phase 1 & 2: Data Foundation & EDA (Completed)

This phase focused on cleaning, imputing, and exploring the sentiment and stock price datasets. The key outcome is a set of clean, merged dataframes for each company, ready for quantitative analysis.

## 3. Phase 3: Multi-Horizon Correlation Analysis (Completed)

This phase quantified the predictive power of sentiment features across multiple investment horizons (1D to 84M) and time lags (-5d to +5d).

*   **Outcome:** A unique "Signal Intelligence Playbook" for each company, identifying which features are leading indicators, contrarian signals, or momentum drivers, and for which time horizons they are most relevant. This playbook is the strategic foundation for all subsequent steps.

## 4. Phase 4: Feature Engineering & Model Input Preparation

This phase will translate the insights from Phase 3 into a powerful set of features for our predictive models.

*   **4.1. Lagged Features:** Based on the optimal lags identified in the lag-correlation plots, create lagged versions of the most predictive features (e.g., `mspr_lag_2`).
*   **4.2. Smart Composite Features:** Engineer a small set of high-impact interaction features tailored to each company's unique signal profile.
    *   **AAPL & GOOGL:** Create `10K_vs_10Q_Divergence` (`10-K_sentiment` - `10-Q_sentiment`) to capture conflicts between long-term health and short-term pessimism.
    *   **NVDA:** Create `Contrarian_Conviction` (`mspr` * -1 * `10-Q_sentiment`) to quantify "smart money buying the dip."
*   **4.3. Rolling Metrics:** For key volatility predictors like `article_volume` and `mspr`, create 90-day and 180-day rolling averages and standard deviations to capture trends.

## 5. Phase 5: The Predictive Ensemble Architecture

This phase focuses on building and training the suite of specialized predictive models.

*   **5.1. Investor Profiling:** The system will be designed around one user-defined dimension: **Investment Horizon** (e.g., "Short-Term: 1-3M", "Medium-Term: 6-18M", "Long-Term: 24M+").

*   **5.2. The Model Triplet:** For each company and each key horizon, we will train a "triplet" of models:
    1.  **Return Model (XGBoost):** Trained to predict the `Future_Return_{n}` for that horizon.
    2.  **Volatility Model (XGBoost):** Trained to predict the `Future_Volatility_{n}` for that horizon.
    3.  **Confidence Model (Logistic Regression):** A simple, interpretable model trained to predict the probability of the Return Model being correct. Its inputs will be the predictions from the other two models plus the most powerful "smart features."

## 6. Phase 6: The Strategic Decision Dashboard

This is the final output of the project—a user-facing dashboard that synthesizes all analysis into a clear, multi-faceted recommendation.

*   **6.1. Core Recommendation ("What"):**
    *   Displays a clear signal (`Strong Buy`, `Cautious Buy`, `Avoid`, etc.) based on the combined outputs of the model triplet.
    *   Shows the predicted return and volatility range for the selected horizon.

*   **6.2. Explanation Layer ("Why"):**
    *   Utilizes **SHAP (SHapley Additive exPlanations)** to generate a force plot or list of top contributing factors.
    *   Provides a human-readable explanation, e.g., "This 'Strong Buy' signal is driven primarily by a strong contrarian signal from the latest 10-Q report."

*   **6.3. Action Window ("When"):**
    *   Leverages the lag analysis from Phase 3 to provide a time-sensitive execution window.
    *   **Example Output:** "The driving signal for this recommendation has a peak predictive power with a 2-day lead time. **Recommended action window: next 2 trading days.**"

*   **6.4. Context Layer ("What If"):**
    *   **Historical Analogs:** Programmatically scans historical data to find past periods with a similar signal profile and displays the subsequent performance.
    *   **Example Output:** "This signal profile is similar to Q2 2020, which was followed by a +150% return over the next 24 months."

    