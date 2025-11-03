# Oxygen Analytics: Master Investment Strategy Plan

## 1. Project Objective

To develop a multi-faceted, data-driven investment decision framework for AAPL, NVDA, and GOOGL. The framework will leverage sentiment analysis and machine learning to provide actionable, horizon-specific investment recommendations, complete with model-driven explanations, confidence scores, and optimal action windows.

## 2. Phase 1 & 2: Data Foundation & EDA (Completed)

This phase focused on cleaning, imputing, and exploring the sentiment and stock price datasets. The key outcome is a set of clean, merged dataframes for each company, ready for quantitative analysis.

## 3. Phase 3: Multi-Horizon Correlation Analysis (Completed)

This phase quantified the predictive power of sentiment features across multiple investment horizons (1D to 84M) and time lags (-5d to +5d).

*   **Outcome:** A unique **"Signal Intelligence Playbook"** for each company. This playbook identifies which sentiment features are leading indicators, contrarian signals, or momentum drivers, and for which time horizons they are most relevant. It serves as the strategic foundation for all subsequent steps.

## 4. Phase 4: Advanced Feature Engineering & Model Input Preparation

This phase will translate the insights from the Signal Intelligence Playbook into a powerful, company-specific set of features for our predictive models.

*   **4.1. Lagged Features:** Based on the optimal lags identified in Phase 3, create lagged versions of the most time-sensitive features. This is less critical for persistent signals like `10-K_sentiment` but crucial for more fleeting signals.

*   **4.2. Smart Composite Features:** Engineer a small set of high-impact interaction features designed to capture the unique dynamics of each company.
    *   **For AAPL & GOOGL:**
        *   **`10K_vs_10Q_Divergence`**: Calculated as `10-K_sentiment - 10-Q_sentiment`. This feature quantifies the divergence between positive long-term outlooks and short-term pessimism, aiming to capture powerful value opportunities.
    *   **For NVDA:**
        *   **`Contrarian_Conviction`**: Calculated as `mspr * (-1 * 10-Q_sentiment)`. This feature is designed to explicitly measure moments of "smart money buying the dip"—when insiders are bullish during periods of official pessimism.

*   **4.3. Rolling Metrics:** For key volatility predictors like `article_volume` and `mspr`, create 90-day and 180-day rolling averages and standard deviations. This captures the trend and consistency of these signals, separating sustained attention from isolated spikes.

## 5. Phase 5: The Predictive Ensemble Architecture

This phase focuses on building and training a suite of specialized predictive models tailored to specific investor profiles.

*   **5.1. Investor Profiling & Model Specialization:** The system will be designed around user-selected **Investment Horizons**. A separate "Model Triplet" will be trained for each key horizon (e.g., 3M, 12M, 24M, 48M) for each company.

*   **5.2. The Model Triplet Architecture:** For each company/horizon pair, we will train a triplet of models that feed into each other in a specific sequence.

    1.  **Return Model (XGBoost):**
        *   **Purpose:** To predict the magnitude and direction of future returns.
        *   **Inputs:** The full set of base sentiment features, lagged features, and smart composite features.
        *   **Target:** `Future_Return_{n}` (e.g., `Future_Return_24M`).
        *   **Output:** A predicted return value (e.g., `+35.5%`).

    2.  **Volatility Model (XGBoost):**
        *   **Purpose:** To predict the expected risk and turbulence over the investment horizon.
        *   **Inputs:** The full set of base sentiment features, lagged features, and smart composite features.
        *   **Target:** `Future_Volatility_{n}` (e.g., `Future_Volatility_24M`).
        *   **Output:** A predicted volatility score (e.g., `0.85`).

    3.  **Confidence Model (Logistic Regression):**
        *   **Purpose:** To provide a statistically learned confidence score in the Return Model's prediction. It answers the question: "Given the current signals, how reliable is the return forecast?"
        *   **Inputs:**
            *   The predicted return from the Return Model.
            *   The predicted volatility from the Volatility Model.
            *   The most powerful "Smart Composite Feature" for that company (e.g., `Contrarian_Conviction` for NVDA).
            *   The **Optimal Lag** (in days) derived programmatically from the lag curves in Phase 3 for the primary driving feature.
        *   **Target:** A binary value indicating if the Return Model's prediction was historically correct (i.e., `sign(predicted_return) == sign(actual_return)`).
        *   **Output:** A **Confidence Score** (probability from 0.0 to 1.0).

## 6. Phase 6: The Strategic Decision Dashboard

This is the final user-facing output, synthesizing the ensemble's analysis into a clear, multi-faceted recommendation.

*   **6.1. Core Recommendation ("What"):**
    *   Displays a clear, synthesized signal (`Strong Buy`, `Cautious Buy`, `Hold/Avoid`, `Cautious Sell`, `Strong Sell`) based on the combined outputs of the model triplet.
    *   Presents the primary model outputs: **Predicted Return**, **Expected Volatility**, and **Confidence Score**.

*   **6.2. Explanation Layer ("Why"):**
    *   Utilizes **SHAP (SHapley Additive exPlanations)** to generate a force plot showing the top 3-5 features driving the return prediction.
    *   Provides a human-readable summary, e.g., "This 'Strong Buy' signal is driven primarily by a strong contrarian signal from the latest 10-Q report, reinforced by a high `Contrarian_Conviction` score."

*   **6.3. Dynamic Action Window ("When"):**
    *   Leverages the full lag-correlation curve of the primary driving feature identified by SHAP.
    *   **Output:** Displays the curve and provides a dynamic timing recommendation.
    *   **Example:** "The driving signal (`10-Q_sentiment`) for this recommendation shows its peak predictive power at a **2-day lead time**. The signal's strength remains above 90% of its peak for 3 days. **Recommended Action Window: Next 2-3 trading days.**"

*   **6.4. Context Layer ("What If"):**
    *   **Historical Analogs:** Programmatically scans historical data to find the top 1-2 past periods where the key feature signals were most similar to the current state.
    *   **Output:** Displays a mini-chart showing the subsequent price action from those past periods.
    *   **Example:** "This signal profile is highly similar to the conditions seen in Q2 2020, which was followed by a +150% return over the subsequent 24 months."