
# Financial Sentiment EDA and Correlation Analysis: Project Roadmap

## 1. Project Overview

**Objective:** To explore, clean, and analyze financial sentiment datasets for Apple (AAPL), NVIDIA (NVDA), and Google (GOOGL). The primary goal is to identify which sentiment and NLP-derived features are most correlated with stock price movements.

**End Goal:** The insights and cleaned data from this analysis will serve as the foundation for feature engineering and selection for a series of company-specific XGBoost models designed to predict daily stock price changes.

**Data Assets:**
*   `AAPL_dataset.csv`, `NVDA_dataset.csv`, `GOOGL_dataset.csv`: Daily sentiment and NLP features.
*   `final_company_dfs.pkl`: A pickle file containing the raw company sentiment dataframes in a dictionary.
*   `Stockprices.csv`: Historical daily stock data (OHLCV) for all three companies.

---

## 2. Phase 1: Setup and Data Unification

This phase focuses on creating clean, standardized, and unified datasets. The process has been refined into a multi-step workflow to handle different data types and missing value patterns robustly.

**2.1. Initial Setup & Verification:**
*   **Action:** Centralize all library installations (`pandas`, `numpy`, `xgboost`, etc.) in the first cell, followed by a mandatory kernel restart.
*   **Action:** Centralize all library imports in the second cell.
*   **Action:** Implement a robust, multi-layer verification script to:
    1.  Confirm NVIDIA driver access via the `nvidia-smi` command.
    2.  Check that the installed `xgboost` package was built with CUDA support.
    3.  Run a small test `fit()` command to confirm the `xgboost` library can successfully initialize the GPU at runtime.

**2.2. Data Loading and Checkpointing:**
*   **Action:** Load the company sentiment data from `final_company_dfs.pkl` and stock price data from `Stockprices.csv`.
*   **Action:** Create separate checkpoint cells to export the final cleaned and imputed dataframe dictionaries (`company_dfs_final` and `stock_dfs_final`) as `.pkl` files.
*   **Action:** Implement a "resume" cell at the beginning of Phase 2 to load these checkpointed `.pkl` files, allowing the user to bypass the entire cleaning pipeline.

**2.3. Sentiment Dataset Imputation Strategy:**
*   A two-tiered imputation strategy is applied:
    *   **Tier 1 (Simple Imputation):** For `article_volume` and `market_average_sentiment`, `0` values are replaced with `NaN` and imputed using `ffill` and `bfill`.
    *   **Tier 2 (Advanced Imputation):** For `average_news_sentiment` and `mspr`, a sophisticated model-based approach is used.

**2.4. Advanced Imputation Workflow:**
*   **Action: Hyperparameter Tuning:** A `tune_xgboost_hyperparameters` function uses `RandomizedSearchCV` to find the optimal hyperparameters for the `XGBRegressor` estimator for each specific company.
*   **Action: Master Imputation Loop:** A master loop iterates through each company, creating a dedicated, uniquely tuned `IterativeImputer` with early stopping to fill missing values.

**2.5. Stock Price Dataset Cleaning and Standardization:**
*   **Action: Clean and Split:** A function cleans the string-based price and volume data into numeric types. The wide-format `stocks_df_raw` is then split into three separate, standardized DataFrames.
*   **Action: Align and Impute:** Each stock DataFrame is re-indexed to a master date range (`2018-01-01` to `2024-12-31`). The resulting `NaN` values for non-trading days are imputed using `ffill` and `bfill`.

---

## 3. Phase 2: Exploratory Data Analysis (EDA)

This phase visually explores feature characteristics and their relationships with stock prices. The strategy is repeated for each company (`AAPL`, `NVDA`, `GOOGL`).

**3.1. Univariate Analysis: Understanding Feature Distributions**
*   **Objective:** To understand the statistical properties of each key feature.
*   **Actions:** For each feature, generate a histogram, a box plot, and descriptive statistics.

**3.2. Bivariate Time-Series Analysis: Finding Predictive Relationships**
*   **Objective:** To visually inspect how sentiment features co-move with stock price and volatility over time.
*   **Methodology:** Merge the sentiment and stock dataframes for each company. Generate a series of time-series plots.

---

## 4. Phase 3: Multi-Horizon Correlation and Predictive Power Analysis

This phase moves beyond visual EDA to **quantify** the predictive power of sentiment features across multiple investment time horizons. The goal is to identify the most promising features, their optimal time lags, and the timeframes over which they are most effective.

**4.1. Engineering a Spectrum of Predictive Targets:**
*   **Objective:** To create a rich set of target variables that capture future returns and volatility over short, medium, and long-term windows.
*   **Actions:**
    1.  **Consolidate Data:** Merge the final sentiment and stock data for each company into a unified `eda_dfs` dictionary.
    2.  **Define Horizons:** Establish a set of lookahead periods: `[1D, 1M, 3M, 6M, 12M, 18M, ..., 84M]`.
    3.  **Engineer Return Targets:** For each horizon `n`, create a `Future_Return_{n}` column, calculating the total percentage return between day `t` and day `t + n_months`.
    4.  **Engineer Volatility Targets:** For each horizon `n`, create a `Future_Volatility_{n}` column, calculating the rolling standard deviation of `daily_return` over the next `n` months.

**4.2. Systematic Multi-Horizon Lag Analysis:**
*   **Objective:** To discover the optimal predictive lag for every combination of sentiment feature and multi-horizon target.
*   **Methodology:**
    1.  **Define Lag Function:** Create a `calculate_lag_correlations` function that iterates through a range of time lags (e.g., `t-5` to `t+5` days), calculating both **Pearson** and **Spearman** correlation for each feature against the full suite of future targets.
*   **Action:** Run this systematic analysis for each company (`AAPL`, `NVDA`, `GOOGL`).

**4.3. "Meta-Analysis" Visualization and Interpretation:**
*   **Objective:** To distill the complex correlation results into clear, high-level summary visualizations.
*   **Methodology:**
    1.  **Define Heatmap Function:** Create a `plot_peak_correlation_heatmap` function. This function finds the maximum absolute correlation for each feature-horizon pair across all tested lags and visualizes the result as a heatmap. Crucially, it must safely handle all-NaN groups to prevent errors.
    2.  **Define Lag Curve Function:** Create a `plot_lag_curves` function to generate "drill-down" plots for specific feature/horizon combinations.
*   **Actions:**
    1.  **Generate Peak Correlation Heatmaps:** For each company, use the heatmap function to generate two primary visualizations (one for Future Returns, one for Future Volatility). Each cell will be annotated with the peak correlation value and the optimal lag at which it occurred (e.g., "-0.89 @ -5d").
    2.  **Drill Down with Lag Plots:** Based on the most interesting signals from the heatmaps, use the lag curve function to generate detailed plots for specific combinations (e.g., `10-Q_sentiment` vs. long-term returns).

**4.4. Synthesis and Action Plan:**
*   **Objective:** To summarize the multi-horizon findings into a precise feature engineering blueprint.
*   **Action:** For each company, create a summary markdown cell structured by investment horizon (Short-Term, Medium-Term, Long-Term, Volatility). This will list the most predictive features, their peak correlation scores, their optimal lags, and a brief interpretation of the strategic implication.

---

## 5. Phase 4: Feature Engineering and Next Steps

Based on the precise, quantitative insights from the Multi-Horizon Correlation Analysis in Phase 3, we will define a highly targeted feature engineering plan.

**5.1. Data-Driven Feature Engineering Plan:**
*   **Objective:** Create new features that capture the predictive signals discovered in Phase 3.
*   **Example Actions:**
    *   **Lagged Features:** If `article_volume` at `t-2` was found to be predictive of `Future_Return_3M`, create a new `article_volume_lag_2` column.
    *   **Rolling Averages on Predictive Lags:** If `average_news_sentiment` at `t-3` is significant for the 6-month horizon, create 7-day and 30-day rolling averages of `average_news_sentiment_lag_3`.
    *   **Momentum Indicators:** Calculate rolling standard deviation (`volatility`) and rate-of-change (`momentum`) for the most promising feature/lag/horizon combinations.

**5.2. Next Steps:**
*   The results from Phase 3 will dictate exactly which of these engineered features we prioritize for each investment horizon.
*   The final, feature-rich datasets will be prepared for input into the XGBoost training pipeline, with a clear understanding of why each feature was created and which prediction horizon it is best suited for.