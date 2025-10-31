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

**2.2. Data Loading and Initial Inspection:**
*   **Action:** Load the company sentiment data from the `final_company_dfs.pkl` file into a dictionary of DataFrames (`company_dfs_raw`).
*   **Action:** Load the `Stockprices.csv` file into a separate DataFrame (`stocks_df_raw`).
*   **Action:** Perform an initial inspection of all raw dataframes to check for missing dates, `NaN` values, and `0` values that represent missing data.

**2.3. Sentiment Dataset Imputation Strategy:**
*   A two-tiered imputation strategy is applied to the company sentiment data:
    *   **Tier 1 (Simple Imputation):** For "less critical" features like `article_volume` and `market_average_sentiment`, `0` values are replaced with `NaN` and then imputed using forward-fill (`ffill`), followed by a backward-fill (`bfill`) to handle any leading `NaN`s.
    *   **Tier 2 (Advanced Imputation):** For "very critical" features (`average_news_sentiment`, `mspr`), a sophisticated model-based approach is used.

**2.4. Advanced Imputation Workflow:**
*   **Action: Subclass `IterativeImputer`:** An "Architect's Solution" is implemented by creating a new `EarlyStoppingIterativeImputer` class that inherits from `sklearn.impute.IterativeImputer`. This provides a stable, reusable, and robust way to enable `early_stopping_rounds` for the internal estimator without fragile monkey-patching.
*   **Action: Hyperparameter Tuning:** A `tune_xgboost_hyperparameters` function is defined. It uses `RandomizedSearchCV` with 3-fold cross-validation on the non-missing data subset to find the optimal hyperparameters for the `XGBRegressor` estimator for each specific company.
*   **Action: Master Imputation Loop:** A master loop iterates through each company:
    1.  Finds the best hyperparameters by tuning the estimator on a proxy task.
    2.  Instantiates a dedicated, uniquely tuned `EarlyStoppingIterativeImputer` for that company using the best parameters.
    3.  Applies the tuned imputer to the real, incomplete dataset.
    4.  Stores the final, fully imputed DataFrame in the `company_dfs_final` dictionary and generates a report.

**2.5. Stock Price Dataset Cleaning and Standardization:**
*   **Action: Clean and Split:** The wide-format `stocks_df_raw` is processed. A function cleans the string-based price and volume data into numeric types. The data is then split into three separate, standardized DataFrames (one for each ticker) and stored in the `stock_dfs_cleaned` dictionary.
*   **Action: Align and Impute:** Each cleaned stock DataFrame is re-indexed to a master date range (`2018-01-01` to `2024-12-31`). The resulting `NaN` values for non-trading days are imputed using `ffill` and `bfill`. The final data is stored in the `stock_dfs_final` dictionary.

---

## 3. Phase 2: Exploratory Data Analysis (EDA)

This phase focuses on understanding the characteristics of our features and their relationship with stock prices, which is critical for downstream modeling. The strategy is divided into two main components, repeated for each company (`AAPL`, `NVDA`, `GOOGL`).

**3.1. Univariate Analysis: Understanding Feature Distributions**
*   **Objective:** To understand the statistical properties and "personality" of each key feature.
*   **Features to Analyze:** `average_news_sentiment`, `market_average_sentiment`, `mspr`, `article_volume`, `10-K_sentiment`, `10-Q_sentiment`.
*   **Actions (for each feature):**
    1.  **Histogram:** Plot a histogram to visualize the value distribution (e.g., skewed, normal, bimodal).
    2.  **Box Plot:** Use a box plot to identify the median, interquartile range (IQR), and outliers, which often correspond to significant real-world events.
    3.  **Descriptive Statistics:** Compute `.describe()` to get precise numerical summaries (mean, std, min, max, quartiles).

**3.2. Bivariate Time-Series Analysis: Finding Predictive Relationships**
*   **Objective:** To visually inspect how features co-move with the stock price over time. This is crucial for identifying potential lead-lag relationships for the XGBoost model.
*   **Methodology:** A series of time-series plots with a shared date axis will be generated for each company.
*   **Key Plots to Generate:**
    1.  **Core Sentiment vs. Price:** Plot `average_news_sentiment` against the stock's `Close` price to find correlations between public narrative and price trends.
    2.  **Attention vs. Price & Volatility:** Plot `article_volume` against the `Close` price and `daily_return` to test the hypothesis that news volume spikes precede price volatility.
    3.  **Insider vs. Public Sentiment:** Plot `mspr` (insider) against `average_news_sentiment` (public) to identify potential divergences where "smart money" moves before the public narrative.
    4.  **Corporate vs. Public Sentiment:** Plot `10-Q_sentiment` against `average_news_sentiment` to analyze how daily sentiment reacts following official quarterly filings.

---

## 4. Phase 3: Correlation Analysis

This phase aims to quantify the relationships between our sentiment features and the stock price.

**4.1. Target Variable Engineering:**
*   **Action:** Create a `daily_return` column, calculated as `Close.pct_change()`. This will be our primary target for correlation.

**4.2. Correlation Matrix Heatmap:**
*   For each company's final merged DataFrame, compute the Pearson correlation matrix.
*   **Action:** Visualize this matrix as a heatmap to identify promising linear relationships.

**4.3. Scatter Plots:**
*   **Action:** For the features that show the highest correlation with `daily_return`, create scatter plots to visually inspect the relationship.

---

## 5. Phase 4: Feature Engineering and Next Steps

Based on the insights from EDA and Correlation Analysis, we will define a concrete plan for feature engineering to improve the predictive power of our XGBoost models.

**5.1. Initial Feature Engineering Plan:**
*   **Sentiment Moving Averages:** Calculate 7-day, 30-day, and 90-day rolling averages for key sentiment features.
*   **Sentiment Momentum & Volatility:** Calculate rolling standard deviation and rate-of-change.
*   **Interaction Features:** Create features like `sentiment_x_volume`.
*   **Sentiment Spread:** Isolate company sentiment from market sentiment (`average_news_sentiment - market_average_sentiment`).

**5.2. Next Steps:**
*   The results from Phase 3 will guide which of these engineered features we prioritize.
*   The final, feature-rich datasets will be prepared for input into the XGBoost training pipeline.