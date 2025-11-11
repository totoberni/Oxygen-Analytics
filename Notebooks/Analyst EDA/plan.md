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

## 5. Phase 5: Stakeholder Visualization & Insight Validation

**Objective:** To translate the quantitative findings from the correlation analysis (Phase 3) into compelling, easy-to-understand visualizations for business stakeholders (hedge fund managers). This phase will validate the identified signals and build the business case for the subsequent feature engineering phase.

**5.1. Core Logic: Isolate Filing Events**
*   **Objective:** The sentiment data is forward-filled. We must first identify the exact dates on which new filings were released.
*   **Action:** Create a reusable Python function `get_filing_dates(df, sentiment_column)`.
*   **Methodology:** This function will take a company DataFrame and a sentiment column name (e.g., `'10-Q_sentiment'`) as input. It will identify event dates by finding where the sentiment score `.diff() != 0`. It must return a new DataFrame containing only the filing dates and their corresponding non-NaN sentiment scores.

**5.2. Visualization Part 1: Event-Study Charts for Short-Term Price Impact**
*   **Objective:** Visualize the average stock price trajectory around filing events, segmented by sentiment, to answer: "How does the market react in the days immediately following a filing, based on its sentiment?"
*   **Actions:**
    1.  **Define a master plotting function:** `plot_event_study(company_df, filing_events_df, filing_type, stock_price_col='Close')`.
    2.  **Data Preparation (within function):** For each filing event date `t`:
        *   Slice a 21-day window (`t-10` to `t+10`) from the main `company_df`.
        *   **Normalize Price:** Normalize the stock price column by dividing all values in the window by the price at `t-1` and multiplying by 100. This converts the price to a common scale showing percentage change relative to the day before the event.
    3.  **Sentiment Grouping (within function):**
        *   Use `pd.qcut` to divide all filing events into three quantiles based on their sentiment scores: 'Low Sentiment' (bottom 33%), 'Mid Sentiment' (middle 34%), and 'High Sentiment' (top 33%).
    4.  **Aggregation (within function):**
        *   For each sentiment group, calculate the `mean` of all the normalized price series at each day from -10 to +10. This produces the average trajectory for each group.
    5.  **Visualization (within function):**
        *   Plot the three average trajectories on a single chart using `matplotlib` or `seaborn`.
        *   Use a clear, intuitive color scheme (e.g., Red for Low, Gray for Mid, Green for High).
        *   Add a vertical dashed line at `x=0` labeled "Filing Date".
        *   Set a descriptive title, legend, and axis labels (`Days Relative to Filing`, `Normalized Price (Day -1 = 100)`).
*   **Execution:** Create a loop to execute this plotting function for each company (AAPL, NVDA, GOOGL) and each filing type (10-K, 10-Q, 8-K), generating 9 distinct event-study charts.

**5.3. Visualization Part 2: Scatter Plots for Long-Term ROI & Volatility Impact**
*   **Objective:** Visually confirm the strong, long-term correlations discovered in Phase 3 to answer: "How predictive is filing sentiment for multi-year returns and volatility?"
*   **Actions:**
    1.  **Identify Key Relationships:** From the Phase 3 markdown summaries, codify the most powerful feature/target pairs into a configuration list. Example: `[('NVDA', '10-Q_sentiment', 'Future_Return_72M'), ('GOOGL', 'mspr', 'Future_Volatility_48M')]`.
    2.  **Define a master plotting function:** `plot_correlation_scatter(company_df, filing_events_df, feature_col, target_col)`.
    3.  **Data Preparation (within function):** Use the `filing_events_df` (from 5.1) to get the specific dates and sentiment scores. For each of these dates, retrieve the corresponding future outcome value (e.g., `Future_Return_72M`) from the main `company_df`.
    4.  **Visualization (within function):**
        *   Use `seaborn.regplot` to create a scatter plot of the `feature_col` (sentiment) vs. the `target_col` (outcome). This function automatically includes the regression line and a confidence interval, which is ideal for this use case.
        *   Set a clear title that includes the company, feature, and target (e.g., "NVDA: 10-Q Sentiment vs. 72-Month Future Return").
*   **Execution:** Iterate through the list of key relationships, calling the plotting function for each one.

**5.4. Synthesis and Final Reporting**
*   **Action:** Create a final summary markdown cell in the `AnalystSentimentEDA.ipynb` notebook.
*   **Content:** This cell will display the most impactful charts generated in this phase. Each chart will be accompanied by a concise, business-friendly takeaway that directly references the visualization and its strategic implication (e.g., "As seen below, pessimistic 10-Q reports for NVDA have historically preceded periods of significant long-term outperformance, highlighting a powerful contrarian signal.").

---
## 6. Phase 6: Prescriptive Strategy Rulebook Generation

**Objective:** To translate the quantitative findings from the correlation analysis into a data-driven, actionable investment rulebook for each company. This rulebook will provide clear, evidence-based instructions for when to act, what action to take, and what outcomes (ROI, Volatility, Probability of Gain) can be expected for various investment horizons.

**Core Concept: From Correlation to Conditional Probability**
*   While Phase 3 identified *that* a relationship exists between a signal (e.g., `10-K_sentiment`) and a future outcome (e.g., `Future_Return_24M`), this phase determines the *nature and probability* of that outcome.
*   We shift from asking "Are these related?" to asking "**Given that we observe signal X, what is the historical probability distribution of outcome Y?**"
*   By segmenting a signal into quantiles (e.g., Top 10%, Bottom 10%), we can analyze the specific historical performance of trades made under those conditions. This allows us to calculate the empirical `Probability of Gain`, `Average ROI`, and `Average Volatility` for each condition, forming the statistical foundation of our rulebook.

**Systematic Implementation Plan**
1.  **6.1: Consolidate and Filter Potent Signals**:
    *   **Action:** Aggregate the correlation results from all three companies (`AAPL`, `NVDA`, `GOOGL`) into a single master DataFrame.
    *   **Action:** Systematically filter this DataFrame to isolate "potent signals," defined as `(Feature, Lag, Target)` triplets that exceed a specific Spearman correlation threshold (e.g., `> 0.20`) and match our desired investment horizons (`12M` to `84M`).
2.  **6.2: Define the Conditional Outcome Analysis Engine**:
    *   **Action:** Create a master function, `analyze_signal_outcomes`.
    *   **Methodology:** This function will take a potent signal as input. It will programmatically segment the signal's historical values into deciles (10% quantiles). For each decile, it will calculate the resulting distribution of outcomes, computing the `Probability of Gain`, `Estimated ROI`, and `Estimated Volatility`. It will then assign a prescriptive `Action` (e.g., 'Strong Buy', 'Sell') based on the decile.
3.  **6.3: Generate Company-Specific Rulebooks**:
    *   **Action:** Create a loop that iterates through each company (`AAPL`, `NVDA`, `GOOGL`).
    *   **Methodology:** Inside the loop, run the `analyze_signal_outcomes` function for every potent signal relevant to that specific company. The results for each company will be stored separately, creating three distinct, machine-readable rulebooks.
4.  **6.4: Synthesize and Display Actionable Rules**:
    *   **Action:** For each company, create a final summary markdown cell.
    *   **Methodology:** This cell will display a clean, formatted, and styled table presenting the most actionable rules (e.g., only 'Buy'/'Sell' signals). The table will be organized by investment horizon and will clearly state the `Signal`, `Condition`, `Action`, `Timing`, `Probability of Gain`, `Estimated ROI`, and `Estimated Volatility`, providing a clear and final prescriptive guide for stakeholders.

**Next Steps:** The generated rulebooks serve as a powerful standalone prescriptive tool. They also provide a robust baseline and feature validation for more complex predictive ensemble models planned in future phases.