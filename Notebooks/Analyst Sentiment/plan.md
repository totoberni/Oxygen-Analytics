# Advanced Financial NLP Implementation Plan: AAPL, NVDA, GOOGL

This document outlines a new, comprehensive strategy for building a financial sentiment analysis pipeline. This revised plan shifts away from relying solely on the Finnhub API and instead adopts a more robust, multi-source approach to data acquisition for `AAPL`, `NVDA`, and `GOOGL` from 2018 to 2024.

The new data gathering strategy is three-pronged, designed to capture sentiment from distinct, complementary perspectives:
1.  **Insider Sentiment:** Using the **Finnhub API** for its reliable historical `stock/insider-sentiment` data.
2.  **Public News Sentiment:** Using pre-compiled **Hugging Face datasets** for broad historical news coverage.
3.  **Corporate Sentiment:** Using **`edgartools`** to download and analyze the text of official company filings (`10-K`, `10-Q`, `8-K`).

---

### **Phase 1: Environment Setup and Configuration**

This phase is updated to include libraries for accessing Hugging Face datasets and the SEC EDGAR database.

1.  **Install Python Libraries:**
    Install all necessary libraries. This now includes `datasets` for Hugging Face and `edgartools` for SEC filings.
    ```bash
    # Use sys.executable to ensure installation into the correct kernel
    import sys
    !{sys.executable} -m pip install finnhub-python pandas transformers torch --quiet
    !{sys.executable} -m pip install datasets --quiet
    !{sys.executable} -m pip install edgartools --quiet
    ```

2.  **Set Up API Keys and Identity:**
    *   **Finnhub:** Store your API key in an environment variable.
    ```bash
    export FINNHUB_API_KEY='your_api_key_here'
    ```
    *   **EDGAR:** The SEC requires a user-agent for all programmatic requests. Set your identity using `edgartools` before making any calls.
        ```python
        from edgar import set_identity
        # Format: "Sample Company Name your.email@example.com"
        set_identity("Financial Analysis Corp analyst@financialanalysis.com")
        ```

---

### **Phase 2: Multi-Source Data Acquisition (2018-2024)**

This phase implements the three distinct data sourcing strategies.

#### **Strategy 2.1: Insider Sentiment via Finnhub API**
This part of the strategy remains. We will loop through each year for each stock to get the historical insider transaction data.
```python
import finnhub
import os

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY"))
all_insider_data = []

# Loop through stocks and years 2018-2024...
# (code from previous implementation remains valid)
```

#### **Strategy 2.2: Historical News via Hugging Face Datasets**
We will load and filter this large, pre-compiled datasets of financial news. Using `streaming=True` is highly recommended to handle these large datasets efficiently. This dataset should retrieve a set of text data for title and text rlated to the specific TARGET TICKET of the AAPL, NVDA, ALPHABET companies. These texts should be organized by Q1-Q4 format over 2018-2024 years. This text data will later be used to gather an average composite sentiment between text and headline, assigning it to a metric for "[RATGET_TICKER] News Sentiment"

**2.2.1: Filtering `Brianferrell787/financial-news-multisource` Dataset**
```python
import json
import pandas as pd
from datasets import load_dataset

TARGET_TICKERS = ['AAPL', 'NVDA', 'GOOGL']

try:
    multisource_dataset = load_dataset("Brianferrell787/financial-news-multisource", name="wikinews_articles", streaming=True, split="train")
    
    filtered_articles = []
    for article in iter(multisource_dataset):
        try:
            extra_data = json.loads(article['extra_fields'])
            if 'stocks' in extra_data and any(ticker in extra_data['stocks'] for ticker in TARGET_TICKERS):
                filtered_articles.append({
                    'date': article['date'],
                    'text': article['text'],
                    'tickers': extra_data.get('stocks', [])
                })
        except (json.JSONDecodeError, TypeError):
            continue
    
    df_multisource = pd.DataFrame(filtered_articles)
    print(f"Extracted {len(df_multisource)} articles from financial-news-multisource.")

        except Exception as e:
    print(f"An error occurred with financial-news-multisource: {e}")
```

**2.2.2: S&P 500 with Financial News Headlines (2008-2024)**
This is a Kaggle pre-imported dataset. It has been downloaded and imported in the ./data directory as it is sufficiently small to do so.\
We will gather 200 headlines for each Q1-Q4 period in the years 2018-2024. These headlines should be generic and refer to the market as a whole.\
Later, we will compute an average market sentiment score for each Q1-Q4 period in years 2018-2024. This will serve as a "General Market Sentiment" metric.

**2.2.3: SEC Filings via `edgartools`**
We will use the `edgartools` library to download the full text of all `10-K`, `10-Q`, and `8-K` filings for our tickers and date range.
This text data too must be collected for the Q1-Q4 periods of the 2018-2024 years. It will be used to compute a composite sentiment score for each form type (10-K, 10-Q, 8-K) for each Q1-Q4 period in years 2018-2024 of each [TARGET_TICKER]. These values will be used to create a new composite sentiment metric named "[TARGET_TICKER] Office Sentiment".

```python
import pandas as pd
from edgar import Company, set_identity

set_identity("Financial Analysis Corp analyst@financialanalysis.com")

TARGET_TICKERS = ['AAPL', 'NVDA', 'GOOGL']
FORM_TYPES = ["10-K", "10-Q", "8-K"]
DATE_RANGE = "2018-01-01:2024-12-31"

all_filings_data = []

for ticker in TARGET_TICKERS:
    try:
        company = Company(ticker)
        filings = company.get_filings().filter(date=DATE_RANGE, form=FORM_TYPES)
        
        for filing in filings:
            all_filings_data.append({
                'ticker': ticker,
                'form': filing.form,
                'filing_date': filing.filing_date,
                'text': filing.text() # Extracts the clean filing text
            })
    except Exception as e:
        print(f"Could not process filings for {ticker}: {e}")

df_filings = pd.DataFrame(all_filings_data)
print(f"Data extraction complete. Total filings extracted: {len(df_filings)}")
```

---

### **Phase 3: NLP Sentiment Analysis on All Textual Data**

The scope of this phase is now expanded. We will apply `FinBERT` not only to news but also to the much larger and more complex text from SEC filings.

1.  **Initialize FinBERT Pipeline:**
    ```python
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="ahmedrachid/FinancialBERT-Sentiment-Analysis")
    ```

2.  **Apply FinBERT to News and Filings:**
    *   **News:** Apply the sentiment pipeline directly to the 'headline' and 'summary' columns of the news DataFrames.
    *   **Filings:** The text from `df_filings` is often too long for `FinBERT`. It must be chunked into smaller segments (e.g., paragraphs) before analysis. The sentiment scores for the chunks can then be averaged to get a score for the entire filing.

    **Example Application (on chunked text):**
    ```python
    # Assume 'filing_chunks' is a list of text strings from a single filing
    # results = sentiment_pipeline(filing_chunks)
    #
    # # Average the scores to get a single sentiment value for the document
    # import numpy as np
    # compound_scores = [ (r['score'] if r['label'] == 'positive' else -r['score']) for r in results ]
    # final_filing_score = np.mean(compound_scores)
    ```

---

### **Phase 4: Data Consolidation and Final Aggregation**

This phase involves integrating our three distinct data sources into a final, quarterly time series.

1.  **Process and Align Data:**
    *   **Insider Sentiment (`insider_df`):** Aggregate to quarterly.
    *   **News Sentiment (`news_df`):** Aggregate daily scores (mean sentiment, count of articles) and then aggregate to quarterly.
    *   **Filing Sentiment (`filings_df`):** Create a DataFrame with filing dates and scores. This is event-driven data.

2.  **Merge and Aggregate:**
    Join the three datasets on their date index using an outer join, then resample the merged DataFrame to a quarterly (`'Q'`) frequency.

3.  **Proposed Final Data Structure:**
    | Quarter | Insider_Sentiment_MSPR_Mean | News_Sentiment_Mean | Market_Sentiment_Mean | Filing_Sentiment_Mean |
    | :--- | :--- | :--- | :--- | :--- |
    | 2023-Q4 | 10.5 | 0.18 | 550 | 0.05 |
    | 2024-Q1 | -5.2 | -0.04 | 620 | 0.12 |

---

### **Phase 5: Strategic Outlook**

1.  **New Challenges:**
    *   **SEC Filing Parsing:** While `edgartools` provides clean text, identifying the most relevant sections (e.g., "Risk Factors," "MD&A") within that text is a key challenge for more nuanced analysis.
    *   **Scalability:** The volume of text from news and filings is substantial. The NLP processing step will be computationally intensive.

2.  **Next Steps:**
    *   Implement the full data pipeline as outlined.
    *   Develop a robust text-chunking strategy for long SEC filings.
    *   Begin exploratory data analysis on the final aggregated dataset.
