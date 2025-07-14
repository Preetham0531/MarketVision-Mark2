# MarketVision: AI-Powered Stock Market Prediction for NSE

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

MarketVision is a comprehensive, AI-driven platform for predicting stock market movements on the National Stock Exchange (NSE) of India. It leverages a multi-faceted approach, combining technical analysis, fundamental data, and real-time news sentiment to deliver actionable insights and multi-horizon forecasts through an intuitive and interactive web interface.

*Your screenshot here. You can add a screenshot of the running application.*
<!-- ![MarketVision Dashboard](link_to_your_screenshot.png) -->

---

## 🌟 Key Features

- **Multi-Horizon Forecasting**: Delivers predictions for multiple timeframes: 1-day, 5-day, and 20-day, catering to both short-term traders and long-term investors.
- **AI-Powered Predictions**: Utilizes a powerful **LightGBM** model for both regression (price forecasting) and classification (trend direction), trained on a rich, high-dimensional dataset.
- **Holistic Data Analysis**: Integrates a wide array of data sources for a 360-degree market view:
  - **Price & Technical Data**: Over 40 technical indicators calculated from historical price data.
  - **News Sentiment Analysis**: Real-time news analysis using a pre-trained **FinBERT** model to gauge market sentiment.
  - **Fundamental & Macro Data**: Incorporates company fundamentals and global market indices (NIFTY, S&P 500, VIX) for broader context.
- **Interactive Dashboard**: A user-friendly Streamlit application that visualizes data, predictions, and model insights in an accessible format.
- **Explainable AI (XAI)**: Provides transparency by showing the top factors (e.g., RSI, news sentiment) that influenced each prediction.
- **Risk Assessment**: Offers dynamic Stop-Loss and Take-Profit recommendations based on predicted volatility to help manage risk.

---

## 🛠️ How It Works: Architecture & Dataflow

MarketVision operates on a sophisticated pipeline that transforms raw data into actionable predictions. The entire process, from data collection to visualization, is automated and modular.

<p align="center">
  [The Mermaid diagram you generated will be displayed here in GitHub]
</p>

1.  **Data Collection**: The system fetches historical price data, global indices, and company fundamentals from `yfinance`, while news articles are scraped from Google News RSS feeds.
2.  **Feature Engineering**:
    -   **Technical Indicators**: Over 40 indicators like RSI, MACD, and Bollinger Bands are computed using `pandas-ta`.
    -   **Sentiment Analysis**: News headlines and summaries are processed by a FinBERT model to generate sentiment scores.
    -   **Macro Context**: Features like Beta (vs. NIFTY 50) and sector performance are calculated to place the stock in the context of the broader market.
3.  **Model Training**: The engineered features are used to train a `MultiOutput` LightGBM model. This model learns the complex relationships between the features and future price movements, allowing it to predict multiple targets (e.g., price, trend, volatility) simultaneously.
4.  **Live Prediction & Interface**: For real-time predictions, the application fetches the latest market data, runs it through the feature engineering pipeline, and feeds the result to the trained model. The output is then beautifully rendered in the Streamlit dashboard.

---

## ⚙️ Technology Stack

-   **Backend & Data Science**: Python, Pandas, NumPy, Scikit-learn
-   **Machine Learning**: LightGBM, XGBoost, Transformers (Hugging Face)
-   **Data Sources**: yfinance, Feedparser (for Google News)
-   **Frontend & Visualization**: Streamlit, Plotly
-   **Development Tools**: Jupyter (for exploration), Git

---

## 🚀 Getting Started

Follow these instructions to set up and run the MarketVision application on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   `pip` and `venv`

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd MarketVision
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once the setup is complete, you can launch the Streamlit application:

```bash
streamlit run marketvision_streamlit.py
```

Your web browser should automatically open to the application's dashboard.

---

## 📂 Project Structure

The project is organized into a modular structure for clarity and maintainability:

```
MarketVision/
├── data/                 # Raw, processed, and final training data
├── interface/            # Scripts for live prediction interfaces (e.g., CLI)
├── logs/                 # Log files, evaluation reports, and charts
├── models/               # Trained models (.pkl) and metadata (.json)
├── training/             # Scripts for the complete ML pipeline
│   ├── fetch_price_data.py
│   ├── compute_indicators.py
│   ├── news_sentiment.py
│   ├── fetch_market_context.py
│   ├── merge_and_label.py
│   └── train_model.py
├── marketvision_streamlit.py # The main Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚖️ Disclaimer

This application is intended for educational and research purposes only. The predictions and recommendations provided are generated by machine learning models based on historical data and should **not** be considered financial advice. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions. Past performance is not indicative of future results.

---

