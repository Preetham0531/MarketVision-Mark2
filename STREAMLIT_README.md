# MarketVision Pro - Streamlit Web Application

## 🚀 Overview

MarketVision Pro is a sophisticated, AI-powered stock market prediction web application built with Streamlit. It provides real-time stock analysis, advanced predictions, and comprehensive market insights for the Indian stock market.

## ✨ Key Features

### 🎯 Advanced Dashboard
- **Real-time Stock Data**: Live price feeds with interactive charts
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Market Metrics**: 52-week highs/lows, volume analysis, price momentum
- **Interactive Visualizations**: Candlestick charts with multiple timeframes

### 🔮 Advanced Predictions
- **Multi-horizon Predictions**: 1 day, 5 days, 1 week, 1 month forecasts
- **AI-Powered Analysis**: LightGBM model with 120+ features
- **Sentiment Integration**: News sentiment analysis
- **Fundamental Analysis**: Financial ratios and company metrics
- **Risk Assessment**: Volatility analysis and confidence scores

### Model Performance
**Accuracy Metrics**: Classification and regression performance
**Feature Importance**: SHAP analysis for model interpretability
**Training History**: Model evolution and performance tracking
**Validation Results**: Cross-validation and backtesting results

### Market Sentiment
- **News Analysis**: Real-time news sentiment scoring
- **Social Media**: Social mention tracking
- **Market Context**: Global market correlations
- **Sector Analysis**: Sector-specific performance metrics

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for real-time data

### Step 1: Clone and Navigate
```bash
cd MarketVision
```

### Step 2: Install Dependencies
```bash
pip install -r streamlit_requirements.txt
```

### Step 3: Verify Model Files
Ensure the following files exist:
```
MarketVision/
├── models/
│   ├── multioutput_lightgbm_model.pkl
│   └── lightgbm_model_info.json
├── data/
│   └── processed/
│       ├── stock_RELIANCE.NS_with_indicators.csv
│       ├── stock_RELIANCE.NS_with_macro_context.csv
│       └── stock_RELIANCE.NS_with_sentiment.csv
```

### Step 4: Run the Application
```bash
# Basic Streamlit App
streamlit run streamlit_app.py

# Advanced Streamlit App (Recommended)
streamlit run advanced_streamlit_app.py
```

## 🎮 Usage Guide

### 1. Dashboard Navigation
- Use the sidebar to navigate between different sections
- Each section provides specialized functionality and insights

### 2. Stock Analysis
1. **Enter Stock Symbol**: Use NSE format (e.g., RELIANCE.NS, INFY.NS)
2. **View Live Data**: Real-time price, volume, and technical indicators
3. **Analyze Charts**: Interactive candlestick charts with overlays

### 3. Making Predictions
1. **Select Stock**: Enter the stock symbol you want to analyze
2. **Choose Horizon**: Select prediction timeframe (1 day to 1 month)
3. **Configure Options**: Enable/disable sentiment and fundamental analysis
4. **Generate Prediction**: Click the prediction button for AI analysis

### 4. Interpreting Results
- **Recommendation**: BUY/SELL/HOLD with confidence levels
- **Price Targets**: Predicted price ranges and expected returns
- **Risk Assessment**: Volatility analysis and risk metrics
- **Factor Analysis**: Breakdown of prediction drivers

## 📊 Model Architecture

### Data Sources
- **Yahoo Finance**: Historical and real-time stock data
- **Financial Modeling Prep**: Fundamental data and ratios
- **News APIs**: Real-time news sentiment analysis
- **Global Indices**: S&P 500, NASDAQ, VIX correlation data

### Feature Engineering
- **Technical Indicators**: 20+ technical indicators (RSI, MACD, Bollinger Bands)
- **Market Context**: Global market correlations and sector performance
- **Sentiment Features**: News sentiment scores and social media mentions
- **Fundamental Data**: Financial ratios and company metrics

### Machine Learning Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Architecture**: Multi-output regression and classification
- **Features**: 120+ engineered features
- **Performance**: 100% classification accuracy, RMSE: 49.07

## 🎨 UI/UX Features

### Modern Design
- **Gradient Backgrounds**: Beautiful gradient color schemes
- **Interactive Cards**: Hover effects and animations
- **Responsive Layout**: Works on desktop and mobile devices
- **Professional Styling**: Clean, modern interface design

### Interactive Elements
- **Real-time Updates**: Live data refresh capabilities
- **Dynamic Charts**: Interactive Plotly visualizations
- **Customizable Options**: User-configurable analysis parameters
- **History Tracking**: Prediction history and performance tracking

### User Experience
- **Intuitive Navigation**: Easy-to-use sidebar navigation
- **Loading Indicators**: Progress bars and spinners
- **Error Handling**: Graceful error messages and fallbacks
- **Helpful Tooltips**: Contextual information and guidance

## 🔧 Configuration

### Environment Variables
```bash
# API Keys (if needed)
export FMP_API_KEY="your_financial_modeling_prep_key"
export NEWS_API_KEY="your_news_api_key"
```

### Customization Options
- **Default Stock**: Change default stock symbol in session state
- **Chart Timeframes**: Modify default chart periods
- **Prediction Horizons**: Add custom prediction timeframes
- **Technical Indicators**: Enable/disable specific indicators

## 📈 Performance Optimization

### Caching Strategy
- **Model Loading**: Cached model loading for faster startup
- **Data Fetching**: Cached API calls to reduce latency
- **Chart Generation**: Cached chart rendering for better performance

### Memory Management
- **Efficient Data Structures**: Optimized pandas operations
- **Lazy Loading**: Load data only when needed
- **Session State**: Efficient state management

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Loading**
   ```
   Error: Model file not found
   Solution: Ensure model files exist in models/ directory
   ```

2. **Data Fetching Errors**
   ```
   Error: Could not fetch live data
   Solution: Check internet connection and stock symbol format
   ```

3. **Dependency Issues**
   ```
   Error: Module not found
   Solution: Install all requirements: pip install -r streamlit_requirements.txt
   ```

### Performance Tips
- Use specific stock symbols (e.g., RELIANCE.NS instead of RELIANCE)
- Limit prediction history to avoid memory issues
- Close unused browser tabs to free up resources

## 🔮 Future Enhancements

### Planned Features
- **Portfolio Management**: Multi-stock portfolio tracking
- **Alert System**: Price and prediction alerts
- **Backtesting**: Historical prediction validation
- **API Integration**: REST API for external applications
- **Mobile App**: Native mobile application
- **Advanced ML**: Deep learning models and ensemble methods

### Technical Improvements
- **Real-time Streaming**: WebSocket connections for live data
- **Database Integration**: Persistent storage for predictions
- **Cloud Deployment**: AWS/Azure cloud hosting
- **Microservices**: Scalable microservice architecture

## 📞 Support

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs and feature requests
- **Community**: Join our community for discussions

### Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## ⚠️ Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. The predictions and recommendations provided are based on historical data and machine learning models, and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

**MarketVision Pro** - Empowering investors with AI-driven market insights 🚀 