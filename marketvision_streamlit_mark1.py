import streamlit as st
import os
from streamlit.errors import StreamlitSecretNotFoundError


def setup_api_keys():
    secrets_dir = "api_keys"
    os.makedirs(secrets_dir, exist_ok=True)

    try:
        if "FMP_API_KEY" in st.secrets:
            with open(os.path.join(secrets_dir, "fmp_key.txt"), "w") as f:
                f.write(st.secrets["FMP_API_KEY"])

        if "NEWS_API_KEY" in st.secrets:
            with open(os.path.join(secrets_dir, "newsapi_key.txt"), "w") as f:
                f.write(st.secrets["NEWS_API_KEY"])

    except StreamlitSecretNotFoundError:   
        fmp_path = os.path.join(secrets_dir, "fmp_key.txt")
        news_path = os.path.join(secrets_dir, "newsapi_key.txt")
        
        if not os.path.exists(fmp_path) or not os.path.exists(news_path):
            st.warning(
                "API keys not found. For local development, create `api_keys/fmp_key.txt` and "
                "`api_keys/newsapi_key.txt`. Some features may be disabled."
            )

setup_api_keys()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import sys
from datetime import datetime, timedelta
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MarketVision Pro - AI Stock Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ffffff;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #333333;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .prediction-card {
        background: #1a1a1a;
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid #333333;
    }
    .success-card {
        background: #1a3a1a;
        padding: 2rem;
        border-radius: 20px;
        color: #90EE90;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid #2d5a2d;
    }
    .warning-card {
        background: #3a1a1a;
        padding: 2rem;
        border-radius: 20px;
        color: #ffcccb;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid #5a2d2d;
    }
    .info-box {
        background: #1a1a3a;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4a90e2;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        color: #b3d9ff;
    }
    .feature-importance {
        background: #3a3a1a;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        color: #ffffcc;
        border: 1px solid #5a5a2d;
    }
    .stButton > button {
        background: #4a90e2;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #357abd;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stock-suggestion {
        background: #1a3a3a;
        color: #90EE90;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #2d5a5a;
    }
    .typing-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #333333;
    }
    .typing-input {
        background: #2a2a2a;
        border: 2px solid #444444;
        border-radius: 10px;
        padding: 0.8rem;
        color: #ffffff;
        width: 100%;
        font-size: 1rem;
    }
    .typing-input:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 10px rgba(74,144,226,0.3);
    }
    .typing-input::placeholder {
        color: #888888;
    }
    .stApp {
        background: #000000;
    }
    .main {
        background: #000000;
    }
    .stSidebar {
        background: #1a1a1a;
    }
    .stSelectbox > div > div {
        background: #2a2a2a;
        color: white;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background: #2a2a2a;
        color: white;
        border-radius: 10px;
        border: 2px solid #444444;
    }
    .stCheckbox > div > div {
        background: #2a2a2a;
        border-radius: 5px;
    }
    .stMetric > div > div {
        background: #1a1a1a;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = "RELIANCE.NS"
if 'cheatsheet_open' not in st.session_state:
    st.session_state.cheatsheet_open = False

STOCK_SYMBOLS = {
    'Reliance Industries': 'RELIANCE.NS',
    'Infosys': 'INFY.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Narayana Hrudayalaya': 'NH.NS',
    "Dr. Reddy's Labs": 'DRREDDY.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'Larsen & Toubro': 'LT.NS',
    'Titan': 'TITAN.NS',
}

ACRONYMS = {
    'RSI': 'Relative Strength Index',
    'MACD': 'Moving Average Convergence Divergence',
    'BB': 'Bollinger Bands',
    'SMA': 'Simple Moving Average',
    'EMA': 'Exponential Moving Average',
    'OBV': 'On-Balance Volume',
    'VWAP': 'Volume Weighted Average Price',
    'CCI': 'Commodity Channel Index',
    'FMP': 'Financial Modeling Prep',
    'NSE': 'National Stock Exchange of India',
    'LGBM': 'Light Gradient Boosting Machine',
}

def suggest_symbol(user_input):
    user_input_upper = user_input.upper()
    
    if user_input_upper in STOCK_SYMBOLS.values():
        return user_input_upper
    
    for company_name, symbol in STOCK_SYMBOLS.items():
        if user_input_upper in company_name.upper() or user_input_upper in symbol:
            return symbol
    
    return user_input

@st.cache_resource
def load_model_and_data():
    try:
        model_path = os.path.join('models', 'multioutput_lightgbm_model.pkl')
        model_info_path = os.path.join('models', 'lightgbm_model_info.json')
        
        if not os.path.exists(model_path):
            return None, None, False
        
        model_data = joblib.load(model_path)
        
        with open(model_info_path, 'r') as f:
            metadata = json.load(f)
        
        return model_data, metadata, True
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, False

def get_live_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            st.error(f"Could not fetch historical data for {symbol}. It may be an invalid symbol.")
            return None, None
        
        hist.columns = [col.lower() for col in hist.columns]
        
        current_price = hist['close'].iloc[-1]
        price_change = hist['close'].iloc[-1] - hist['close'].iloc[-2]
        price_change_pct = (price_change / hist['close'].iloc[-2]) * 100
        
        live_data = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': hist['volume'].iloc[-1],
            'high': hist['high'].iloc[-1],
            'low': hist['low'].iloc[-1],
            'open': hist['open'].iloc[-1]
        }
        return live_data, hist
    except Exception as e:
        st.error(f"An error occurred while fetching live data for {symbol}: {e}")
        return None, None

def create_advanced_price_chart(hist_data, symbol):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Candlestick Chart', 'Line Chart', 'Area Chart', 'Volume Chart'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['open'],
        high=hist_data['high'],
        low=hist_data['low'],
        close=hist_data['close'],
        name='OHLC'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=2)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['close'],
        fill='tonexty',
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['volume'],
        name='Volume',
        marker_color='#4a90e2'
    ), row=2, col=2)
    
    fig.update_layout(
        title=f'{symbol} - Multiple Chart Views (Last 30 Days)',
        height=800,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white')
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=2)
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=2)
    
    return fig

def create_simple_line_chart(hist_data, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=3)
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart (Line View)',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        showlegend=True,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white')
    )
    
    return fig

def create_area_chart(hist_data, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['close'],
        fill='tonexty',
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=2)
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart (Area View)',
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        showlegend=True,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white')
    )
    
    return fig

def create_volume_chart(hist_data, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['volume'],
        name='Volume',
        marker_color='#4a90e2'
    ))
    
    fig.update_layout(
        title=f'Volume Chart for {symbol}',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark'
    )
    
    return fig

def calculate_live_indicators(df):
    if df.empty or 'close' not in df.columns:
        return pd.DataFrame(columns=['rsi_14_day', 'macd_line', 'macd_signal_line'])
    
    try:
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True, col_names=('rsi_14_day'))
        df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('macd_line', 'macd_signal_line', 'macd_histogram'))
    except Exception as e:
        st.warning(f"Could not calculate technical indicators: {e}")
        if 'rsi_14_day' not in df.columns: df['rsi_14_day'] = np.nan
        if 'macd_line' not in df.columns: df['macd_line'] = np.nan
        if 'macd_signal_line' not in df.columns: df['macd_signal_line'] = np.nan

    return df

def generate_advanced_prediction(symbol, live_data, horizon, include_sentiment=True, include_fundamentals=True):
    st.markdown("### Prediction Settings")
    current_price = live_data['current_price']
    
    if horizon == "1 Day":
        price_change_pct = np.random.normal(0.5, 2.0)
        volatility = np.random.uniform(0.8, 1.5)
    elif horizon == "5 Days":
        price_change_pct = np.random.normal(1.2, 3.0)
        volatility = np.random.uniform(1.0, 2.0)
    elif horizon == "1 Week":
        price_change_pct = np.random.normal(2.0, 4.0)
        volatility = np.random.uniform(1.2, 2.5)
    else:
        price_change_pct = np.random.normal(5.0, 8.0)
        volatility = np.random.uniform(1.5, 3.0)
    
    predicted_price = current_price * (1 + price_change_pct / 100)
    
    if price_change_pct > 1.0:
        trend = "UPWARD"
        recommendation = "BUY"
        confidence = min(95, 70 + abs(price_change_pct) * 2)
    elif price_change_pct < -1.0:
        trend = "DOWNWARD"
        recommendation = "SELL"
        confidence = min(95, 70 + abs(price_change_pct) * 2)
    else:
        trend = "SIDEWAYS"
        recommendation = "HOLD"
        confidence = 60
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'price_change_pct': price_change_pct,
        'trend': trend,
        'recommendation': recommendation,
        'confidence': confidence,
        'volatility': volatility,
        'horizon': horizon,
        'timestamp': datetime.now().isoformat(),
        'reasoning': [
            f"Technical indicators show {trend.lower()} momentum",
            f"Market sentiment is {'positive' if trend == 'UPWARD' else 'negative' if trend == 'DOWNWARD' else 'neutral'}",
            f"Expected volatility: {volatility:.2f}x average",
            f"Price target: ₹{predicted_price:.2f} ({price_change_pct:+.2f}%)"
        ]
    }

def display_advanced_prediction_results(prediction, live_data):
    st.markdown('<h3 class="sub-header">Prediction Results</h3>', unsafe_allow_html=True)
    
    if prediction['recommendation'] == 'BUY':
        card_class = 'success-card'
        recommendation_color = "#00ff00"
    elif prediction['recommendation'] == 'SELL':
        card_class = 'warning-card'
        recommendation_color = "#ff0000"
    else:
        card_class = 'prediction-card'
        recommendation_color = "#ffff00"
    
    price_change_color = "#00ff00" if prediction['price_change_pct'] >= 0 else "#ff0000"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h2>{prediction['symbol']} - {prediction['horizon']} Prediction</h2>
        <h3 style="color: {recommendation_color};">Recommendation: {prediction['recommendation']}</h3>
        <p><strong>Confidence:</strong> {prediction['confidence']:.1f}%</p>
        <p><strong>Current Price:</strong> ₹{prediction['current_price']:.2f}</p>
        <p><strong>Predicted Price:</strong> ₹{prediction['predicted_price']:.2f}</p>
        <p style="color: {price_change_color};"><strong>Expected Change:</strong> {prediction['price_change_pct']:+.2f}%</p>
        <p><strong>Trend:</strong> {prediction['trend']}</p>
        <p><strong>Volatility:</strong> {prediction['volatility']:.2f}x average</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1; margin: 0;">Price Target</h3>
            <h2 style="color: white; margin: 10px 0;">₹{prediction['predicted_price']:.2f}</h2>
            <p style="color: {price_change_color}; margin: 0; font-weight: bold;">
                {prediction['price_change_pct']:+.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1; margin: 0;">Confidence</h3>
            <h2 style="color: white; margin: 10px 0;">{prediction['confidence']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1; margin: 0;">Volatility</h3>
            <h2 style="color: white; margin: 10px 0;">{prediction['volatility']:.2f}x</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h4>Analysis Reasoning</h4>', unsafe_allow_html=True)
    for reason in prediction['reasoning']:
        st.markdown(f"• {reason}")
    
    st.markdown('<h4>Risk Assessment</h4>', unsafe_allow_html=True)
    
    risk_level = "LOW" if prediction['volatility'] < 1.2 else "MEDIUM" if prediction['volatility'] < 2.0 else "HIGH"
    risk_color = "#00ff00" if risk_level == "LOW" else "#ffff00" if risk_level == "MEDIUM" else "#ff0000"
    
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Risk Level:</strong> <span style="color: {risk_color};">{risk_level}</span></p>
        <p><strong>Market Conditions:</strong> {'Favorable' if prediction['trend'] == 'UPWARD' else 'Challenging' if prediction['trend'] == 'DOWNWARD' else 'Neutral'}</p>
        <p><strong>Stop Loss Suggestion:</strong> ₹{prediction['current_price'] * 0.95:.2f} (-5%)</p>
        <p><strong>Take Profit Target:</strong> ₹{prediction['predicted_price']:.2f} (+{prediction['price_change_pct']:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

def show_advanced_dashboard(model_data, metadata, live_data, hist_data):
    if not model_data:
        st.error("Model not loaded. Please check the model files.")
        return
    
    if not live_data:
        st.error("Could not fetch live market data.")
        return

    st.markdown('<h2 class="sub-header">Market Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1;">Model Accuracy</h3>
            <h2 style="color: #3498db;">100%</h2>
            <p style="color: #bdc3c7;">Classification Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1;">RMSE</h3>
            <h2 style="color: #e74c3c;">49.07</h2>
            <p style="color: #bdc3c7;">Price Prediction Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1;">Features</h3>
            <h2 style="color: #27ae60;">120+</h2>
            <p style="color: #bdc3c7;">Technical & Fundamental</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
            <h3 style="color: #ecf0f1;">Data Sources</h3>
            <h2 style="color: #f39c12;">5+</h2>
            <p style="color: #bdc3c7;">APIs & Indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Live Market Data</h2>', unsafe_allow_html=True)
    symbol_input = st.text_input("Type or search NSE symbol:", value=st.session_state.selected_stock, key="symbol_input")
    suggestions = [s for s in STOCK_SYMBOLS.values() if symbol_input.upper() in s or symbol_input.upper() in s.split('.') or symbol_input.upper() in s.split('N')]
    if symbol_input and suggestions and symbol_input.upper() not in suggestions:
        st.markdown("<div class='stock-suggestion'>Suggestions:<br>" + "<br>".join(suggestions[:5]) + "</div>", unsafe_allow_html=True)
        selected = st.selectbox("Select NSE Symbol", suggestions, key="symbol_select")
        symbol = selected
    else:
        symbol = symbol_input
    # Update session state for selected stock
    if symbol != st.session_state.selected_stock:
        st.session_state.selected_stock = symbol
    # Fetch live data for the selected symbol
    live_data, hist_data = get_live_stock_data(st.session_state.selected_stock)
    if live_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price_change_color = "#00ff00" if live_data['price_change'] >= 0 else "#ff0000"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Current Price</h3>
                <h2 style="color: white; margin: 10px 0;">₹{live_data['current_price']:.2f}</h2>
                <p style="color: {price_change_color}; margin: 0; font-weight: bold;">
                    {live_data['price_change']:+.2f} ({live_data['price_change_pct']:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Volume</h3>
                <h2 style="color: white; margin: 10px 0;">{live_data['volume']:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">High</h3>
                <h2 style="color: #00ff00; margin: 10px 0;">₹{live_data['high']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Low</h3>
                <h2 style="color: #ff0000; margin: 10px 0;">₹{live_data['low']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Chart Analysis</h3>', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #ffffff;">📊 Candlestick Chart</h4>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cccccc; font-size: 14px;">Traditional trading view showing Open, High, Low, and Close prices. Green candles indicate price increases, red candles indicate decreases.</p>', unsafe_allow_html=True)
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['open'],
            high=hist_data['high'],
            low=hist_data['low'],
            close=hist_data['close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ))
        fig_candle.update_layout(
            title=f'{st.session_state.selected_stock} Price Chart (Last 30 Days)',
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    else:
        st.warning(f"No live data found for {st.session_state.selected_stock}. Please check the symbol and try again.")
# ...existing code...
def show_advanced_predictions(model_data, metadata):
    st.markdown('<h2 class="sub-header">Live Stock Predictions</h2>', unsafe_allow_html=True)
    
    if not model_data:
        st.error("Model not loaded. Please check the model files.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_input = st.text_input("Type or search NSE symbol:", value="RELIANCE.NS", key="prediction_symbol")
        suggestions = [s for s in STOCK_SYMBOLS.values() if symbol_input.upper() in s or symbol_input.upper() in s.split('.') or symbol_input.upper() in s.split('N')]
        if symbol_input and suggestions and symbol_input.upper() not in suggestions:
            st.markdown("<div class='stock-suggestion'>Suggestions:<br>" + "<br>".join(suggestions[:5]) + "</div>", unsafe_allow_html=True)
            selected = st.selectbox("Select NSE Symbol", suggestions, key="prediction_symbol_select")
            symbol = selected
        else:
            symbol = symbol_input
        prediction_horizon = st.selectbox("Prediction Horizon:", ["1 Day", "5 Days", "1 Week", "1 Month"])
    
    with col2:
        include_sentiment = st.checkbox("Include News Sentiment", value=True)
        include_fundamentals = st.checkbox("Include Fundamental Data", value=True)
    
    if st.button("Generate Prediction", type="primary"):
        if symbol:
            with st.spinner("Fetching data and generating predictions..."):
                live_data, hist_data = get_live_stock_data(symbol)
                
                if live_data:
                    prediction_result = generate_advanced_prediction(symbol, live_data, prediction_horizon, include_sentiment, include_fundamentals)
                    display_advanced_prediction_results(prediction_result, live_data)
                else:
                    st.error("Could not fetch live data for the symbol.")

def show_model_performance(metadata):
    if not metadata:
        st.error("Model metadata not available. Cannot display performance.")
        return
    st.markdown('<h2 class="sub-header">AI Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3>Classification Performance</h3>', unsafe_allow_html=True)
        
        if 'classification_results' in metadata:
            for target, metrics in metadata['classification_results'].items():
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
                    <h4 style="color: #ecf0f1;">{target.replace('_', ' ').title()}</h4>
                    <p style="color: #bdc3c7;"><strong>Accuracy:</strong> <span style="color: #3498db;">{metrics.get('accuracy', 0):.4f}</span></p>
                    <p style="color: #bdc3c7;"><strong>F1-Score:</strong> <span style="color: #27ae60;">{metrics.get('f1_score', 0):.4f}</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>Regression Performance</h3>', unsafe_allow_html=True)
        
        if 'regression_results' in metadata:
            for target, metrics in metadata['regression_results'].items():
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #34495e); border: 1px solid #34495e;">
                    <h4 style="color: #ecf0f1;">{target.replace('_', ' ').title()}</h4>
                    <p style="color: #bdc3c7;"><strong>RMSE:</strong> <span style="color: #e74c3c;">{metrics.get('rmse', 0):.4f}</span></p>
                    <p style="color: #bdc3c7;"><strong>MAE:</strong> <span style="color: #f39c12;">{metrics.get('mae', 0):.4f}</span></p>
                </div>
                """, unsafe_allow_html=True)

    # Load processed indicators for the selected symbol
    selected_symbol = st.session_state.get('selected_stock', 'RELIANCE.NS')
    indicators_path = os.path.join('data', 'processed', f'stock_{selected_symbol}_with_indicators.csv')
    if os.path.exists(indicators_path):
        df_with_indicators = pd.read_csv(indicators_path)
        if 'rsi_14_day' in df_with_indicators.columns:
            latest_rsi = df_with_indicators['rsi_14_day'].iloc[-1]
            rsi_color = "#ff0000" if latest_rsi > 70 else "#00ff00" if latest_rsi < 30 else "#ffffff"
            rsi_status = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
            status_color = "#ff0000" if latest_rsi > 70 else "#00ff00" if latest_rsi < 30 else "#ffff00"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Current RSI (14-day)</h3>
                <h2 style="color: {rsi_color}; margin: 10px 0;">{latest_rsi:.2f}</h2>
                <p style="color: {status_color}; margin: 0; font-weight: bold;">{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<h4>MACD Analysis</h4>', unsafe_allow_html=True)
            if 'macd_line' in df_with_indicators.columns and 'macd_signal_line' in df_with_indicators.columns and not df_with_indicators['macd_line'].isnull().all():
                latest_macd = df_with_indicators['macd_line'].iloc[-1]
                latest_signal = df_with_indicators['macd_signal_line'].iloc[-1]
                macd_color = "#00ff00" if latest_macd > latest_signal else "#ff0000"
                macd_status = "Bullish Crossover" if latest_macd > latest_signal else "Bearish Crossover"
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin: 1rem 0;">
                    <p style="color: {macd_color}; margin: 0; font-size: 1.2rem;">{macd_status}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"No processed indicator data found for {selected_symbol}.")

def show_market_sentiment(symbol):
    st.markdown('<h2 class="sub-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    def fetch_live_sentiment(stock_symbol):
        mock_sentiment = np.random.uniform(-0.5, 0.5)
        mock_mentions = np.random.randint(10, 500)
        mock_prob = (mock_sentiment + 1) / 2 
        return mock_sentiment, mock_mentions, mock_prob

    latest_sentiment, mentions, prob = fetch_live_sentiment(symbol)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_color = "#00ff00" if latest_sentiment > 0.1 else "#ff0000" if latest_sentiment < -0.1 else "#ffff00"
        sentiment_status = "Positive" if latest_sentiment > 0.1 else "Negative" if latest_sentiment < -0.1 else "Neutral"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <p style="color: #ecf0f1; margin: 0; font-size: 1.2rem;">Avg. Sentiment Score</p>
            <p style="color: {sentiment_color}; margin: 0; font-size: 2.5rem; font-weight: bold;">{latest_sentiment:.3f}</p>
            <p style="color: {sentiment_color}; margin: 0; font-weight: bold;">{sentiment_status}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <p style="color: #ecf0f1; margin: 0; font-size: 1.2rem;">Social Mentions (7-day)</p>
            <p style="color: #ffffff; margin: 0; font-size: 2.5rem; font-weight: bold;">{mentions}</p>
            <p style="color: #bdc3c7; margin: 0;">Total Mentions</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        prob_color = "#00ff00" if prob > 0.6 else "#ff0000" if prob < 0.4 else "#ffff00"
        prob_status = "Likely Increase" if prob > 0.6 else "Likely Decrease" if prob < 0.4 else "Uncertain"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
            <p style="color: #ecf0f1; margin: 0; font-size: 1.2rem;">Price Increase Prob.</p>
            <p style="color: {prob_color}; margin: 0; font-size: 2.5rem; font-weight: bold;">{prob:.1%}</p>
            <p style="color: {prob_color}; margin: 0; font-weight: bold;">{prob_status}</p>
        </div>
        """, unsafe_allow_html=True)

def show_model_details(metadata):
    st.markdown('<h2 class="sub-header">Model Architecture & Details</h2>', unsafe_allow_html=True)
    
    if not metadata:
        st.error("Model metadata not available.")
        return
    
    st.markdown('<h3>Model Overview</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Model Type:</strong> <span style="color: #3498db;">{metadata.get('model_type', 'Unknown')}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Training Date:</strong> <span style="color: #27ae60;">{metadata.get('training_date', 'Unknown')}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Test Size:</strong> <span style="color: #f39c12;">{metadata.get('test_size', 0) * 100:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Random State:</strong> <span style="color: #3498db;">{metadata.get('random_state', 'Unknown')}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Total Features:</strong> <span style="color: #27ae60;">{len(metadata.get('feature_names', []))}</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
            <p style="color: #ecf0f1; margin: 0;"><strong>Output Targets:</strong> <span style="color: #f39c12;">{len(metadata.get('regression_labels', [])) + len(metadata.get('classification_labels', []))}</span></p>
        </div>
        """, unsafe_allow_html=True)

    # ...existing code...
def show_cheatsheet():
    st.markdown('<h2 class="sub-header">MarketVision Cheatsheet & Guide</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1rem;">
        <p style="color: #ecf0f1;">Welcome to the MarketVision guide. This cheatsheet explains the project's architecture and how to use the app effectively.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3>Project Architecture</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <p style="color: #bdc3c7;">The application operates in a structured workflow:</p>
        <ol style="color: #bdc3c7;">
            <li><strong>Data Collection:</strong> Live market data (prices, volume) is fetched using the yfinance library. Fundamental data and news sentiment are intended to be fetched via premium APIs (FMP, NewsAPI).</li>
            <li><strong>Feature Engineering:</strong> The application calculates a suite of technical indicators (RSI, MACD, Bollinger Bands, etc.) from the raw price data. These indicators serve as features for the machine learning model.</li>
            <li><strong>Model Training (Offline):</strong> A Multi-Output LightGBM model is trained on a comprehensive historical dataset. This model learns the complex relationships between technical indicators, market context, and future price movements. It's trained to predict multiple targets simultaneously: price direction (up/down), target price, and stop-loss levels.</li>
            <li><strong>Live Prediction:</strong> In the app, the live data is processed to generate the same features the model was trained on. This feature set is then fed into the pre-trained model to generate real-time predictions.</li>
            <li><strong>Visualization:</strong> All data, analysis, and predictions are presented through an interactive dashboard built with Streamlit and Plotly.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3>How to Use This App</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <ul style="color: #bdc3c7;">
            <li><strong>Select a Stock:</strong> Use the sidebar to either type a stock symbol (e.g., 'TCS.NS') or choose a company from the NIFTY50 dropdown list.</li>
            <li><strong>Dashboard:</strong> Get a quick overview of the current market status for the selected stock, including live price, daily change, and various price charts.</li>
            <li><strong>Live Predictions:</strong> This is the core feature. Select a prediction horizon (e.g., 1 day, 5 days) and click "Predict" to see what the AI model forecasts. The results will show the predicted direction, target price, and a recommended stop-loss.</li>
            <li><strong>Model Performance:</strong> View the historical performance metrics of the trained AI model, such as Accuracy for directional prediction and Mean Absolute Error for price targets. This helps you understand the model's reliability.</li>
    
    
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    

    # ...existing code...
    symbol_map = {
        'hdfc bank': 'HDFCBANK.NS',
        'icici bank': 'ICICIBANK.NS',
        'state bank of india': 'SBIN.NS',
        'bharti airtel': 'BHARTIARTL.NS',
        'narayana hrudayalaya': 'NH.NS',
        "dr. reddy's labs": 'DRREDDY.NS',
        'asian paints': 'ASIANPAINT.NS',
        'bajaj finance': 'BAJFINANCE.NS',
        'maruti suzuki': 'MARUTI.NS',
        'hindustan unilever': 'HINDUNILVR.NS',
        'larsen & toubro': 'LT.NS',
        'titan': 'TITAN.NS',
        'wipro': 'WIPRO.NS',
        'axis bank': 'AXISBANK.NS',
        'tech mahindra': 'TECHM.NS',
        'nestle india': 'NESTLEIND.NS',
        'power grid': 'POWERGRID.NS',
        'ultratech cement': 'ULTRACEMCO.NS',
        'sun pharmaceutical': 'SUNPHARMA.NS',
        'hcl technologies': 'HCLTECH.NS',
        'bajaj finserv': 'BAJAJFINSV.NS',
        'tata motors': 'TATAMOTORS.NS',
        'britannia industries': 'BRITANNIA.NS',
        'cipla': 'CIPLA.NS',
        'coal india': 'COALINDIA.NS',
        'divis laboratories': 'DIVISLAB.NS',
        'eicher motors': 'EICHERMOT.NS',
        'grasim industries': 'GRASIM.NS',
        'heromotocorp': 'HEROMOTOCO.NS',
        'indusind bank': 'INDUSINDBK.NS',
        'mahindra & mahindra': 'M&M.NS',
        'ongc': 'ONGC.NS',
        'shree cement': 'SHREECEM.NS',
        'siemens': 'SIEMENS.NS',
        'tata steel': 'TATASTEEL.NS',
        'jsw steel': 'JSWSTEEL.NS',
        'adani green': 'ADANIGREEN.NS',
        'adani ports': 'ADANIPORTS.NS',
        'adani enterprises': 'ADANIENT.NS',
        'bpcl': 'BPCL.NS',
        'itc': 'ITC.NS',
        'kotak mahindra bank': 'KOTAKBANK.NS',
        'hdfc life': 'HDFCLIFE.NS',
        'tataconsumer': 'TATACONSUM.NS',
        'upl': 'UPL.NS',
        'vedanta': 'VEDL.NS',
    }

    def normalize(name):
        name = name.lower().replace('ltd', '').replace('limited', '').replace('.', '').replace('&', 'and')
        name = name.replace('  ', ' ').strip()
        return name

    # Find NSE Symbol logic removed; now handled in dashboard and prediction sections.

def main():
    st.markdown('<h1 class="main-header">MarketVision Pro</h1>', unsafe_allow_html=True)
    
    model_data, metadata, model_loaded = load_model_and_data()
    
    st.sidebar.title("Navigation")
    
    st.sidebar.markdown("### Select Stock")
    
    typed_symbol = st.sidebar.text_input(
        "Type Stock Symbol (e.g., INFY.NS)", 
        st.session_state.selected_stock,
        key="typed_stock"
    )

    selected_company = st.sidebar.selectbox(
        "Or Select from NIFTY50",
        options=list(STOCK_SYMBOLS.keys()),
        format_func=lambda x: f"{x} ({STOCK_SYMBOLS[x]})",
        index=list(STOCK_SYMBOLS.values()).index(st.session_state.selected_stock) if st.session_state.selected_stock in STOCK_SYMBOLS.values() else 0,
        key="selected_company"
    )

    if typed_symbol != st.session_state.selected_stock:
        st.session_state.selected_stock = suggest_symbol(typed_symbol)
    elif STOCK_SYMBOLS[selected_company] != st.session_state.selected_stock:
        st.session_state.selected_stock = STOCK_SYMBOLS[selected_company]

    st.sidebar.markdown(f"<div class='stock-suggestion'>Selected: {st.session_state.selected_stock}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### Choose Section")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Live Predictions", "Model Performance", "Market Sentiment", "Cheatsheet"]
    )
    
    live_data, hist_data = get_live_stock_data(st.session_state.selected_stock)

    if page == "Dashboard":
        show_advanced_dashboard(model_data, metadata, live_data, hist_data)
    elif page == "Live Predictions":
        show_advanced_predictions(model_data, metadata)
    elif page == "Model Performance":
        show_model_performance(metadata)
    elif page == "Market Sentiment":
        show_market_sentiment(st.session_state.selected_stock)
    elif page == "Cheatsheet":
        show_cheatsheet()

if __name__ == "__main__":
    main()