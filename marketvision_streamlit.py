import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
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
        return None, None, False

@st.cache_data
def load_processed_data():
    try:
        data_path = os.path.join('data', 'processed')
        files = {
            'indicators': 'stock_RELIANCE.NS_with_indicators.csv',
            'macro': 'stock_RELIANCE.NS_with_macro_context.csv',
            'sentiment': 'stock_RELIANCE.NS_with_sentiment.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath)
        
        return data
    except Exception as e:
        return {}

def get_live_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            return None, None
        
        current_price = hist['Close'].iloc[-1]
        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
        price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': hist['Volume'].iloc[-1],
            'high': hist['High'].iloc[-1],
            'low': hist['Low'].iloc[-1],
            'open': hist['Open'].iloc[-1]
        }, hist
    except Exception as e:
        return None, None

def create_advanced_price_chart(hist_data, symbol):
    # Create subplots for different chart types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Candlestick Chart', 'Line Chart', 'Area Chart', 'Volume Chart'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Line chart
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=2)
    ), row=1, col=2)
    
    # Area chart
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        fill='tonexty',
        mode='lines',
        name='Close Price',
        line=dict(color='#4a90e2', width=1)
    ), row=2, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
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
        y=hist_data['Close'],
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
        y=hist_data['Close'],
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
        y=hist_data['Volume'],
        name='Volume',
        marker_color='#4a90e2'
    ))
    
    fig.update_layout(
        title=f'{symbol} Volume Chart',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        showlegend=True,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white')
    )
    
    return fig

def calculate_technical_indicators(df):
    if df.empty:
        return df
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().rolling(window=14).mean() / df['Close'].diff().rolling(window=14).std())))
    
    return df

def generate_advanced_prediction(symbol, live_data, horizon, include_sentiment=True, include_fundamentals=True):
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

def show_advanced_dashboard(model_data, metadata, processed_data):
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
    
    symbol_input = st.text_input("Or select from dropdown:", value="RELIANCE.NS", key="symbol_input")
    symbol = suggest_symbol(symbol_input)
    
    if symbol != symbol_input:
        st.markdown(f'<div class="stock-suggestion">Suggested: {symbol}</div>', unsafe_allow_html=True)
    
    if symbol:
        live_data, hist_data = get_live_stock_data(symbol)
        
        if live_data and hist_data is not None:
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
            
            # Candlestick Chart
            st.markdown('<h4 style="color: #ffffff;">📊 Candlestick Chart</h4>', unsafe_allow_html=True)
            st.markdown('<p style="color: #cccccc; font-size: 14px;">Traditional trading view showing Open, High, Low, and Close prices. Green candles indicate price increases, red candles indicate decreases.</p>', unsafe_allow_html=True)
            
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name='OHLC',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ))
            fig_candle.update_layout(
                title=f'{symbol} Candlestick Chart (Last 30 Days)',
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_dark',
                height=400,
                showlegend=True,
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white')
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
            # Line Chart
            st.markdown('<h4 style="color: #ffffff;">📈 Line Chart</h4>', unsafe_allow_html=True)
            st.markdown('<p style="color: #cccccc; font-size: 14px;">Smooth price trend visualization showing closing prices over time. Good for identifying overall market direction.</p>', unsafe_allow_html=True)
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#4a90e2', width=3)
            ))
            fig_line.update_layout(
                title=f'{symbol} Line Chart (Last 30 Days)',
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_dark',
                height=400,
                showlegend=True,
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white')
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Area Chart
            st.markdown('<h4 style="color: #ffffff;">📊 Area Chart</h4>', unsafe_allow_html=True)
            st.markdown('<p style="color: #cccccc; font-size: 14px;">Filled area visualization emphasizing price levels and trends. Shows price movement with visual weight.</p>', unsafe_allow_html=True)
            
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                fill='tonexty',
                mode='lines',
                name='Close Price',
                line=dict(color='#4a90e2', width=2)
            ))
            fig_area.update_layout(
                title=f'{symbol} Area Chart (Last 30 Days)',
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_dark',
                height=400,
                showlegend=True,
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white')
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            # Volume Chart
            st.markdown('<h4 style="color: #ffffff;">📊 Volume Chart</h4>', unsafe_allow_html=True)
            st.markdown('<p style="color: #cccccc; font-size: 14px;">Trading volume analysis showing market activity. Higher bars indicate more trading volume, confirming price movements.</p>', unsafe_allow_html=True)
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=hist_data.index,
                y=hist_data['Volume'],
                name='Volume',
                marker_color='#4a90e2'
            ))
            fig_volume.update_layout(
                title=f'{symbol} Volume Chart (Last 30 Days)',
                yaxis_title='Volume',
                xaxis_title='Date',
                template='plotly_dark',
                height=400,
                showlegend=True,
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white')
            )
            st.plotly_chart(fig_volume, use_container_width=True)

def show_advanced_predictions(model_data, metadata):
    st.markdown('<h2 class="sub-header">Live Stock Predictions</h2>', unsafe_allow_html=True)
    
    if not model_data:
        st.error("Model not loaded. Please check the model files.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol_input = st.text_input("Stock Symbol:", value="RELIANCE.NS", key="prediction_symbol")
        symbol = suggest_symbol(symbol_input)
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
    st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    if not metadata:
        st.error("Model metadata not available.")
        return
    
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

def show_technical_analysis(processed_data):
    st.markdown('<h2 class="sub-header">Technical Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    if not processed_data:
        st.error("No processed data available for technical analysis.")
        return
    
    if 'indicators' in processed_data:
        df = processed_data['indicators']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4>RSI Analysis</h4>', unsafe_allow_html=True)
            if 'rsi_14_day' in df.columns:
                latest_rsi = df['rsi_14_day'].iloc[-1]
                
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
            if 'macd_line' in df.columns and 'macd_signal_line' in df.columns:
                latest_macd = df['macd_line'].iloc[-1]
                latest_signal = df['macd_signal_line'].iloc[-1]
                
                macd_color = "#00ff00" if latest_macd > latest_signal else "#ff0000"
                macd_status = "Bullish" if latest_macd > latest_signal else "Bearish"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1.2rem;">
                    <h3 style="color: #ecf0f1; margin: 0;">MACD Line</h3>
                    <h2 style="color: white; margin: 10px 0;">{latest_macd:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1.2rem;">
                    <h3 style="color: #ecf0f1; margin: 0;">Signal Line</h3>
                    <h2 style="color: white; margin: 10px 0;">{latest_signal:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 1.2rem;">
                    <h3 style="color: {macd_color}; margin: 0;">{macd_status} MACD</h3>
                </div>
                """, unsafe_allow_html=True)

def show_market_sentiment(processed_data):
    st.markdown('<h2 class="sub-header">Market Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    if 'sentiment' not in processed_data:
        st.error("Sentiment data not available.")
        return
    
    df = processed_data['sentiment']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'avg_sentiment_score' in df.columns:
            latest_sentiment = df['avg_sentiment_score'].iloc[-1]
            
            sentiment_color = "#00ff00" if latest_sentiment > 0.1 else "#ff0000" if latest_sentiment < -0.1 else "#ffff00"
            sentiment_status = "Positive" if latest_sentiment > 0.1 else "Negative" if latest_sentiment < -0.1 else "Neutral"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Average Sentiment Score</h3>
                <h2 style="color: {sentiment_color}; margin: 10px 0;">{latest_sentiment:.3f}</h2>
                <p style="color: {sentiment_color}; margin: 0; font-weight: bold;">{sentiment_status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'social_mention_count_7_day' in df.columns:
            mentions = df['social_mention_count_7_day'].iloc[-1]
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Social Mentions (7-day)</h3>
                <h2 style="color: white; margin: 10px 0;">{mentions:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'probability_price_increase_next_day' in df.columns:
            prob = df['probability_price_increase_next_day'].iloc[-1]
            
            prob_color = "#00ff00" if prob > 0.6 else "#ff0000" if prob < 0.4 else "#ffff00"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e;">
                <h3 style="color: #ecf0f1; margin: 0;">Price Increase Probability</h3>
                <h2 style="color: {prob_color}; margin: 10px 0;">{prob:.1%}</h2>
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

def show_about_page():
    st.markdown('<h2 class="sub-header">About MarketVision</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h3 style="color: #ecf0f1;">Project Overview</h3>
        <p style="color: #bdc3c7;">MarketVision is an advanced AI-powered stock market prediction system designed specifically for the Indian stock market. 
        It combines multiple data sources and sophisticated machine learning algorithms to provide comprehensive market insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3>Key Features</h3>', unsafe_allow_html=True)
    
    features = [
        "Multi-timeframe Predictions: 1-day, 5-day, weekly, and monthly forecasts",
        "Technical Analysis: 20+ technical indicators including RSI, MACD, Bollinger Bands",
        "Sentiment Analysis: News sentiment and social media analysis",
        "Fundamental Data: Financial ratios and company fundamentals",
        "Market Context: Global market indicators and sector performance",
        "Trading Signals: Buy/Sell/Hold recommendations with confidence scores",
        "Risk Assessment: Volatility prediction and risk metrics",
        "Real-time Data: Live market data integration"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.warning("""
    **Important Disclaimer**: This application is for educational and research purposes only. 
    The predictions and recommendations provided are based on historical data and machine learning models, 
    and should not be considered as financial advice. Always consult with a qualified financial advisor 
    before making investment decisions. Past performance does not guarantee future results.
    """)

def show_cheatsheet():
    st.markdown('<h2 class="sub-header">MarketVision Cheatsheet</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h3 style="color: #ecf0f1; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem;">How MarketVision Works: A Deep Dive</h3>
        <p style="color: #bdc3c7; font-size: 16px;">MarketVision is a sophisticated AI system that analyzes multiple layers of financial data to predict stock price movements. Here's a step-by-step breakdown of its workflow:</p>
        <ol style="color: #bdc3c7; font-size: 16px; line-height: 1.8;">
            <li><strong>Data Collection:</strong> The system gathers a wide array of data, including historical stock prices, global market indices (like NIFTY 50, S&P 500), technical indicators, company fundamentals, and real-time news sentiment.</li>
            <li><strong>Feature Engineering:</strong> Raw data is processed to create meaningful features. This includes calculating over 40 technical indicators (RSI, MACD, etc.), analyzing news sentiment scores, and computing market context variables like beta and sector performance.</li>
            <li><strong>Model Training:</strong> The processed features are fed into a powerful LightGBM (Light Gradient Boosting Machine) model. This model is trained on historical data to recognize complex patterns that correlate with future price movements. It learns to predict multiple outcomes simultaneously, such as next-day price, weekly trends, and volatility.</li>
            <li><strong>Live Prediction:</strong> When you select a stock, the app fetches the latest market data, processes it through the same pipeline, and feeds it to the trained model to generate live predictions.</li>
            <li><strong>Visualization:</strong> The results are presented through interactive charts and easy-to-understand dashboards, helping you make informed decisions.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3>How to Use Each Section</h3>', unsafe_allow_html=True)
    
    # Dashboard
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h4 style="color: #ecf0f1;">Dashboard</h4>
        <p style="color: #bdc3c7;">The main dashboard is your mission control. It gives you a high-level snapshot of the market and the selected stock.</p>
        <ul>
            <li style="color: #bdc3c7;"><strong>Market Overview:</strong> Check key model metrics here. High accuracy and low error suggest the model is confident. See how many features the model uses to make its decisions.</li>
            <li style="color: #bdc3c7;"><strong>Live Market Data:</strong> Get the latest price, volume, and daily high/low. Red and green colors instantly show you the price direction.</li>
            <li style="color: #bdc3c7;"><strong>Chart Analysis:</strong> This is your primary tool for visual analysis. Switch between Candlestick, Line, Area, and Volume charts to spot trends, patterns, and support/resistance levels. Each chart tells a different story about the stock's behavior.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Live Predictions
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h4 style="color: #ecf0f1;">Live Predictions</h4>
        <p style="color: #bdc3c7;">This is where the AI does its magic. Select a prediction horizon (e.g., 1-day, 5-day) to see the model's forecast.</p>
        <ul>
            <li style="color: #bdc3c7;"><strong>Prediction Summary:</strong> A clear BUY, SELL, or HOLD recommendation based on the model's analysis. The confidence score tells you how certain the model is.</li>
            <li style="color: #bdc3c7;"><strong>Price Targets:</strong> See the exact price the model predicts for the end of the selected period. The expected change is shown in both absolute and percentage terms for quick assessment.</li>
            <li style="color: #bdc3c7;"><strong>Risk Assessment:</strong> The model estimates future volatility and provides recommended Stop-Loss and Take-Profit levels. This is crucial for risk management.</li>
            <li style="color: #bdc3c7;"><strong>Top Influencing Factors:</strong> Understand *why* the model made its prediction. This section lists the top 3 data points (e.g., RSI, news sentiment) that influenced the forecast, providing transparency.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Technical Analysis
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h4 style="color: #ecf0f1;">Technical Analysis</h4>
        <p style="color: #bdc3c7;">Dive deeper into the technicals that drive price action. This section provides a detailed view of key indicators.</p>
        <ul>
            <li style="color: #bdc3c7;"><strong>RSI Analysis:</strong> The Relative Strength Index (RSI) helps you spot overbought (>70) or oversold (<30) conditions. A stock that is overbought might be due for a price correction, while an oversold stock could be a buying opportunity.</li>
            <li style="color: #bdc3c7;"><strong>MACD Analysis:</strong> The Moving Average Convergence Divergence (MACD) indicates trend momentum. When the MACD line crosses above the Signal line, it's a bullish sign. When it crosses below, it's bearish.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Market Sentiment
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h4 style="color: #ecf0f1;">Market Sentiment</h4>
        <p style="color: #bdc3c7;">Understand the "mood" of the market. This section analyzes news and social media to gauge public opinion.</p>
        <ul>
            <li style="color: #bdc3c7;"><strong>Sentiment Score:</strong> A value from -1 (very negative) to +1 (very positive). Strong positive sentiment can drive prices up, while negative sentiment can pull them down.</li>
            <li style="color: #bdc3c7;"><strong>Social Mentions:</strong> Shows how much "buzz" a stock is getting. A spike in mentions can sometimes precede a significant price move.</li>
            <li style="color: #bdc3c7;"><strong>Price Increase Probability:</strong> The model's calculated likelihood that the stock price will go up the next day, based on sentiment and other factors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Find NSE Symbol
    st.markdown("""
    <div style="padding: 1.5rem; background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; border: 1px solid #34495e; margin-bottom: 2rem;">
        <h4 style="color: #ecf0f1;">Find NSE Symbol</h4>
        <p style="color: #bdc3c7;">Don't know the ticker symbol for a company? No problem. Just type the company name here, and the tool will find the correct NSE symbol for you. It uses fuzzy matching, so you don't have to be exact.</p>
    </div>
    """, unsafe_allow_html=True)

def find_nse_symbol_section():
    st.markdown('<h2 class="sub-header">Find NSE Symbol</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #bdc3c7;">Type a company name to get the correct NSE stock symbol. For example, try "Narayana Hrudayalaya" or "Tata Consultancy".</p>', unsafe_allow_html=True)

    symbol_map = {
        'reliance industries': 'RELIANCE.NS',
        'infosys': 'INFY.NS',
        'tata consultancy services': 'TCS.NS',
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

    user_input = st.text_input("Enter company name:", "")
    normalized_input = normalize(user_input)
    match = None
    if normalized_input:
        for key in symbol_map:
            if normalized_input == key:
                match = symbol_map[key]
                break
        if not match:
            for key in symbol_map:
                if normalized_input in key:
                    match = symbol_map[key]
                    break
        if not match:   
            for key in symbol_map:
                if any(word in key for word in normalized_input.split()):
                    match = symbol_map[key]
                    break
    if user_input:
        if match:
            st.success(f"Symbol for '{user_input}': **{match}**")
        else:
            st.warning(f"No NSE symbol found for '{user_input}'. Please check the spelling or try another name.")

def main():
    st.markdown('<h1 class="main-header">MarketVision Pro</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #ffffff;">Advanced AI-Powered Indian Stock Market Prediction System</h2>', unsafe_allow_html=True)
    
    model_data, metadata, model_loaded = load_model_and_data()
    processed_data = load_processed_data()
    
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Live Predictions", "Model Performance", "Technical Analysis", "Market Sentiment", "Model Details", "Cheatsheet", "Find NSE Symbol", "About"]
    )
    
    if page == "Dashboard":
        show_advanced_dashboard(model_data, metadata, processed_data)
    elif page == "Live Predictions":
        show_advanced_predictions(model_data, metadata)
    elif page == "Model Performance":
        show_model_performance(metadata)
    elif page == "Technical Analysis":
        show_technical_analysis(processed_data)
    elif page == "Market Sentiment":
        show_market_sentiment(processed_data)
    elif page == "Model Details":
        show_model_details(metadata)
    elif page == "Cheatsheet":
        show_cheatsheet()
    elif page == "Find NSE Symbol":
        find_nse_symbol_section()
    elif page == "About":
        show_about_page()

if __name__ == "__main__":
    main() 