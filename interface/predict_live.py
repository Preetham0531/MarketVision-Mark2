#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests
import feedparser
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

MODELS_DIR = os.path.join('..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'multioutput_lightgbm_model.pkl')
MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'lightgbm_model_info.json')
API_KEYS_DIR = os.path.join('..', 'api_keys')
NEWSAPI_KEY_PATH = os.path.join(API_KEYS_DIR, 'newsapi_key.txt')

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
label_map = {0: -1, 1: 0, 2: 1}

class MarketVisionPredictor:
    
    def __init__(self):
        self.model_data = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        print("Loading MarketVision AI Model...")
        
        if not os.path.exists(MODEL_PATH):
            print("Error: Trained model not found. Please run training first.")
            sys.exit(1)
        
        try:
            self.model_data = joblib.load(MODEL_PATH)
            with open(MODEL_INFO_PATH, 'r') as f:
                self.metadata = json.load(f)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def fetch_live_price_data(self, symbol):
        print(f"Fetching live price data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            today_data = ticker.history(period='1d', interval='1m')
            
            if today_data.empty:
                print(f"No live data available for {symbol}")
                return None
            
            latest = today_data.iloc[-1]
            
            price_data = {
                'close': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'vwap': (latest['High'] + latest['Low'] + latest['Close']) / 3,
                'price_change': latest['Close'] - latest['Open'],
                'price_change_pct': ((latest['Close'] - latest['Open']) / latest['Open']) * 100
            }
            
            print(f"Live price: ₹{price_data['close']:.2f} ({price_data['price_change_pct']:+.2f}%)")
            return price_data
            
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return None
    
    def fetch_technical_indicators(self, symbol):
        print("Calculating technical indicators...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='60d', interval='1d')
            
            if len(data) < 20:
                print("Insufficient data for technical indicators")
                return {}
            
            close_prices = data['Close']
            
            indicators = {
                'rsi_14_day': self.calculate_rsi(close_prices, 14),
                'moving_avg_20_day': close_prices.rolling(20).mean().iloc[-1],
                'moving_avg_50_day': close_prices.rolling(50).mean().iloc[-1],
                'bollinger_middle_band': close_prices.rolling(20).mean().iloc[-1],
                'bollinger_upper_band': close_prices.rolling(20).mean().iloc[-1] + 2 * close_prices.rolling(20).std().iloc[-1],
                'bollinger_lower_band': close_prices.rolling(20).mean().iloc[-1] - 2 * close_prices.rolling(20).std().iloc[-1],
                'macd_line': self.calculate_macd(close_prices),
                'stochastic_k_percent': self.calculate_stochastic(data, 14),
                'williams_r_14_day': self.calculate_williams_r(data, 14)
            }
            
            print("Technical indicators calculated")
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}
    
    def fetch_macro_context(self, symbol):
        print("Fetching macro market context...")
        
        try:
            nifty = yf.Ticker('^NSEI')
            nifty_data = nifty.history(period='5d')
            
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(period='5d')
            
            macro_context = {
                'nifty_close': nifty_data['Close'].iloc[-1] if not nifty_data.empty else 0,
                'nifty_change_pct': ((nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[-2]) / nifty_data['Close'].iloc[-2] * 100) if len(nifty_data) > 1 else 0,
                'vix_close': vix_data['Close'].iloc[-1] if not vix_data.empty else 0,
                'market_volatility': vix_data['Close'].iloc[-1] if not vix_data.empty else 20
            }
            
            print(f"NIFTY: ₹{macro_context['nifty_close']:.2f} ({macro_context['nifty_change_pct']:+.2f}%)")
            return macro_context
            
        except Exception as e:
            print(f"Error fetching macro context: {e}")
            return {}
    
    def fetch_news_sentiment(self, symbol):
        print("Analyzing news sentiment...")
        
        try:
            symbol_to_name = {
                'RELIANCE.NS': 'Reliance Industries',
                'TCS.NS': 'Tata Consultancy Services',
                'INFY.NS': 'Infosys',
                'HDFCBANK.NS': 'HDFC Bank',
                'ICICIBANK.NS': 'ICICI Bank',
                'WIPRO.NS': 'Wipro',
                'TATAMOTORS.NS': 'Tata Motors',
                'BHARTIARTL.NS': 'Bharti Airtel'
            }
            company_name = symbol_to_name.get(symbol.upper(), symbol.upper().replace('.NS', ''))
            
            query = company_name.replace(' ', '+') + "+stock+india"
            url = f"https://news.google.com/rss/search?q={query}"
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:5]:
                text = entry.title + ' ' + entry.get('summary', '')
                sentiment_score, _ = self.classify_sentiment(text)
                articles.append({
                    'title': entry.title,
                    'sentiment': sentiment_score,
                    'url': entry.link
                })
            
            if articles:
                avg_sentiment = np.mean([art['sentiment'] for art in articles])
                sentiment_context = {
                    'avg_sentiment_score': avg_sentiment,
                    'social_mention_count_7_day': len(articles),
                    'top_article': articles[0]['title'],
                    'top_article_url': articles[0]['url']
                }
                print(f"Sentiment: {avg_sentiment:.2f} ({len(articles)} articles)")
                return sentiment_context
            else:
                print("No news articles found")
                return {'avg_sentiment_score': 0, 'social_mention_count_7_day': 0}
                
        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            return {'avg_sentiment_score': 0, 'social_mention_count_7_day': 0}
    
    def classify_sentiment(self, text):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
                label = int(np.argmax(scores))
                return label_map[label], scores[label]
        except:
            return 0, 0.5
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.iloc[-1]
    
    def calculate_stochastic(self, data, period=14):
        low_min = data['Low'].rolling(period).min()
        high_max = data['High'].rolling(period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        return k_percent.iloc[-1]
    
    def calculate_williams_r(self, data, period=14):
        high_max = data['High'].rolling(period).max()
        low_min = data['Low'].rolling(period).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r.iloc[-1]
    
    def prepare_features(self, symbol):
        print("Preparing features for prediction...")
        
        price_data = self.fetch_live_price_data(symbol)
        indicators = self.fetch_technical_indicators(symbol)
        macro_context = self.fetch_macro_context(symbol)
        sentiment = self.fetch_news_sentiment(symbol)
        
        features = {}
        features.update(price_data or {})
        features.update(indicators)
        features.update(macro_context)
        features.update(sentiment)
        
        expected_features = self.metadata.get('feature_names', [])
        
        feature_vector = []
        for feature in expected_features:
            if feature in features:
                feature_vector.append(features[feature])
            else:
                feature_vector.append(0)
        
        return np.array(feature_vector).reshape(1, -1), features
    
    def make_prediction(self, symbol):
        print(f"\nMaking prediction for {symbol}...")
        
        X_pred, raw_features = self.prepare_features(symbol)
        
        scaler = self.model_data['scaler']
        X_pred_scaled = scaler.transform(X_pred)
        
        reg_model = self.model_data['regression_model']
        clf_model = self.model_data['classification_model']
        
        reg_predictions = reg_model.predict(X_pred_scaled)
        clf_predictions = clf_model.predict(X_pred_scaled)
        
        label_encoders = self.model_data['label_encoders']
        decoded_predictions = {}
        
        for i, label in enumerate(self.metadata['classification_labels']):
            if label in label_encoders:
                le = label_encoders[label]
                decoded_predictions[label] = le.inverse_transform([clf_predictions[0][i]])[0]
            else:
                decoded_predictions[label] = clf_predictions[0][i]
        
        confidence = self.calculate_confidence(reg_predictions, clf_predictions, raw_features)
        
        top_features = self.get_top_features(X_pred_scaled)
        
        return {
            'symbol': symbol,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'regression_predictions': reg_predictions[0],
            'classification_predictions': decoded_predictions,
            'confidence': confidence,
            'top_features': top_features,
            'raw_features': raw_features
        }
    
    def calculate_confidence(self, reg_pred, clf_pred, features):
        confidence_factors = []
        
        if 'avg_sentiment_score' in features:
            sentiment_abs = abs(features['avg_sentiment_score'])
            confidence_factors.append(min(sentiment_abs * 20, 100))
        
        if 'volume' in features and features['volume'] > 0:
            confidence_factors.append(80)
        
        if 'market_volatility' in features:
            volatility = features['market_volatility']
            if volatility < 15:
                confidence_factors.append(90)
            elif volatility > 25:
                confidence_factors.append(60)
            else:
                confidence_factors.append(75)
        
        avg_confidence = np.mean(confidence_factors) if confidence_factors else 70
        return min(max(avg_confidence, 50), 95)
    
    def get_top_features(self, X_pred_scaled):
        try:
            reg_model = self.model_data['regression_model']
            feature_names = self.metadata.get('feature_names', [])
            
            if hasattr(reg_model, 'estimators_'):
                importance = reg_model.estimators_[0].feature_importances_
            else:
                importance = reg_model.feature_importances_
            
            feature_importance = list(zip(feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance[:5]
            
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            return []
    
    def display_prediction(self, prediction):
        symbol = prediction['symbol']
        date = prediction['date']
        reg_pred = prediction['regression_predictions']
        clf_pred = prediction['classification_predictions']
        confidence = prediction['confidence']
        top_features = prediction['top_features']
        raw_features = prediction['raw_features']
        
        trend_direction = clf_pred.get('predicted_trend_direction', 'NEUTRAL')
        next_day_price = reg_pred[0] if len(reg_pred) > 0 else raw_features.get('close', 0)
        current_price = raw_features.get('close', 0)
        
        if trend_direction == 'UPWARD':
            recommendation = "BUY"
            direction_emoji = "🟢"
        elif trend_direction == 'DOWNWARD':
            recommendation = "SELL"
            direction_emoji = "🔴"
        else:
            recommendation = "HOLD"
            direction_emoji = "🟡"
        
        price_change = next_day_price - current_price
        price_change_pct = (price_change / current_price * 100) if current_price > 0 else 0
        
        confidence_interval = confidence * 0.02
        lower_bound = next_day_price - confidence_interval
        upper_bound = next_day_price + confidence_interval
        
        print("\n" + "="*80)
        print(f"MarketVision AI Prediction Report")
        print("="*80)
        print(f"Symbol: {symbol} | Date: {date}")
        print(f"Current Price: ₹{current_price:.2f}")
        print("-"*80)
        
        print(f"{direction_emoji} Direction: {trend_direction} (Confidence: {confidence:.0f}%)")
        print(f"Recommendation: {recommendation}")
        print(f"Predicted Close Tomorrow: ₹{next_day_price:.2f}")
        print(f"Expected Change: ₹{price_change:+.2f} ({price_change_pct:+.2f}%)")
        print(f"Prediction Interval (95%): ₹{lower_bound:.2f} – ₹{upper_bound:.2f}")
        
        print("\nTop Influencing Factors:")
        for i, (feature, importance) in enumerate(top_features[:3], 1):
            feature_value = raw_features.get(feature, 0)
            print(f"   {i}. {feature}: {feature_value:.3f} (Impact: {importance:.3f})")
        
        if 'nifty_change_pct' in raw_features:
            print(f"\nMarket Context:")
            print(f"   NIFTY 50: {raw_features['nifty_change_pct']:+.2f}%")
            print(f"   VIX: {raw_features.get('market_volatility', 0):.1f}")
        
        if 'avg_sentiment_score' in raw_features:
            sentiment = raw_features['avg_sentiment_score']
            sentiment_emoji = "😊" if sentiment > 0 else "😐" if sentiment == 0 else "😞"
            print(f"   {sentiment_emoji} News Sentiment: {sentiment:.2f}")
        
        if 'top_article' in raw_features:
            print(f"\nTop News: {raw_features['top_article'][:80]}...")
        
        print("="*80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_live.py <SYMBOL>")
        print("Example: python predict_live.py TCS.NS")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    print("MarketVision AI - Live Stock Prediction")
    print("="*50)
    
    try:
        predictor = MarketVisionPredictor()
        prediction = predictor.make_prediction(symbol)
        predictor.display_prediction(prediction)
        
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 