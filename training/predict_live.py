import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join('..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'multioutput_lightgbm_model.pkl')
MODEL_INFO_PATH = os.path.join(MODELS_DIR, 'lightgbm_model_info.json')
FINAL_DATA_PATH = os.path.join('..', 'data', 'final_training_data.csv')

def load_trained_model():
    print("[INFO] Loading trained model...")
    
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Trained model not found. Please run train_model.py first.")
        return None, None, None, None, None
    
    model_data = joblib.load(MODEL_PATH)
    reg_model = model_data['regression_model']
    clf_model = model_data['classification_model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            metadata = json.load(f)
    else:
        print("[WARNING] Model metadata not found. Using default labels.")
        metadata = {
            'regression_labels': ['predicted_close_price_t_plus_1'],
            'classification_labels': ['predicted_trend_direction']
        }
    
    return reg_model, clf_model, scaler, label_encoders, metadata

def prepare_prediction_data(model_data):
    print("[INFO] Preparing prediction data...")
    
    if not os.path.exists(FINAL_DATA_PATH):
        print("[ERROR] Final training data not found. Please run the data preparation scripts first.")
        return None, None
    
    df = pd.read_csv(FINAL_DATA_PATH)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    
    latest_row = df.iloc[-1:].copy()
    
    _, _, _, _, metadata = model_data
    if 'feature_names' in metadata:
        feature_names = metadata['feature_names']
    else:
        print("[WARNING] Feature names not found in metadata. Using all non-target columns.")
        target_cols = ['predicted_close_price_t_plus_1', 'predicted_close_price_t_plus_5', 
                       'predicted_close_price_t_plus_20', 'predicted_daily_return_t_plus_1',
                       'predicted_weekly_return_t_plus_5', 'predicted_volatility_next_20_days',
                       'predicted_trend_direction', 'buy_sell_hold_recommendation', 
                       'predicted_volatility_regime']
        feature_names = [col for col in df.columns if col not in target_cols + ['date']]
    
    print(f"[INFO] Using {len(feature_names)} features for prediction")
    
    X_pred = latest_row[feature_names].copy()
    
    for col in feature_names:
        if col not in X_pred.columns:
            print(f"[WARNING] Feature '{col}' not found in data, filling with 0")
            X_pred[col] = 0
    
    return X_pred, latest_row

def make_predictions(model_data, X_pred, latest_row):
    print("[INFO] Making predictions...")
    
    reg_model, clf_model, scaler, label_encoders, metadata = model_data
    
    X_pred_scaled = scaler.transform(X_pred)
    
    reg_predictions = reg_model.predict(X_pred_scaled)
    clf_predictions = clf_model.predict(X_pred_scaled)
    
    decoded_predictions = {}
    for i, label in enumerate(metadata['classification_labels']):
        if label in label_encoders:
            le = label_encoders[label]
            decoded_predictions[label] = le.inverse_transform([clf_predictions[0][i]])[0]
        else:
            decoded_predictions[label] = clf_predictions[0][i]
    
    results = {
        'date': latest_row['date'].iloc[0],
        'symbol': 'RELIANCE.NS'  
    }
    
    for i, label in enumerate(metadata['regression_labels']):
        results[label] = reg_predictions[0][i]
    
    for label, value in decoded_predictions.items():
        results[label] = value
    
    return results

def generate_trading_recommendation(predictions):
    print("[INFO] Generating trading recommendation...")
    
    recommendation = {
        'timestamp': datetime.now().isoformat(),
        'symbol': predictions['symbol'],
        'current_date': predictions['date'],
        'recommendation': 'HOLD',
        'confidence': 'MEDIUM',
        'reasoning': [],
        'price_predictions': {},
        'risk_assessment': {}
    }
    
    if 'predicted_close_price_t_plus_1' in predictions:
        predicted_price = predictions['predicted_close_price_t_plus_1']
        recommendation['price_predictions']['next_day'] = {
            'predicted_price': predicted_price,
            'expected_change_pct': 0  
        }
        
        recommendation['reasoning'].append(f"Predicted next day price: ₹{predicted_price:.2f}")
    
    if 'predicted_trend_direction' in predictions:
        trend = predictions['predicted_trend_direction']
        recommendation['trend_analysis'] = trend
        
        if trend == 'UPWARD':
            recommendation['recommendation'] = 'BUY'
            recommendation['reasoning'].append("Upward trend predicted")
        elif trend == 'DOWNWARD':
            recommendation['recommendation'] = 'SELL'
            recommendation['reasoning'].append("Downward trend predicted")
    
    if 'predicted_volatility_regime' in predictions:
        volatility = predictions['predicted_volatility_regime']
        recommendation['risk_assessment']['volatility_regime'] = volatility
        
        if volatility == 'HIGH':
            recommendation['confidence'] = 'LOW'
            recommendation['reasoning'].append("High volatility expected - lower confidence")
    
    return recommendation

def print_prediction_summary(predictions, recommendation):
    print("\n" + "=" * 60)
    print("LIVE PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"Symbol: {predictions['symbol']}")
    print(f"Date: {predictions['date']}")
    print(f"Timestamp: {recommendation['timestamp']}")
    
    print("\n--- PRICE PREDICTIONS ---")
    for label in ['predicted_close_price_t_plus_1', 'predicted_close_price_t_plus_5', 'predicted_close_price_t_plus_20']:
        if label in predictions:
            print(f"{label}: ₹{predictions[label]:.2f}")
    
    print("\n--- TRADING RECOMMENDATION ---")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']}")
    print(f"Trend: {recommendation.get('trend_analysis', 'N/A')}")
    print(f"Volatility: {recommendation['risk_assessment'].get('volatility_regime', 'N/A')}")
    
    if recommendation['reasoning']:
        print("\n--- REASONING ---")
        for reason in recommendation['reasoning']:
            print(f"• {reason}")
    
    print("=" * 60)

def main(): 
    print("=" * 60)
    print("LIVE PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        model_data = load_trained_model()
        if model_data[0] is None:
            return
        
        X_pred, latest_row = prepare_prediction_data(model_data)
        if X_pred is None:
            return
        
        predictions = make_predictions(model_data, X_pred, latest_row)
        
        recommendation = generate_trading_recommendation(predictions)
        
        print_prediction_summary(predictions, recommendation)
        
        print(f"\n[SUCCESS] Live prediction completed!")
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 