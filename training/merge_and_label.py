import os
import sys
import json
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join('..', 'data', 'processed')
RAW_FUNDAMENTALS_DIR = os.path.join('..', 'data', 'raw', 'fundamentals')
FINAL_DATA_PATH = os.path.join('..', 'data', 'final_training_data.csv')

def load_fundamentals(symbol):
    for s in [symbol, symbol.replace('.NS', '')]:
        path = os.path.join(RAW_FUNDAMENTALS_DIR, f'{s}.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

def add_fundamentals(df, fundamentals):
    for k, v in fundamentals.items():
        if k == 'symbol':
            continue
        df[k] = v
    return df.ffill()

def generate_labels(df):
    df['predicted_close_price_t_plus_1'] = df['close'].shift(-1)
    df['predicted_close_price_t_plus_5'] = df['close'].shift(-5)
    df['predicted_close_price_t_plus_20'] = df['close'].shift(-20)
    df['predicted_daily_return_t_plus_1'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['predicted_weekly_return_t_plus_5'] = (df['close'].shift(-5) - df['close']) / df['close']
    df['predicted_monthly_return_t_plus_20'] = (df['close'].shift(-20) - df['close']) / df['close']
    df['predicted_cumulative_return_next_quarter'] = df['close'].shift(-60) / df['close'] - 1
    df['predicted_volatility_next_20_days'] = df['close'].pct_change().rolling(20).std().shift(-20)
    df['predicted_maximum_drawdown_next_30_days'] = (
        df['close'].rolling(30).max().shift(-30) - df['close'].rolling(30).min().shift(-30)
    ) / df['close'].rolling(30).max().shift(-30)
    df['predicted_value_at_risk_95_percent_confidence'] = df['close'].pct_change().rolling(20).quantile(0.05).shift(-20)
    df['predicted_conditional_value_at_risk_95_percent'] = df['close'].pct_change().rolling(20).apply(lambda x: x[x <= np.quantile(x, 0.05)].mean() if len(x) > 0 else np.nan).shift(-20)
    df['probability_price_increase_next_day'] = (df['predicted_daily_return_t_plus_1'] > 0).astype(int)
    df['predicted_trend_direction'] = np.where(df['predicted_weekly_return_t_plus_5'] > 0.03, 'bullish',
                                               np.where(df['predicted_weekly_return_t_plus_5'] < -0.03, 'bearish', 'neutral'))
    df['predicted_volatility_regime'] = pd.cut(df['predicted_volatility_next_20_days'],
                                               bins=[-np.inf, 0.015, 0.03, np.inf],
                                               labels=['low', 'medium', 'high'])
    df['buy_sell_hold_recommendation'] = np.where(
        (df['predicted_trend_direction'] == 'bullish') & (df['avg_sentiment_score'] > 0.5), 'strong_buy',
        np.where((df['predicted_trend_direction'] == 'bearish') & (df['predicted_volatility_regime'] == 'high'), 'strong_sell', 'hold')
    )
    df['prediction_confidence_interval_lower_bound_95_percent'] = np.nan
    df['prediction_confidence_interval_upper_bound_95_percent'] = np.nan
    df['model_prediction_uncertainty_score'] = np.nan
    return df

def main(symbol):
    ind_path = os.path.join(PROCESSED_DIR, f'stock_{symbol}_with_indicators.csv')
    macro_path = os.path.join(PROCESSED_DIR, f'stock_{symbol}_with_macro_context.csv')
    sent_path = os.path.join(PROCESSED_DIR, f'stock_{symbol}_with_sentiment.csv')
    fundamentals = load_fundamentals(symbol)
    if not (os.path.exists(ind_path) and os.path.exists(macro_path) and os.path.exists(sent_path)):
        print("[ERROR] One or more input files are missing.")
        return
    ind_df = pd.read_csv(ind_path)
    macro_df = pd.read_csv(macro_path)
    sent_df = pd.read_csv(sent_path)
    df = ind_df.merge(macro_df, on='date', how='inner', suffixes=('', '_macro'))
    df = df.merge(sent_df, on='date', how='inner', suffixes=('', '_sent'))
    df = add_fundamentals(df, fundamentals)
    df = df.dropna(subset=['close'])
    df = generate_labels(df)
    df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"[INFO] Final merged and labeled dataset saved to {FINAL_DATA_PATH}")
    print(f"[INFO] Final shape: {df.shape}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python merge_and_label.py <SYMBOL>")
        sys.exit(1)
    main(sys.argv[1]) 