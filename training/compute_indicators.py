import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')

def plot_indicators(df: pd.DataFrame, symbol: str):
    charts_dir = os.path.join('..', 'logs', 'indicator_charts', symbol)
    os.makedirs(charts_dir, exist_ok=True)
    
    indicator_cols = df.columns.drop(['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'vwap', 'obv'])
    
    logging.info(f"Generating {len(indicator_cols)} indicator charts in '{charts_dir}'...")

    for col in indicator_cols:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df[col], label=col)
        plt.title(f'{symbol} - {col}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        chart_path = os.path.join(charts_dir, f"{col}.png")
        plt.savefig(chart_path)
        plt.close()
        
    logging.info("Chart generation complete.")


def calculate_technical_indicators(df):
    if df.empty:
        return df
    
    df = df.copy()
    
    df['rsi_14_day'] = ta.rsi(df['close'], length=14)
    df['rsi_30_day'] = ta.rsi(df['close'], length=30)
    
    macd = ta.macd(df['close'])
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal_line'] = macd['MACDs_12_26_9']
    df['macd_histogram'] = macd['MACDh_12_26_9']
    
    bb = ta.bbands(df['close'], length=20)
    df['bollinger_upper_band'] = bb['BBU_20_2.0']
    df['bollinger_middle_band'] = bb['BBM_20_2.0']
    df['bollinger_lower_band'] = bb['BBL_20_2.0']
    
    df['moving_avg_5_day'] = ta.sma(df['close'], length=5)
    df['moving_avg_20_day'] = ta.sma(df['close'], length=20)
    df['moving_avg_50_day'] = ta.sma(df['close'], length=50)
    
    df['exponential_moving_avg_12_day'] = ta.ema(df['close'], length=12)
    df['exponential_moving_avg_26_day'] = ta.ema(df['close'], length=26)
    
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stochastic_k_percent'] = stoch['STOCHk_14_3_3']
    df['stochastic_d_percent'] = stoch['STOCHd_14_3_3']
    
    df['williams_r_14_day'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    df['commodity_channel_index_20_day'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    df['on_balance_volume'] = ta.obv(df['close'], df['volume'])
    df['volume_weighted_avg_price'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    df['average_true_range'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['average_directional_index'] = ta.adx(df['high'], df['low'], df['close'], length=14)
    
    df['money_flow_index'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    df['chaikin_money_flow'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
    
    df['price_rate_of_change'] = ta.roc(df['close'], length=10)
    df['momentum'] = ta.mom(df['close'], length=10)
    
    df['ultimate_oscillator'] = ta.uo(df['high'], df['low'], df['close'])
    df['trix'] = ta.trix(df['close'], length=18)
    
    df['kst'] = ta.kst(df['close'])
    df['tsi'] = ta.tsi(df['close'])
    
    df['aroon_up'] = ta.aroon(df['high'], length=25)['AROONU_25']
    df['aroon_down'] = ta.aroon(df['high'], length=25)['AROOND_25']
    df['aroon_oscillator'] = ta.aroon(df['high'], length=25)['AROONOSC_25']
    
    df['ichimoku_a'] = ta.ichimoku(df['high'], df['low'], df['close'])['ISA_9']
    df['ichimoku_b'] = ta.ichimoku(df['high'], df['low'], df['close'])['ISB_26']
    df['ichimoku_base'] = ta.ichimoku(df['high'], df['low'], df['close'])['ITS_26']
    df['ichimoku_conversion'] = ta.ichimoku(df['high'], df['low'], df['close'])['IKS_26']
    
    df['psar'] = ta.psar(df['high'], df['low'], df['close'])
    
    df['supertrend'] = ta.supertrend(df['high'], df['low'], df['close'])
    
    df['squeeze'] = ta.squeeze(df['high'], df['low'], df['close'], df['volume'])
    
    df['vwma'] = ta.vwma(df['close'], df['volume'], length=20)
    df['hma'] = ta.hma(df['close'], length=20)
    df['ema'] = ta.ema(df['close'], length=20)
    df['dema'] = ta.dema(df['close'], length=20)
    df['tema'] = ta.tema(df['close'], length=20)
    
    df['kama'] = ta.kama(df['close'], length=20)
    df['zlema'] = ta.zlema(df['close'], length=20)
    
    df['wma'] = ta.wma(df['close'], length=20)
    df['hma'] = ta.hma(df['close'], length=20)
    
    df['rsi_divergence'] = df['rsi_14_day'].diff()
    df['macd_divergence'] = df['macd_line'].diff()
    
    df['price_above_ma20'] = (df['close'] > df['moving_avg_20_day']).astype(int)
    df['price_above_ma50'] = (df['close'] > df['moving_avg_50_day']).astype(int)
    df['ma20_above_ma50'] = (df['moving_avg_20_day'] > df['moving_avg_50_day']).astype(int)
    
    df['rsi_overbought'] = (df['rsi_14_day'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14_day'] < 30).astype(int)
    
    df['macd_bullish'] = (df['macd_line'] > df['macd_signal_line']).astype(int)
    df['macd_bearish'] = (df['macd_line'] < df['macd_signal_line']).astype(int)
    
    df['bb_upper_breakout'] = (df['close'] > df['bollinger_upper_band']).astype(int)
    df['bb_lower_breakout'] = (df['close'] < df['bollinger_lower_band']).astype(int)
    
    df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['price_volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    return df

def process_stock_data(symbol):
    data_path = os.path.join('..', 'data', 'raw', 'stock_prices', f'{symbol}.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df = calculate_technical_indicators(df)
    
    output_path = os.path.join('..', 'data', 'processed', f'stock_{symbol}_with_indicators.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Technical indicators computed and saved to: {output_path}")
    
    return df

def main():
    symbols = ['RELIANCE.NS']
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        process_stock_data(symbol)

if __name__ == "__main__":
    main() 