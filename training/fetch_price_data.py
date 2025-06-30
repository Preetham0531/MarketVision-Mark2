import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def fetch_stock_data(symbol, period="2y"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        
        data = data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })
        
        data['symbol'] = symbol
        
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_stock_data(data, symbol):
    output_dir = os.path.join('..', 'data', 'raw', 'stock_prices')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{symbol}.csv')
    data.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

def main():
    symbols = ['RELIANCE.NS', 'INFY.NS']
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data = fetch_stock_data(symbol)
        
        if data is not None:
            save_stock_data(data, symbol)
            print(f"Successfully fetched {len(data)} records for {symbol}")
        else:
            print(f"Failed to fetch data for {symbol}")
        
        time.sleep(1)

if __name__ == "__main__":
    main() 