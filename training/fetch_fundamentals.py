import os
import json
import requests
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def fetch_fundamentals(symbol, api_key):
    url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]
    return None

def save_fundamentals(data, symbol):
    output_dir = os.path.join('..', 'data', 'raw', 'fundamentals')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{symbol}.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Fundamentals saved to: {output_path}")

def main():
    api_key_path = os.path.join('..', 'api_keys', 'fmp_key.txt')
    if not os.path.exists(api_key_path):
        print("API key file not found.")
        return
    with open(api_key_path, 'r') as f:
        api_key = f.read().strip()
    symbols = ['RELIANCE.NS']
    for symbol in symbols:
        print(f"Fetching fundamentals for {symbol}...")
        data = fetch_fundamentals(symbol, api_key)
        if data:
            save_fundamentals(data, symbol)
        else:
            print(f"Failed to fetch fundamentals for {symbol}")

if __name__ == "__main__":
    main() 