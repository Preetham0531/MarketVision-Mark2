import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GLOBAL_TICKERS = {
    'sp500': '^GSPC',
    'nasdaq': '^IXIC',
    'vix': '^VIX',
    'nifty': '^NSEI'
}
SECTOR_ETFS = {
    'IT': 'INFY.NS',
    'FMCG': 'HINDUNILVR.NS',
    'PHARMA': 'SUNPHARMA.NS',
}
NIFTY50_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'LT.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS',
    'BAJFINANCE.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'AXISBANK.NS', 'MARUTI.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'TATAMOTORS.NS', 'WIPRO.NS', 'NTPC.NS', 'POWERGRID.NS',
    'JSWSTEEL.NS', 'TATASTEEL.NS', 'DIVISLAB.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS',
    'ADANIENT.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'INDUSINDBK.NS', 'M&M.NS', 'NESTLEIND.NS', 'ONGC.NS', 'SHREECEM.NS', 'SIEMENS.NS',
    'TATACONSUM.NS', 'TECHM.NS', 'UPL.NS', 'VEDL.NS'
]

RAW_MACRO_DIR = os.path.join('..', 'data', 'raw', 'global_indicators')
SECTOR_DIR = os.path.join(RAW_MACRO_DIR, 'sector_etf')
PROCESSED_DIR = os.path.join('..', 'data', 'processed')

os.makedirs(RAW_MACRO_DIR, exist_ok=True)
os.makedirs(SECTOR_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def fetch_index_data(ticker, name):
    path = os.path.join(RAW_MACRO_DIR, f'{name}.csv')
    try:
        df = yf.download(ticker, period='90d', interval='1d', progress=False)
        if df.empty:
            logging.warning(f"Download failed or returned no data for {ticker} ({name})")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == ticker or col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        df = df.reset_index()
        
        df = df.tail(60).copy()
        
        if 'Date' in df.columns and 'Close' in df.columns:
            df = df[['Date', 'Close']].copy()
            df.rename(columns={'Date': 'date', 'Close': f'{name}_close'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df.to_csv(path, index=False)
            logging.info(f"Successfully fetched {name} data: {len(df)} records")
            return df
        else:
            logging.error(f"Required columns missing for {ticker} ({name}). Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error fetching data for {ticker} ({name}): {e}")
        return pd.DataFrame()

def fetch_sector_etf(sector, ticker):
    path = os.path.join(SECTOR_DIR, f'{sector}.csv')
    try:
        df = yf.download(ticker, period='90d', interval='1d', progress=False)
        if df.empty:
            logging.warning(f"Download failed or returned no data for {ticker} (sector {sector})")
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == ticker or col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        df = df.reset_index()
        df = df.tail(60).copy()
        
        if 'Date' in df.columns and 'Close' in df.columns:
            df = df[['Date', 'Close']].copy()
            df.rename(columns={'Date': 'date', 'Close': f'{sector}_close'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            df['sector_etf_performance'] = df[f'{sector}_close'].pct_change()
            
            df.to_csv(path, index=False)
            logging.info(f"Successfully fetched {sector} ETF data: {len(df)} records")
            return df
        else:
            logging.error(f"Required columns missing for {ticker} (sector {sector}). Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error fetching sector ETF data for {ticker} ({sector}): {e}")
        return pd.DataFrame()

def get_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('marketCap', np.nan)
    except Exception as e:
        logging.warning(f"Could not fetch market cap for {symbol}: {e}")
        return np.nan

def get_nifty_market_caps():
    caps = {}
    for sym in NIFTY50_SYMBOLS:
        caps[sym] = get_market_cap(sym)
    return caps

def calculate_percentile_rank(market_cap, all_caps):
    caps = np.array([v for v in all_caps.values() if not np.isnan(v)])
    if np.isnan(market_cap) or len(caps) == 0:
        return np.nan
    return (rankdata([market_cap] + list(caps), method='min')[0] - 1) / len(caps)

def calculate_beta(stock_returns, index_returns):
    if len(stock_returns) != len(index_returns):
        return np.nan
    
    stock_returns = np.array(stock_returns)
    index_returns = np.array(index_returns)
    
    mask = ~(np.isnan(stock_returns) | np.isnan(index_returns))
    stock_returns_clean = stock_returns[mask]
    index_returns_clean = index_returns[mask]
    
    if len(stock_returns_clean) < 2:
        return np.nan
        
    cov = np.cov(stock_returns_clean, index_returns_clean)[0][1]
    var = np.var(index_returns_clean)
    return cov / var if var != 0 else np.nan

def create_lagged_features(df, column_name, prefix, num_lags=5):
    result_df = df.copy()
    
    for lag in range(1, num_lags + 1):
        colname = f'{prefix}_t-{lag}'
        if column_name in df.columns:
            result_df[colname] = df[column_name].shift(lag)
        else:
            result_df[colname] = np.nan
    
    norm_cols = [f'{prefix}_t-{i}' for i in range(1, num_lags + 1)]
    for col in norm_cols:
        if col in result_df.columns and not result_df[col].isna().all():
            scaler = MinMaxScaler()
            result_df[col] = scaler.fit_transform(result_df[[col]].fillna(0))
        else:
            result_df[col] = np.nan
    
    return result_df

def fetch_market_context():
    context = {}
    base_path = os.path.join('..', 'data', 'raw', 'global_indicators')
    files = ['nifty.csv', 'sp500.csv', 'nasdaq.csv', 'vix.csv']
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            context[file.replace('.csv', '')] = pd.read_csv(file_path)
    return context

def save_market_context(context):
    output_dir = os.path.join('..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    for key, df in context.items():
        output_path = os.path.join(output_dir, f'{key}_context.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {key} context to: {output_path}")

def main(symbol, sector):
    logging.info("Fetching global indices...")
    sp500 = fetch_index_data(GLOBAL_TICKERS['sp500'], 'sp500')
    nasdaq = fetch_index_data(GLOBAL_TICKERS['nasdaq'], 'nasdaq')
    vix = fetch_index_data(GLOBAL_TICKERS['vix'], 'vix')
    nifty = fetch_index_data(GLOBAL_TICKERS['nifty'], 'nifty')
    
    if sector in SECTOR_ETFS:
        sector_etf = fetch_sector_etf(sector, SECTOR_ETFS[sector])
    else:
        logging.warning(f"Sector {sector} not found in SECTOR_ETFS. Available sectors: {list(SECTOR_ETFS.keys())}")
        sector_etf = pd.DataFrame()

    stock_path = os.path.join(PROCESSED_DIR, f'stock_{symbol}_with_indicators.csv')
    if not os.path.exists(stock_path):
        logging.error(f"Processed stock file not found: {stock_path}")
        return
    
    stock_df = pd.read_csv(stock_path)
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.strftime('%Y-%m-%d')
    logging.info(f"Loaded stock data with shape: {stock_df.shape}")

    macro_data = [
        (sp500, 'sp500', 'sp500_close'),
        (nasdaq, 'nasdaq', 'nasdaq_close'),
        (vix, 'vix', 'vix_close')
    ]
    
    for df, prefix, close_col in macro_data:
        if df.empty:
            logging.warning(f"{prefix} DataFrame is empty. Skipping merge for this index.")
            for lag in range(1, 6):
                stock_df[f'{prefix}_t-{lag}'] = np.nan
            continue
        
        df_with_lags = create_lagged_features(df, close_col, prefix, 5)
        
        lag_cols = [f'{prefix}_t-{i}' for i in range(1, 6)]
        merge_cols = ['date'] + lag_cols
        df_to_merge = df_with_lags[merge_cols].copy()
        
        stock_df = stock_df.merge(df_to_merge, on='date', how='left')
        logging.info(f"Merged {prefix} data. New shape: {stock_df.shape}")

    if not sector_etf.empty and 'sector_etf_performance' in sector_etf.columns:
        stock_df = stock_df.merge(
            sector_etf[['date', 'sector_etf_performance']], 
            on='date', 
            how='left'
        )
        logging.info("Merged sector ETF performance data")
    else:
        logging.warning("Sector ETF DataFrame is empty or missing required columns. Adding NaN column.")
        stock_df['sector_etf_performance'] = np.nan

    logging.info("Calculating market cap percentile rank...")
    all_caps = get_nifty_market_caps()
    stock_cap = get_market_cap(symbol)
    percentile_rank = calculate_percentile_rank(stock_cap, all_caps)
    stock_df['market_cap_percentile_rank'] = percentile_rank
    logging.info(f"Market cap percentile rank: {percentile_rank}")

    logging.info("Calculating beta coefficient vs NIFTY 50...")
    if nifty.empty or 'nifty_close' not in nifty.columns:
        logging.warning("NIFTY DataFrame is empty or missing required columns. Setting beta to NaN.")
        stock_df['beta_coefficient_60_day'] = np.nan
    else:
        stock_df['stock_return'] = stock_df['close'].pct_change()
        nifty['nifty_return'] = nifty['nifty_close'].pct_change()
        
        merged_for_beta = stock_df[['date', 'stock_return']].merge(
            nifty[['date', 'nifty_return']], 
            on='date', 
            how='inner'
        )
        
        if merged_for_beta.empty or len(merged_for_beta) < 2:
            logging.warning("Insufficient overlapping return data for beta calculation. Setting beta to NaN.")
            beta = np.nan
        else:
            beta = calculate_beta(
                merged_for_beta['stock_return'].values, 
                merged_for_beta['nifty_return'].values
            )
        
        stock_df['beta_coefficient_60_day'] = beta
        logging.info(f"Beta coefficient: {beta}")

    stock_df.drop(columns=['stock_return'], inplace=True, errors='ignore')
    
    out_path = os.path.join(PROCESSED_DIR, f'stock_{symbol}_with_macro_context.csv')
    stock_df.to_csv(out_path, index=False, encoding='utf-8')
    logging.info(f"Final macro-enriched file saved to {out_path}")
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Final shape: {stock_df.shape}")
    print(f"Columns added: {[col for col in stock_df.columns if any(x in col for x in ['sp500', 'nasdaq', 'vix', 'sector', 'market_cap', 'beta'])]}")
    print("\nFirst few rows:")
    print(stock_df.head())
    print("\nLast few rows:")
    print(stock_df.tail())
    
    numeric_cols = stock_df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric columns count: {len(numeric_cols)}")
    print(f"Missing values per column:")
    missing_counts = stock_df.isnull().sum()
    for col in missing_counts[missing_counts > 0].index:
        print(f"  {col}: {missing_counts[col]} ({missing_counts[col]/len(stock_df)*100:.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fetch_market_context.py <SYMBOL> <SECTOR>")
        print("Example: python fetch_market_context.py RELIANCE.NS IT")
        print(f"Available sectors: {list(SECTOR_ETFS.keys())}")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    sector = sys.argv[2].upper()
    
    logging.info(f"Starting market context fetch for {symbol} in {sector} sector")
    main(symbol, sector)