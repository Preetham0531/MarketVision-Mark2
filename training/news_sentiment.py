import os
import sys
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')

RAW_SENTIMENT_DIR = os.path.join('..', 'data', 'raw', 'sentiment')
PROCESSED_DIR = os.path.join('..', 'data', 'processed')
API_KEYS_DIR = os.path.join('..', 'api_keys')
NEWSAPI_KEY_PATH = os.path.join(API_KEYS_DIR, 'newsapi_key.txt')

os.makedirs(RAW_SENTIMENT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
label_map = {0: -1, 1: 0, 2: 1}  # FinBERT: 0=negative, 1=neutral, 2=positive

def get_newsapi_key():
    if os.path.exists(NEWSAPI_KEY_PATH):
        with open(NEWSAPI_KEY_PATH) as f:
            return f.read().strip()
    return None

def fetch_newsapi_articles(company_name, from_date, to_date, api_key, max_articles=100):
    url = (
        f"https://newsapi.org/v2/everything?q=\"{company_name}\" AND (site:moneycontrol.com OR site:business-standard.com)"
        f"&from={from_date}&to={to_date}&language=en&sortBy=publishedAt&pageSize=100&apiKey={api_key}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return []
    data = r.json()
    articles = data.get('articles', [])[:max_articles]
    return [
        {
            'date': a['publishedAt'][:10],
            'headline': a['title'],
            'description': a.get('description', ''),
            'source': a['source']['name'],
            'url': a['url']
        }
        for a in articles
    ]

def fetch_google_news_rss(company_name):
    query = company_name.replace(' ', '+') + "+stock"
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        date = entry.get('published', entry.get('updated', ''))[:10]
        articles.append({
            'date': date,
            'headline': entry.title,
            'description': entry.get('summary', ''),
            'source': 'Google News',
            'url': entry.link
        })
    return articles

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
        label = int(np.argmax(scores))
        return label_map[label], scores[label]

def aggregate_daily_sentiment(articles, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    daily = {d.strftime('%Y-%m-%d'): {'scores': [], 'count': 0} for d in date_range}
    for art in articles:
        date = art['date']
        if date in daily:
            daily[date]['scores'].append(art['sentiment_score'])
            daily[date]['count'] += 1
    rows = []
    for date in daily:
        scores = daily[date]['scores']
        avg_score = np.mean(scores) if scores else 0
        count = daily[date]['count']
        rows.append({
            'date': date,
            'avg_sentiment_score': avg_score,
            'social_mention_count_7_day': count,
            'net_insider_trading_activity': ''  
        })
    return pd.DataFrame(rows)

def generate_sentiment_data(symbol, days=60):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    sentiment_data = []
    
    for date in dates:
        if date.weekday() < 5:
            sentiment_score = np.random.normal(0.1, 0.3)
            sentiment_score = max(-1, min(1, sentiment_score))
            
            news_count = np.random.poisson(15)
            social_mentions = np.random.poisson(50)
            
            probability_increase = 0.5 + (sentiment_score * 0.3)
            probability_increase = max(0, min(1, probability_increase))
            
            sentiment_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'avg_sentiment_score': sentiment_score,
                'news_count_7_day': news_count,
                'social_mention_count_7_day': social_mentions,
                'probability_price_increase_next_day': probability_increase
            })
    
    return pd.DataFrame(sentiment_data)

def process_sentiment_data(symbol):
    df = generate_sentiment_data(symbol)
    
    output_path = os.path.join('..', 'data', 'processed', f'stock_{symbol}_with_sentiment.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sentiment data saved to: {output_path}")
    
    return df

def main():
    symbols = ['RELIANCE.NS']
    
    for symbol in symbols:
        print(f"Processing sentiment data for {symbol}...")
        process_sentiment_data(symbol)

if __name__ == "__main__":
    main() 