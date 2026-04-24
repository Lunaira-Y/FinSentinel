import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import sys
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# --- 导入项目统一配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import FULL_DATASET_PATH, FINBERT_PATH, SIGNAL_OUTPUT_DIR

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCORE_MAP = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

def format_stock_code(code):
    code = str(code).strip().zfill(6)
    if code.startswith('6'): return f"{code}.SS"
    elif code.startswith(('0', '3')): return f"{code}.SZ"
    return code

def main():
    if not os.path.exists(FULL_DATASET_PATH): return
    df = pd.read_csv(FULL_DATASET_PATH)

    tokenizer = BertTokenizer.from_pretrained(FINBERT_PATH)
    model = BertForSequenceClassification.from_pretrained(FINBERT_PATH)
    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    
    FINBERT_LABELS = {'LABEL_0': 'Neutral', 'LABEL_1': 'Positive', 'LABEL_2': 'Negative'}
    
    sentiments = []
    for text in tqdm(df['sentence'].astype(str).tolist(), desc="Sentiment Inference"):
        try:
            res = nlp(text[:512])[0]
            sentiments.append(FINBERT_LABELS.get(res['label'].upper(), 'Neutral'))
        except: sentiments.append('Neutral')
    
    df['sentiment'] = sentiments
    df['sentiment_score'] = df['sentiment'].map(SCORE_MAP)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    daily_sentiment = df.groupby(['stock_code', 'date'])['sentiment_score'].mean().reset_index()

    unique_stocks = daily_sentiment['stock_code'].unique()
    all_price_data = []
    for sc in unique_stocks:
        ticker = format_stock_code(sc)
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if hist.empty: continue
            hist['Next_Day_Return'] = hist['Close'].pct_change().shift(-1)
            hist = hist.reset_index()
            hist['date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            hist['stock_code'] = sc
            all_price_data.append(hist[['stock_code', 'date', 'Close', 'Next_Day_Return']])
        except: pass

    if not all_price_data: return
    price_df = pd.concat(all_price_data)
    final_df = pd.merge(daily_sentiment, price_df, on=['stock_code', 'date'], how='inner').dropna()

    final_df.to_csv(SIGNAL_OUTPUT_DIR / "final_signals_with_price.csv", index=False)

    # Plot Ranking
    rank_df = final_df.groupby('stock_code')['sentiment_score'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rank_df, x='stock_code', y='sentiment_score', hue='stock_code', palette='RdYlGn_r', legend=False)
    plt.savefig(SIGNAL_OUTPUT_DIR / "stock_sentiment_ranking.png")

    # Plot Scatter
    plt.figure(figsize=(8, 6))
    sns.regplot(data=final_df, x='sentiment_score', y='Next_Day_Return')
    plt.savefig(SIGNAL_OUTPUT_DIR / "correlation_scatter.png")

if __name__ == "__main__":
    main()
