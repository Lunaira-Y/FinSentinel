import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import pearsonr
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# ==========================================
# 0. 环境与路径配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径计算
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(BASE_DIR, "analysis_reports", "full_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "finbert_local")

# --- 🔥 核心修改：将产出物打包放入 analysis_reports 下的新文件夹 ---
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_reports", "sentiment_price_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ------------------------------------------------------------

# 情感分值映射
SCORE_MAP = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# ==========================================
# 1. 辅助函数
# ==========================================
def format_stock_code(code):
    """中国 A 股代码补全后缀"""
    code = str(code).strip().zfill(6)
    if code.startswith('6'):
        return f"{code}.SS"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    return code

# ==========================================
# 2. 核心逻辑实现
# ==========================================
def main():
    print("🚀 启动金融情绪-股价相关性分析流水线...")

    # --- A. 数据载入与 FinBERT 推理 ---
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 找不到数据集: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"📂 已载入数据: {len(df)} 条")

    # 加载模型
    print(f"🤖 正在从 {MODEL_PATH} 加载 FinBERT 模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        device = 0 if torch.cuda.is_available() else -1
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        
        # 标签映射 (根据 finbert-tone 常用映射)
        FINBERT_LABELS = {
            'LABEL_0': 'Neutral', 'LABEL_1': 'Positive', 'LABEL_2': 'Negative',
            'NEUTRAL': 'Neutral', 'POSITIVE': 'Positive', 'NEGATIVE': 'Negative'
        }
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 推理过程
    print("🧠 正在进行全量数据情感推理...")
    sentiments = []
    for text in tqdm(df['sentence'].astype(str).tolist(), desc="Sentiment Inference"):
        try:
            res = nlp(text[:512])[0]
            label = FINBERT_LABELS.get(res['label'].upper(), 'Neutral')
            sentiments.append(label)
        except:
            sentiments.append('Neutral')
    
    df['sentiment'] = sentiments
    df['sentiment_score'] = df['sentiment'].map(SCORE_MAP)

    # --- B. 情绪聚合 ---
    print("📊 正在按日期和代码聚合情绪...")
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    daily_sentiment = df.groupby(['stock_code', 'date'])['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['stock_code', 'date', 'mean_sentiment']

    # --- C. 股价匹配 ---
    print("📈 正在抓取股价数据 (yfinance)...")
    unique_stocks = daily_sentiment['stock_code'].unique()
    all_price_data = []

    for sc in unique_stocks:
        ticker_symbol = format_stock_code(sc)
        print(f"  🔍 抓取 {ticker_symbol}...")
        try:
            # 抓取比数据集范围多出几天的股价，以便计算次日收益率
            start_date = daily_sentiment[daily_sentiment['stock_code']==sc]['date'].min()
            end_date = daily_sentiment[daily_sentiment['stock_code']==sc]['date'].max()
            
            # 扩展日期范围
            fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            fetch_end = (pd.to_datetime(end_date) + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
            
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(start=fetch_start, end=fetch_end)
            
            if hist.empty:
                continue
                
            # 计算次日收益率 T+1
            hist['Next_Day_Return'] = hist['Close'].pct_change().shift(-1)
            hist = hist.reset_index()
            hist['date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            hist['stock_code'] = sc
            
            all_price_data.append(hist[['stock_code', 'date', 'Close', 'Next_Day_Return']])
        except Exception as e:
            print(f"  ⚠️ {ticker_symbol} 抓取失败: {e}")

    if not all_price_data:
        print("❌ 未能获取到任何股价数据，请检查网络。")
        return

    price_df = pd.concat(all_price_data)
    
    # 合并情绪与股价
    final_df = pd.merge(daily_sentiment, price_df, on=['stock_code', 'date'], how='inner')
    final_df = final_df.dropna(subset=['Next_Day_Return'])

    # --- D. 相关性分析 ---
    if len(final_df) > 1:
        corr, p_value = pearsonr(final_df['mean_sentiment'], final_df['Next_Day_Return'])
        print(f"\n✨ 相关性分析结果 (Pearson):")
        print(f"   系数: {corr:.4f}")
        print(f"   P值: {p_value:.4f} ({'显著' if p_value < 0.05 else '不显著'})")
    else:
        print("⚠️ 样本量不足，无法进行相关性分析。")

    # --- E. 存储与可视化 ---
    csv_out = os.path.join(OUTPUT_DIR, "final_signals_with_price.csv")
    final_df.to_csv(csv_out, index=False)
    print(f"💾 结果已保存至: {csv_out}")

    # 绘图 1: 股票情绪得分排名柱状图 (Factor Exposure Ranking)
    # 按股票代码聚合计算整体平均情绪
    rank_df = final_df.groupby('stock_code')['mean_sentiment'].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(14, 7))
    # 自定义调色板：正数红，负数绿，零灰
    colors = ['#FF4B4B' if x > 0 else '#00C853' if x < 0 else '#9E9E9E' for x in rank_df['mean_sentiment']]
    
    # 修复 FutureWarning: 指定 hue 并关闭 legend
    sns.barplot(data=rank_df, x='stock_code', y='mean_sentiment', hue='stock_code', palette=colors, legend=False)
    
    plt.title("各股票情绪得分排名 (因子暴露度排序)", fontsize=15, pad=20)
    plt.xlabel("股票代码", fontsize=12)
    plt.ylabel("平均情绪得分 (-1 到 1)", fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linewidth=0.8)  # 增加 0 轴基准线
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 在柱状图上方添加数值标注
    for i, v in enumerate(rank_df['mean_sentiment']):
        plt.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:.2f}", ha='center', va='bottom' if v>=0 else 'top', fontsize=10, fontweight='bold')

    plot1_path = os.path.join(OUTPUT_DIR, "stock_sentiment_ranking.png")
    plt.tight_layout()
    plt.savefig(plot1_path)
    print(f"🖼️ 图表1 (情绪排名柱状图) 已保存: {plot1_path}")

    # 绘图 2: 相关性散点图
    plt.figure(figsize=(8, 6))
    sns.regplot(data=final_df, x='mean_sentiment', y='Next_Day_Return', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f"情绪分值 vs 次日收益率 相关性 (Corr: {corr:.4f})")
    plt.xlabel('平均情绪得分 (-1 到 1)')
    plt.ylabel('次日收益率 (%)')
    
    plot2_path = os.path.join(OUTPUT_DIR, "correlation_scatter.png")
    plt.savefig(plot2_path)
    print(f"🖼️ 图表2已保存: {plot2_path}")

    print("\n✅ 全流程运行结束。")

if __name__ == "__main__":
    main()
