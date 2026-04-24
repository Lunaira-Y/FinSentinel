import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 导入项目统一配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, REPORT_DIR

COLOR_MAP = {
    "利好 (Positive)": "#FF4B4B",
    "利空 (Negative)": "#00C853",
    "中性 (Neutral)": "#9E9E9E"
}

def analyze_distribution():
    csv_files = glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True)
    if not csv_files: return

    df_list = [pd.read_csv(f, on_bad_lines='skip', engine='python') for f in csv_files]
    df_all = pd.concat(df_list, ignore_index=True)
    
    target_col = next((c for col in ['llm_label', 'sentiment', 'label'] if (c := col) in df_all.columns), None)
    if not target_col: return

    def standardize_label(val):
        val = str(val).strip()
        if any(x in val for x in ['正面', 'Positive', '利好']): return "利好 (Positive)"
        if any(x in val for x in ['负面', 'Negative', '利空']): return "利空 (Negative)"
        if any(x in val for x in ['中性', 'Neutral']): return "中性 (Neutral)"
        return "其他"

    df_all['clean_sentiment'] = df_all[target_col].apply(standardize_label)
    df_valid = df_all[df_all['clean_sentiment'] != "其他"].copy()
    
    stats = df_valid['clean_sentiment'].value_counts()
    
    plt.figure(figsize=(10, 8), dpi=120)
    plt.pie(stats, labels=stats.index, autopct='%1.1f%%', startangle=140, colors=[COLOR_MAP.get(l) for l in stats.index])
    plt.title(f"金融情感数据集分布 - 共 {len(df_valid)} 条")
    
    save_path = REPORT_DIR / "sentiment_distribution.png"
    plt.savefig(save_path)
    
    with open(REPORT_DIR / "analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Generated at: {pd.Timestamp.now()}\nTotal Samples: {len(df_valid)}\n{stats.to_string()}")

if __name__ == "__main__":
    analyze_distribution()
