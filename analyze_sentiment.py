import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体（解决绘图乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 路径与配置区
# ==========================================
# 动态获取项目根目录 (假设脚本放在 project 文件夹下)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# 创建分析结果存放文件夹（与 data 同级）
REPORT_DIR = BASE_DIR / "analysis_reports"
REPORT_DIR.mkdir(exist_ok=True)

# 金融配色方案：红色利好（Positive），绿色利空（Negative），灰色中性（Neutral）
COLOR_MAP = {
    "利好 (Positive)": "#FF4B4B",  # 鲜艳红
    "利空 (Negative)": "#00C853",  # 森林绿
    "中性 (Neutral)": "#9E9E9E"    # 稳重灰
}

# ==========================================
# 2. 数据读取与逻辑处理
# ==========================================
def analyze_distribution():
    print(f"🔍 正在从 {DATA_DIR} 扫描 CSV 文件...")
    
    # 获取所有 CSV 文件路径
    csv_files = glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ 错误：未在 data 目录下找到任何 CSV 数据文件！")
        return

    df_list = []
    for file in csv_files:
        try:
            # 兼容处理坏行问题
            df_temp = pd.read_csv(file, on_bad_lines='skip', engine='python')
            df_list.append(df_temp)
        except Exception as e:
            print(f"⚠️ 跳过无法读取的文件 {file}: {e}")

    # 合并所有数据
    df_all = pd.concat(df_list, ignore_index=True)
    total_raw = len(df_all)
    
    # 自动识别情绪列 (优先级: llm_label > sentiment > label)
    target_col = None
    for col in ['llm_label', 'sentiment', 'label']:
        if col in df_all.columns:
            target_col = col
            break

    if not target_col:
        print(f"❌ 错误：在数据中未找到情绪标签列（llm_label/sentiment/label）。现有列: {df_all.columns.tolist()}")        return

    # 数据清洗：标准化标签名
    def standardize_label(val):
        val = str(val).strip()
        if '正面' in val or 'Positive' in val or '利好' in val:
            return "利好 (Positive)"
        if '负面' in val or 'Negative' in val or '利空' in val:
            return "利空 (Negative)"
        if '中性' in val or 'Neutral' in val:
            return "中性 (Neutral)"
        return "异常/其他"

    df_all['clean_sentiment'] = df_all[target_col].apply(standardize_label)
    
    # 过滤掉“异常/其他”和空值
    df_valid = df_all[df_all['clean_sentiment'] != "异常/其他"].copy()
    
    # ==========================================
    # 3. 统计分析
    # ==========================================
    stats = df_valid['clean_sentiment'].value_counts()
    stats_percent = df_valid['clean_sentiment'].value_counts(normalize=True) * 100
    
    print("\n" + "="*40)
    print(f"📊 金融情绪数据集统计报告")
    print(f"总计扫描样本数: {total_raw}")
    print(f"清洗后有效样本: {len(df_valid)}")
    print("-" * 40)
    for label in stats.index:
        print(f"{label:<15}: {stats[label]:>5} 条 ({stats_percent[label]:.1f}%)")
    print("="*40 + "\n")

    # ==========================================
    # 4. 可视化生成
    # ==========================================
    plt.figure(figsize=(10, 8), dpi=120)
    
    # 准备绘图数据
    plot_data = stats
    colors = [COLOR_MAP.get(label, "#333333") for label in plot_data.index]

    # 自定义百分比显示格式（显示数值 + 百分比）
    def func(pct, allvals):
        absolute = int(round(pct/100.*sum(allvals)))
        return f"{absolute:d} 条\n({pct:.1f}%)"

    wedges, texts, autotexts = plt.pie(
        plot_data, 
        labels=plot_data.index,
        autopct=lambda pct: func(pct, plot_data),
        startangle=140,
        colors=colors,
        explode=[0.05 if i == 0 else 0 for i in range(len(plot_data))], # 第一个切片突出显示
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    # 美化标签颜色
    plt.setp(autotexts, size=11, color="white")
    
    plt.title(f"金融情感数据集分布 - 共 {len(df_valid)} 条样本", fontsize=15, pad=20)
    plt.legend(wedges, plot_data.index, title="情绪分类", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 保存饼图
    save_path = REPORT_DIR / "sentiment_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 饼图已保存至: {save_path}")

    # 保存文字版统计报告
    report_file = REPORT_DIR / "analysis_summary.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Financial Sentiment Analysis Report\n")
        f.write("="*35 + "\n")
        f.write(f"Generated at: {pd.Timestamp.now()}\n")
        f.write(f"Total CSVs found: {len(csv_files)}\n")
        f.write(f"Total valid samples: {len(df_valid)}\n")
        f.write("-" * 35 + "\n")
        f.write(stats.to_string())
    print(f"✅ 统计报告已保存至: {report_file}")

if __name__ == "__main__":
    analyze_distribution()
