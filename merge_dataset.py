import os
import glob
import pandas as pd
import csv
from pathlib import Path

# ==========================================
# 1. 路径配置
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "analysis_reports"
REPORT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "full_dataset.csv"

# ==========================================
# 2. 合并执行逻辑
# ==========================================
def merge_all_csv():
    print(f"🚀 正在启动全量数据整理流水线...")
    
    # 获取所有 CSV 文件路径
    csv_files = glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ 错误：未在 data 目录下找到任何 CSV 数据文件！")
        return

    print(f"📂 发现 {len(csv_files)} 个数据分片，正在执行合并...")

    df_list = []
    success_count = 0
    
    for file in csv_files:
        try:
            # 使用鲁棒性读取参数：跳过坏行，使用 python 引擎
            df_temp = pd.read_csv(file, on_bad_lines='skip', engine='python')
            df_list.append(df_temp)
            success_count += 1
        except Exception as e:
            print(f"⚠️ 跳过解析失败的文件 {file}: {e}")

    if not df_list:
        print("❌ 错误：没有成功读取到任何有效数据。")
        return

    # 执行垂直合并
    df_full = pd.concat(df_list, ignore_index=True)
    
    # 简单的“整理”逻辑：去重（基于 sentence 内容，防止多日重复采集带来的干扰）
    original_count = len(df_full)
    if 'sentence' in df_full.columns:
        df_full = df_full.drop_duplicates(subset=['sentence'])
    
    # 强制安全写入：使用 QUOTE_MINIMAL 避免逗号冲突
    df_full.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"✅ 数据整理完成！")
    print(f"原始记录总数: {original_count}")
    print(f"去重后有效总数: {len(df_full)}")
    print(f"最终输出路径: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    merge_all_csv()
