import os
import glob
import pandas as pd
import csv
import sys
from pathlib import Path

# --- 导入项目统一配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, FULL_DATASET_PATH

def merge_all_csv():
    print(f"🚀 正在启动全量数据整理流水线...")
    csv_files = glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ 错误：未在 data 目录下找到任何 CSV 数据文件！")
        return

    print(f"📂 发现 {len(csv_files)} 个数据分片，正在执行合并...")
    df_list = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(file, on_bad_lines='skip', engine='python')
            df_list.append(df_temp)
        except Exception as e:
            print(f"⚠️ 跳过解析失败的文件 {file}: {e}")

    if not df_list:
        print("❌ 错误：没有成功读取到任何有效数据。")
        return

    df_full = pd.concat(df_list, ignore_index=True)
    original_count = len(df_full)
    if 'sentence' in df_full.columns:
        df_full = df_full.drop_duplicates(subset=['sentence'])
    
    df_full.to_csv(FULL_DATASET_PATH, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"✅ 数据整理完成！")
    print(f"原始记录总数: {original_count}")
    print(f"去重后有效总数: {len(df_full)}")
    print(f"最终输出路径: {FULL_DATASET_PATH}")
    print("="*40)

if __name__ == "__main__":
    merge_all_csv()
