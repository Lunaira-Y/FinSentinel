import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, pipeline

# ==========================================
# 1. 路径与配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATASET_PATH = os.path.join(CURRENT_DIR, "analysis_reports", "full_dataset.csv")
REPORT_DIR = os.path.join(CURRENT_DIR, "analysis_reports")
MODEL_PATH = os.path.join(CURRENT_DIR, "finbert_local")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据准备
# ==========================================
def prepare_dataset():
    if not os.path.exists(FULL_DATASET_PATH):
        raise FileNotFoundError(f"未找到数据集")
    df = pd.read_csv(FULL_DATASET_PATH)
    df = df.dropna(subset=['sentence', 'llm_label'])
    df = df[df['llm_label'].isin(['Positive', 'Negative', 'Neutral'])]
    df['label'] = df['llm_label']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, test_df

# ==========================================
# 3. 推理逻辑 (保持原有高性能配置)
# ==========================================
def run_method_a(train_df, test_df):
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_test = vectorizer.transform(test_df['sentence'])
    model = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced', C=0.5)
    model.fit(X_train, train_df['label'])
    latency = (time.time() - start_time) / len(test_df) * 1000
    preds = model.predict(X_test)
    return preds, latency

def run_method_b(test_df):
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        config = BertConfig.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
        device_id = 0 if torch.cuda.is_available() else -1
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device_id)
        
        FINAL_MAP = {}
        if hasattr(config, 'id2label'):
            for idx, label in config.id2label.items():
                std_label = "Neutral"
                l_lower = label.lower()
                if "pos" in l_lower: std_label = "Positive"
                elif "neg" in l_lower: std_label = "Negative"
                FINAL_MAP[label.upper()] = std_label
                FINAL_MAP[f"LABEL_{idx}"] = std_label
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return ["Error"] * len(test_df), 0.0

    sentences = test_df['sentence'].tolist()
    preds = []
    start_time = time.time()
    for text in tqdm(sentences, desc="FinBERT 推理"):
        try:
            res = nlp(str(text), truncation=True, max_length=512, padding=True)[0]
            preds.append(FINAL_MAP.get(res['label'].upper(), FINAL_MAP.get(f"LABEL_{res['label']}", "Neutral")))
        except:
            preds.append("Neutral")
    latency = (time.time() - start_time) / len(test_df) * 1000
    return preds, latency

# ==========================================
# 4. 主流程 (只产出单一 CSV)
# ==========================================
def main():
    print(f"🚀 启动 Benchmarking (运行设备: {device})...")
    train_df, test_df = prepare_dataset()
    y_true = test_df['label'].tolist()
    
    # 执行评估
    preds_lr, lat_lr = run_method_a(train_df, test_df)
    preds_bert, lat_bert = run_method_b(test_df)
    
    # --- 构建结构化数据 ---
    # 1. 汇总指标表
    summary_data = [
        {"Model": "TF-IDF+LR", "Accuracy": accuracy_score(y_true, preds_lr), "Macro_F1": f1_score(y_true, preds_lr, average='macro'), "Latency_ms": lat_lr},
        {"Model": "HKUST-FinBERT", "Accuracy": accuracy_score(y_true, preds_bert), "Macro_F1": f1_score(y_true, preds_bert, average='macro'), "Latency_ms": lat_bert}
    ]
    df_summary = pd.DataFrame(summary_data)

    # 2. 详细分类报告 (转换为长表格式)
    def get_report_df(y_true, y_pred, model_name):
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        rows = []
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                row = {"Model": model_name, "Class": label}
                row.update(metrics)
                rows.append(row)
        return pd.DataFrame(rows)

    df_report_lr = get_report_df(y_true, preds_lr, "TF-IDF+LR")
    df_report_bert = get_report_df(y_true, preds_bert, "HKUST-FinBERT")

    # --- 合并全量结果并保存 ---
    final_output_path = os.path.join(REPORT_DIR, "benchmark_all_results.csv")
    
    # 使用 pd.ExcelWriter 的逻辑思路，但在 CSV 中我们通过分段写入或拼接
    # 这里采用拼接方式，增加一个 'Type' 列用于区分
    df_summary['Type'] = "OVERALL_SUMMARY"
    df_report_lr['Type'] = "DETAILED_METRICS"
    df_report_bert['Type'] = "DETAILED_METRICS"
    
    df_all = pd.concat([df_summary, df_report_lr, df_report_bert], axis=0, ignore_index=True)
    
    # 保存单一 CSV
    df_all.to_csv(final_output_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ 任务完成！已将汇总指标及详细报告合并存入单一 CSV：")
    print(f"📍 {final_output_path}")
    print("="*50)
    
    # 终端依然显示关键汇总，方便快速查看
    print("\n快速预览:")
    print(df_summary[["Model", "Accuracy", "Macro_F1", "Latency_ms"]])

if __name__ == "__main__":
    main()
