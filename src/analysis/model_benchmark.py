import os
import time
import torch
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, pipeline

# --- 导入项目统一配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import FULL_DATASET_PATH, REPORT_DIR, FINBERT_PATH

def prepare_dataset():
    if not os.path.exists(FULL_DATASET_PATH):
        raise FileNotFoundError(f"未找到数据集")
    df = pd.read_csv(FULL_DATASET_PATH)
    df = df.dropna(subset=['sentence', 'llm_label'])
    df = df[df['llm_label'].isin(['Positive', 'Negative', 'Neutral'])]
    df['label'] = df['llm_label']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, test_df

def run_method_a(train_df, test_df):
    start_time = time.time()
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_test = vectorizer.transform(test_df['sentence'])
    model = LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
    model.fit(X_train, train_df['label'])
    latency = (time.time() - start_time) / len(test_df) * 1000
    preds = model.predict(X_test)
    return preds, latency

def run_method_b(test_df):
    tokenizer = BertTokenizer.from_pretrained(FINBERT_PATH)
    model = BertForSequenceClassification.from_pretrained(FINBERT_PATH)
    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    
    sentences = test_df['sentence'].tolist()
    preds = []
    start_time = time.time()
    for text in tqdm(sentences, desc="FinBERT Inference"):
        try:
            res = nlp(str(text), truncation=True, max_length=512)[0]
            label = res['label'].upper()
            std_label = "Neutral"
            if "POS" in label: std_label = "Positive"
            elif "NEG" in label: std_label = "Negative"
            preds.append(std_label)
        except:
            preds.append("Neutral")
    latency = (time.time() - start_time) / len(test_df) * 1000
    return preds, latency

def main():
    train_df, test_df = prepare_dataset()
    y_true = test_df['label'].tolist()
    
    preds_lr, lat_lr = run_method_a(train_df, test_df)
    preds_bert, lat_bert = run_method_b(test_df)
    
    summary_data = [
        {"Model": "TF-IDF+LR", "Accuracy": accuracy_score(y_true, preds_lr), "Macro_F1": f1_score(y_true, preds_lr, average='macro'), "Latency_ms": lat_lr},
        {"Model": "HKUST-FinBERT", "Accuracy": accuracy_score(y_true, preds_bert), "Macro_F1": f1_score(y_true, preds_bert, average='macro'), "Latency_ms": lat_bert}
    ]
    
    df_all = pd.DataFrame(summary_data)
    df_all.to_csv(REPORT_DIR / "benchmark_all_results.csv", index=False)
    print(df_all)

if __name__ == "__main__":
    main()
