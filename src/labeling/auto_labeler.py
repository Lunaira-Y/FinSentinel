import pandas as pd
import time
import json
import os
import glob
import sys
import csv
import random
from collections import deque
from datetime import datetime
from openai import OpenAI

# Rich 核心组件
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich import box

# --- 导入项目统一配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, DEEPSEEK_API_KEY

# 系统运行参数
SAVE_INTERVAL = 10 
MAX_CONSECUTIVE_ERRORS = 3 
MAX_RETRIES = 5 

# ==========================================
# 2. 核心逻辑
# ==========================================
system_prompt = """
你是一位专业的量化金融情感分析师。请判断以下中国A股财经新闻句子的情绪倾向。
规则：
1. 只能输出"正面"、"负面"或"中性"三个词中的一个。
2. 必须以严格的 JSON 格式返回，键名为 "sentiment"。
示例：{"sentiment": "正面"}
"""

def get_sentiment(client, text, log_callback=None):
    for n in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"新闻：{text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=30.0
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("sentiment", "解析失败"), None
        except Exception as e:
            error_msg = str(e)
            if n < MAX_RETRIES - 1:
                wait_time = (2 ** n) + random.uniform(0, 1)
                if log_callback:
                    log_callback(f"[bold yellow][Retry][/bold yellow] 第 {n+1} 次失败，等待后重试...")
                time.sleep(wait_time)
            else:
                return f"Error", error_msg

# ==========================================
# 3. UI 构造
# ==========================================
def make_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=12)
    )
    layout["body"].split_row(
        Layout(name="status_pane", ratio=1),
        Layout(name="progress_pane", ratio=2)
    )
    return layout

def generate_log_table(recent_logs):
    table = Table(expand=True, box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("时间", style="dim", width=10)
    table.add_column("源文件", style="blue", width=15)
    table.add_column("文本预览", ratio=1)
    table.add_column("结果", width=25, justify="center")

    for log in recent_logs:
        res = log['sentiment']
        color = "white"
        if "正面" in res: color = "green"
        elif "负面" in res: color = "red"
        elif "中性" in res: color = "yellow"
        table.add_row(log['time'], log['filename'], log['sentence'][:30], f"[{color}]{res}[/{color}]")
    return table

def safe_save(df, file_path):
    temp_path = str(file_path) + ".tmp"
    try:
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_MINIMAL)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)
    except: pass

def main():
    console = Console()
    layout = make_layout()
    recent_logs = deque(maxlen=6)
    consecutive_errors = 0
    
    def add_ui_log(filename, sentence, sentiment):
        recent_logs.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "filename": filename,
            "sentence": sentence,
            "sentiment": sentiment
        })
        layout["footer"].update(Panel(generate_log_table(recent_logs), title="实时处理数据流"))

    progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console, expand=True)
    
    with Live(layout, refresh_per_second=4, console=console):
        layout["header"].update(Panel("[bold cyan]FinSentinel Auto-Labeler V4.2[/bold cyan]", border_style="cyan"))

        if "your_deepseek" in DEEPSEEK_API_KEY:
            layout["status_pane"].update(Panel("[bold red]❌ 未配置 API KEY[/bold red]", title="配置错误"))
            time.sleep(3)
            return

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        all_csv_files = glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True)
        all_tasks = []

        for file_path in all_csv_files:
            try:
                df = pd.read_csv(file_path, engine='python')
                if 'llm_label' not in df.columns: df['llm_label'] = ""
                mask = df['llm_label'].isna() | (df['llm_label'].astype(str).str.strip() == "")
                todo_indices = df[mask].index.tolist()
                if todo_indices: all_tasks.append({'path': file_path, 'df': df, 'indices': todo_indices})
            except: continue

        total_todo = sum(len(t['indices']) for t in all_tasks)
        if total_todo == 0:
            layout["status_pane"].update(Panel("[bold green]✅ 所有文件已标注完成", title="系统状态"))
            time.sleep(2)
            return

        layout["status_pane"].update(Panel(f"待标注总量: [bold magenta]{total_todo}[/bold magenta]", title="任务规模"))
        task_id = progress.add_task("AI 标注进度", total=total_todo)
        layout["progress_pane"].update(Panel(progress, title="总进度"))

        try:
            for task in all_tasks:
                current_path = task['path']
                current_df = task['df']
                filename = os.path.basename(current_path)
                
                for idx in task['indices']:
                    text = str(current_df.loc[idx, 'sentence'])
                    label, err = get_sentiment(client, text, log_callback=lambda m: add_ui_log(filename, text[:20], m))

                    if err: consecutive_errors += 1
                    else: consecutive_errors = 0 

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        sys.exit(1)

                    current_df.at[idx, 'llm_label'] = label
                    add_ui_log(filename, text, label)
                    progress.update(task_id, advance=1, description=f"处理: {filename}")
                    time.sleep(0.3)
                
                safe_save(current_df, current_path)
        except KeyboardInterrupt:
            if 'current_df' in locals(): safe_save(current_df, current_path)
            sys.exit(0)

if __name__ == "__main__":
    main()
