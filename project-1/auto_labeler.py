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

# ==========================================
# 1. 基础配置区
# ==========================================
# 💡 安全建议：建议通过环境变量获取 API_KEY，或在此处填入占位符后再上传
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")

# 动态计算路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))

# 系统运行参数
SAVE_INTERVAL = 10 
MAX_CONSECUTIVE_ERRORS = 3 
MAX_RETRIES = 5  # API 最大重试次数

# ==========================================
# 2. 核心逻辑与指数退避重试
# ==========================================
system_prompt = """
你是一位专业的量化金融情感分析师。请判断以下中国A股财经新闻句子的情绪倾向。
规则：
1. 只能输出"正面"、"负面"或"中性"三个词中的一个。
2. 必须以严格的 JSON 格式返回，键名为 "sentiment"。
示例：{"sentiment": "正面"}
"""

def get_sentiment(client, text, log_callback=None):
    """支持指数退避重试的情感分析函数"""
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
                timeout=30.0  # 显式超时设置
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("sentiment", "解析失败"), None
        except Exception as e:
            error_msg = str(e)
            # 如果不是最后一次重试，则执行退避等待
            if n < MAX_RETRIES - 1:
                wait_time = (2 ** n) + random.uniform(0, 1)
                if log_callback:
                    log_callback(f"[bold yellow][Retry][/bold yellow] 第 {n+1} 次尝试失败，{wait_time:.1f}s 后重试...")
                time.sleep(wait_time)
            else:
                return f"Error: {error_msg[:50]}", error_msg

# ==========================================
# 3. UI 构造器
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
    table.add_column("结果/故障", width=25, justify="center")

    for log in recent_logs:
        res = log['sentiment']
        color = "white"
        if "正面" in res: color = "green"
        elif "负面" in res: color = "red"
        elif "中性" in res: color = "yellow"
        elif "Error" in res: color = "bold red"
        elif "Retry" in res: color = "bold yellow"

        table.add_row(
            log['time'],
            log['filename'],
            (log['sentence'][:30] + "...") if len(log['sentence']) > 30 else log['sentence'],
            f"[{color}]{res}[/{color}]"
        )
    return table

def safe_save(df, file_path, console=None):
    temp_path = file_path + ".tmp"
    try:
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_MINIMAL)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)
    except Exception as e:
        if console:
            console.print(f"[bold red]保存失败 {file_path}: {e}[/bold red]")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    console = Console()
    layout = make_layout()
    recent_logs = deque(maxlen=6)
    consecutive_errors = 0
    
    # 辅助函数：更新 UI 日志流
    def add_ui_log(filename, sentence, sentiment):
        recent_logs.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "filename": filename,
            "sentence": sentence,
            "sentiment": sentiment
        })
        layout["footer"].update(Panel(generate_log_table(recent_logs), title="实时处理数据流 (Top 6)", border_style="dim"))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(style="bright_black", complete_style="orange1"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True
    )
    
    with Live(layout, refresh_per_second=4, console=console, screen=False) as live:
        layout["header"].update(Panel(
            "[bold cyan]FinSentinel Auto-Labeler V4.2[/bold cyan] | 稳定性与容错增强版", 
            border_style="cyan"
        ))

        if "填在这里" in API_KEY:
            layout["status_pane"].update(Panel("[bold red]❌ 未配置 API KEY[/bold red]\n请先修改脚本中的 API_KEY 变量", title="配置错误"))
            time.sleep(3)
            return

        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

        layout["status_pane"].update(Panel("[yellow]正在扫描数据源...", title="系统状态"))
        search_pattern = os.path.join(DATA_DIR, '**', '*.csv')
        all_csv_files = glob.glob(search_pattern, recursive=True)
        all_tasks = []

        for file_path in all_csv_files:
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
                if 'llm_label' not in df.columns:
                    df['llm_label'] = ""
                # 初始扫描：筛选未标注行
                mask = df['llm_label'].isna() | (df['llm_label'].astype(str).str.strip() == "")
                todo_indices = df[mask].index.tolist()
                if todo_indices:
                    all_tasks.append({'path': file_path, 'df': df, 'indices': todo_indices})
            except:
                continue

        total_todo = sum(len(t['indices']) for t in all_tasks)
        if total_todo == 0:
            layout["status_pane"].update(Panel("[bold green]✅ 所有文件已完成标注", title="系统状态"))
            time.sleep(2)
            return

        status_info = f"文件总数: {len(all_csv_files)}\n待处理文件: {len(all_tasks)}\n待标注总量: [bold magenta]{total_todo}[/bold magenta]"
        layout["status_pane"].update(Panel(status_info, title="任务规模", border_style="magenta"))

        task_id = progress.add_task("AI 标注进度", total=total_todo)
        layout["progress_pane"].update(Panel(progress, title="总进度", border_style="blue"))

        try:
            for task in all_tasks:
                current_path = task['path']
                current_df = task['df']
                indices = task['indices']
                filename = os.path.basename(current_path)
                
                save_counter = 0
                for idx in indices:
                    row = current_df.loc[idx]
                    text = str(row['sentence'])
                    
                    # 3. 强化“断点续传”逻辑：二次校验是否已标注
                    current_val = str(current_df.at[idx, 'llm_label']).strip()
                    if current_val != "" and current_val != "nan":
                        progress.update(task_id, advance=1)
                        continue

                    # 定义重试时的 UI 回调
                    def retry_cb(msg):
                        add_ui_log(filename, text[:20], msg)

                    # 1 & 2. 执行带重试与超时的 API 调用
                    label, err = get_sentiment(client, text, log_callback=retry_cb)

                    # 4. 熔断判定升级
                    if err:
                        consecutive_errors += 1
                        last_error_detail = err
                    else:
                        consecutive_errors = 0 

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        layout["status_pane"].update(Panel(
                            f"[bold red]❌ 熔断触发！[/bold red]\n连续错误: {consecutive_errors}\n最后报错: {last_error_detail[:100]}", 
                            title="紧急停机", border_style="red"
                        ))
                        safe_save(current_df, current_path)
                        time.sleep(3)
                        sys.exit(1)

                    # 更新结果并反馈 UI
                    current_df.at[idx, 'llm_label'] = label
                    add_ui_log(filename, text, label)
                    
                    progress.update(task_id, advance=1, description=f"处理: {filename}")
                    
                    # 定期保存
                    save_counter += 1
                    if save_counter >= SAVE_INTERVAL:
                        safe_save(current_df, current_path, console=console)
                        save_counter = 0
                    
                    time.sleep(0.3)
                
                # 文件结束强制保存
                safe_save(current_df, current_path, console=console)

        except KeyboardInterrupt:
            # 5. 存储安全：捕获中断并强制保存当前文件进度
            layout["status_pane"].update(Panel("[bold yellow]检测到用户中断，正在安全保存进度并退出...", title="紧急停止"))
            # 寻找当前正在处理的文件并保存
            if 'current_df' in locals() and 'current_path' in locals():
                safe_save(current_df, current_path, console=console)
            time.sleep(1.5)
            sys.exit(0)

    console.print("\n[bold green]🎉 任务已安全完成。[/bold green]")

if __name__ == "__main__":
    main()
