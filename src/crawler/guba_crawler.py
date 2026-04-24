import argparse
import datetime
import logging
import os
import random
import time
import re
import sys
from typing import List

import pandas as pd
import requests
import schedule
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Rich 视觉增强组件
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

# --- 导入项目统一配置 ---
# 这里的 sys.path 处理是为了让脚本在直接运行时也能找到 config 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATA_DIR, LOG_DIR

UA_LIBRARY = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
]

BASE_URL = "https://guba.eastmoney.com/list,{code},{page},f.html"
# 默认股票列表
DEFAULT_STOCK_LIST = ["000001", "600519", "300750", "000002", "002230"]
TOP_SAMPLE_LIMIT = 10

# --- 核心引擎 (Engine) ---
class StockNewsCrawler:
    def __init__(self, logger):
        self.session = self._init_session()
        self.logger = logger

    def _init_session(self):
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_detail(self, url: str) -> str:
        try:
            time.sleep(random.uniform(3.0, 5.0))
            headers = {
                "User-Agent": random.choice(UA_LIBRARY),
                "Referer": "https://guba.eastmoney.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            if len(response.text) < 500:
                self.logger.warning(f"⚠️ 收到异常短响应: {url}")
            
            if response.status_code != 200:
                return f"ERROR_HTTP_{response.status_code}"
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for s in soup(["script", "style"]): s.decompose()

            selectors = ['.zwconbody', '.article-body', '#zw_body', '.mainbody', '.article-content', '#ContentBody']
            content = ""
            for selector in selectors:
                node = soup.select_one(selector)
                if node:
                    content = node.get_text().strip()
                    break
            
            if not content:
                p_tags = soup.find_all('p')
                content = "".join([p.get_text().strip() for p in p_tags if p.get_text().strip()])

            return content.replace('\n', '').replace('\r', '').replace('\t', ' ').strip() if content else "ERROR_PARSE_EMPTY"
        except Exception as e:
            self.logger.error(f"详情页异常: {url} -> {e}")
            return f"ERROR_EXCEPTION"

    def fetch_page(self, stock_code: str) -> pd.DataFrame:
        url = BASE_URL.format(code=stock_code, page=1)
        try:
            time.sleep(random.uniform(1.0, 2.0))
            self.logger.info(f"正在扫描 {stock_code} 资讯列表页...")
            headers = {"User-Agent": random.choice(UA_LIBRARY), "Referer": "https://guba.eastmoney.com/"}
            response = self.session.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            raw_links = []
            for a in soup.find_all('a'):
                title = a.get_text().strip()
                href = a.get('href', '')
                if len(title) > 6 and ("news," in href or "article/" in href):
                    full_url = href if href.startswith('http') else "https://guba.eastmoney.com" + (href if href.startswith('/') else '/' + href)
                    raw_links.append({
                        "stock_code": stock_code,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "title": title,
                        "url": full_url
                    })
                if len(raw_links) >= TOP_SAMPLE_LIMIT: break
            
            self.logger.info(f"🔎 [列表页] 成功提取到 {len(raw_links)} 条潜在资讯链接")
            
            processed_data = []
            for i, item in enumerate(raw_links):
                self.logger.info(f"  [资讯采集 {i+1}/{TOP_SAMPLE_LIMIT}] 正在抓取: {item['title'][:12]}...")
                item['content'] = self.fetch_detail(item['url'])
                processed_data.append(item)
            
            return pd.DataFrame(processed_data)
        except Exception as e:
            self.logger.error(f"列表页解析失败: {e}")
            return pd.DataFrame()

# --- NLP 处理引擎 (NLP Pipeline) ---
class NLPPipeline:
    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df

        def basic_clean(text: str) -> str:
            return text.replace(',', '，').replace('"', '”')

        df['content'] = df['content'].apply(basic_clean)
        df = df[~df['content'].str.contains("ERROR_PARSE_EMPTY", na=False)]

        def split_to_sentences(text: str) -> List[str]:
            sentences = re.split(r'[。！？!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) >= 10]

        df['sentence'] = df['content'].apply(split_to_sentences)
        df = df.explode('sentence')
        df = df.dropna(subset=['sentence'])
        df = df[df['sentence'].str.len() >= 10]
        
        return df[['stock_code', 'date', 'title', 'sentence', 'url']]

# --- 系统控制器 ---
class FinSentinel:
    def __init__(self, stock_codes: List[str]):
        self.stock_codes = stock_codes
        self.log_buffer = []
        self.logger = self._setup_logging()
        self.crawler = StockNewsCrawler(self.logger)
        self.stats = {code: {"articles": 0, "sentences": 0} for code in stock_codes}
        self.system_status = "系统就绪"
        self.total_runs = 0
        self.start_time = datetime.datetime.now()

    def _setup_logging(self):
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        log_file = os.path.join(LOG_DIR, f"crawler_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        logger = logging.getLogger("FinSentinel")
        logger.setLevel(logging.INFO)
        
        class BufferHandler(logging.Handler):
            def __init__(self, buffer):
                super().__init__()
                self.buffer = buffer
            def emit(self, record):
                self.buffer.append(self.format(record))
                if len(self.buffer) > 15: self.buffer.pop(0)
        
        bh = BufferHandler(self.log_buffer)
        bh.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(bh)
        logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
        return logger

    def update_dashboard(self, layout: Layout):
        layout["header"].update(Panel(f"[bold cyan]FinSentinel v4.1[/bold cyan] | 状态: [green]{self.system_status}[/green]", border_style="blue"))
        layout["logs"].update(Panel("\n".join(self.log_buffer), title="爬虫与 NLP 流水线日志"))
        
        table = Table(expand=True)
        table.add_column("股票代码")
        table.add_column("采样篇数", justify="right")
        table.add_column("生成句子数", justify="right")
        for code, s in self.stats.items():
            table.add_row(code, str(s['articles']), str(s['sentences']))
        layout["stats"].update(Panel(table, title="实时统计"))

    def run_task(self):
        self.logger.info(f">>> 启动资讯采集任务")
        self.system_status = "执行中"
        for code in self.stock_codes:
            raw_df = self.crawler.fetch_page(code)
            if not raw_df.empty:
                self.stats[code]["articles"] = len(raw_df)
                self.logger.info(f"开始 NLP 句子级处理: {code}")
                final_df = NLPPipeline.process(raw_df)
                self.stats[code]["sentences"] = len(final_df)
                
                date_str = datetime.datetime.now().strftime("%Y%m%d")
                save_dir = os.path.join(DATA_DIR, date_str)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                output_path = os.path.join(save_dir, f"news_{code}_{date_str}.csv")
                final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"导出成功: {output_path}")
            
        self.total_runs += 1
        self.system_status = "待机"
        self.logger.info("<<< 任务轮次结束")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", type=str, default=",".join(DEFAULT_STOCK_LIST))
    args = parser.parse_args()
    
    codes = args.codes.split(",")
    sentinel = FinSentinel(codes)
    schedule.every().day.at("15:30").do(sentinel.run_task)
    
    layout = Layout()
    layout.split(Layout(name="header", size=3), Layout(name="body", ratio=1))
    layout["body"].split_row(Layout(name="logs", ratio=1), Layout(name="stats", ratio=1))
    
    with Live(layout, refresh_per_second=2):
        import threading
        threading.Thread(target=sentinel.run_task, daemon=True).start()
        try:
            while True:
                schedule.run_pending()
                sentinel.update_dashboard(layout)
                time.sleep(1)
        except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
