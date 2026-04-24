import os
from pathlib import Path

# ==========================================
# 1. 路径锚点 (Path Anchoring)
# ==========================================
# 自动定位项目根目录 (FinSentinel/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "analysis_reports"
MODEL_DIR = BASE_DIR / "finbert_local"

# 确保核心目录存在
for path in [DATA_DIR, LOG_DIR, REPORT_DIR]:
    path.mkdir(exist_ok=True)

# ==========================================
# 2. 核心配置 (Core Config)
# ==========================================
# 从环境变量获取 API KEY，默认占位符供上传使用
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")

# 模型相关配置
FINBERT_PATH = str(MODEL_DIR)

# ==========================================
# 3. 数据集配置 (Dataset Config)
# ==========================================
FULL_DATASET_PATH = REPORT_DIR / "full_dataset.csv"
SIGNAL_OUTPUT_DIR = REPORT_DIR / "sentiment_price_analysis"
SIGNAL_OUTPUT_DIR.mkdir(exist_ok=True)
