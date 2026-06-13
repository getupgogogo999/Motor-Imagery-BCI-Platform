"""
项目配置文件 - 路径、标签映射、EEG 参数等常量
"""
from pathlib import Path

# 项目根目录（本文件所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# 默认数据集路径（合并版 CSV）
DEFAULT_DATA_PATH = PROJECT_ROOT / "BCICIV_2a_all_patients.csv"

# 分患者 CSV 目录
PATIENTS_DIR = PROJECT_ROOT / "patients (1)" / "patients"

# BCI Competition IV 2a 原始 GDF 目录
GDF_DIR = PROJECT_ROOT / "BCICIV_2a_gdf"

# 模型与输出目录
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "motor_imagery_model.pkl"

# GDF 运动想象事件码 -> 标签名
MI_EVENT_CODES = {
    769: "left",
    770: "right",
    771: "foot",
    772: "tongue",
}

# GDF epoch 参数（cue 后 0.5~2.5 秒为经典 MI 窗口）
GDF_EPOCH_TMIN = 0.5
GDF_EPOCH_TMAX = 2.5
GDF_BANDPASS = (8.0, 30.0)
GDF_CSP_COMPONENTS = 6

# 可能的标签列名（CSV 用）
LABEL_COLUMN_CANDIDATES = ["label", "class", "target", "y", "category"]

# 原始标签 -> 可读名称
LABEL_DISPLAY_NAMES = {
    "left": "Left Hand",
    "right": "Right Hand",
    "foot": "Feet",
    "tongue": "Tongue",
}

# 可读名称 -> 控制命令
COMMAND_MAPPING = {
    "Left Hand": "Move Left",
    "Right Hand": "Move Right",
    "Feet": "Move Forward",
    "Tongue": "Select",
}

# 训练参数
RANDOM_STATE = 42
TEST_SIZE = 0.2

# CSV EEG 信号参数
SAMPLE_RATE = 250
MI_TIME_MIN = 0.0

FREQUENCY_BANDS = [
    ("mu", 8, 13),
    ("beta", 13, 30),
    ("mu_beta", 8, 30),
]

EXCLUDED_PATIENT_IDS = {4}
